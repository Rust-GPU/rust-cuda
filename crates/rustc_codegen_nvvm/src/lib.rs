#![feature(rustc_private)]
// crate is perma-unstable because of rustc_private so might as well
// make our lives a lot easier for llvm ffi with this. And since rustc's core infra
// relies on it its almost guaranteed to not be removed/broken
#![feature(extern_types)]
#![feature(slice_as_array)]

extern crate rustc_abi;
extern crate rustc_arena;
extern crate rustc_ast;
extern crate rustc_attr_data_structures;
extern crate rustc_attr_parsing;
extern crate rustc_codegen_ssa;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_fs_util;
extern crate rustc_hash;
extern crate rustc_hashes;
extern crate rustc_hir;
extern crate rustc_index;
extern crate rustc_interface;
extern crate rustc_macros;
extern crate rustc_metadata;
extern crate rustc_middle;
extern crate rustc_query_system;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_symbol_mangling;
extern crate rustc_target;
extern crate rustc_type_ir;

mod abi;
mod allocator;
mod asm;
mod attributes;
mod back;
mod builder;
mod common;
mod const_ty;
mod consts;
mod context;
mod ctx_intrinsics;
mod debug_info;
mod init;
mod int_replace;
mod intrinsic;
mod link;
mod llvm;
mod lto;
mod mono_item;
mod nvvm;
mod override_fns;
mod ptx_filter;
mod target;
mod ty;

use abi::readjust_fn_abi;
use back::target_machine_factory;
use lto::ThinBuffer;
use rustc_ast::expand::allocator::AllocatorKind;
use rustc_ast::expand::autodiff_attrs::AutoDiffItem;
use rustc_codegen_ssa::{
    CodegenResults, CompiledModule, ModuleCodegen, TargetConfig,
    back::{
        lto::{SerializedModule, ThinModule},
        write::{CodegenContext, FatLtoInput, ModuleConfig, OngoingCodegen},
    },
    traits::{CodegenBackend, ExtraBackendMethods, WriteBackendMethods},
};
use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::{DiagCtxtHandle, FatalError};
use rustc_metadata::creader::MetadataLoaderDyn;
use rustc_middle::util::Providers;
use rustc_middle::{
    dep_graph::{WorkProduct, WorkProductId},
    ty::TyCtxt,
};
use rustc_session::{
    Session,
    config::{self, OutputFilenames},
};
use tracing::debug;

use std::ffi::CString;
use std::path::PathBuf;

// codegen dylib entrypoint
#[unsafe(no_mangle)]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    rustc_driver::install_ice_hook(
        "https://github.com/Rust-GPU/rust-cuda/issues/new",
        |handler| {
            handler.handle().note(concat!(
                "`rust-cuda` version `",
                env!("CARGO_PKG_VERSION"),
                "`"
            ));
        },
    );
    Box::new(NvvmCodegenBackend)
}

#[derive(Clone)]
pub struct NvvmCodegenBackend;

unsafe impl Send for NvvmCodegenBackend {}
unsafe impl Sync for NvvmCodegenBackend {}

impl CodegenBackend for NvvmCodegenBackend {
    fn locale_resource(&self) -> &'static str {
        ""
    }

    fn init(&self, sess: &Session) {
        let filter = tracing_subscriber::EnvFilter::from_env("NVVM_LOG");
        let subscriber = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .without_time()
            .with_ansi(false)
            .compact()
            .finish();

        tracing::subscriber::set_global_default(subscriber).expect("no default subscriber");
        init::init(sess);
    }

    // FIXME If we can use the default metadata loader in the LLVM backend
    // we can remove this and use the default provided impl instead.
    fn metadata_loader(&self) -> Box<MetadataLoaderDyn> {
        Box::new(link::NvvmMetadataLoader)
    }

    fn provide(&self, providers: &mut Providers) {
        // Synthesize compute capability target features from the architecture specified in llvm-args.
        // This enables code to use `#[cfg(target_feature = "compute_60")]` etc. for conditional compilation.
        // Following NVIDIA semantics, we enable "at least this capability" matching - for example,
        // when targeting compute_70, we also enable compute_60, compute_50, and all lower capabilities.
        // This allows libraries to gate features based on minimum required compute capability.
        providers.global_backend_features = |tcx, ()| {
            let mut features = vec![];

            // Parse CodegenArgs to get the architecture from llvm-args (e.g., "-arch=compute_70")
            let args = context::CodegenArgs::from_session(tcx.sess);

            // Find the architecture option and synthesize all implied features
            for opt in &args.nvvm_options {
                if let ::nvvm::NvvmOption::Arch(arch) = opt {
                    // Add all features up to and including the current architecture
                    features.extend(arch.all_target_features());
                    break;
                }
            }

            features
        };

        providers.fn_abi_of_fn_ptr = |tcx, key| {
            let result = (rustc_interface::DEFAULT_QUERY_PROVIDERS.fn_abi_of_fn_ptr)(tcx, key);
            Ok(readjust_fn_abi(tcx, result?))
        };
        providers.fn_abi_of_instance = |tcx, key| {
            let result = (rustc_interface::DEFAULT_QUERY_PROVIDERS.fn_abi_of_instance)(tcx, key);
            Ok(readjust_fn_abi(tcx, result?))
        };
    }

    fn codegen_crate(&self, tcx: TyCtxt<'_>) -> Box<dyn std::any::Any> {
        debug!("Codegen crate");
        Box::new(rustc_codegen_ssa::base::codegen_crate(
            Self,
            tcx,
            tcx.sess
                .opts
                .cg
                .target_cpu
                .clone()
                .unwrap_or_else(|| tcx.sess.target.cpu.to_string()),
        ))
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn std::any::Any>,
        sess: &Session,
        _outputs: &OutputFilenames,
    ) -> (CodegenResults, FxIndexMap<WorkProductId, WorkProduct>) {
        debug!("Join codegen");
        let (codegen_results, work_products) = ongoing_codegen
            .downcast::<OngoingCodegen<Self>>()
            .expect("Expected OngoingCodegen, found Box<Any>")
            .join(sess);

        // sess.compile_status();

        (codegen_results, work_products)
    }

    fn link(
        &self,
        sess: &rustc_session::Session,
        codegen_results: rustc_codegen_ssa::CodegenResults,
        metadata: rustc_metadata::EncodedMetadata,
        outputs: &config::OutputFilenames,
    ) {
        link::link(
            sess,
            &codegen_results,
            outputs,
            codegen_results.crate_info.local_crate_name.as_str(),
            metadata,
        );
    }

    fn target_config(&self, sess: &Session) -> TargetConfig {
        // Parse target features from command line
        let cmdline = sess.opts.cg.target_feature.split(',');
        let cfg = sess.target.options.features.split(',');

        let mut target_features: Vec<_> = cfg
            .chain(cmdline)
            .filter(|l| l.starts_with('+'))
            .map(|l| &l[1..])
            .filter(|l| !l.is_empty())
            .map(rustc_span::Symbol::intern)
            .collect();

        // Add backend-synthesized features (e.g., hierarchical compute capabilities)
        // Parse CodegenArgs to get the architecture from llvm-args
        let args = context::CodegenArgs::from_session(sess);
        for opt in &args.nvvm_options {
            if let ::nvvm::NvvmOption::Arch(arch) = opt {
                // Add all features up to and including the current architecture
                let backend_features = arch.all_target_features();
                target_features.extend(
                    backend_features
                        .iter()
                        .map(|f| rustc_span::Symbol::intern(f)),
                );
                break;
            }
        }

        // For NVPTX, all target features are stable
        let unstable_target_features = target_features.clone();

        TargetConfig {
            target_features,
            unstable_target_features,
            has_reliable_f16: false,
            has_reliable_f16_math: false,
            has_reliable_f128: false,
            has_reliable_f128_math: false,
        }
    }
}

impl WriteBackendMethods for NvvmCodegenBackend {
    type Module = LlvmMod;
    type ModuleBuffer = lto::ModuleBuffer;
    type TargetMachine = &'static mut llvm::TargetMachine;
    type TargetMachineError = String;
    type ThinData = ();
    type ThinBuffer = ThinBuffer;

    fn run_link(
        _cgcx: &CodegenContext<Self>,
        _diag_handler: DiagCtxtHandle<'_>,
        _modules: Vec<ModuleCodegen<Self::Module>>,
    ) -> Result<ModuleCodegen<Self::Module>, FatalError> {
        // TODO(Rdambrosio016):
        // we can probably call the llvm codegen to do this, but cgcx
        // is a codegen context of NvvmCodegenBackend not LlvmCodegenBackend
        // and to make a new cgcx we need to make a new LlvmCodegenBackend which
        // cannot be done through the API currently
        todo!();
    }

    fn run_and_optimize_fat_lto(
        _cgcx: &CodegenContext<Self>,
        _exported_symbols_for_lto: &[String],
        _each_linked_rlib_for_lto: &[PathBuf],
        _modules: Vec<FatLtoInput<Self>>,
        _diff_fncs: Vec<AutoDiffItem>,
    ) -> Result<ModuleCodegen<Self::Module>, FatalError> {
        todo!()
    }

    fn run_thin_lto(
        cgcx: &CodegenContext<Self>,
        // FIXME: Limit LTO exports to these symbols
        _exported_symbols_for_lto: &[String],
        // FIXME: handle these? but only relevant for non-thin LTO?
        _each_linked_rlib_for_lto: &[PathBuf],
        modules: Vec<(String, Self::ThinBuffer)>,
        cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>,
    ) -> Result<(Vec<ThinModule<Self>>, Vec<WorkProduct>), FatalError> {
        lto::run_thin(cgcx, modules, cached_modules)
    }

    fn print_pass_timings(&self) {
        // Not applicable, nvvm doesnt expose pass timing info, maybe we could print llvm pass stuff here.
    }

    fn print_statistics(&self) {
        // Not applicable, nvvm doesnt expose pass timing info, maybe we could print llvm pass stuff here.
    }

    fn optimize(
        cgcx: &CodegenContext<Self>,
        diag_handler: DiagCtxtHandle<'_>,
        module: &mut ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
    ) -> Result<(), FatalError> {
        unsafe { back::optimize(cgcx, diag_handler, module, config) }
    }

    fn optimize_thin(
        cgcx: &CodegenContext<Self>,
        thin_module: ThinModule<Self>,
    ) -> Result<ModuleCodegen<Self::Module>, FatalError> {
        unsafe { lto::optimize_thin(cgcx, thin_module) }
    }

    fn codegen(
        cgcx: &CodegenContext<Self>,
        module: ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
    ) -> Result<CompiledModule, FatalError> {
        unsafe { back::codegen(cgcx, module, config) }
    }

    fn prepare_thin(
        module: ModuleCodegen<Self::Module>,
        _want_summary: bool,
    ) -> (String, Self::ThinBuffer) {
        debug!("Prepare thin");
        unsafe {
            (
                module.name,
                lto::ThinBuffer::new(module.module_llvm.llmod.as_ref().unwrap()),
            )
        }
    }

    fn serialize_module(module: ModuleCodegen<Self::Module>) -> (String, Self::ModuleBuffer) {
        debug!("Serializing module");
        unsafe {
            (
                module.name,
                lto::ModuleBuffer::new(module.module_llvm.llmod.as_ref().unwrap()),
            )
        }
    }
}

impl ExtraBackendMethods for NvvmCodegenBackend {
    fn codegen_allocator(
        &self,
        tcx: TyCtxt<'_>,
        module_name: &str,
        kind: AllocatorKind,
        alloc_error_handler_kind: AllocatorKind,
    ) -> LlvmMod {
        let mut module_llvm = LlvmMod::new(module_name);
        unsafe {
            allocator::codegen(
                tcx,
                &mut module_llvm,
                module_name,
                kind,
                alloc_error_handler_kind,
            );
        }
        module_llvm
    }

    fn compile_codegen_unit(
        &self,
        tcx: TyCtxt<'_>,
        cgu_name: rustc_span::Symbol,
    ) -> (rustc_codegen_ssa::ModuleCodegen<Self::Module>, u64) {
        back::compile_codegen_unit(tcx, cgu_name)
    }

    fn target_machine_factory(
        &self,
        sess: &Session,
        opt_level: config::OptLevel,
        _target_features: &[String],
    ) -> rustc_codegen_ssa::back::write::TargetMachineFactoryFn<Self> {
        target_machine_factory(sess, opt_level)
    }
}

/// Create the LLVM module for the rest of the compilation, this houses
/// the LLVM bitcode we then add to the NVVM program and feed to libnvvm.
/// LLVM's codegen is never actually called.
pub(crate) unsafe fn create_module<'ll>(
    llcx: &'ll llvm::Context,
    mod_name: &str,
) -> &'ll llvm::Module {
    debug!("Creating llvm module with name `{}`", mod_name);
    let mod_name = CString::new(mod_name).expect("nul in module name");
    let llmod = unsafe { llvm::LLVMModuleCreateWithNameInContext(mod_name.as_ptr(), llcx) };

    let data_layout = CString::new(target::DATA_LAYOUT).unwrap();
    unsafe { llvm::LLVMSetDataLayout(llmod, data_layout.as_ptr()) };

    let target = CString::new(target::TARGET_TRIPLE).unwrap();
    unsafe { llvm::LLVMSetTarget(llmod, target.as_ptr()) };

    llmod
}

/// Wrapper over raw llvm structures
pub struct LlvmMod {
    llcx: &'static mut llvm::Context,
    llmod: *const llvm::Module,
}

unsafe impl Send for LlvmMod {}
unsafe impl Sync for LlvmMod {}

impl LlvmMod {
    pub fn new(name: &str) -> Self {
        unsafe {
            // TODO(RDambrosio016): does shouldDiscardNames affect NVVM at all?
            let llcx = llvm::LLVMRustContextCreate(false);
            let llmod = create_module(llcx, name) as *const _;
            LlvmMod { llcx, llmod }
        }
    }
}

impl Drop for LlvmMod {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMContextDispose(&mut *(self.llcx as *mut _));
        }
    }
}
