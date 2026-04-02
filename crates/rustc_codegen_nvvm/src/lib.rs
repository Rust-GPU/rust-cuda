#![feature(rustc_private)]
// crate is perma-unstable because of rustc_private so might as well
// make our lives a lot easier for llvm ffi with this. And since rustc's core infra
// relies on it its almost guaranteed to not be removed/broken
#![feature(extern_types)]

extern crate rustc_abi;
extern crate rustc_arena;
extern crate rustc_ast;
extern crate rustc_attr_parsing;
extern crate rustc_codegen_ssa;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_fs_util;
extern crate rustc_hashes;
extern crate rustc_hir;
extern crate rustc_index;
extern crate rustc_interface;
extern crate rustc_macros;
extern crate rustc_metadata;
extern crate rustc_middle;
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
use rustc_ast::expand::allocator::AllocatorMethod;
use rustc_codegen_ssa::{
    CompiledModule, CompiledModules, CrateInfo, ModuleCodegen, TargetConfig,
    back::{
        lto::{SerializedModule, ThinModule},
        write::{CodegenContext, FatLtoInput, ModuleConfig, OngoingCodegen},
    },
    traits::{CodegenBackend, ExtraBackendMethods, WriteBackendMethods},
};
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_errors::DiagCtxtHandle;
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
    fn name(&self) -> &'static str {
        "nvvm"
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
        providers.queries.global_backend_features = |tcx, ()| {
            let mut features = vec![];

            // Parse CodegenArgs to get the architecture from llvm-args (e.g., "-arch=compute_70")
            let args = context::CodegenArgs::from_session(tcx.sess);

            // Find the architecture option and synthesize all implied features
            for opt in &args.nvvm_options {
                if let ::nvvm::NvvmOption::Arch(arch) = opt {
                    // Add all features up to and including the current architecture
                    features.extend(
                        arch.all_target_features()
                            .into_iter()
                            .map(|s| s.target_feature().to_string()),
                    );
                    break;
                }
            }

            features
        };

        providers.queries.fn_abi_of_fn_ptr = |tcx, key| {
            let result = (rustc_interface::DEFAULT_QUERY_PROVIDERS
                .queries
                .fn_abi_of_fn_ptr)(tcx, key);
            Ok(readjust_fn_abi(tcx, result?))
        };
        providers.queries.fn_abi_of_instance_raw = |tcx, key| {
            let result = (rustc_interface::DEFAULT_QUERY_PROVIDERS
                .queries
                .fn_abi_of_instance_raw)(tcx, key);
            Ok(readjust_fn_abi(tcx, result?))
        };
    }

    fn target_cpu(&self, sess: &Session) -> String {
        sess.opts
            .cg
            .target_cpu
            .clone()
            .unwrap_or_else(|| sess.target.cpu.to_string())
    }

    fn codegen_crate(&self, tcx: TyCtxt<'_>, crate_info: &CrateInfo) -> Box<dyn std::any::Any> {
        debug!("Codegen crate");
        Box::new(rustc_codegen_ssa::base::codegen_crate(
            Self, tcx, crate_info,
        ))
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn std::any::Any>,
        sess: &Session,
        _outputs: &OutputFilenames,
    ) -> (CompiledModules, FxIndexMap<WorkProductId, WorkProduct>) {
        debug!("Join codegen");
        let (compiled_modules, work_products) = ongoing_codegen
            .downcast::<OngoingCodegen<Self>>()
            .expect("Expected OngoingCodegen, found Box<Any>")
            .join(sess);

        (compiled_modules, work_products)
    }

    fn link(
        &self,
        sess: &rustc_session::Session,
        compiled_modules: CompiledModules,
        crate_info: CrateInfo,
        metadata: rustc_metadata::EncodedMetadata,
        outputs: &config::OutputFilenames,
    ) {
        link::link(sess, compiled_modules, crate_info, metadata, outputs);
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
                        .map(|f| rustc_span::Symbol::intern(f.target_feature())),
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
    type ThinData = ();

    fn target_machine_factory(
        &self,
        sess: &Session,
        opt_level: config::OptLevel,
        _target_features: &[String],
    ) -> rustc_codegen_ssa::back::write::TargetMachineFactoryFn<Self> {
        target_machine_factory(sess, opt_level)
    }

    fn optimize_and_codegen_fat_lto(
        _cgcx: &CodegenContext,
        _prof: &SelfProfilerRef,
        _shared_emitter: &rustc_codegen_ssa::back::write::SharedEmitter,
        _tm_factory: rustc_codegen_ssa::back::write::TargetMachineFactoryFn<Self>,
        _exported_symbols_for_lto: &[String],
        _each_linked_rlib_for_lto: &[PathBuf],
        _modules: Vec<FatLtoInput<Self>>,
    ) -> CompiledModule {
        todo!()
    }

    fn run_thin_lto(
        cgcx: &CodegenContext,
        _prof: &SelfProfilerRef,
        _dcx: DiagCtxtHandle<'_>,
        _exported_symbols_for_lto: &[String],
        _each_linked_rlib_for_lto: &[PathBuf],
        modules: Vec<(String, Self::ModuleBuffer)>,
        cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>,
    ) -> (Vec<ThinModule<Self>>, Vec<WorkProduct>) {
        lto::run_thin(cgcx, modules, cached_modules)
    }

    fn optimize(
        cgcx: &CodegenContext,
        prof: &SelfProfilerRef,
        shared_emitter: &rustc_codegen_ssa::back::write::SharedEmitter,
        module: &mut ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
    ) {
        unsafe { back::optimize(cgcx, prof, shared_emitter, module, config) }
            .unwrap_or_else(|err| err.raise())
    }

    fn optimize_and_codegen_thin(
        cgcx: &CodegenContext,
        prof: &SelfProfilerRef,
        shared_emitter: &rustc_codegen_ssa::back::write::SharedEmitter,
        tm_factory: rustc_codegen_ssa::back::write::TargetMachineFactoryFn<Self>,
        thin: ThinModule<Self>,
    ) -> CompiledModule {
        unsafe { lto::optimize_and_codegen_thin(cgcx, prof, shared_emitter, tm_factory, thin) }
    }

    fn codegen(
        cgcx: &CodegenContext,
        prof: &SelfProfilerRef,
        shared_emitter: &rustc_codegen_ssa::back::write::SharedEmitter,
        module: ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
    ) -> CompiledModule {
        unsafe { back::codegen(cgcx, prof, shared_emitter, module, config) }
            .unwrap_or_else(|err| err.raise())
    }

    fn serialize_module(module: Self::Module, is_thin: bool) -> Self::ModuleBuffer {
        debug!("Serializing module");
        unsafe { lto::ModuleBuffer::new(module.llmod.as_ref().unwrap(), is_thin) }
    }
}

impl ExtraBackendMethods for NvvmCodegenBackend {
    fn codegen_allocator(
        &self,
        tcx: TyCtxt<'_>,
        module_name: &str,
        methods: &[AllocatorMethod],
    ) -> LlvmMod {
        let mut module_llvm = LlvmMod::new(module_name);
        unsafe {
            allocator::codegen(tcx, &mut module_llvm, module_name, methods);
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
