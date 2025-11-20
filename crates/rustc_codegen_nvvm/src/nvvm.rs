//! Final steps in codegen, coalescing modules and feeding them to libnvvm.

use crate::back::demangle_callback;
use crate::builder::unnamed;
use crate::common::AsCCharPtr;
use crate::context::CodegenArgs;
use crate::llvm::*;
use crate::lto::ThinBuffer;
use nvvm::*;
use rustc_codegen_ssa::traits::ThinBufferMethods;
use rustc_session::{Session, config::DebugInfo};
use std::fmt::Display;
use std::marker::PhantomData;
use std::ptr;
use tracing::debug;

// see libintrinsics.ll on what this is.
const LIBINTRINSICS: &[u8] = include_bytes!("../libintrinsics.bc");

pub enum CodegenErr {
    Nvvm(NvvmError),
    Io(std::io::Error),
}

impl From<std::io::Error> for CodegenErr {
    fn from(v: std::io::Error) -> Self {
        Self::Io(v)
    }
}

impl From<NvvmError> for CodegenErr {
    fn from(v: NvvmError) -> Self {
        Self::Nvvm(v)
    }
}

impl Display for CodegenErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nvvm(err) => std::fmt::Display::fmt(&err, f),
            Self::Io(err) => std::fmt::Display::fmt(&err, f),
        }
    }
}

/// Take a list of bitcode module bytes and their names and codegen it
/// into PTX bytes. The final PTX *should* be utf8, but just to be on the safe side
/// it returns a vector of bytes.
///
/// Note that this will implicitly try to find libdevice and add it, so don't do that
/// step before this. It will fatal error if it cannot find it.
pub fn codegen_bitcode_modules(
    args: &CodegenArgs,
    sess: &Session,
    modules: Vec<Vec<u8>>,
    llcx: &Context,
) -> Result<Vec<u8>, CodegenErr> {
    debug!("Codegenning bitcode to PTX");

    // Make sure the nvvm version is high enough so users don't get confusing compilation errors.
    let (major, minor) = nvvm::ir_version();

    if major <= 2 && minor < 0 {
        sess.dcx()
            .fatal("rustc_codegen_nvvm requires at least libnvvm 2.0 (CUDA 12.0)");
    }

    // First, create the nvvm program we will add modules to.
    let prog = NvvmProgram::new()?;

    let module = merge_llvm_modules(modules, llcx);
    unsafe {
        internalize_pass(module, llcx);
        dce_pass(module);

        if sess.opts.debuginfo != DebugInfo::None {
            cleanup_dicompileunit(module);
        }

        let (dbg_major, dbg_minor) = nvvm::dbg_version();

        // needed for debug info or else nvvm complains about ir version mismatch for some
        // reason. It works if you don't use debug info though...
        let ty_i32 = LLVMInt32TypeInContext(llcx);
        let major = LLVMConstInt(ty_i32, major as u64, False);
        let minor = LLVMConstInt(ty_i32, minor as u64, False);
        let dbg_major = LLVMConstInt(ty_i32, dbg_major as u64, False);
        let dbg_minor = LLVMConstInt(ty_i32, dbg_minor as u64, False);
        let vals = [major, minor, dbg_major, dbg_minor];
        let node = LLVMMDNodeInContext(llcx, vals.as_ptr(), vals.len() as u32);

        LLVMAddNamedMetadataOperand(module, c"nvvmir.version".as_ptr().cast(), node);

        if let Some(path) = &args.final_module_path {
            let out = path.to_str().unwrap();
            let result =
                LLVMRustPrintModule(module, out.as_c_char_ptr(), out.len(), demangle_callback);
            result
                .into_result()
                .expect("Failed to write final llvm module output");
        }
    }

    let buf = ThinBuffer::new(module);

    prog.add_module(buf.data(), "merged".to_string())?;
    prog.add_lazy_module(LIBDEVICE_BITCODE, "libdevice".to_string())?;
    prog.add_lazy_module(LIBINTRINSICS, "libintrinsics".to_string())?;

    // for now, while the codegen is young, we always run verification on the program.
    // This is to make debugging much easier, libnvvm tends to infinitely loop or segfault on invalid programs
    // which makes debugging extremely hard. This way, if a malformed program is created, it is caught before
    // giving it to libnvvm. Then to debug codegen failures, we can just ask the user to provide the corresponding llvm ir
    // file with --emit=llvm-ir

    let verification_res = prog.verify();
    if verification_res.is_err() {
        let log = prog.compiler_log().unwrap().unwrap_or_default();
        let footer = "If you plan to submit a bug report please re-run the codegen with `RUSTFLAGS=\"--emit=llvm-ir\" and include the .ll file corresponding to the .o file mentioned in the log";
        panic!(
            "Malformed NVVM IR program rejected by libnvvm, dumping verifier log:\n\n{log}\n\n{footer}"
        );
    }

    let res = match prog.compile(&args.nvvm_options) {
        Ok(b) => b,
        Err(error) => {
            let log = prog.compiler_log().unwrap().unwrap_or_default();
            // this should never happen, if it does, something went really bad or its a bug on libnvvm's end
            panic!(
                "libnvvm returned an error that was not previously caught by the verifier: {error:?} {log:?}"
            );
        }
    };

    Ok(res)
}

unsafe fn cleanup_dicompileunit(module: &Module) {
    unsafe {
        let mut cu1 = ptr::null_mut();
        let mut cu2 = ptr::null_mut();
        LLVMRustThinLTOGetDICompileUnit(module, &mut cu1, &mut cu2);
        LLVMRustThinLTOPatchDICompileUnit(module, cu1);
    }
}

// Merging and DCE (dead code elimination) logic. Inspired a lot by rust-ptx-linker.
//
// This works in a couple of steps starting from the bitcode of every single module (crate), then:
// - Merge all of the modules into a single large module, basically fat LTO. In the future we could probably lazily-load only
// the things we need using dependency graphs, like we used to do for libnvvm.
// - Iterate over every function in the module and:
//      - If it is not a kernel and it is not a declaration (i.e. an extern fn) then mark its linkage as internal and its visiblity as default
// - Iterate over every global in the module and:
//      - Same as functions, if it is not an external declaration, mark it as internal.
// - run LLVM's global DCE pass, this will remove any functions and globals that are not directly or indirectly used by kernels.

fn merge_llvm_modules(modules: Vec<Vec<u8>>, llcx: &Context) -> &Module {
    let module = unsafe { crate::create_module(llcx, "merged_modules") };
    for merged_module in modules {
        unsafe {
            let tmp = LLVMRustParseBitcodeForLTO(
                llcx,
                merged_module.as_ptr(),
                merged_module.len(),
                unnamed(),
                0,
            )
            .expect("Failed to parse module bitcode");
            LLVMLinkModules2(module, tmp);
        }
    }
    module
}

struct FunctionIter<'a, 'll> {
    module: PhantomData<&'a &'ll Module>,
    next: Option<&'ll Value>,
}

struct GlobalIter<'a, 'll> {
    module: PhantomData<&'a &'ll Module>,
    next: Option<&'ll Value>,
}

impl<'a, 'll> FunctionIter<'a, 'll> {
    pub fn new(module: &'a &'ll Module) -> Self {
        FunctionIter {
            module: PhantomData,
            next: unsafe { LLVMGetFirstFunction(module) },
        }
    }
}

impl<'ll> Iterator for FunctionIter<'_, 'll> {
    type Item = &'ll Value;

    fn next(&mut self) -> Option<&'ll Value> {
        let next = self.next;

        self.next = match next {
            Some(next) => unsafe { LLVMGetNextFunction(next) },
            None => None,
        };

        next
    }
}

impl<'a, 'll> GlobalIter<'a, 'll> {
    pub fn new(module: &'a &'ll Module) -> Self {
        GlobalIter {
            module: PhantomData,
            next: unsafe { LLVMGetFirstGlobal(module) },
        }
    }
}

impl<'ll> Iterator for GlobalIter<'_, 'll> {
    type Item = &'ll Value;

    fn next(&mut self) -> Option<&'ll Value> {
        let next = self.next;

        self.next = match next {
            Some(next) => unsafe { LLVMGetNextGlobal(next) },
            None => None,
        };

        next
    }
}

unsafe fn internalize_pass(module: &Module, cx: &Context) {
    unsafe {
        // collect the values of all the declared kernels
        let num_operands =
            LLVMGetNamedMetadataNumOperands(module, c"nvvm.annotations".as_ptr().cast()) as usize;
        let mut operands = Vec::with_capacity(num_operands);
        LLVMGetNamedMetadataOperands(
            module,
            c"nvvm.annotations".as_ptr().cast(),
            operands.as_mut_ptr(),
        );
        operands.set_len(num_operands);
        let mut kernels = Vec::with_capacity(num_operands);
        let kernel_str = LLVMMDStringInContext(cx, "kernel".as_ptr().cast(), 6);

        for mdnode in operands {
            let num_operands = LLVMGetMDNodeNumOperands(mdnode) as usize;
            let mut operands = Vec::with_capacity(num_operands);
            LLVMGetMDNodeOperands(mdnode, operands.as_mut_ptr());
            operands.set_len(num_operands);

            if operands.get(1) == Some(&kernel_str) {
                kernels.push(operands[0]);
            }
        }

        // see what functions are marked as externally visible by the user.
        let num_operands =
            LLVMGetNamedMetadataNumOperands(module, c"cg_nvvm_used".as_ptr().cast()) as usize;
        let mut operands = Vec::with_capacity(num_operands);
        LLVMGetNamedMetadataOperands(
            module,
            c"cg_nvvm_used".as_ptr().cast(),
            operands.as_mut_ptr(),
        );
        operands.set_len(num_operands);
        let mut used_funcs = Vec::with_capacity(num_operands);

        for mdnode in operands {
            let num_operands = LLVMGetMDNodeNumOperands(mdnode) as usize;
            let mut operands = Vec::with_capacity(num_operands);
            LLVMGetMDNodeOperands(mdnode, operands.as_mut_ptr());
            operands.set_len(num_operands);

            used_funcs.push(operands[0]);
        }

        let iter = FunctionIter::new(&module);
        for func in iter {
            let is_kernel = kernels.contains(&func);
            let is_decl = LLVMIsDeclaration(func) == True;
            let is_used = used_funcs.contains(&func);

            if !is_decl && !is_kernel {
                LLVMRustSetLinkage(func, Linkage::InternalLinkage);
                LLVMRustSetVisibility(func, Visibility::Default);
            }

            // explicitly set it to external just in case the codegen set them to internal for some reason
            if is_used {
                LLVMRustSetLinkage(func, Linkage::ExternalLinkage);
                LLVMRustSetVisibility(func, Visibility::Default);
            }
        }

        let iter = GlobalIter::new(&module);
        for func in iter {
            let is_decl = LLVMIsDeclaration(func) == True;

            if !is_decl {
                LLVMRustSetLinkage(func, Linkage::InternalLinkage);
                LLVMRustSetVisibility(func, Visibility::Default);
            }
        }
    }
}

unsafe fn dce_pass(module: &Module) {
    unsafe {
        let pass_manager = LLVMCreatePassManager();

        LLVMAddGlobalDCEPass(pass_manager);

        LLVMRunPassManager(pass_manager, module);
        LLVMDisposePassManager(pass_manager);
    }
}
