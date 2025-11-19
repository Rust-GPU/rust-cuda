use crate::llvm::{self, Type};
use rustc_target::spec::{
    LinkerFlavor, MergeFunctions, PanicStrategy, Target, TargetMetadata, TargetOptions,
};

// This data layout must match `datalayout` in `crates/rustc_codegen_nvvm/libintrinsics.ll`.
pub const DATA_LAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64";
pub const TARGET_TRIPLE: &str = "nvptx64-nvidia-cuda";
pub const POINTER_WIDTH: u32 = 64;

/// The pointer width of the current target
pub(crate) unsafe fn usize_ty(llcx: &'_ llvm::Context) -> &'_ Type {
    unsafe { llvm::LLVMInt64TypeInContext(llcx) }
}

pub fn target() -> Target {
    let mut options = TargetOptions::default();

    options.os = "cuda".into();
    options.vendor = "nvidia".into();
    options.linker_flavor = LinkerFlavor::Ptx;
    // nvvm does all the linking for us, but technically its not a linker
    options.linker = None;
    options.cpu = "sm_30".into();
    options.max_atomic_width = Some(64);
    // Unwinding on CUDA is neither feasible nor useful.
    options.panic_strategy = PanicStrategy::Abort;
    // Needed to use `dylib` and `bin` crate types and the linker.
    options.dynamic_linking = true;
    options.executables = true;
    options.only_cdylib = true;

    // nvvm does all the work of turning the bitcode into ptx
    options.obj_is_bitcode = true;

    options.dll_prefix = "".into();
    options.dll_suffix = ".ptx".into();
    options.exe_suffix = ".ptx".into();

    // Disable MergeFunctions LLVM optimisation pass because it can
    // produce kernel functions that call other kernel functions.
    // This behavior is not supported by PTX ISA.
    options.merge_functions = MergeFunctions::Disabled;

    Target {
        arch: "nvptx".into(),
        data_layout: DATA_LAYOUT.into(),
        llvm_target: "nvptx64-nvidia-cuda".into(),
        pointer_width: POINTER_WIDTH,
        options,
        metadata: TargetMetadata {
            description: Some("NVIDIA CUDA".into()),
            ..Default::default()
        },
    }
}
