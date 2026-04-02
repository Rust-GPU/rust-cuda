use crate::llvm::{self, Type};
use rustc_target::spec::{Target, TargetTuple};

// This data layout must match `datalayout` in `crates/rustc_codegen_nvvm/libintrinsics.ll`.
pub const DATA_LAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64";
pub const TARGET_TRIPLE: &str = "nvptx64-nvidia-cuda";
pub const POINTER_WIDTH: u32 = 64;

/// The pointer width of the current target
pub(crate) unsafe fn usize_ty(llcx: &'_ llvm::Context) -> &'_ Type {
    unsafe { llvm::LLVMInt64TypeInContext(llcx) }
}

pub fn target() -> Target {
    let mut target = Target::expect_builtin(&TargetTuple::TargetTuple(TARGET_TRIPLE.into()));
    target.data_layout = DATA_LAYOUT.into();
    target.pointer_width = POINTER_WIDTH as u16;
    target
}
