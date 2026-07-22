// build-pass
// compile-flags: -Cllvm-args=--disassemble-entry=leading_zeros_u64 --error-format=human -Cdebuginfo=0

// Regression test: like `ctpop`, `core::intrinsics::ctlz` returns `u32` for every operand
// width, so the `i64` result of `llvm.ctlz.i64` has to be truncated before it is stored into
// the `u32` result slot. Without that truncation the oversized store is dropped and the
// `clz` is eliminated as dead, so `leading_zeros` silently returns `0` on device.
//
// The kernel below must contain a `clz.b64`.

use cuda_std::kernel;

#[kernel]
pub unsafe fn leading_zeros_u64(buffer: *const u64, out: *mut u32) {
    unsafe {
        *out = (*buffer).leading_zeros();
    }
}
