// build-pass
// compile-flags: -Cllvm-args=--disassemble-entry=count_ones_u64 --error-format=human -Cdebuginfo=0

// Regression test: `core::intrinsics::ctpop` returns `u32` for every operand width, so the
// `i64` result of `llvm.ctpop.i64` has to be truncated before it is stored into the `u32`
// result slot. Without that truncation the oversized store is dropped and the `popc` is
// eliminated as dead, so `count_ones` silently returns `0` on device.
//
// The kernel below must contain a `popc.b64`.

use cuda_std::kernel;

#[kernel]
pub unsafe fn count_ones_u64(buffer: *const u64, out: *mut u32) {
    unsafe {
        *out = (*buffer).count_ones();
    }
}
