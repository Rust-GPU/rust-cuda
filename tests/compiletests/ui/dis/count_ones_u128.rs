// build-pass
// compile-flags: -Cllvm-args=--disassemble-entry=count_ones_u128 --error-format=human -Cdebuginfo=0

// Regression test for the emulated 128-bit path: `handle_128_bit_intrinsic` returns an `i128`,
// which must be truncated to the `u32` that `core::intrinsics::ctpop` is declared to return.
// Without the truncation the 16-byte store into the 4-byte result slot is discarded and both
// `popc`s are eliminated as dead, so `u128::count_ones` silently returns `0` on device.
//
// The kernel below must contain two `popc.b64`s and the `add` that combines them.

use cuda_std::kernel;

#[kernel]
pub unsafe fn count_ones_u128(buffer: *const u128, out: *mut u32) {
    unsafe {
        *out = (*buffer).count_ones();
    }
}
