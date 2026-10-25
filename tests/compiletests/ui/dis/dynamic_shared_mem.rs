// build-pass
// compile-flags: -Cllvm-args=--disassemble-globals --error-format=human -Cdebuginfo=0
// normalize-stderr-test "sm_\d+" -> "sm_XX"

// Regression test: the extern behind `dynamic_shared_mem` must be unsized (`[]`) in the PTX,
// like nvcc's `extern __shared__` arrays. A sized one is laid out among the kernel's static
// shared variables, so the dynamic region overlaps them.

use cuda_std::{kernel, shared};

#[kernel]
pub unsafe fn dynamic_shared(out: *mut u32) {
    unsafe {
        *out = *shared::dynamic_shared_mem::<u32>();
    }
}
