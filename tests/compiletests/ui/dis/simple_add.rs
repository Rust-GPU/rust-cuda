// build-pass
// compile-flags: -Cllvm-args=--disassemble-entry=simple_add_kernel --error-format=human

// This test verifies PTX generation for a simple kernel

use cuda_std::kernel;

#[kernel]
pub unsafe fn simple_add_kernel(a: *const f32, b: *const f32, c: *mut f32) {
    unsafe {
        let sum = *a + *b;
        *c = sum;
    }
}
