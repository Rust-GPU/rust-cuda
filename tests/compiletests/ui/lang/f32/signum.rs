// Test that `signum` works.
// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_signum(i: f32, o: *mut f32) {
    unsafe {
        *o = i.signum();
    }
}
