// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn add_one(x: *mut f32) {
    unsafe {
        *x = *x + 1.0;
    }
}
