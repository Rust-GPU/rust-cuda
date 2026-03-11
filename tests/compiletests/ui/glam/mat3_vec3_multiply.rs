// Tests multiplying a `Mat3` by a `Vec3`.
// build-pass

use cuda_std::glam;
use cuda_std::kernel;

#[kernel]
pub unsafe fn mat3_vec3_multiply(input: glam::Mat3, output: *mut glam::Vec3) {
    let vector = input * glam::Vec3::new(1.0, 2.0, 3.0);
    unsafe {
        *output = vector;
    }
}
