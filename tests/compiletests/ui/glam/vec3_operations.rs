// build-pass

// This test verifies glam Vec3 operations work correctly in CUDA kernels

use cuda_std::glam::Vec3;
use cuda_std::kernel;

#[kernel]
pub unsafe fn vec3_basic_ops(
    a: Vec3,
    b: Vec3,
    result_add: *mut Vec3,
    result_dot: *mut f32,
    result_cross: *mut Vec3,
) {
    // Vector addition
    let sum = a + b;
    unsafe {
        *result_add = sum;
    }

    // Dot product
    let dot = a.dot(b);
    unsafe {
        *result_dot = dot;
    }

    // Cross product
    let cross = a.cross(b);
    unsafe {
        *result_cross = cross;
    }
}

#[kernel]
pub unsafe fn vec3_normalization(
    input: Vec3,
    result_normalized: *mut Vec3,
    result_length: *mut f32,
) {
    // Get length
    let len = input.length();
    unsafe {
        *result_length = len;
    }

    // Normalize
    let normalized = input.normalize();
    unsafe {
        *result_normalized = normalized;
    }
}
