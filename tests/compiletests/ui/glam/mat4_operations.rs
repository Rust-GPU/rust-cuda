// build-pass

// This test verifies glam Mat4 operations work correctly in CUDA kernels

use cuda_std::glam::{Mat4, Vec3, Vec4};
use cuda_std::kernel;

#[kernel]
pub unsafe fn mat4_transform_operations(
    matrix: Mat4,
    point: Vec3,
    vector: Vec4,
    result_point: *mut Vec3,
    result_vector: *mut Vec4,
    result_determinant: *mut f32,
) {
    // Transform a 3D point (w=1 implied)
    let transformed_point = matrix.transform_point3(point);
    unsafe {
        *result_point = transformed_point;
    }

    // Transform a 4D vector
    let transformed_vector = matrix * vector;
    unsafe {
        *result_vector = transformed_vector;
    }

    // Calculate determinant
    let det = matrix.determinant();
    unsafe {
        *result_determinant = det;
    }
}

#[kernel]
pub unsafe fn mat4_construction(
    translation: Vec3,
    scale: Vec3,
    angle_radians: f32,
    axis: Vec3,
    result_translation: *mut Mat4,
    result_scale: *mut Mat4,
    result_rotation: *mut Mat4,
) {
    // Create translation matrix
    let trans_mat = Mat4::from_translation(translation);
    unsafe {
        *result_translation = trans_mat;
    }

    // Create scale matrix
    let scale_mat = Mat4::from_scale(scale);
    unsafe {
        *result_scale = scale_mat;
    }

    // Create rotation matrix
    let rot_mat = Mat4::from_axis_angle(axis, angle_radians);
    unsafe {
        *result_rotation = rot_mat;
    }
}
