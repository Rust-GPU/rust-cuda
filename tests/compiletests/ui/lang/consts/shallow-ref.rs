// Test `&'static T` constants where the `T` values don't themselves contain
// references, and where the `T` values aren't immediately loaded from.

// build-pass

use cuda_std::glam::{Mat2, Vec2};
use cuda_std::kernel;

#[inline(never)]
fn scalar_load(r: &'static u32) -> u32 {
    *r
}

const ROT90: Mat2 = Mat2::from_cols_array_2d(&[[0.0, 1.0], [-1.0, 0.0]]);

#[kernel]
pub unsafe fn test_shallow_ref(
    scalar_out: *mut u32,
    vec_in: Vec2,
    bool_out: *mut u32,
    vec_out: *mut Vec2,
) {
    unsafe {
        *scalar_out = scalar_load(&123);
        *bool_out = (vec_in == Vec2::ZERO) as u32;
        *vec_out = ROT90.transpose() * vec_in;
    }
}
