//! Generic math utilities.

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;
use glam::Vec3;
use gpu_rand::{DefaultRand, GpuRand};

/// Converts a float in the range of [0.0, 1.0] to a range of [-1.0, 1.0].
pub fn norm_f32_to_snorm(x: f32) -> f32 {
    x * 2.0 - 1.0
}

pub fn random_unit_vec(state: &mut DefaultRand) -> Vec3 {
    let [x, y] = state.normal_f32_2();
    let z = state.normal_f32();
    Vec3::new(x, y, z)
}

/// Creates a random vector with each element being in the range of [-1.0, 1.0] (signed normalized).
pub fn random_snorm_vec(state: &mut DefaultRand) -> Vec3 {
    random_unit_vec(state).map(norm_f32_to_snorm)
}

pub fn random_in_unit_sphere(state: &mut DefaultRand) -> Vec3 {
    loop {
        let p = random_snorm_vec(state);
        if p.length_squared() >= 1.0 {
            continue;
        }
        return p;
    }
}

pub fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    v - 2.0 * v.dot(n) * n
}

pub fn refract(v: Vec3, n: Vec3, ni_over_nt: f32) -> Option<Vec3> {
    let uv = v.normalize();
    let dt = uv.dot(n);
    let discriminant = 1.0 - ni_over_nt * ni_over_nt * (1.0 - dt * dt);
    if discriminant > 0.0 {
        Some(ni_over_nt * (uv - n * dt) - discriminant.sqrt() * n)
    } else {
        None
    }
}

pub fn schlick(cos: f32, ref_idx: f32) -> f32 {
    let r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    let r0sq = r0 * r0;
    r0sq + (1.0 - r0sq) * (1.0 - cos).powf(5.0)
}
