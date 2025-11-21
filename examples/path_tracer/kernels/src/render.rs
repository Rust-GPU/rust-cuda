use crate::*;
use cuda_std::glam::UVec2;

const BACKGROUND_BLUE_MULTIPLIER: f32 = 0.7;

pub fn color(ray: Ray) -> Vec3 {
    let unit = ray.dir.normalize();
    let t = BACKGROUND_BLUE_MULTIPLIER * (unit.y + 1.0);
    (1.0 - t) * Vec3::ONE + t * Vec3::new(0.5, 0.7, 1.0)
}

pub fn generate_ray(idx: UVec2, view: &Viewport, offset: Vec2) -> Ray {
    let uv = (idx.as_vec2() + offset) / view.bounds.as_vec2();
    Ray {
        origin: view.origin,
        dir: view.lower_left + uv.x * view.horizontal + uv.y * view.vertical - view.origin,
    }
}
