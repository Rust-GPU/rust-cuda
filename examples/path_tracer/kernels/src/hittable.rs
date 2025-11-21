use crate::{Ray, Vec3};
use enum_dispatch::enum_dispatch;

#[derive(Clone, Copy, PartialEq)]
pub struct HitRecord {
    pub material_handle: usize,
    pub t: f32,
    pub point: Vec3,
    pub normal: Vec3,
}

#[enum_dispatch]
pub trait Hittable {
    fn material(&self) -> usize;
    fn hit(&self, ray: Ray, t_min: f32, t_max: f32) -> Option<HitRecord>;
}
