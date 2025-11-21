use crate::hittable::{HitRecord, Hittable};
use crate::*;
#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;
use cust_core::DeviceCopy;

#[derive(Clone, Copy, PartialEq, DeviceCopy)]
pub struct Sphere {
    pub center: Point,
    pub radius: f32,
    pub mat: usize,
}

impl Sphere {
    pub fn new(center: Point, radius: f32, mat: usize) -> Self {
        Self {
            center,
            radius,
            mat,
        }
    }
}

impl Hittable for Sphere {
    fn material(&self) -> usize {
        self.mat
    }

    fn hit(&self, ray: Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let oc = ray.origin - self.center;
        let a = ray.dir.dot(ray.dir);
        let b = oc.dot(ray.dir);
        let c = oc.dot(oc) - self.radius * self.radius;
        let discriminant = b * b - a * c;

        if discriminant > 0.0 {
            let temp = (-b - discriminant.sqrt()) / a;
            if temp < t_max && temp > t_min {
                return Some(HitRecord {
                    t: temp,
                    point: ray.at(temp),
                    normal: (ray.at(temp) - self.center) / self.radius,
                    material_handle: self.mat,
                });
            }
            let temp = (-b + discriminant.sqrt()) / a;
            if temp < t_max && temp > t_min {
                return Some(HitRecord {
                    t: temp,
                    point: ray.at(temp),
                    normal: (ray.at(temp) - self.center) / self.radius,
                    material_handle: self.mat,
                });
            }
        }
        None
    }
}
