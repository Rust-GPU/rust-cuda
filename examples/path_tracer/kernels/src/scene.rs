use crate::material::*;
use crate::*;
use cust_core::DeviceCopy;
use gpu_rand::DefaultRand;

pub const MAX_BOUNCES: u32 = 5;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Scene<'a> {
    pub objects: &'a [Object],
    pub materials: &'a [MaterialKind],
}

/// SAFETY: the slice is created from unified memory so it works on the GPU too.
unsafe impl DeviceCopy for Scene<'_> {}

impl Scene<'_> {
    pub fn hit(&self, ray: Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let mut hit = None;
        let mut closest_so_far = t_max;

        for obj in self.objects {
            if let Some(rec) = obj.hit(ray, t_min, closest_so_far) {
                hit = Some(rec);
                closest_so_far = rec.t;
            }
        }
        hit
    }

    /// Casts a ray into the scene and returns the object hit by the ray.
    pub fn raycast(&self, ray: Ray) -> Option<&Object> {
        let mut hit = None;
        let mut closest_so_far = f32::INFINITY;

        for obj in self.objects {
            if let Some(rec) = obj.hit(ray, 0.001, closest_so_far) {
                hit = Some(obj);
                closest_so_far = rec.t;
            }
        }
        hit
    }

    pub fn ray_color(&self, ray: Ray, rng: &mut DefaultRand) -> Vec3 {
        let mut cur_ray = ray;
        let mut attenuation = Vec3::ONE;

        for _ in 0..MAX_BOUNCES {
            if let Some(hit) = self.hit(cur_ray, 0.001, f32::INFINITY) {
                let material = self.materials[hit.material_handle];
                let (hit_attenuation, scattered) = material.scatter(cur_ray, hit, rng);
                if let Some(scattered) = scattered {
                    attenuation *= hit_attenuation;
                    cur_ray = scattered;
                } else {
                    return Vec3::ZERO;
                }
            } else {
                return attenuation * render::color(cur_ray);
            }
        }
        Vec3::ZERO
    }
}
