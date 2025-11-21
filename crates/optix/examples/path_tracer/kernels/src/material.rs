use crate::{hittable::HitRecord, math::*, Ray, Vec3};
use approx::{AbsDiffEq, RelativeEq};
use cust_core::DeviceCopy;
use enum_dispatch::enum_dispatch;
use gpu_rand::{DefaultRand, GpuRand};

#[enum_dispatch]
pub trait Material {
    /// Optionally scatters a ray and returns an attenuation color and an optional ray
    fn scatter(&self, incoming: Ray, hit: HitRecord, rng: &mut DefaultRand) -> (Vec3, Option<Ray>);
}

#[derive(Clone, Copy, PartialEq, DeviceCopy)]
#[enum_dispatch(Material)]
pub enum MaterialKind {
    Diffuse(DiffuseMaterial),
    Metallic(MetallicMaterial),
    Dielectric(DielectricMaterial),
}

#[derive(Clone, Copy, PartialEq, DeviceCopy)]
pub struct DiffuseMaterial {
    pub color: Vec3,
}

impl Material for DiffuseMaterial {
    fn scatter(&self, _: Ray, hit: HitRecord, rng: &mut DefaultRand) -> (Vec3, Option<Ray>) {
        let mut scatter_dir = hit.normal + random_in_unit_sphere(rng);
        let epsilon = f32::default_epsilon();
        let max_rel = f32::default_max_relative();
        let four_epsilon = epsilon + epsilon + epsilon + epsilon;
        let four_max_rel = max_rel + max_rel + max_rel + max_rel;
        if scatter_dir
            .length_squared()
            .relative_eq(&0., four_epsilon, four_max_rel)
        {
            scatter_dir = hit.normal;
        }

        let attenuation = self.color;
        let ray = Ray {
            origin: hit.point,
            dir: scatter_dir,
        };
        (attenuation, Some(ray))
    }
}

#[derive(Clone, Copy, PartialEq, DeviceCopy)]
pub struct MetallicMaterial {
    pub color: Vec3,
    pub roughness: f32,
}

impl Material for MetallicMaterial {
    fn scatter(&self, incoming: Ray, hit: HitRecord, rng: &mut DefaultRand) -> (Vec3, Option<Ray>) {
        let reflected = reflect(incoming.dir.normalize(), hit.normal);
        let scattered = Ray {
            origin: hit.point,
            dir: reflected + self.roughness * random_in_unit_sphere(rng),
        };
        let attenuation = self.color;
        if scattered.dir.dot(hit.normal) > 0.0 {
            (attenuation, Some(scattered))
        } else {
            (attenuation, None)
        }
    }
}

#[derive(Clone, Copy, PartialEq, DeviceCopy)]
pub struct DielectricMaterial {
    pub ior: f32,
    pub color: Vec3,
}

impl Material for DielectricMaterial {
    fn scatter(&self, incoming: Ray, hit: HitRecord, rng: &mut DefaultRand) -> (Vec3, Option<Ray>) {
        let outward_norm;
        let ni_over_nt: f32;
        let cos: f32;

        if incoming.dir.dot(hit.normal) > 0.0 {
            outward_norm = -hit.normal;
            ni_over_nt = self.ior;
            cos = self.ior * incoming.dir.dot(hit.normal) / incoming.dir.length();
        } else {
            outward_norm = hit.normal;
            ni_over_nt = 1.0 / self.ior;
            cos = -incoming.dir.dot(hit.normal) / incoming.dir.length();
        }

        if let Some(refracted) = refract(incoming.dir, outward_norm, ni_over_nt) {
            if rng.normal_f32() > schlick(cos, self.ior) {
                return (
                    self.color,
                    Some(Ray {
                        origin: hit.point,
                        dir: refracted,
                    }),
                );
            }
        }

        (
            self.color,
            Some(Ray {
                origin: hit.point,
                dir: reflect(incoming.dir.normalize(), hit.normal),
            }),
        )
    }
}
