pub mod common;
pub mod cpu;
pub mod cuda;
// See the OPTIX_DISABLED comment.
// pub mod optix;
pub mod renderer;
pub mod viewer;

use common::Camera;
use glam::Vec3;
use path_tracer_kernels::{
    Object,
    material::{DielectricMaterial, DiffuseMaterial, MaterialKind, MetallicMaterial},
    scene::Scene,
    sphere::Sphere,
};
use std::error::Error;

pub const WIDTH: u32 = 1920;
pub const HEIGHT: u32 = 1080;

fn main() -> Result<(), Box<dyn Error>> {
    let camera = Camera {
        origin: Vec3::new(0.0, 0.5, 2.0),
        lookat: Vec3::new(0.0, 0.0, -0.5),
        vup: Vec3::Y,
        fov: 70.0,
        aspect_ratio: (WIDTH as f32) / (HEIGHT as f32),
    };

    let materials = vec![
        MaterialKind::Metallic(MetallicMaterial {
            color: Vec3::new(1.0, 0.85, 0.45),
            roughness: 0.0,
        }),
        MaterialKind::Diffuse(DiffuseMaterial {
            color: Vec3::new(0.5, 0.5, 1.0),
        }),
        MaterialKind::Metallic(MetallicMaterial {
            color: Vec3::new(1.0, 0.7, 0.7),
            roughness: 0.2,
        }),
        MaterialKind::Dielectric(DielectricMaterial {
            ior: 1.0,
            color: Vec3::new(1.0, 1.0, 1.0),
        }),
    ];

    let objects = vec![
        Object::Sphere(Sphere::new(Vec3::new(0.0, 0.0, -1.0), 0.5, 0)),
        Object::Sphere(Sphere::new(Vec3::new(1.1, 0.2, -0.7), 0.2, 2)),
        Object::Sphere(Sphere::new(Vec3::new(-0.5, 0.4, -0.9), 0.3, 3)),
        Object::Sphere(Sphere::new(Vec3::new(0.7, 0.0, -0.2), 0.2, 1)),
        // Object::Sphere(Sphere::new(Vec3::new(0.0, -200.5, -1.0), 200.0, 1)),
    ];
    let cpu_scene = Scene {
        objects: &objects,
        materials: &materials,
    };

    viewer::run(&camera, &cpu_scene);
}
