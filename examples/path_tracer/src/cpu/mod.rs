use std::time::Duration;

use glam::{U8Vec3, USizeVec2, UVec2, Vec2, Vec3};
use gpu_rand::{DefaultRand, GpuRand};
use imgui::Ui;
use path_tracer_kernels::{
    material::MaterialKind, render::generate_ray, scene::Scene, Object, Viewport,
};
use rayon::prelude::*;
use sysinfo::System;

use crate::{common::Camera, cuda::SEED};

pub struct CpuRenderer {
    // this is basically the cuda buffers but not gpu buffers.
    accumulated_buffer: Vec<Vec3>,
    out_buffer: Vec<U8Vec3>,

    viewport: Viewport,
    objects: Vec<Object>,
    materials: Vec<MaterialKind>,
    rand_states: Vec<DefaultRand>,
}

impl CpuRenderer {
    pub fn new(dimensions: USizeVec2, camera: &Camera, scene: &Scene) -> Self {
        let accumulated_buffer = vec![Vec3::ZERO; dimensions.element_product()];
        let out_buffer = vec![U8Vec3::ZERO; dimensions.element_product()];

        let rand_states = DefaultRand::initialize_states(SEED, dimensions.element_product());

        let mut viewport = Viewport::default();
        camera.as_viewport(&mut viewport);
        viewport.bounds = dimensions;

        Self {
            accumulated_buffer,
            out_buffer,
            viewport,
            objects: scene.objects.to_vec(),
            materials: scene.materials.to_vec(),
            rand_states,
        }
    }

    pub fn info(&self, ui: &Ui, system: &System) {
        let cores = system.physical_core_count().unwrap();
        let processor = &system.cpus()[0];

        let group = ui.begin_group();
        ui.text(format!(
            "CPU: {}MHz, {} cores",
            processor.frequency(),
            cores
        ));
        group.end();
    }

    /// Resets and reallocates the entire scene.
    pub fn reset_scene(&mut self, scene: &Scene) {
        self.objects = scene.objects.to_vec();
        self.materials = scene.materials.to_vec();
    }

    pub fn update_camera(&mut self, new_camera: &Camera) {
        self.accumulated_buffer.fill(Vec3::ZERO);
        new_camera.as_viewport(&mut self.viewport);
    }

    pub fn resize(&mut self, dimensions: USizeVec2) {
        self.accumulated_buffer
            .resize(dimensions.element_product(), Vec3::ZERO);
        self.out_buffer
            .resize(dimensions.element_product(), U8Vec3::ZERO);
        self.viewport.bounds = dimensions;
    }

    /// Swaps out a material at a specific index.
    pub fn update_material(&mut self, idx: usize, new: MaterialKind) {
        self.materials[idx] = new;
    }

    /// Swaps out an object at a specific index.
    pub fn update_object(&mut self, idx: usize, new: Object) {
        self.objects[idx] = new;
    }

    pub fn final_image(&mut self, cur_sample: usize) -> (&[U8Vec3], Duration) {
        let start = std::time::Instant::now();

        let Self {
            accumulated_buffer,
            out_buffer,
            ..
        } = self;

        out_buffer
            .par_iter_mut()
            .zip(accumulated_buffer.par_iter())
            .for_each(|(px, acc)| {
                let scaled = acc / cur_sample as f32;
                let gamma_corrected = scaled.map(f32::sqrt);

                *px = (gamma_corrected * 255.0)
                    .clamp(Vec3::ZERO, Vec3::splat(255.0))
                    .as_u8vec3();
            });

        (&self.out_buffer, start.elapsed())
    }

    pub fn render(&mut self) -> Duration {
        // rustc has some problems with borrows even though it should be fine in this case,
        // so we just destructure to tell it its disjoint.
        let Self {
            accumulated_buffer,
            viewport,
            objects,
            materials,
            rand_states,
            ..
        } = self;
        let start = std::time::Instant::now();

        let scene = Scene { objects, materials };

        accumulated_buffer
            .par_iter_mut()
            .zip(rand_states.par_iter_mut())
            .enumerate()
            .for_each(|(idx, (px, rng))| {
                let x = idx % viewport.bounds.x;
                let y = idx / viewport.bounds.x;
                let idx = UVec2::new(x as u32, y as u32);

                let offset = Vec2::from(rng.normal_f32_2());

                let ray = generate_ray(idx, viewport, offset);

                let color = scene.ray_color(ray, rng);
                *px += color;
            });

        start.elapsed()
    }
}
