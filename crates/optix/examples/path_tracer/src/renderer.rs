use glam::USizeVec2;
use glium::glutin::{event::Event, event_loop::ControlFlow};
use imgui::Ui;
use path_tracer_kernels::scene::Scene;
use sysinfo::System;

use crate::{
    common::{Camera, CameraController},
    cpu::CpuRenderer,
    cuda::CudaRenderer,
};

pub struct Renderer {
    cuda: CudaRenderer,
    cpu: CpuRenderer,
    running_on_gpu: bool,
    pub use_optix: bool,
    pub denoise: bool,
    accumulated_samples: usize,
    camera: Camera,
    controller: CameraController,
    system: System,
}

impl Renderer {
    pub fn new(dimensions: USizeVec2, camera: &Camera, scene: &Scene) -> Self {
        Self {
            cuda: CudaRenderer::new(dimensions, camera, scene)
                .expect("Failed to make CUDA renderer"),
            cpu: CpuRenderer::new(dimensions, camera, scene),
            running_on_gpu: true,
            denoise: false,
            accumulated_samples: 0,
            camera: *camera,
            controller: CameraController::new(dimensions),
            system: System::new_all(),
            use_optix: false,
        }
    }

    pub fn resize(&mut self, new: USizeVec2) {
        self.accumulated_samples = 0;
        self.cpu.resize(new);
        self.cuda
            .resize(new)
            .expect("Failed to resize CUDA renderer");
    }

    /// Renders the scene and returns a final image buffer that can be displayed.
    pub fn render(&mut self, ui: &Ui) -> &[u8] {
        self.cuda.info(ui);
        self.cpu.info(ui, &self.system);
        self.accumulated_samples += 1;

        ui.separator();
        ui.text(format!("Camera Pos: {:?}", self.camera.origin));
        ui.text(format!("Camera Lookat: {:?}", self.camera.lookat));
        ui.new_line();
        ui.separator();
        ui.text(format!("Current Sample: {}", self.accumulated_samples));
        ui.separator();

        let switched = ui.checkbox("Use CUDA", &mut self.running_on_gpu);

        if switched {
            self.clear_view(true);
        }

        if self.running_on_gpu {
            ui.separator();
            ui.text("Running on GPU");
            ui.checkbox("Use OptiX", &mut self.use_optix);
            ui.checkbox("OptiX Denoise", &mut self.denoise);
            ui.separator();

            let duration = self
                .cuda
                .render(self.use_optix)
                .expect("Failed to render using CUDA backend");

            ui.text(format!(
                "Sampling time: {:.2}ms",
                duration.as_secs_f32() * 1000.0
            ));

            let (output, denoising_time, postprocessing_time) = self
                .cuda
                .final_image(self.accumulated_samples, self.denoise)
                .expect("Failed to get final image");

            if self.denoise {
                ui.text(format!(
                    "Denoising time: {:.2}ms",
                    denoising_time.as_secs_f32() * 1000.0
                ));
            } else {
                ui.text("Denoising time: N/A");
            }

            ui.text(format!(
                "Postprocessing time: {:.2}ms",
                postprocessing_time.as_secs_f32() * 1000.0
            ));

            ui.text(format!(
                "Total: {:.2}ms",
                (postprocessing_time.as_secs_f32()
                    + denoising_time.as_secs_f32()
                    + duration.as_secs_f32())
                    * 1000.0
            ));

            // bytemuck could do this but vek doesnt have bytemuck
            unsafe { std::slice::from_raw_parts(output.as_ptr().cast(), output.len() * 3) }
        } else {
            ui.separator();
            ui.text("Running on CPU");
            ui.separator();

            let duration = self.cpu.render();

            ui.text(format!(
                "Sampling time: {:.2}ms",
                duration.as_secs_f32() * 1000.0
            ));

            let (output, postprocessing_time) = self.cpu.final_image(self.accumulated_samples);

            ui.text(format!(
                "Postprocessing time: {:.2}ms",
                postprocessing_time.as_secs_f32() * 1000.0
            ));

            ui.text(format!(
                "Total: {:.2}ms",
                (duration + postprocessing_time).as_secs_f32() * 1000.0
            ));

            unsafe { std::slice::from_raw_parts(output.as_ptr().cast(), output.len() * 3) }
        }
    }

    fn clear_view(&mut self, force: bool) {
        if force {
            self.cuda.update_camera(&self.camera).unwrap();
            self.cpu.update_camera(&self.camera);
        } else if self.running_on_gpu {
            self.cuda.update_camera(&self.camera).unwrap();
        } else {
            self.cpu.update_camera(&self.camera);
        }
        self.accumulated_samples = 0;
    }

    pub fn process_event(&mut self, event: &Event<()>, _control_flow: &mut ControlFlow) {
        if self.controller.process_event(event, &mut self.camera) {
            self.clear_view(false);
        }
    }
}
