mod data;

pub use data::*;
use imgui::Ui;

use std::time::Duration;

use crate::{common::Camera, optix::OptixRenderer};
use anyhow::Result;
use cust::{
    error::CudaResult,
    event::{Event, EventFlags},
    function::{BlockSize, GridSize},
    memory::DeviceBox,
    prelude::*,
};
use glam::{U8Vec3, USizeVec2};
use optix::{
    context::DeviceContext,
    denoiser::{Denoiser, DenoiserModelKind, Image, ImageFormat},
};
use path_tracer_kernels::scene::Scene;
/// Seed for the random states
pub const SEED: u64 = 932174513921034;

/// How many pixels a single thread block should process, in each axis.
/// That is to say, 8 will dispatch 8x8 threads (in a 2d config) per block.
/// This should always be a multiple of warp size (32) to maximize occupancy.
const THREAD_BLOCK_AXIS_LENGTH: usize = 16;

pub(crate) static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

pub struct CudaRenderer {
    stream: Stream,
    module: Module,
    denoiser: Denoiser,
    _optix_context: DeviceContext,
    _context: Context,

    buffers: CudaRendererBuffers,
    cpu_image: Vec<U8Vec3>,
    optix_renderer: OptixRenderer,
}

impl CudaRenderer {
    pub fn new(dimensions: USizeVec2, camera: &Camera, scene: &Scene) -> Result<Self> {
        let context = cust::quick_init()?;
        optix::init().unwrap();

        let mut optix_context = DeviceContext::new(&context, false).unwrap();

        let module = Module::from_ptx(PTX, &[]).unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let mut denoiser =
            Denoiser::new(&optix_context, DenoiserModelKind::Ldr, Default::default()).unwrap();

        denoiser
            .setup_state(&stream, dimensions.x as u32, dimensions.y as u32, false)
            .unwrap();

        let buffers = CudaRendererBuffers::new(dimensions, camera, scene)?;
        let cpu_image = vec![U8Vec3::ZERO; dimensions.element_product()];

        let optix_renderer = OptixRenderer::new(&mut optix_context, &stream, scene)?;

        Ok(Self {
            _context: context,
            _optix_context: optix_context,
            denoiser,
            module,
            stream,
            buffers,
            cpu_image,
            optix_renderer,
        })
    }

    pub fn info(&self, ui: &Ui) {
        let device = Device::get_device(0).expect("Failed to retrieve device");
        let name = device.name().unwrap();
        let mem = device.total_memory().unwrap();

        let group = ui.begin_group();
        ui.text(format!("CUDA Device: {}", name));
        ui.text(format!("Total VRAM: {}mb", mem / 1_000_000));
        group.end();
    }

    /// Update the camera of the renderer and reset any accumulated buffers.
    pub fn update_camera(&mut self, camera: &Camera) -> CudaResult<()> {
        self.buffers.update_camera(camera)
    }

    /// Resize the image-specific data for a new size
    pub fn resize(&mut self, new_size: USizeVec2) -> CudaResult<()> {
        self.buffers.resize(new_size)?;
        self.cpu_image
            .resize(new_size.element_product(), U8Vec3::ZERO);

        self.denoiser
            .setup_state(&self.stream, new_size.x as u32, new_size.y as u32, false)
            .unwrap();
        Ok(())
    }

    /// calculate an optimal launch configuration for an image kernel
    fn launch_dimensions(&self) -> (GridSize, BlockSize) {
        let threads = USizeVec2::splat(THREAD_BLOCK_AXIS_LENGTH);
        let blocks = (self.buffers.viewport.bounds / threads) + 1;
        (blocks.into(), threads.into())
    }

    /// Run postprocessing on the accumulated buffer, color correct it, and divide it by the total
    /// samples to yield a final image that can be displayed.
    ///
    /// Also returns the denoising time and postprocessing time.
    pub fn final_image(
        &mut self,
        cur_sample: usize,
        denoise: bool,
    ) -> CudaResult<(&[U8Vec3], Duration, Duration)> {
        let module = &self.module;
        let stream = &self.stream;

        let (blocks, threads) = self.launch_dimensions();
        let width = self.buffers.viewport.bounds.x as u32;
        let height = self.buffers.viewport.bounds.y as u32;

        let start = Event::new(EventFlags::DEFAULT)?;
        let denoising_stop = Event::new(EventFlags::DEFAULT)?;
        let postprocessing_stop = Event::new(EventFlags::DEFAULT)?;

        start.record(stream)?;

        unsafe {
            launch!(
                module.scale_buffer<<<blocks, threads, 0, stream>>>(
                    self.buffers.accumulated_buffer.as_device_ptr(),
                    self.buffers.scaled_buffer.as_device_ptr(),
                    cur_sample,
                    self.buffers.viewport
                )
            )?;
        }

        let input_buf = if denoise {
            let input_image = Image::new(
                &self.buffers.scaled_buffer,
                ImageFormat::Float3,
                width,
                height,
            );

            self.denoiser
                .invoke(
                    stream,
                    Default::default(),
                    input_image,
                    Default::default(),
                    &mut self.buffers.denoised_buffer,
                )
                .unwrap();

            self.buffers.denoised_buffer.as_device_ptr()
        } else {
            self.buffers.scaled_buffer.as_device_ptr()
        };

        denoising_stop.record(stream)?;

        unsafe {
            launch!(
                module.postprocess<<<blocks, threads, 0, stream>>>(
                    input_buf,
                    self.buffers.out_buffer.as_device_ptr(),
                    self.buffers.viewport
                )
            )?;
        }

        postprocessing_stop.record(stream)?;
        postprocessing_stop.synchronize()?;

        let denoising_time = denoising_stop.elapsed(&start)?;
        let postprocessing_time = postprocessing_stop.elapsed(&denoising_stop)?;

        self.buffers.out_buffer.copy_to(&mut self.cpu_image)?;

        Ok((&self.cpu_image, denoising_time, postprocessing_time))
    }

    /// Render another sample of the image, adding it on top of the already accumulated buffer.
    pub fn render(&mut self, use_optix: bool) -> Result<Duration> {
        let module = &self.module;
        let stream = &self.stream;

        let (blocks, threads) = self.launch_dimensions();

        // record how long each render sample took using events

        let start = Event::new(EventFlags::DEFAULT)?;
        let stop = Event::new(EventFlags::DEFAULT)?;

        start.record(stream)?;

        if use_optix {
            self.optix_renderer.render(stream, &mut self.buffers)?;
        } else {
            unsafe {
                let scene = DeviceBox::new_async(
                    &Scene {
                        objects: &self.buffers.objects,
                        materials: &self.buffers.materials,
                    },
                    stream,
                )?;

                launch!(
                    module.render<<<blocks, threads, 0, stream>>>(
                        self.buffers.accumulated_buffer.as_device_ptr(),
                        self.buffers.viewport,
                        scene.as_device_ptr(),
                        self.buffers.rand_states.as_unified_ptr()
                    )
                )?;

                scene.drop_async(stream)?;
            }
        }

        stop.record(stream)?;
        // dont need to synchronize the stream, the event can do it for us.
        stop.synchronize()?;
        Ok(stop.elapsed(&start)?)
    }
}
