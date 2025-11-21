use crate::common::Camera;
use bytemuck::Zeroable;
use cust::{
    error::CudaResult,
    memory::{DeviceBuffer, DeviceCopy, UnifiedBuffer},
    util::SliceExt,
};
use glam::{U8Vec3, USizeVec2, Vec3};
use gpu_rand::DefaultRand;
use path_tracer_kernels::{material::MaterialKind, scene::Scene, Object, Viewport};

use super::SEED;

/// The various buffers held by the CUDA renderer.
///
/// You could put these in the CUDA renderer but we separate them out for code readability.
pub struct CudaRendererBuffers {
    /// The buffer of accumulated colors, every sample/render call adds its color to this buffer.
    pub accumulated_buffer: DeviceBuffer<Vec3>,
    /// The scaled buffer of colors, this is just the accumulated colors divided by sample count.
    pub scaled_buffer: DeviceBuffer<Vec3>,
    /// The final image buffer after denoising and postprocessing.
    pub out_buffer: DeviceBuffer<U8Vec3>,
    /// The scaled buffer but denoised. In the future we will use the same buffer for this.
    pub denoised_buffer: DeviceBuffer<Vec3>,

    /// The viewport used by the render kernel to emit rays.
    pub viewport: Viewport,
    /// Allocated buffer of objects in the scene.
    pub objects: UnifiedBuffer<Object>,
    /// Allocated buffer of the materials in the scene.
    pub materials: UnifiedBuffer<MaterialKind>,
    /// Per-thread randomness states.
    pub rand_states: UnifiedBuffer<DefaultRand>,
}

impl CudaRendererBuffers {
    pub fn new(dimensions: USizeVec2, camera: &Camera, scene: &Scene) -> CudaResult<Self> {
        let accumulated_buffer = Self::image_buffer(dimensions)?;
        let out_buffer = Self::image_buffer(dimensions)?;
        let denoised_buffer = Self::image_buffer(dimensions)?;
        let scaled_buffer = Self::image_buffer(dimensions)?;

        let objects = scene.objects.as_unified_buf()?;
        let materials = scene.materials.as_unified_buf()?;

        let mut viewport = Viewport::default();
        camera.as_viewport(&mut viewport);
        viewport.bounds = dimensions;

        let rand_states = DefaultRand::initialize_states(SEED, dimensions.element_product())
            .as_slice()
            .as_unified_buf()?;

        Ok(Self {
            accumulated_buffer,
            scaled_buffer,
            out_buffer,
            denoised_buffer,
            viewport,
            objects,
            materials,
            rand_states,
        })
    }

    /// Resets and reallocates the entire scene. This may be slow because it needs to reallocate
    /// all of the GPU scene buffers.
    pub fn reset_scene(&mut self, scene: &Scene) -> CudaResult<()> {
        self.objects = scene.objects.as_unified_buf()?;
        self.materials = scene.materials.as_unified_buf()?;

        Ok(())
    }

    /// Reset the renderer's view, in the buffer's case this means clearing accumulated buffers from previous samples.
    /// As well as changing the viewport.
    pub fn update_camera(&mut self, new_camera: &Camera) -> CudaResult<()> {
        self.accumulated_buffer = DeviceBuffer::zeroed(self.accumulated_buffer.len())?;
        new_camera.as_viewport(&mut self.viewport);
        Ok(())
    }

    /// Resize the image-specific buffers for a new image size.
    pub fn resize(&mut self, new: USizeVec2) -> CudaResult<()> {
        self.viewport.bounds = new;
        self.accumulated_buffer = Self::image_buffer(new)?;
        self.out_buffer = Self::image_buffer(new)?;
        self.denoised_buffer = Self::image_buffer(new)?;
        self.scaled_buffer = Self::image_buffer(new)?;
        self.rand_states = DefaultRand::initialize_states(SEED, new.element_product())
            .as_slice()
            .as_unified_buf()?;
        Ok(())
    }

    /// Swaps out a material at a specific index.
    pub fn update_material(&mut self, idx: usize, new: MaterialKind) {
        self.materials[idx] = new;
    }

    /// Swaps out an object at a specific index.
    pub fn update_object(&mut self, idx: usize, new: Object) {
        self.objects[idx] = new;
    }

    // could also use the convenience method on optix::denoiser::Image for this
    fn image_buffer<T: DeviceCopy + Zeroable>(
        dimensions: USizeVec2,
    ) -> CudaResult<DeviceBuffer<T>> {
        DeviceBuffer::zeroed(dimensions.element_product())
    }
}
