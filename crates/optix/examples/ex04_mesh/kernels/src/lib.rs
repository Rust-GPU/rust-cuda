#![allow(non_snake_case, clippy::missing_safety_doc)]

use cuda_std::kernel;
use optix_device::{
    closesthit, get_launch_index,
    glam::*,
    payload,
    trace::TraversableHandle,
    trace::{trace, RayFlags},
    util::*,
};

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct LaunchParams {
    pub frame: Frame,
    pub camera: Camera,
    pub traversable: TraversableHandle,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Frame {
    pub color_buf: *mut Vec4,
    pub size: UVec2,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Camera {
    pub position: Vec3,
    pub direction: Vec3,
    pub horizontal: Vec3,
    pub vertical: Vec3,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum RayType {
    SurfaceRay = 0,
}

fn get_color_buf() -> *mut Vec4 {
    unsafe { unpack_pointer(payload::get_payload(0), payload::get_payload(1)) }
}

fn random_color(i: u32) -> Vec4 {
    let r = i * 13 * 17 + 0x234235;
    let g = i * 7 * 3 * 5 + 0x773477;
    let b = i * 11 * 19 + 0x223766;
    Vec4::new(
        (r & 255) as f32 / 255.0,
        (g & 255) as f32 / 255.0,
        (b & 255) as f32 / 255.0,
        1.0,
    )
}

#[kernel]
pub unsafe fn __closesthit__radiance() {
    let prim_id = closesthit::primitive_index();
    let buf = get_color_buf();
    *buf = random_color(prim_id);
}

#[kernel]
pub unsafe fn __anyhit__radiance() {}

#[kernel]
pub unsafe fn __miss__radiance() {
    let buf = get_color_buf();
    // pure white background
    *buf = Vec4::ONE;
}

extern "C" {
    #[cfg_attr(target_os = "cuda", nvvm_internal::addrspace(4))]
    static PARAMS: LaunchParams;
}

#[kernel]
pub unsafe fn __raygen__renderFrame() {
    let i = get_launch_index();
    let i = UVec2::new(i.x, i.y);

    let camera = PARAMS.camera;

    let px_color = Vec3::ZERO;
    let (mut p0, mut p1) = pack_pointer(&px_color as *const _ as *mut Vec3);

    let screen = (i.as_vec2() + Vec2::splat(0.5)) / PARAMS.frame.size.as_vec2();
    let ray_dir = (camera.direction
        + (screen.x - 0.5) * camera.horizontal
        + (screen.y - 0.5) * camera.vertical)
        .normalize();

    trace(
        PARAMS.traversable,
        camera.position,
        ray_dir,
        0.0,
        1e20,
        0.0,
        255,
        RayFlags::DISABLE_ANYHIT,
        RayType::SurfaceRay as u32,
        1,
        RayType::SurfaceRay as u32,
        [&mut p0, &mut p1],
    );

    let fb_index = i.x + i.y * PARAMS.frame.size.x;
    *PARAMS.frame.color_buf.add(fb_index as usize) =
        Vec4::new(px_color.x, px_color.y, px_color.z, 1.0);
}
