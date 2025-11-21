// #![deny(warnings)]
#![allow(clippy::missing_safety_doc)]

use cuda_std::*;
use cust_core::DeviceCopy;
use optix_device as optix;

extern crate alloc;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct LaunchParams {
    pub frame_id: i32,
    pub fb_size: [u32; 2],
    pub color_buffer: u64,
}

unsafe impl DeviceCopy for LaunchParams {}

#[no_mangle]
static PARAMS: LaunchParams = LaunchParams {
    frame_id: 88,
    fb_size: [1, 1],
    color_buffer: 0,
};

extern "C" {
    pub fn vprintf(format: *const u8, valist: *const core::ffi::c_void) -> i32;
}

#[kernel]
pub unsafe fn __closesthit__radiance() {}

#[kernel]
pub unsafe fn __anyhit__radiance() {}

#[kernel]
pub unsafe fn __miss__radiance() {}

#[kernel]
pub unsafe fn __raygen__renderFrame() {
    // let ix = _optix_get_launch_index_x();
    // let iy = _optix_get_launch_index_y();

    let idx = optix::get_launch_index();

    if idx[0] == 3 && idx[1] == 4 {
        vprintf(
            c"Hello from Rust kernel!\n".as_ptr().cast(),
            core::ptr::null::<core::ffi::c_void>(),
        );

        #[repr(C)]
        struct PrintArgs(i32);

        vprintf(
            c"frame id is %d\n".as_ptr().cast(),
            &PrintArgs(core::ptr::read_volatile(&PARAMS.frame_id)) as *const PrintArgs
                as *const core::ffi::c_void,
        );
    }
}

// #[kernel]
// pub unsafe fn render(fb: *mut Vec3<f32>, view: &Viewport) {}
