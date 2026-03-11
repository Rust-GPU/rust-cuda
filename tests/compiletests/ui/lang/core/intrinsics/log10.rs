// Test log10 intrinsic
// build-pass

#![allow(internal_features)]
#![feature(core_intrinsics)]
#![no_std]

use cuda_std::kernel;
use cuda_std::GpuFloat;

#[kernel]
pub unsafe fn test_log10(input: *const f32, output: *mut f32) {
    unsafe {
        *output = (*input).log10();
    }
}
