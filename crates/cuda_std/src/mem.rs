//! Support for allocating memory and using `alloc` using CUDA memory allocation system-calls.

use crate::gpu_only;
#[cfg(target_arch = "nvptx64")]
use alloc::alloc::*;
#[cfg(target_arch = "nvptx64")]
use core::ffi::c_void;

#[cfg(target_arch = "nvptx64")]
unsafe extern "C" {
    // implicitly defined by cuda.
    pub fn malloc(size: usize) -> *mut c_void;

    pub fn free(ptr: *mut c_void);
}

pub struct CUDAAllocator;

#[cfg(target_arch = "nvptx64")]
unsafe impl GlobalAlloc for CUDAAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { malloc(layout.size()) as *mut u8 }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        unsafe {
            free(ptr as *mut _);
        }
    }
}

#[cfg(target_arch = "nvptx64")]
#[global_allocator]
pub static GLOBAL_ALLOCATOR: CUDAAllocator = CUDAAllocator;

/// Returns the amount of shared memory that has been dynamically allocated
/// by the caller of the kernel for every thread block (CTA).
#[gpu_only]
#[inline(always)]
pub fn dynamic_smem_size() -> u32 {
    let mut out;
    unsafe {
        core::arch::asm!(
            "mov.u32 {}, %dynamic_smem_size",
            out(reg32) out
        )
    }
    out
}

/// Returns the amount of total shared memory that has been allocated
/// for every thread block for this kernel. This includes both static and dynamic
/// shared memory. The returned number will be a multiple of static memory allocation unit size:
/// - 128 bytes on sm_2x and sm_8x
/// - 256 bytes on sm_3x, sm_5x, sm_6x, and sm_7x
#[gpu_only]
#[inline(always)]
pub fn total_smem_size() -> u32 {
    let mut out;
    unsafe {
        core::arch::asm!(
            "mov.u32 {}, %total_smem_size",
            out(reg32) out
        )
    }
    out
}
