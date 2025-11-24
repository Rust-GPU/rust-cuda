// Test CUDA shared memory allocations compile correctly
// build-pass
//
// FIXME: The default of `-Cdebuginfo=2` causes a seg fault, for unclear reasons
// compile-flags: -Cdebuginfo=1

use core::mem::MaybeUninit;
use cuda_std::{address_space, kernel, thread};

#[kernel]
pub unsafe fn test_static_shared_memory() {
    // Allocate static shared memory for 256 i32 values
    #[address_space(shared)]
    static mut SHARED_DATA: [MaybeUninit<i32>; 256] = [MaybeUninit::uninit(); 256];

    let tid = thread::thread_idx_x() as usize;

    // Write to shared memory
    SHARED_DATA[tid].write(tid as i32);

    // Synchronize threads before reading
    thread::sync_threads();

    // Read from shared memory
    let _value = SHARED_DATA[tid].assume_init();
}

#[kernel]
pub unsafe fn test_different_types() {
    // Test different array types
    static mut _SHARED_U32: [MaybeUninit<u32>; 128] = [MaybeUninit::uninit(); 128];
    static mut _SHARED_F32: [MaybeUninit<f32>; 64] = [MaybeUninit::uninit(); 64];
    static mut _SHARED_U8: [MaybeUninit<u8>; 512] = [MaybeUninit::uninit(); 512];

    // Test arrays of arrays
    static mut _SHARED_VEC3: [MaybeUninit<[f32; 3]>; 32] = [MaybeUninit::uninit(); 32];
}
