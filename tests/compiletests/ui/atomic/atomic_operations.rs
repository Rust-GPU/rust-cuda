// Test CUDA atomic operations compile correctly
// build-pass

use core::sync::atomic::Ordering;
use cuda_std::atomic::{
    AtomicF32, AtomicF64, BlockAtomicF32, BlockAtomicF64, SystemAtomicF32, SystemAtomicF64,
};
use cuda_std::kernel;

#[kernel]
pub unsafe fn test_cuda_atomic_floats() {
    // Device-scoped atomic float
    let atomic_f32 = AtomicF32::new(3.14);
    let _old = atomic_f32.fetch_add(1.0, Ordering::Relaxed);
    let _val = atomic_f32.load(Ordering::Relaxed);
    atomic_f32.store(2.718, Ordering::Relaxed);

    // Block-scoped atomic float
    let block_atomic = BlockAtomicF32::new(1.5);
    let _old = unsafe { block_atomic.fetch_add(0.5, Ordering::Relaxed) };

    // System-scoped atomic float
    let system_atomic = SystemAtomicF32::new(0.0);
    let _old = system_atomic.fetch_add(1.0, Ordering::Relaxed);

    // Test f64 as well
    let atomic_f64 = AtomicF64::new(3.14159);
    let _old = atomic_f64.fetch_add(1.0, Ordering::Relaxed);

    // Test block-scoped f64
    let block_f64 = BlockAtomicF64::new(2.718);
    let _old = unsafe { block_f64.fetch_sub(0.5, Ordering::Relaxed) };

    // Test bitwise operations on floats
    let _old = atomic_f32.fetch_and(3.14, Ordering::Relaxed);
    let _old = atomic_f32.fetch_or(1.0, Ordering::Relaxed);
    let _old = atomic_f32.fetch_xor(2.0, Ordering::Relaxed);
}
