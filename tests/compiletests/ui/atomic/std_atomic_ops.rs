// Test CUDA atomic operations compile correctly
// build-pass
// compile-flags: -Z verify-llvm-ir 
use core::sync::atomic::{AtomicUsize,Ordering};

use cuda_std::atomic::{
    AtomicF32, AtomicF64, BlockAtomicF32, BlockAtomicF64, SystemAtomicF32, SystemAtomicF64,
};
use cuda_std::kernel;
static GLOBAL:AtomicUsize = AtomicUsize::new(0);
#[kernel]
pub unsafe fn test_cuda_atomic_floats() {
    let local = AtomicUsize::new(0);
    // `compare_exchange` should succeed
    local.compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed);
    // `compare_exchange` should fail
    local.compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed);
    // `compare_exchange` should succeed
    GLOBAL.compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed);
    // `compare_exchange` should fail
    GLOBAL.compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed);
    // Ops
    local.swap(1, Ordering::Relaxed);
    GLOBAL.swap(1, Ordering::Relaxed);
    local.fetch_add(1, Ordering::Relaxed);
    GLOBAL.fetch_add(1, Ordering::Relaxed);
    local.fetch_sub(1, Ordering::Relaxed);
    GLOBAL.fetch_sub(1, Ordering::Relaxed);
    local.fetch_and(1, Ordering::Relaxed);
   GLOBAL.fetch_and(1, Ordering::Relaxed);
    local.fetch_and(1, Ordering::Relaxed);
    GLOBAL.fetch_and(1, Ordering::Relaxed);
    local.fetch_or(1, Ordering::Relaxed);
    GLOBAL.fetch_or(1, Ordering::Relaxed);
    local.fetch_xor(1, Ordering::Relaxed);
    GLOBAL.fetch_xor(1, Ordering::Relaxed);
    local.fetch_max(1, Ordering::Relaxed);
    GLOBAL.fetch_max(1, Ordering::Relaxed);
    local.fetch_min(1, Ordering::Relaxed);
    GLOBAL.fetch_min(1, Ordering::Relaxed);
    // Loads:
    local.load(Ordering::Relaxed);
    GLOBAL.load(Ordering::Relaxed);
    local.store(1, Ordering::Relaxed);
    GLOBAL.store(1, Ordering::Relaxed);
    // Atomic NAND is not supported quite yet
    //local.fetch_nand(1, Ordering::Relaxed);
    //GLOBAL.fetch_nand(1, Ordering::Relaxed);
}
