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

}
