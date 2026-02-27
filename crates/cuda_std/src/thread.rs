//! Functions for dealing with the parallel thread execution model employed by CUDA.
//!
//! # CUDA thread model
//!
//! CUDA organizes execution into three hierarchical levels:
//! - Threads
//! - Thread blocks
//! - Grids
//!
//! ## Threads
//!
//! Threads are the fundamental unit of execution. Every thread runs the same kernel
//! code, typically operating on different data. Threads identify their work via
//! their indices and the dimensions of their block and grid.
//!
//! ## Thread blocks
//!
//! Threads are arranged into one-, two-, or three-dimensional blocks. The dimensionality
//! of a block usually mirrors the data layout (e.g., 2D blocks for images). The number of
//! threads per block is configurable and device-dependent (commonly up to 1024 total threads).
//!
//! Thread blocks are the primary unit of scheduling. Any block can be scheduled on any of the
//! GPU’s streaming multiprocessors (SMs). If no SM is available, the block waits in a queue.
//! Because blocks may execute in any order and at different times, they must be designed to run
//! independently of one another.
//!
//! Threads within the same block can cooperate via shared memory and block-wide barriers.
//! The kernel can retrieve a thread’s index within its block via `thread_idx_x`, `thread_idx_y`,
//! and `thread_idx_z`, and the block’s dimensions via `block_dim_x`, `block_dim_y`, and
//! `block_dim_z`.
//!
//! ## Grids
//!
//! A grid is an array (1D/2D/3D) of thread blocks. Grids define how many blocks are launched
//! and how they are arranged.
//!
//! The kernel can retrieve the block’s index within the grid via `block_idx_x`, `block_idx_y`,
//! and `block_idx_z`, and the grid’s dimensions via `grid_dim_x`, `grid_dim_y`, and `grid_dim_z`.
//! Combined with the `thread_*` and `block_dim_*` values, these indices are used to compute
//! which portion of the input data a thread should process.
//!
//! ## Computing global indices (examples)
//!
//! 1D global thread index:
//! ```no_run
//! # use cuda_std::kernel;
//! ##[kernel]
//! pub unsafe fn f1d() {
//!     use cuda_std::thread;
//!     let gx = thread::block_idx_x() * thread::block_dim_x() + thread::thread_idx_x();
//! }
//! ```
//!
//! 2D global coordinates (x, y):
//! ```rust
//! # use cuda_std::kernel;
//! ##[kernel]
//! pub unsafe fn f2d() {
//!     use cuda_std::thread;
//!     let x = thread::block_idx_x() * thread::block_dim_x() + thread::thread_idx_x();
//!     let y = thread::block_idx_y() * thread::block_dim_y() + thread::thread_idx_y();
//! }
//! ```
//!
//! Note: Hardware limits for block dimensions, grid dimensions, and total threads per block
//! vary by device. Query device properties when you need exact limits.
//!
use cuda_std_macros::gpu_only;
use glam::{USizeVec2, USizeVec3};

// different calling conventions dont exist in nvptx, so we just use C as a placeholder.
unsafe extern "C" {
    // defined in libintrinsics.ll
    fn __nvvm_warp_size() -> u32;

    fn __nvvm_block_barrier();

    fn __nvvm_grid_fence();
    fn __nvvm_device_fence();
    fn __nvvm_system_fence();
}

#[cfg(target_os = "cuda")]
macro_rules! in_range {
    // The bounds were taken mostly from the cuda C++ programming guide. I also
    // double-checked with what cuda clang does by checking its emitted llvm ir's scalar metadata.
    ($func_name:path, $range:expr) => {{
        let val = unsafe { $func_name() as u32 };
        if !$range.contains(&val) {
            // SAFETY: this condition is declared unreachable by compute capability max bound.
            // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
            // We do this to potentially allow for better optimizations by LLVM.
            unsafe { core::hint::unreachable_unchecked() }
        } else {
            val
        }
    }};
}

#[gpu_only]
#[inline(always)]
pub fn thread_idx_x() -> usize {
    // The range is derived from the `block_dim_x` range.
    in_range!(core::arch::nvptx::_thread_idx_x, 0..1024) as usize
}

#[gpu_only]
#[inline(always)]
pub fn thread_idx_y() -> usize {
    // The range is derived from the `block_dim_y` range.
    in_range!(core::arch::nvptx::_thread_idx_y, 0..1024) as usize
}

#[gpu_only]
#[inline(always)]
pub fn thread_idx_z() -> usize {
    // The range is derived from the `block_dim_z` range.
    in_range!(core::arch::nvptx::_thread_idx_z, 0..64) as usize
}

#[gpu_only]
#[inline(always)]
pub fn block_idx_x() -> usize {
    // The range is derived from the `grid_dim_x` range.
    in_range!(core::arch::nvptx::_block_idx_x, 0..2147483647) as usize
}

#[gpu_only]
#[inline(always)]
pub fn block_idx_y() -> usize {
    // The range is derived from the `grid_dim_y` range.
    in_range!(core::arch::nvptx::_block_idx_y, 0..65535) as usize
}

#[gpu_only]
#[inline(always)]
pub fn block_idx_z() -> usize {
    // The range is derived from the `grid_dim_z` range.
    in_range!(core::arch::nvptx::_block_idx_z, 0..65535) as usize
}

#[gpu_only]
#[inline(always)]
pub fn block_dim_x() -> usize {
    // CUDA Compute Capabilities: "Maximum x- or y-dimensionality of a block" is 1024.
    in_range!(core::arch::nvptx::_block_dim_x, 1..=1024) as usize
}

#[gpu_only]
#[inline(always)]
pub fn block_dim_y() -> usize {
    // CUDA Compute Capabilities: "Maximum x- or y-dimensionality of a block" is 1024.
    in_range!(core::arch::nvptx::_block_dim_y, 1..=1024) as usize
}

#[gpu_only]
#[inline(always)]
pub fn block_dim_z() -> usize {
    // CUDA Compute Capabilities: "Maximum z-dimension of a block" is 64.
    in_range!(core::arch::nvptx::_block_dim_z, 1..=64) as usize
}

#[gpu_only]
#[inline(always)]
pub fn grid_dim_x() -> usize {
    // CUDA Compute Capabilities: "Maximum x-dimension of a grid of thread blocks" is 2^32 - 1.
    in_range!(core::arch::nvptx::_grid_dim_x, 1..=2147483647) as usize
}

#[gpu_only]
#[inline(always)]
pub fn grid_dim_y() -> usize {
    // CUDA Compute Capabilities: "Maximum y- or z-dimension of a grid of thread blocks" is 65535.
    in_range!(core::arch::nvptx::_grid_dim_y, 1..=65535) as usize
}

#[gpu_only]
#[inline(always)]
pub fn grid_dim_z() -> usize {
    // CUDA Compute Capabilities: "Maximum y- or z-dimension of a grid of thread blocks" is 65535.
    in_range!(core::arch::nvptx::_grid_dim_z, 1..=65535) as usize
}

/// Gets the 3d index of the thread currently executing the kernel.
#[gpu_only]
#[inline(always)]
pub fn thread_idx() -> USizeVec3 {
    USizeVec3::new(thread_idx_x(), thread_idx_y(), thread_idx_z())
}

/// Gets the 3d index of the block that the thread currently executing the kernel is located in.
#[gpu_only]
#[inline(always)]
pub fn block_idx() -> USizeVec3 {
    USizeVec3::new(block_idx_x(), block_idx_y(), block_idx_z())
}

/// Gets the 3d layout of the thread blocks executing this kernel. In other words,
/// how many threads exist in each thread block in every direction.
#[gpu_only]
#[inline(always)]
pub fn block_dim() -> USizeVec3 {
    USizeVec3::new(block_dim_x(), block_dim_y(), block_dim_z())
}

/// Gets the 3d layout of the block grids executing this kernel. In other words,
/// how many thread blocks exist in each grid in every direction.
#[gpu_only]
#[inline(always)]
pub fn grid_dim() -> USizeVec3 {
    USizeVec3::new(grid_dim_x(), grid_dim_y(), grid_dim_z())
}

/// Gets the overall thread index, accounting for 1d/2d/3d block/grid dimensions. This
/// value is most commonly used for indexing into data and this index is guaranteed to
/// be unique for every single thread executing this kernel no matter the launch configuration.
///
/// For very simple kernels it may be faster to use a more simple index calculation, however,
/// it will be unsound if the kernel launches in a 2d/3d configuration.
#[gpu_only]
#[rustfmt::skip]
#[inline(always)]
pub fn index() -> usize {
    let grid_dim = grid_dim();
    let block_idx = block_idx();
    let block_dim = block_dim();
    let thread_idx = thread_idx();

    let block_id = block_idx.x + block_idx.y * grid_dim.x
                       + grid_dim.x * grid_dim.y * block_idx.z;

    block_id * block_dim.element_product()
    + (thread_idx.z * (block_dim.x * block_dim.y))
    + (thread_idx.y * block_dim.x) + thread_idx.x
}

#[inline(always)]
pub fn index_1d() -> usize {
    thread_idx_x() + block_idx_x() * block_dim_x()
}

#[inline(always)]
pub fn index_2d() -> USizeVec2 {
    let i = thread_idx_x() + block_idx_x() * block_dim_x();
    let j = thread_idx_y() + block_idx_y() * block_dim_y();
    USizeVec2::new(i, j)
}

#[inline(always)]
pub fn index_3d() -> USizeVec3 {
    let i = thread_idx_x() + block_idx_x() * block_dim_x();
    let j = thread_idx_y() + block_idx_y() * block_dim_y();
    let k = thread_idx_z() + block_idx_z() * block_dim_z();
    USizeVec3::new(i, j, k)
}

/// Whether this is the first thread (not the first thread to be executing). This function is guaranteed
/// to only return true in a single thread that is invoking it. This is useful for only doing something
/// once.
#[inline(always)]
pub fn first() -> bool {
    block_idx() == USizeVec3::ZERO && thread_idx() == USizeVec3::ZERO
}

/// Gets the number of threads inside of a warp. Currently 32 threads on every GPU architecture.
#[gpu_only]
#[inline(always)]
pub fn warp_size() -> u32 {
    unsafe { __nvvm_warp_size() }
}

/// Waits until all threads in the thread block have reached this point. This guarantees
/// that any global or shared mem accesses are visible to every thread after this call.
///
/// Be careful when using sync_threads in conditional code. It will be perfectly fine if
/// all threads evaluate to the same path, but if they dont, execution will halt
/// or produce odd results (but should not produce undefined behavior).
#[gpu_only]
#[inline(always)]
pub fn sync_threads() {
    unsafe { __nvvm_block_barrier() }
}

/// Identical to [`sync_threads`] but with the additional feature that it evaluates
/// the predicate for every thread and returns the number of threads in which it evaluated to a non-zero number.
#[gpu_only]
#[inline(always)]
pub fn sync_threads_count(predicate: u32) -> u32 {
    unsafe extern "C" {
        #[link_name = "llvm.nvvm.barrier0.popc"]
        fn __nvvm_sync_threads_count(predicate: u32) -> u32;
    }

    unsafe { __nvvm_sync_threads_count(predicate) }
}

/// Identical to [`sync_threads`] but with the additional feature that it evaluates
/// the predicate for every thread and returns a non-zero integer if every predicate evaluates to non-zero for all threads.
#[gpu_only]
#[inline(always)]
pub fn sync_threads_and(predicate: u32) -> u32 {
    unsafe extern "C" {
        #[link_name = "llvm.nvvm.barrier0.and"]
        fn __nvvm_sync_threads_and(predicate: u32) -> u32;
    }

    unsafe { __nvvm_sync_threads_and(predicate) }
}

/// Identical to [`sync_threads`] but with the additional feature that it evaluates
/// the predicate for every thread and returns a non-zero integer if at least one predicate in a thread evaluates
/// to non-zero.
#[gpu_only]
#[inline(always)]
pub fn sync_threads_or(predicate: u32) -> u32 {
    unsafe extern "C" {
        #[link_name = "llvm.nvvm.barrier0.or"]
        fn __nvvm_sync_threads_or(predicate: u32) -> u32;
    }

    unsafe { __nvvm_sync_threads_or(predicate) }
}

/// Acts as a memory fence at the grid level (all threads inside of a kernel execution).
///
/// Note that this is NOT an execution synchronization like [`sync_threads`]. It is not possible
/// to sync threads at a grid level. It is simply a memory fence.
#[gpu_only]
#[inline(always)]
pub fn grid_fence() {
    unsafe { __nvvm_grid_fence() }
}

/// Acts as a memory fence at the device level.
#[gpu_only]
#[inline(always)]
pub fn device_fence() {
    unsafe { __nvvm_device_fence() }
}

/// Acts as a memory fence at the system level.
#[gpu_only]
#[inline(always)]
pub fn system_fence() {
    unsafe { __nvvm_system_fence() }
}

/// Suspends the calling thread for a duration (in nanoseconds) approximately close to `nanos`.
///
/// This is useful for implementing something like a mutex with exponential back-off.
#[gpu_only]
#[inline(always)]
pub fn nanosleep(nanos: u32) {
    unsafe {
        core::arch::asm!(
            "nanosleep {}",
            in(reg32) nanos
        )
    }
}
