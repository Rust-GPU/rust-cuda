// A sample Rust CUDA kernel for Compiler Explorer.
//
// This demonstrates shared-memory tiling, thread indexing, and
// synchronisation -- the core patterns used in GPU programming
// with rust-cuda.

use cuda_std::prelude::*;
use core::mem::MaybeUninit;

const TILE: usize = 16;

/// Tiled matrix-vector multiply: y = A * x.
///
/// Each block collaboratively loads a tile of A into shared memory,
/// then each thread accumulates its dot-product contribution.
///
/// - `a`:  row-major matrix, m rows x n cols
/// - `x`:  input vector, length n
/// - `y`:  output vector, length m (must be pre-zeroed)
/// - `m`:  number of rows
/// - `n`:  number of columns
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn matvec(a: &[f32], x: &[f32], y: *mut f32, m: usize, n: usize) {
    #[address_space(shared)]
    static mut TILE_A: [MaybeUninit<f32>; TILE] = [MaybeUninit::uninit(); TILE];

    let row = thread::block_idx_x() as usize * thread::block_dim_x() as usize
        + thread::thread_idx_x() as usize;
    let tx = thread::thread_idx_x() as usize;

    let mut sum = 0.0f32;

    // Walk across the columns in tiles of size TILE.
    let mut col = 0usize;
    while col < n {
        // Collaboratively load one tile of x into shared memory.
        if col + tx < n {
            unsafe {
                TILE_A[tx].write(x[col + tx]);
            }
        } else {
            unsafe {
                TILE_A[tx].write(0.0);
            }
        }
        thread::sync_threads();

        // Each thread accumulates the dot product for its row.
        if row < m {
            let mut k = 0usize;
            while k < TILE && col + k < n {
                sum += a[row * n + (col + k)] * unsafe { TILE_A[k].assume_init() };
                k += 1;
            }
        }
        thread::sync_threads();

        col += TILE;
    }

    if row < m {
        let out = unsafe { &mut *y.add(row) };
        *out = sum;
    }
}

/// Element-wise vector addition (simple baseline for comparison).
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn vecadd(a: &[f32], b: &[f32], c: *mut f32) {
    let idx = thread::index_1d() as usize;
    if idx < a.len() {
        let elem = unsafe { &mut *c.add(idx) };
        *elem = a[idx] + b[idx];
    }
}
