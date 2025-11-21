use cuda_std::address_space;
use cuda_std::kernel;
use cuda_std::thread;

#[kernel]
#[allow(improper_ctypes_definitions)]
/// Tiled GEMM kernel for C = alpha * A * B + beta * C.
///
/// This kernel uses shared memory tiling to improve memory access patterns and performance.
///
/// # Safety
/// CUDA kernel requires unsafe.
///
/// # Parameters
/// - `mat_a`: Input matrix A, shape (m x k), row-major order.
/// - `mat_b`: Input matrix B, shape (k x n), row-major order.
/// - `mat_c`: Output matrix C, shape (m x n), row-major order. Must be valid for writes.
/// - `m`: Number of rows in A and C.
/// - `n`: Number of columns in B and C.
/// - `k`: Number of columns in A and rows in B.
/// - `alpha`: Scalar multiplier for A * B.
/// - `beta`: Scalar multiplier for C.
///
/// # Tiling
/// Each block computes a TILE_SIZE x TILE_SIZE tile of C using shared memory for A and B tiles.
/// Threads within a block collaboratively load tiles and compute partial sums.
///
/// # Thread Mapping
/// Each thread computes one element of the output tile.
pub unsafe fn gemm_tiled(
    mat_a: &[f32],
    mat_b: &[f32],
    mat_c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    const TILE_SIZE: usize = 16;

    #[address_space(shared)]
    static mut TILE_A: [f32; TILE_SIZE * TILE_SIZE] = [0.; TILE_SIZE * TILE_SIZE];
    #[address_space(shared)]
    static mut TILE_B: [f32; TILE_SIZE * TILE_SIZE] = [0.; TILE_SIZE * TILE_SIZE];

    // Thread indices within the block.
    let tx = thread::thread_idx_x() as usize;
    let ty = thread::thread_idx_y() as usize;

    // Calculate row and column in the mat_c.
    let row = thread::block_idx_x() as usize * TILE_SIZE + ty;
    let col = thread::block_idx_y() as usize * TILE_SIZE + tx;

    let mut sum = 0.0f32;
    // Loop over tiles of mat_a and mat_b in the k dimension.
    for kk in (0..k).step_by(TILE_SIZE) {
        // Collaborative loading of tiles into shared memory.
        if row < m && (kk + tx) < k {
            unsafe { TILE_A[ty * TILE_SIZE + tx] = mat_a[row * k + (kk + tx)] };
        } else {
            unsafe { TILE_A[ty * TILE_SIZE + tx] = 0.0f32 };
        }
        if col < n && (kk + ty) < k {
            unsafe { TILE_B[ty * TILE_SIZE + tx] = mat_b[(kk + ty) * n + col] };
        } else {
            unsafe { TILE_B[ty * TILE_SIZE + tx] = 0.0f32 };
        }
        thread::sync_threads();

        // Perform the computation on the tile.
        for i in 0..TILE_SIZE {
            sum += unsafe { TILE_A[ty * TILE_SIZE + i] * TILE_B[i * TILE_SIZE + tx] };
        }
        thread::sync_threads();
    }

    // Write the result back to mat_c with alpha and beta scaling.
    if row < m && col < n {
        let c = unsafe { mat_c.add(row * n + col) };
        unsafe { *c = alpha * sum + beta * *c };
    }
}
