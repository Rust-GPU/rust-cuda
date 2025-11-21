use cuda_std::kernel;
use cuda_std::thread;

#[kernel]
#[allow(improper_ctypes_definitions)]
/// Naive GEMM kernel for C = alpha * A * B + beta * C.
///
/// This kernel computes each element of the output matrix C independently, without any memory coalescing or tiling optimizations.
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
/// # Thread Mapping
/// Each thread computes one element of C at (row, col).
pub unsafe fn gemm_naive(
    mat_a: &[f32],
    mat_b: &[f32],
    mat_c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    let row = (thread::block_dim_x() * thread::block_idx_x() + thread::thread_idx_x()) as usize;
    let col = (thread::block_dim_y() * thread::block_idx_y() + thread::thread_idx_y()) as usize;

    if row < m && col < n {
        let mut sum = 0.0f32;
        for i in 0..k {
            sum += mat_a[row * k + i] * mat_b[i * n + col];
        }
        let elem = unsafe { &mut *mat_c.add((row * n + col) as usize) };
        *elem = alpha * sum + beta * *elem;
    }
}
