use crate::{
    GemmDatatype, MatrixOp,
    context::CublasContext,
    error::{Error, ToResult},
    raw::GemmOps,
};
use cust::memory::{GpuBox, GpuBuffer};
use cust::stream::Stream;

type Result<T = (), E = Error> = std::result::Result<T, E>;

#[track_caller]
fn check_gemm<T: GemmDatatype + GemmOps>(
    m: usize,
    n: usize,
    k: usize,
    a: &impl GpuBuffer<T>,
    lda: usize,
    op_a: MatrixOp,
    b: &impl GpuBuffer<T>,
    ldb: usize,
    op_b: MatrixOp,
    c: &mut impl GpuBuffer<T>,
    ldc: usize,
) {
    assert!(m > 0 && n > 0 && k > 0, "m, n, and k must be at least 1");

    if op_a == MatrixOp::None {
        assert!(lda >= m, "lda must be at least m if op_a is None");

        assert!(
            a.len() >= lda * k,
            "matrix A's length must be at least lda * k"
        );
    } else {
        assert!(lda >= k, "lda must be at least k if op_a is None");

        assert!(
            a.len() >= lda * m,
            "matrix A's length must be at least lda * m"
        );
    }

    if op_b == MatrixOp::None {
        assert!(ldb >= k, "ldb must be at least k if op_b is None");

        assert!(
            b.len() >= ldb * n,
            "matrix B's length must be at least ldb * n"
        );
    } else {
        assert!(ldb >= n, "ldb must be at least n if op_b is None");

        assert!(
            a.len() >= ldb * k,
            "matrix B's length must be at least ldb * k"
        );
    }

    assert!(ldc >= m, "ldc must be at least m");

    assert!(
        c.len() >= ldc * n,
        "matrix C's length must be at least ldc * n"
    );
}

impl CublasContext {
    /// Generic Matrix Multiplication.
    ///
    /// # Panics
    ///
    /// Panics if any of the following conditions are not met:
    /// - `m > 0 && n > 0 && k > 0`
    /// - `lda >= m` if `op_a == MatrixOp::None`
    /// - `a.len() >= lda * k` if `op_a == MatrixOp::None`
    /// - `lda >= k` if `op_a == MatrixOp::Transpose` or `MatrixOp::ConjugateTranspose`
    /// - `a.len() >= lda * m` if `op_a == MatrixOp::Transpose` or `MatrixOp::ConjugateTranspose`
    /// - `ldb >= k` if `op_b == MatrixOp::None`
    /// - `b.len() >= ldb * n` if `op_b == MatrixOp::None`
    /// - `ldb >= n` if `op_b == MatrixOp::Transpose` or `MatrixOp::ConjugateTranspose`
    /// - `b.len() >= ldb * k` if `op_b == MatrixOp::Transpose` or `MatrixOp::ConjugateTranspose`
    /// - `ldc >= m`
    /// - `c.len() >= ldc * n`
    ///
    /// # Errors
    ///
    /// Returns an error if the kernel execution failed or the selected precision is `half` and the device does not support half precision.
    #[track_caller]
    pub fn gemm<T: GemmDatatype + GemmOps>(
        &mut self,
        stream: &Stream,
        m: usize,
        n: usize,
        k: usize,
        alpha: &impl GpuBox<T>,
        a: &impl GpuBuffer<T>,
        lda: usize,
        op_a: MatrixOp,
        beta: &impl GpuBox<T>,
        b: &impl GpuBuffer<T>,
        ldb: usize,
        op_b: MatrixOp,
        c: &mut impl GpuBuffer<T>,
        ldc: usize,
    ) -> Result {
        check_gemm(m, n, k, a, lda, op_a, b, ldb, op_b, c, ldc);

        let transa = op_a.to_raw();
        let transb = op_b.to_raw();

        self.with_stream(stream, |ctx| unsafe {
            Ok(T::gemm(
                ctx.raw,
                transa,
                transb,
                m as i32,
                n as i32,
                k as i32,
                alpha.as_device_ptr().as_ptr(),
                a.as_device_ptr().as_ptr(),
                lda as i32,
                b.as_device_ptr().as_ptr(),
                ldb as i32,
                beta.as_device_ptr().as_ptr(),
                c.as_device_ptr().as_mut_ptr(),
                ldc as i32,
            )
            .to_result()?)
        })
    }
}
