use std::os::raw::c_int;

use cust_raw::cublas_sys::*;
use num_complex::{Complex32, Complex64};

use crate::GemmDatatype;

pub trait GemmOps: GemmDatatype {
    unsafe fn gemm(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const Self,
        a: *const Self,
        lda: c_int,
        b: *const Self,
        ldb: c_int,
        beta: *const Self,
        c: *mut Self,
        ldc: c_int,
    ) -> cublasStatus_t;
}

#[cfg(feature = "half")]
impl GemmOps for half::f16 {
    unsafe fn gemm(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const Self,
        a: *const Self,
        lda: c_int,
        b: *const Self,
        ldb: c_int,
        beta: *const Self,
        c: *mut Self,
        ldc: c_int,
    ) -> cublasStatus_t {
        // for some weird reason cublas only defines Hgemm if __cplusplus is defined, no idea why
        // but for now we just link against it manually, in the future we should figure out why
        unsafe extern "C" {
            fn cublasHgemm(
                handle: cublasHandle_t,
                transa: cublasOperation_t,
                transb: cublasOperation_t,
                m: c_int,
                n: c_int,
                k: c_int,
                alpha: *const half::f16,
                a: *const half::f16,
                lda: c_int,
                b: *const half::f16,
                ldb: c_int,
                beta: *const half::f16,
                c: *mut half::f16,
                ldc: c_int,
            ) -> cublasStatus_t;
        }
        unsafe {
            cublasHgemm(
                handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
            )
        }
    }
}

impl GemmOps for f32 {
    unsafe fn gemm(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const Self,
        a: *const Self,
        lda: c_int,
        b: *const Self,
        ldb: c_int,
        beta: *const Self,
        c: *mut Self,
        ldc: c_int,
    ) -> cublasStatus_t {
        unsafe {
            cublasSgemm(
                handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
            )
        }
    }
}

impl GemmOps for f64 {
    unsafe fn gemm(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const Self,
        a: *const Self,
        lda: c_int,
        b: *const Self,
        ldb: c_int,
        beta: *const Self,
        c: *mut Self,
        ldc: c_int,
    ) -> cublasStatus_t {
        unsafe {
            cublasDgemm(
                handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
            )
        }
    }
}

impl GemmOps for Complex32 {
    unsafe fn gemm(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const Self,
        a: *const Self,
        lda: c_int,
        b: *const Self,
        ldb: c_int,
        beta: *const Self,
        c: *mut Self,
        ldc: c_int,
    ) -> cublasStatus_t {
        unsafe {
            cublasCgemm(
                handle,
                transa,
                transb,
                m,
                n,
                k,
                alpha.cast(),
                a.cast(),
                lda,
                b.cast(),
                ldb,
                beta.cast(),
                c.cast(),
                ldc,
            )
        }
    }
}

impl GemmOps for Complex64 {
    unsafe fn gemm(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const Self,
        a: *const Self,
        lda: c_int,
        b: *const Self,
        ldb: c_int,
        beta: *const Self,
        c: *mut Self,
        ldc: c_int,
    ) -> cublasStatus_t {
        unsafe {
            cublasCgemm(
                handle,
                transa,
                transb,
                m,
                n,
                k,
                alpha.cast(),
                a.cast(),
                lda,
                b.cast(),
                ldb,
                beta.cast(),
                c.cast(),
                ldc,
            )
        }
    }
}
