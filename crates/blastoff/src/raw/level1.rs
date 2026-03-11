use std::os::raw::c_int;

use cust_raw::cublas_sys::*;
use num_complex::{Complex32, Complex64};

use crate::BlasDatatype;

pub trait Level1: BlasDatatype {
    unsafe fn amax(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t;
    unsafe fn amin(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t;
    unsafe fn axpy(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const Self,
        x: *const Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t;
    unsafe fn copy(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t;
    unsafe fn nrm2(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut Self::FloatTy,
    ) -> cublasStatus_t;
    unsafe fn rot(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
        c: *const Self::FloatTy,
        s: *const Self::FloatTy,
    ) -> cublasStatus_t;
    unsafe fn rotg(
        handle: cublasHandle_t,
        a: *mut Self,
        b: *mut Self,
        c: *mut Self::FloatTy,
        s: *mut Self,
    ) -> cublasStatus_t;
    unsafe fn rotm(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
        param: *const Self::FloatTy,
    ) -> cublasStatus_t;
    unsafe fn rotmg(
        handle: cublasHandle_t,
        d1: *mut Self,
        d2: *mut Self,
        x1: *mut Self,
        y1: *const Self,
        param: *mut Self,
    ) -> cublasStatus_t;
    unsafe fn scal(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const Self,
        x: *mut Self,
        incx: c_int,
    ) -> cublasStatus_t;
    unsafe fn swap(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t;
}

impl Level1 for f32 {
    unsafe fn amax(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t {
        unsafe { cublasIsamax(handle, n, x, incx, result) }
    }
    unsafe fn amin(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t {
        unsafe { cublasIsamin(handle, n, x, incx, result) }
    }
    unsafe fn axpy(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const Self,
        x: *const Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        unsafe { cublasSaxpy(handle, n, alpha, x, incx, y, incy) }
    }
    unsafe fn copy(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        unsafe { cublasScopy(handle, n, x, incx, y, incy) }
    }
    unsafe fn nrm2(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut Self::FloatTy,
    ) -> cublasStatus_t {
        unsafe { cublasSnrm2(handle, n, x, incx, result) }
    }
    unsafe fn rot(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
        c: *const Self::FloatTy,
        s: *const Self,
    ) -> cublasStatus_t {
        unsafe { cublasSrot(handle, n, x, incx, y, incy, c, s) }
    }
    unsafe fn rotg(
        handle: cublasHandle_t,
        a: *mut Self,
        b: *mut Self,
        c: *mut Self::FloatTy,
        s: *mut Self,
    ) -> cublasStatus_t {
        unsafe { cublasSrotg(handle, a, b, c, s) }
    }
    unsafe fn rotm(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
        param: *const Self::FloatTy,
    ) -> cublasStatus_t {
        unsafe { cublasSrotm(handle, n, x, incx, y, incy, param) }
    }
    unsafe fn rotmg(
        handle: cublasHandle_t,
        d1: *mut Self,
        d2: *mut Self,
        x1: *mut Self,
        y1: *const Self,
        param: *mut Self,
    ) -> cublasStatus_t {
        unsafe { cublasSrotmg(handle, d1, d2, x1, y1, param) }
    }
    unsafe fn scal(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const Self,
        x: *mut Self,
        incx: c_int,
    ) -> cublasStatus_t {
        unsafe { cublasSscal(handle, n, alpha, x, incx) }
    }
    unsafe fn swap(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        unsafe { cublasSswap(handle, n, x, incx, y, incy) }
    }
}

impl Level1 for f64 {
    unsafe fn amax(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t {
        unsafe { cublasIdamax(handle, n, x, incx, result) }
    }
    unsafe fn amin(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t {
        unsafe { cublasIdamin(handle, n, x, incx, result) }
    }
    unsafe fn axpy(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const Self,
        x: *const Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        unsafe { cublasDaxpy(handle, n, alpha, x, incx, y, incy) }
    }
    unsafe fn copy(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        unsafe { cublasDcopy(handle, n, x, incx, y, incy) }
    }
    unsafe fn nrm2(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut Self::FloatTy,
    ) -> cublasStatus_t {
        unsafe { cublasDnrm2(handle, n, x, incx, result) }
    }
    unsafe fn rot(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
        c: *const Self::FloatTy,
        s: *const Self,
    ) -> cublasStatus_t {
        unsafe { cublasDrot(handle, n, x, incx, y, incy, c, s) }
    }
    unsafe fn rotg(
        handle: cublasHandle_t,
        a: *mut Self,
        b: *mut Self,
        c: *mut Self::FloatTy,
        s: *mut Self,
    ) -> cublasStatus_t {
        unsafe { cublasDrotg(handle, a, b, c, s) }
    }
    unsafe fn rotm(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
        param: *const Self::FloatTy,
    ) -> cublasStatus_t {
        unsafe { cublasDrotm(handle, n, x, incx, y, incy, param) }
    }
    unsafe fn rotmg(
        handle: cublasHandle_t,
        d1: *mut Self,
        d2: *mut Self,
        x1: *mut Self,
        y1: *const Self,
        param: *mut Self,
    ) -> cublasStatus_t {
        unsafe { cublasDrotmg(handle, d1, d2, x1, y1, param) }
    }
    unsafe fn scal(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const Self,
        x: *mut Self,
        incx: c_int,
    ) -> cublasStatus_t {
        unsafe { cublasDscal(handle, n, alpha, x, incx) }
    }
    unsafe fn swap(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        unsafe { cublasDswap(handle, n, x, incx, y, incy) }
    }
}

impl Level1 for Complex32 {
    unsafe fn amax(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t {
        unsafe { cublasIcamax(handle, n, x.cast(), incx, result) }
    }
    unsafe fn amin(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t {
        unsafe { cublasIcamin(handle, n, x.cast(), incx, result) }
    }
    unsafe fn axpy(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const Self,
        x: *const Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        unsafe { cublasCaxpy(handle, n, alpha.cast(), x.cast(), incx, y.cast(), incy) }
    }
    unsafe fn copy(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        unsafe { cublasCcopy(handle, n, x.cast(), incx, y.cast(), incy) }
    }
    unsafe fn nrm2(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut Self::FloatTy,
    ) -> cublasStatus_t {
        unsafe { cublasScnrm2(handle, n, x.cast(), incx, result) }
    }
    unsafe fn rot(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
        c: *const Self::FloatTy,
        s: *const Self::FloatTy,
    ) -> cublasStatus_t {
        unsafe { cublasCsrot(handle, n, x.cast(), incx, y.cast(), incy, c, s) }
    }
    unsafe fn rotg(
        handle: cublasHandle_t,
        a: *mut Self,
        b: *mut Self,
        c: *mut Self::FloatTy,
        s: *mut Self,
    ) -> cublasStatus_t {
        unsafe { cublasCrotg(handle, a.cast(), b.cast(), c, s.cast()) }
    }
    unsafe fn rotm(
        _handle: cublasHandle_t,
        _n: c_int,
        _x: *mut Self,
        _incx: c_int,
        _y: *mut Self,
        _incy: c_int,
        _param: *const Self::FloatTy,
    ) -> cublasStatus_t {
        unreachable!()
    }
    unsafe fn rotmg(
        _handle: cublasHandle_t,
        _d1: *mut Self,
        _d2: *mut Self,
        _x1: *mut Self,
        _y1: *const Self,
        _param: *mut Self,
    ) -> cublasStatus_t {
        unreachable!()
    }
    unsafe fn scal(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const Self,
        x: *mut Self,
        incx: c_int,
    ) -> cublasStatus_t {
        unsafe { cublasCscal(handle, n, alpha.cast(), x.cast(), incx) }
    }
    unsafe fn swap(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        unsafe { cublasCswap(handle, n, x.cast(), incx, y.cast(), incy) }
    }
}

impl Level1 for Complex64 {
    unsafe fn amax(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t {
        unsafe { cublasIzamax(handle, n, x.cast(), incx, result) }
    }
    unsafe fn amin(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t {
        unsafe { cublasIzamin(handle, n, x.cast(), incx, result) }
    }
    unsafe fn axpy(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const Self,
        x: *const Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        unsafe { cublasZaxpy(handle, n, alpha.cast(), x.cast(), incx, y.cast(), incy) }
    }
    unsafe fn copy(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        unsafe { cublasZcopy(handle, n, x.cast(), incx, y.cast(), incy) }
    }
    unsafe fn nrm2(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut Self::FloatTy,
    ) -> cublasStatus_t {
        unsafe { cublasDznrm2(handle, n, x.cast(), incx, result) }
    }
    unsafe fn rot(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
        c: *const Self::FloatTy,
        s: *const Self::FloatTy,
    ) -> cublasStatus_t {
        unsafe { cublasZdrot(handle, n, x.cast(), incx, y.cast(), incy, c, s) }
    }
    unsafe fn rotg(
        handle: cublasHandle_t,
        a: *mut Self,
        b: *mut Self,
        c: *mut Self::FloatTy,
        s: *mut Self,
    ) -> cublasStatus_t {
        unsafe { cublasZrotg(handle, a.cast(), b.cast(), c, s.cast()) }
    }
    unsafe fn rotm(
        _handle: cublasHandle_t,
        _n: c_int,
        _x: *mut Self,
        _incx: c_int,
        _y: *mut Self,
        _incy: c_int,
        _param: *const Self::FloatTy,
    ) -> cublasStatus_t {
        unreachable!()
    }
    unsafe fn rotmg(
        _handle: cublasHandle_t,
        _d1: *mut Self,
        _d2: *mut Self,
        _x1: *mut Self,
        _y1: *const Self,
        _param: *mut Self,
    ) -> cublasStatus_t {
        unreachable!()
    }
    unsafe fn scal(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const Self,
        x: *mut Self,
        incx: c_int,
    ) -> cublasStatus_t {
        unsafe { cublasZscal(handle, n, alpha.cast(), x.cast(), incx) }
    }
    unsafe fn swap(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        unsafe { cublasZswap(handle, n, x.cast(), incx, y.cast(), incy) }
    }
}

/// Level-1 Methods exclusive to complex numbers.
pub trait ComplexLevel1: BlasDatatype {
    unsafe fn dotu(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *const Self,
        incy: c_int,
        result: *mut Self,
    ) -> cublasStatus_t;
    unsafe fn dotc(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *const Self,
        incy: c_int,
        result: *mut Self,
    ) -> cublasStatus_t;
}

impl ComplexLevel1 for Complex32 {
    unsafe fn dotu(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *const Self,
        incy: c_int,
        result: *mut Self,
    ) -> cublasStatus_t {
        unsafe { cublasCdotu(handle, n, x.cast(), incx, y.cast(), incy, result.cast()) }
    }
    unsafe fn dotc(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *const Self,
        incy: c_int,
        result: *mut Self,
    ) -> cublasStatus_t {
        unsafe { cublasCdotc(handle, n, x.cast(), incx, y.cast(), incy, result.cast()) }
    }
}

impl ComplexLevel1 for Complex64 {
    unsafe fn dotu(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *const Self,
        incy: c_int,
        result: *mut Self,
    ) -> cublasStatus_t {
        unsafe { cublasZdotu(handle, n, x.cast(), incx, y.cast(), incy, result.cast()) }
    }
    unsafe fn dotc(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *const Self,
        incy: c_int,
        result: *mut Self,
    ) -> cublasStatus_t {
        unsafe { cublasZdotc(handle, n, x.cast(), incx, y.cast(), incy, result.cast()) }
    }
}

/// Level-1 Methods exclusive to floats.
pub trait FloatLevel1: BlasDatatype {
    unsafe fn dot(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *const Self,
        incy: c_int,
        result: *mut Self,
    ) -> cublasStatus_t;
}

impl FloatLevel1 for f32 {
    unsafe fn dot(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *const Self,
        incy: c_int,
        result: *mut Self,
    ) -> cublasStatus_t {
        unsafe { cublasSdot(handle, n, x, incx, y, incy, result) }
    }
}

impl FloatLevel1 for f64 {
    unsafe fn dot(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *const Self,
        incy: c_int,
        result: *mut Self,
    ) -> cublasStatus_t {
        unsafe { cublasDdot(handle, n, x, incx, y, incy, result) }
    }
}
