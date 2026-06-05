use std::marker::PhantomData;
use std::mem::MaybeUninit;

use cust::memory::GpuBuffer;

use crate::{CufftError, IntoResult};

mod sealed {
    pub trait Sealed {}
}

pub trait FftType: sealed::Sealed {
    #[doc(hidden)]
    fn fft_type() -> cufft_raw::cufftType;
}

/// Marker type for single-precision complex-to-complex transforms.
pub struct C2C;
/// Marker type for single-precision real-to-complex transforms.
pub struct R2C;
/// Marker type for single-precision complex-to-real transforms.
pub struct C2R;
/// Marker type for double-precision real-to-complex transforms.
pub struct D2Z;
/// Marker type for double-precision complex-to-real transforms.
pub struct Z2D;
/// Marker type for double-precision complex-to-complex transforms.
pub struct Z2Z;

impl sealed::Sealed for C2C {}
impl sealed::Sealed for R2C {}
impl sealed::Sealed for C2R {}
impl sealed::Sealed for D2Z {}
impl sealed::Sealed for Z2D {}
impl sealed::Sealed for Z2Z {}

impl FftType for C2C {
    fn fft_type() -> cufft_raw::cufftType {
        cufft_raw::cufftType::CUFFT_C2C
    }
}

impl FftType for R2C {
    fn fft_type() -> cufft_raw::cufftType {
        cufft_raw::cufftType::CUFFT_R2C
    }
}

impl FftType for C2R {
    fn fft_type() -> cufft_raw::cufftType {
        cufft_raw::cufftType::CUFFT_C2R
    }
}

impl FftType for D2Z {
    fn fft_type() -> cufft_raw::cufftType {
        cufft_raw::cufftType::CUFFT_D2Z
    }
}

impl FftType for Z2D {
    fn fft_type() -> cufft_raw::cufftType {
        cufft_raw::cufftType::CUFFT_Z2D
    }
}

impl FftType for Z2Z {
    fn fft_type() -> cufft_raw::cufftType {
        cufft_raw::cufftType::CUFFT_Z2Z
    }
}

/// FFT direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Forward DFT (CUFFT_FORWARD = -1).
    Forward,
    /// Inverse DFT (CUFFT_INVERSE = +1).
    Inverse,
}

impl Direction {
    pub(crate) fn into_raw(self) -> i32 {
        match self {
            Direction::Forward => cufft_raw::CUFFT_FORWARD,
            Direction::Inverse => cufft_raw::CUFFT_INVERSE as i32,
        }
    }
}

/// Wrapper for a `cufftHandle`.
#[derive(Debug)]
pub struct FftPlan<T> {
    raw: cufft_raw::cufftHandle,
    _marker: PhantomData<T>,
}

impl<T: FftType> FftPlan<T> {
    /// Creates a 1-D FFT plan.
    ///
    /// # Reference
    ///
    /// [cufftPlan1d](https://docs.nvidia.com/cuda/cufft/index.html#cufftplan1d)
    pub fn plan_1d(nx: i32, batch: i32) -> Result<Self, CufftError> {
        let mut raw = MaybeUninit::uninit();

        unsafe {
            cufft_raw::cufftPlan1d(raw.as_mut_ptr(), nx, T::fft_type(), batch).into_result()?;

            Ok(Self {
                raw: raw.assume_init(),
                _marker: PhantomData,
            })
        }
    }

    /// Creates a 2-D FFT plan.
    ///
    /// # Reference
    ///
    /// [cufftPlan2d](https://docs.nvidia.com/cuda/cufft/index.html#cufftplan2d)
    pub fn plan_2d(nx: i32, ny: i32) -> Result<Self, CufftError> {
        let mut raw = MaybeUninit::uninit();

        unsafe {
            cufft_raw::cufftPlan2d(raw.as_mut_ptr(), nx, ny, T::fft_type()).into_result()?;

            Ok(Self {
                raw: raw.assume_init(),
                _marker: PhantomData,
            })
        }
    }

    /// Creates a 3-D FFT plan.
    ///
    /// # Reference
    ///
    /// [cufftPlan3d](https://docs.nvidia.com/cuda/cufft/index.html#cufftplan3d)
    pub fn plan_3d(nx: i32, ny: i32, nz: i32) -> Result<Self, CufftError> {
        let mut raw = MaybeUninit::uninit();

        unsafe {
            cufft_raw::cufftPlan3d(raw.as_mut_ptr(), nx, ny, nz, T::fft_type()).into_result()?;

            Ok(Self {
                raw: raw.assume_init(),
                _marker: PhantomData,
            })
        }
    }

    /// Creates a batched N-D FFT plan with full stride/distance control.
    ///
    /// # Reference
    ///
    /// [cufftPlanMany](https://docs.nvidia.com/cuda/cufft/index.html#cufftplanmany)
    #[allow(clippy::too_many_arguments)]
    pub fn plan_many(
        rank: i32,
        n: &[i32],
        inembed: Option<&[i32]>,
        istride: i32,
        idist: i32,
        onembed: Option<&[i32]>,
        ostride: i32,
        odist: i32,
        batch: i32,
    ) -> Result<Self, CufftError> {
        let mut raw = MaybeUninit::uninit();

        let inembed_ptr = inembed.map_or(std::ptr::null_mut(), |s| s.as_ptr().cast_mut());
        let onembed_ptr = onembed.map_or(std::ptr::null_mut(), |s| s.as_ptr().cast_mut());

        unsafe {
            cufft_raw::cufftPlanMany(
                raw.as_mut_ptr(),
                rank,
                n.as_ptr().cast_mut(),
                inembed_ptr,
                istride,
                idist,
                onembed_ptr,
                ostride,
                odist,
                T::fft_type(),
                batch,
            )
            .into_result()?;

            Ok(Self {
                raw: raw.assume_init(),
                _marker: PhantomData,
            })
        }
    }

    /// Sets the CUDA stream for the plan.
    ///
    /// # Reference
    ///
    /// [cufftSetStream](https://docs.nvidia.com/cuda/cufft/index.html#cufftsetstream)
    pub fn set_stream(&mut self, stream: &cust::stream::Stream) -> Result<(), CufftError> {
        unsafe {
            cufft_raw::cufftSetStream(
                self.raw,
                stream.as_inner() as *mut _ as cufft_raw::cudaStream_t,
            )
            .into_result()
        }
    }

    /// Returns the raw `cufftHandle`.
    pub fn as_raw(&self) -> cufft_raw::cufftHandle {
        self.raw
    }
}

impl FftPlan<C2C> {
    /// Executes a single-precision C2C FFT.
    ///
    /// # Reference
    ///
    /// [cufftExecC2C](https://docs.nvidia.com/cuda/cufft/index.html#cufftexecc2c-and-cufftexecz2z)
    pub fn exec(
        &self,
        idata: &impl GpuBuffer<cufft_raw::cufftComplex>,
        odata: &mut impl GpuBuffer<cufft_raw::cufftComplex>,
        direction: Direction,
    ) -> Result<(), CufftError> {
        unsafe {
            cufft_raw::cufftExecC2C(
                self.raw,
                idata.as_device_ptr().as_mut_ptr(),
                odata.as_device_ptr().as_mut_ptr(),
                direction.into_raw(),
            )
            .into_result()
        }
    }
}

impl FftPlan<R2C> {
    /// Executes a single-precision R2C FFT.
    ///
    /// # Reference
    ///
    /// [cufftExecR2C](https://docs.nvidia.com/cuda/cufft/index.html#cufftexecr2c-and-cufftexecd2z)
    pub fn exec(
        &self,
        idata: &impl GpuBuffer<cufft_raw::cufftReal>,
        odata: &mut impl GpuBuffer<cufft_raw::cufftComplex>,
    ) -> Result<(), CufftError> {
        unsafe {
            cufft_raw::cufftExecR2C(
                self.raw,
                idata.as_device_ptr().as_mut_ptr(),
                odata.as_device_ptr().as_mut_ptr(),
            )
            .into_result()
        }
    }
}

impl FftPlan<C2R> {
    /// Executes a single-precision C2R inverse FFT.
    ///
    /// # Reference
    ///
    /// [cufftExecC2R](https://docs.nvidia.com/cuda/cufft/index.html#cufftexecc2r-and-cufftexecz2d)
    pub fn exec(
        &self,
        idata: &impl GpuBuffer<cufft_raw::cufftComplex>,
        odata: &mut impl GpuBuffer<cufft_raw::cufftReal>,
    ) -> Result<(), CufftError> {
        unsafe {
            cufft_raw::cufftExecC2R(
                self.raw,
                idata.as_device_ptr().as_mut_ptr(),
                odata.as_device_ptr().as_mut_ptr(),
            )
            .into_result()
        }
    }
}

impl FftPlan<Z2Z> {
    /// Executes a double-precision Z2Z FFT.
    ///
    /// # Reference
    ///
    /// [cufftExecZ2Z](https://docs.nvidia.com/cuda/cufft/index.html#cufftexecc2c-and-cufftexecz2z)
    pub fn exec(
        &self,
        idata: &impl GpuBuffer<cufft_raw::cufftDoubleComplex>,
        odata: &mut impl GpuBuffer<cufft_raw::cufftDoubleComplex>,
        direction: Direction,
    ) -> Result<(), CufftError> {
        unsafe {
            cufft_raw::cufftExecZ2Z(
                self.raw,
                idata.as_device_ptr().as_mut_ptr(),
                odata.as_device_ptr().as_mut_ptr(),
                direction.into_raw(),
            )
            .into_result()
        }
    }
}

impl FftPlan<D2Z> {
    /// Executes a double-precision D2Z FFT.
    ///
    /// # Reference
    ///
    /// [cufftExecD2Z](https://docs.nvidia.com/cuda/cufft/index.html#cufftexecr2c-and-cufftexecd2z)
    pub fn exec(
        &self,
        idata: &impl GpuBuffer<cufft_raw::cufftDoubleReal>,
        odata: &mut impl GpuBuffer<cufft_raw::cufftDoubleComplex>,
    ) -> Result<(), CufftError> {
        unsafe {
            cufft_raw::cufftExecD2Z(
                self.raw,
                idata.as_device_ptr().as_mut_ptr(),
                odata.as_device_ptr().as_mut_ptr(),
            )
            .into_result()
        }
    }
}

impl FftPlan<Z2D> {
    /// Executes a double-precision Z2D inverse FFT.
    ///
    /// # Reference
    ///
    /// [cufftExecZ2D](https://docs.nvidia.com/cuda/cufft/index.html#cufftexecc2r-and-cufftexecz2d)
    pub fn exec(
        &self,
        idata: &impl GpuBuffer<cufft_raw::cufftDoubleComplex>,
        odata: &mut impl GpuBuffer<cufft_raw::cufftDoubleReal>,
    ) -> Result<(), CufftError> {
        unsafe {
            cufft_raw::cufftExecZ2D(
                self.raw,
                idata.as_device_ptr().as_mut_ptr(),
                odata.as_device_ptr().as_mut_ptr(),
            )
            .into_result()
        }
    }
}

impl<T> Drop for FftPlan<T> {
    /// Destroys the plan.
    ///
    /// # Reference
    ///
    /// [cufftDestroy](https://docs.nvidia.com/cuda/cufft/index.html#cufftdestroy)
    fn drop(&mut self) {
        unsafe {
            let _ = cufft_raw::cufftDestroy(self.raw);
        }
    }
}
