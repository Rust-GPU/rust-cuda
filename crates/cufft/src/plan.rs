use std::mem::MaybeUninit;

use cust::memory::GpuBuffer;

use crate::{CufftError, IntoResult};

/// cuFFT transform type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FftType {
    /// Single-precision real-to-complex.
    R2C,
    /// Single-precision complex-to-real.
    C2R,
    /// Single-precision complex-to-complex.
    C2C,
    /// Double-precision real-to-complex.
    D2Z,
    /// Double-precision complex-to-real.
    Z2D,
    /// Double-precision complex-to-complex.
    Z2Z,
}

impl FftType {
    fn into_raw(self) -> cufft_raw::cufftType {
        use cufft_raw::cufftType::*;
        match self {
            FftType::R2C => CUFFT_R2C,
            FftType::C2R => CUFFT_C2R,
            FftType::C2C => CUFFT_C2C,
            FftType::D2Z => CUFFT_D2Z,
            FftType::Z2D => CUFFT_Z2D,
            FftType::Z2Z => CUFFT_Z2Z,
        }
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
pub struct FftPlan {
    pub(crate) raw: cufft_raw::cufftHandle,
}

impl FftPlan {
    /// Creates a 1-D FFT plan.
    ///
    /// # Reference
    ///
    /// [cufftPlan1d](https://docs.nvidia.com/cuda/cufft/index.html#cufftplan1d)
    pub fn plan_1d(nx: i32, fft_type: FftType, batch: i32) -> Result<Self, CufftError> {
        let mut raw = MaybeUninit::uninit();

        unsafe {
            cufft_raw::cufftPlan1d(raw.as_mut_ptr(), nx, fft_type.into_raw(), batch)
                .into_result()?;

            Ok(Self {
                raw: raw.assume_init(),
            })
        }
    }

    /// Creates a 2-D FFT plan.
    ///
    /// # Reference
    ///
    /// [cufftPlan2d](https://docs.nvidia.com/cuda/cufft/index.html#cufftplan2d)
    pub fn plan_2d(nx: i32, ny: i32, fft_type: FftType) -> Result<Self, CufftError> {
        let mut raw = MaybeUninit::uninit();

        unsafe {
            cufft_raw::cufftPlan2d(raw.as_mut_ptr(), nx, ny, fft_type.into_raw()).into_result()?;

            Ok(Self {
                raw: raw.assume_init(),
            })
        }
    }

    /// Creates a 3-D FFT plan.
    ///
    /// # Reference
    ///
    /// [cufftPlan3d](https://docs.nvidia.com/cuda/cufft/index.html#cufftplan3d)
    pub fn plan_3d(nx: i32, ny: i32, nz: i32, fft_type: FftType) -> Result<Self, CufftError> {
        let mut raw = MaybeUninit::uninit();

        unsafe {
            cufft_raw::cufftPlan3d(raw.as_mut_ptr(), nx, ny, nz, fft_type.into_raw())
                .into_result()?;

            Ok(Self {
                raw: raw.assume_init(),
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
        fft_type: FftType,
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
                fft_type.into_raw(),
                batch,
            )
            .into_result()?;

            Ok(Self {
                raw: raw.assume_init(),
            })
        }
    }

    /// Set the CUDA stream for the plan.
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

    /// Executes a single-precision C2C FFT.
    ///
    /// # Reference
    ///
    /// [cufftExecC2C](https://docs.nvidia.com/cuda/cufft/index.html#cufftexecc2c-and-cufftexecz2z)
    pub fn exec_c2c(
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

    /// Executes a single-precision R2C FFT.
    ///
    /// # Reference
    ///
    /// [cufftExecR2C](https://docs.nvidia.com/cuda/cufft/index.html#cufftexecr2c-and-cufftexecd2z)
    pub fn exec_r2c(
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

    /// Executes a single-precision C2R inverse FFT.
    ///
    /// # Reference
    ///
    /// [cufftExecC2R](https://docs.nvidia.com/cuda/cufft/index.html#cufftexecc2r-and-cufftexecz2d)
    pub fn exec_c2r(
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

    /// Executes a double-precision Z2Z FFT.
    ///
    /// # Reference
    ///
    /// [cufftExecZ2Z](https://docs.nvidia.com/cuda/cufft/index.html#cufftexecc2c-and-cufftexecz2z)
    pub fn exec_z2z(
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

    /// Executes a double-precision D2Z FFT.
    ///
    /// # Reference
    ///
    /// [cufftExecD2Z](https://docs.nvidia.com/cuda/cufft/index.html#cufftexecr2c-and-cufftexecd2z)
    pub fn exec_d2z(
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

    /// Executes a double-precision Z2D inverse FFT.
    ///
    /// # Reference
    ///
    /// [cufftExecZ2D](https://docs.nvidia.com/cuda/cufft/index.html#cufftexecc2r-and-cufftexecz2d)
    pub fn exec_z2d(
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

impl Drop for FftPlan {
    /// Destroys the plan.
    ///
    /// # Reference
    ///
    /// [cufftDestroy)(https://docs.nvidia.com/cuda/cufft/index.html#cufftdestroy)
    fn drop(&mut self) {
        unsafe {
            let _ = cufft_raw::cufftDestroy(self.raw);
        }
    }
}
