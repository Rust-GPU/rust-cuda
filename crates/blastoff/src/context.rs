use std::ffi::CString;
use std::mem::{self, MaybeUninit};
use std::os::raw::c_char;
use std::ptr;

use cust::stream::Stream;
use cust_raw::cublas_sys;
use cust_raw::driver_sys;

use super::error::DropResult;
use super::error::ToResult as _;

type Result<T, E = super::error::Error> = std::result::Result<T, E>;

bitflags::bitflags! {
    /// Configures precision levels for the math in cuBLAS.
    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct MathMode: u32 {
        /// Highest performance mode which uses compute and intermediate storage precisions
        /// with at least the same number of mantissa and exponent bits as requested. Will
        /// also use tensor cores when possible.
        const DEFAULT = 0;
        /// Mode which uses prescribed precision and standardized arithmetic for all phases of calculations
        /// and is primarily intended for numerical robustness studies, testing, and debugging. This mode
        /// might not be as performant as the other modes.
        const PEDANTIC = 1;
        /// Enable acceleration of single precision routines using TF32 Tensor Cores.
        const TF32_TENSOR_OP = 3;
        /// Forces any reductions during matrix multiplication to use the accumulator type (i.e. the compute type)
        /// and not the output type in case of mixed precision routines where output precision is less than compute
        /// type precision.
        const DISALLOW_REDUCED_PRECISION_REDUCTION = 16;
    }
}

/// The central structure required to do anything with cuBLAS. It holds and manages internal memory allocations
///
/// # Multithreaded Usage
///
/// While it is technically allowed to use the same context across threads, it is very suboptimal and dangerous
/// so we do not expose this functionality. Instead, you should create a context for every thread (as the cuBLAS docs reccomend).
///
/// # Multi-Device Usage
///
/// cuBLAS contexts are tied to the current device (through the current CUDA Context), therefore, for multi-device usage you should
/// create a context for every device.
///
/// # Drop Cost
///
/// cuBLAS contexts hold internal memory allocations required by the library, and will free those allocations on drop. They will
/// also synchronize the entire device when dropping the context. Therefore, you should minimize both the amount of contexts, and the
/// amount of context drops. You should generally allocate all the contexts at once, and drop them all at once.
///
/// # Methods
///
/// ## Level 1 Methods (Scalar/Vector-based operations)
/// - [Index of smallest element by absolute value <span style="float:right;">`amin`</span>](CublasContext::amin)
/// - [Index of largest element by absolute value <span style="float:right;">`amax`</span>](CublasContext::amax)
/// - [$\alpha \boldsymbol{x} + \boldsymbol{y}$ <span style="float:right;">`axpy`</span>](CublasContext::axpy)
/// - [Copy $n$ elements from $\boldsymbol{x}$ into $\boldsymbol{y}$ <span style="float:right;">`copy`</span>](CublasContext::copy)
/// - [Dot Product <span style="float:right;">`dot`</span>](CublasContext::dot)
/// - [Unconjugated Complex Dot Product <span style="float:right;">`dotu`</span>](CublasContext::dotu)
/// - [Conjugated Complex Dot Product <span style="float:right;">`dotc`</span>](CublasContext::dotc)
/// - [Euclidian Norm <span style="float:right;">`nrm2`</span>](CublasContext::nrm2)
/// - [Rotate points in the xy-plane using a Givens rotation matrix <span style="float:right;">`rot`</span>](CublasContext::rot)
/// - [Construct the givens rotation matrix that zeros the second entry of a vector<span style="float:right;">`rotg`</span>](CublasContext::rotg)
/// - [Apply the modified Givens transformation to vectors <span style="float:right;">`rotm`</span>](CublasContext::rotm)
/// - [Construct the modified givens rotation matrix that zeros the second entry of a vector<span style="float:right;">`rotmg`</span>](CublasContext::rotmg)
/// - [Scale a vector by a scalar <span style="float:right;">`scal`</span>](CublasContext::scal)
/// - [Swap two vectors <span style="float:right;">`swap`</span>](CublasContext::swap)
///
/// ## Level 3 Methods (Matrix-based operations)
/// - [Matrix Multiplication <span style="float:right;">`gemm`</span>](CublasContext::gemm)
#[derive(Debug)]
pub struct CublasContext {
    pub(crate) raw: cublas_sys::cublasHandle_t,
}

impl CublasContext {
    /// Creates a new cuBLAS context, allocating all of the required host and device memory.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _ctx = cust::quick_init()?;
    /// use blastoff::CublasContext;
    /// let ctx = CublasContext::new()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new() -> Result<Self> {
        let mut raw = MaybeUninit::uninit();
        unsafe {
            cublas_sys::cublasCreate(raw.as_mut_ptr()).to_result()?;
            cublas_sys::cublasSetPointerMode(
                raw.assume_init(),
                cublas_sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE,
            )
            .to_result()?;
            Ok(Self {
                raw: raw.assume_init(),
            })
        }
    }

    /// Tries to destroy a [`CublasContext`], returning an error if it fails.
    pub fn drop(mut ctx: CublasContext) -> DropResult<CublasContext> {
        if ctx.raw.is_null() {
            return Ok(());
        }

        unsafe {
            let inner = mem::replace(&mut ctx.raw, ptr::null_mut());
            match cublas_sys::cublasDestroy(inner).to_result() {
                Ok(()) => {
                    mem::forget(ctx);
                    Ok(())
                }
                Err(e) => Err((e, CublasContext { raw: inner })),
            }
        }
    }

    /// Returns the major, minor, and patch versions of the cuBLAS library.
    pub fn version(&self) -> (u32, u32, u32) {
        let mut raw = MaybeUninit::<u32>::uninit();
        unsafe {
            // getVersion can't fail
            cublas_sys::cublasGetVersion(self.raw, raw.as_mut_ptr().cast())
                .to_result()
                .unwrap();

            let raw = raw.assume_init();
            (raw / 1000, (raw % 1000) / 100, raw % 100)
        }
    }

    /// Executes a given closure in a specific CUDA [`Stream`], specifically, it sets the current cublas stream
    /// for the context, runs the closure, then unsets the stream back to NULL.
    pub fn with_stream<T, F: FnOnce(&mut Self) -> Result<T>>(
        &mut self,
        stream: &Stream,
        func: F,
    ) -> Result<T> {
        unsafe {
            // cudaStream_t is the same as CUstream
            cublas_sys::cublasSetStream(
                self.raw,
                mem::transmute::<*mut driver_sys::CUstream_st, *mut cublas_sys::CUstream_st>(
                    stream.as_inner(),
                ),
            )
            .to_result()?;
            let res = func(self)?;
            // reset the stream back to NULL just in case someone calls with_stream, then drops the stream, and tries to
            // execute a raw sys function with the context's handle.
            cublas_sys::cublasSetStream(self.raw, ptr::null_mut()).to_result()?;
            Ok(res)
        }
    }

    /// Sets whether the cuBLAS library is allowed to use atomics for certain routines such as `symv` or `hemv`.
    ///
    /// cuBLAS has specialized versions of functions that use atomics to accumulate results, which is generally significantly
    /// faster than not using atomics. However, atomics generate results that are not strictly identical from one run to another.
    /// Such differences are mathematically insignificant, but when debugging, the differences are less than ideal.
    ///
    /// This function sets whether atomics usage is allowed or not, unless explicitly specified in function docs, functions
    /// do not have an atomic specialization of the function.
    ///
    /// This is `false` by default (cuBLAS will not use atomics unless explicitly set to be allowed to do so).
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _ctx = cust::quick_init()?;
    /// use blastoff::CublasContext;
    /// let ctx = CublasContext::new()?;
    /// // allows cuBLAS to use atomics to speed up functions at the cost of determinism.
    /// ctx.set_atomics_mode(true)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_atomics_mode(&self, allowed: bool) -> Result<()> {
        unsafe {
            Ok(cublas_sys::cublasSetAtomicsMode(
                self.raw,
                if allowed {
                    cublas_sys::cublasAtomicsMode_t::CUBLAS_ATOMICS_ALLOWED
                } else {
                    cublas_sys::cublasAtomicsMode_t::CUBLAS_ATOMICS_NOT_ALLOWED
                },
            )
            .to_result()?)
        }
    }

    /// Returns whether the context is set to be allowed to use atomics per [`CublasContext::set_atomics_mode`].
    /// Returns `false` unless previously explicitly set to `true`.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _ctx = cust::quick_init()?;
    /// use blastoff::CublasContext;
    /// let ctx = CublasContext::new()?;
    /// ctx.set_atomics_mode(true)?;
    /// assert!(ctx.get_atomics_mode()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_atomics_mode(&self) -> Result<bool> {
        let mut mode = MaybeUninit::uninit();
        unsafe {
            cublas_sys::cublasGetAtomicsMode(self.raw, mode.as_mut_ptr()).to_result()?;
            Ok(match mode.assume_init() {
                cublas_sys::cublasAtomicsMode_t::CUBLAS_ATOMICS_ALLOWED => true,
                cublas_sys::cublasAtomicsMode_t::CUBLAS_ATOMICS_NOT_ALLOWED => false,
            })
        }
    }

    /// Sets the precision level for different routines in cuBLAS. See [`MathMode`] for more info.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _ctx = cust::quick_init()?;
    /// use blastoff::{CublasContext, MathMode};
    /// let ctx = CublasContext::new()?;
    /// ctx.set_math_mode(MathMode::DEFAULT | MathMode::DISALLOW_REDUCED_PRECISION_REDUCTION)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_math_mode(&self, math_mode: MathMode) -> Result<()> {
        unsafe {
            Ok(cublas_sys::cublasSetMathMode(
                self.raw,
                mem::transmute::<u32, cublas_sys::cublasMath_t>(math_mode.bits()),
            )
            .to_result()?)
        }
    }

    /// Gets the precision level that was previously set by [`CublasContext::set_math_mode`].
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _ctx = cust::quick_init()?;
    /// use blastoff::{CublasContext, MathMode};
    /// let ctx = CublasContext::new()?;
    /// ctx.set_math_mode(MathMode::DEFAULT | MathMode::DISALLOW_REDUCED_PRECISION_REDUCTION)?;
    /// assert_eq!(ctx.get_math_mode()?, MathMode::DEFAULT | MathMode::DISALLOW_REDUCED_PRECISION_REDUCTION);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_math_mode(&self) -> Result<MathMode> {
        let mut mode = MaybeUninit::uninit();
        unsafe {
            cublas_sys::cublasGetMathMode(self.raw, mode.as_mut_ptr()).to_result()?;
            Ok(MathMode::from_bits(mode.assume_init() as u32)
                .expect("Invalid MathMode from cuBLAS"))
        }
    }

    /// Configures cuBLAS logging.
    ///
    /// - `enable` will enable or disable logging completely. Off by default.
    /// - `log_to_stdout` will turn on/off logging to standard output. Off by default.
    /// - `log_to_stderr` will turn on/off logging to standard error. Off by default.
    /// - `log_file_name` will turn on/off logging to a file in the file system. None by default.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _ctx = cust::quick_init()?;
    /// use blastoff::{CublasContext, MathMode};
    /// let ctx = CublasContext::new()?;
    /// // turn off logging completely
    /// ctx.configure_logger(false, false, false, None);
    /// // log to stdout and stderr
    /// ctx.configure_logger(true, true, true, None);
    /// // log to a file
    /// ctx.configure_logger(true, false, false, Some("./log.txt"));
    /// # Ok(())
    /// # }
    /// ```
    pub fn configure_logger(
        &self,
        enable: bool,
        log_to_stdout: bool,
        log_to_stderr: bool,
        log_file_name: Option<&str>,
    ) {
        unsafe {
            let path = log_file_name.map(|p| CString::new(p).expect("nul in log_file_name"));
            let path_ptr = path.map_or(ptr::null(), |s| s.as_ptr());

            cublas_sys::cublasLoggerConfigure(
                enable as i32,
                log_to_stdout as i32,
                log_to_stderr as i32,
                path_ptr,
            )
            .to_result()
            .expect("logger configure failed");
        }
    }

    /// Sets a function for the logger callback.
    ///
    /// # Safety
    ///
    /// The callback must not panic and unwind.
    pub unsafe fn set_logger_callback(callback: Option<unsafe extern "C" fn(*const c_char)>) {
        unsafe {
            cublas_sys::cublasSetLoggerCallback(callback)
                .to_result()
                .unwrap();
        }
    }

    /// Gets the logger callback that was previously set.
    pub fn get_logger_callback() -> Option<unsafe extern "C" fn(*const c_char)> {
        let mut cb = MaybeUninit::uninit();
        unsafe {
            cublas_sys::cublasGetLoggerCallback(cb.as_mut_ptr())
                .to_result()
                .unwrap();
            cb.assume_init()
        }
    }
}

impl Drop for CublasContext {
    fn drop(&mut self) {
        unsafe {
            let _ = cublas_sys::cublasDestroy(self.raw);
        }
    }
}
