use std::{ffi::CStr, fmt::Display};

use cust::error::CudaError;
use cust_raw::cublas_sys;

/// Result that contains the un-dropped value on error.
pub type DropResult<T> = std::result::Result<(), (CublasError, T)>;

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CublasError {
    NotInitialized,
    AllocFailed,
    InvalidValue,
    ArchMismatch,
    MappingError,
    ExecutionFailed,
    InternalError,
    NotSupported,
    LicenseError,
}

impl std::error::Error for CublasError {}

impl Display for CublasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            let ptr = cublas_sys::cublasGetStatusString(self.into_raw());
            let cow = CStr::from_ptr(ptr).to_string_lossy();
            f.write_str(cow.as_ref())
        }
    }
}

pub trait ToResult {
    fn to_result(self) -> Result<(), CublasError>;
}

impl ToResult for cublas_sys::cublasStatus_t {
    fn to_result(self) -> Result<(), CublasError> {
        use CublasError::*;
        use cust_raw::cublas_sys::cublasStatus_t::*;

        Err(match self {
            CUBLAS_STATUS_SUCCESS => return Ok(()),
            CUBLAS_STATUS_NOT_INITIALIZED => NotInitialized,
            CUBLAS_STATUS_ALLOC_FAILED => AllocFailed,
            CUBLAS_STATUS_INVALID_VALUE => InvalidValue,
            CUBLAS_STATUS_ARCH_MISMATCH => ArchMismatch,
            CUBLAS_STATUS_MAPPING_ERROR => MappingError,
            CUBLAS_STATUS_EXECUTION_FAILED => ExecutionFailed,
            CUBLAS_STATUS_INTERNAL_ERROR => InternalError,
            CUBLAS_STATUS_NOT_SUPPORTED => NotSupported,
            CUBLAS_STATUS_LICENSE_ERROR => LicenseError,
        })
    }
}

impl CublasError {
    pub fn into_raw(self) -> cublas_sys::cublasStatus_t {
        use CublasError::*;
        use cust_raw::cublas_sys::cublasStatus_t::*;

        match self {
            NotInitialized => CUBLAS_STATUS_NOT_INITIALIZED,
            AllocFailed => CUBLAS_STATUS_ALLOC_FAILED,
            InvalidValue => CUBLAS_STATUS_INVALID_VALUE,
            ArchMismatch => CUBLAS_STATUS_ARCH_MISMATCH,
            MappingError => CUBLAS_STATUS_MAPPING_ERROR,
            ExecutionFailed => CUBLAS_STATUS_EXECUTION_FAILED,
            InternalError => CUBLAS_STATUS_INTERNAL_ERROR,
            NotSupported => CUBLAS_STATUS_NOT_SUPPORTED,
            LicenseError => CUBLAS_STATUS_LICENSE_ERROR,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Error {
    Cublas(CublasError),
    Cuda(CudaError),
}

impl From<CublasError> for Error {
    fn from(err: CublasError) -> Self {
        Self::Cublas(err)
    }
}

impl From<CudaError> for Error {
    fn from(err: CudaError) -> Self {
        Self::Cuda(err)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cublas(e) => Some(e),
            Self::Cuda(e) => Some(e),
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cublas(_) => write!(f, "cuBLAS error"),
            Self::Cuda(_) => write!(f, "CUDA error"),
        }
    }
}
