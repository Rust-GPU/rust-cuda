use std::error::Error;
use std::fmt::Display;

/// Error type for cuFFT operations.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CufftError {
    /// The plan handle is invalid.
    InvalidPlan,
    /// Memory allocation failed.
    AllocFailed,
    /// The transform type is invalid.
    InvalidType,
    /// An invalid value was provided.
    InvalidValue,
    /// An internal cuFFT error occurred.
    InternalError,
    /// The transform failed to execute.
    ExecFailed,
    /// The library failed to initialize.
    SetupFailed,
    /// The transform size is invalid.
    InvalidSize,
    /// The device is invalid.
    InvalidDevice,
    /// No workspace has been provided.
    NoWorkspace,
    /// This feature is not implemented.
    NotImplemented,
    /// This feature is not supported.
    NotSupported,
}

impl Display for CufftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            CufftError::InvalidPlan => "invalid plan handle",
            CufftError::AllocFailed => "allocation failed",
            CufftError::InvalidType => "invalid type",
            CufftError::InvalidValue => "invalid value",
            CufftError::InternalError => "internal error",
            CufftError::ExecFailed => "exec failed",
            CufftError::SetupFailed => "setup failed",
            CufftError::InvalidSize => "invalid size",
            CufftError::InvalidDevice => "invalid device",
            CufftError::NoWorkspace => "no workspace",
            CufftError::NotImplemented => "not implemented",
            CufftError::NotSupported => "not supported",
        };
        f.write_str(msg)
    }
}

impl Error for CufftError {}

pub trait IntoResult {
    fn into_result(self) -> Result<(), CufftError>;
}

impl IntoResult for cufft_raw::cufftResult {
    fn into_result(self) -> Result<(), CufftError> {
        use cufft_raw::cufftResult::*;
        Err(match self {
            CUFFT_SUCCESS => return Ok(()),
            CUFFT_INVALID_PLAN => CufftError::InvalidPlan,
            CUFFT_ALLOC_FAILED => CufftError::AllocFailed,
            CUFFT_INVALID_TYPE => CufftError::InvalidType,
            CUFFT_INVALID_VALUE => CufftError::InvalidValue,
            CUFFT_INTERNAL_ERROR => CufftError::InternalError,
            CUFFT_EXEC_FAILED => CufftError::ExecFailed,
            CUFFT_SETUP_FAILED => CufftError::SetupFailed,
            CUFFT_INVALID_SIZE => CufftError::InvalidSize,
            CUFFT_UNALIGNED_DATA => CufftError::InvalidValue,
            CUFFT_INVALID_DEVICE => CufftError::InvalidDevice,
            CUFFT_NO_WORKSPACE => CufftError::NoWorkspace,
            CUFFT_NOT_IMPLEMENTED => CufftError::NotImplemented,
            CUFFT_NOT_SUPPORTED => CufftError::NotSupported,
            _ => CufftError::InternalError,
        })
    }
}
