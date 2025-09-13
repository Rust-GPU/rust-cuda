//! Raw and High level bindings to the CUDA NVPTX compiler used to compile PTX to
//! cubin files.

use std::ffi::c_char;
use std::mem::MaybeUninit;

use cust_raw::nvptx_compiler_sys;

trait ToResult {
    fn to_result(self) -> Result<(), NvptxError>;
}

impl ToResult for nvptx_compiler_sys::nvPTXCompileResult {
    fn to_result(self) -> Result<(), NvptxError> {
        use cust_raw::nvptx_compiler_sys::nvPTXCompileResult::*;
        match self {
            NVPTXCOMPILE_SUCCESS => Ok(()),
            NVPTXCOMPILE_ERROR_INVALID_INPUT => Err(NvptxError::InvalidInput),
            NVPTXCOMPILE_ERROR_COMPILATION_FAILURE => Err(NvptxError::CompilationFailure),
            NVPTXCOMPILE_ERROR_INTERNAL => Err(NvptxError::Internal),
            NVPTXCOMPILE_ERROR_OUT_OF_MEMORY => Err(NvptxError::OutOfMemory),
            NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION => Err(NvptxError::UnsupportedPtxVersion),
            // these two are statically prevented so they should never happen
            NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE => {
                unreachable!("nvptx yielded an incomplete invocation error")
            }
            NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE => {
                unreachable!("nvptx yielded an invalid handle err")
            }
            _ => unreachable!(),
        }
    }
}

pub type NvptxResult<T> = Result<T, NvptxError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NvptxError {
    InvalidInput,
    CompilationFailure,
    Internal,
    OutOfMemory,
    UnsupportedPtxVersion,
}

#[repr(transparent)]
#[derive(Debug)]
pub struct NvptxCompiler {
    raw: nvptx_compiler_sys::nvPTXCompilerHandle,
}

impl NvptxCompiler {
    /// Create a new compiler from a ptx string.
    pub fn new(ptx: impl AsRef<str>) -> NvptxResult<Self> {
        let ptx = ptx.as_ref();
        let mut raw = MaybeUninit::uninit();

        unsafe {
            nvptx_compiler_sys::nvPTXCompilerCreate(
                raw.as_mut_ptr(),
                ptx.len(),
                ptx.as_ptr().cast(),
            )
            .to_result()?;
            let raw = raw.assume_init();
            Ok(Self { raw })
        }
    }
}

impl Drop for NvptxCompiler {
    fn drop(&mut self) {
        unsafe {
            nvptx_compiler_sys::nvPTXCompilerDestroy(&mut self.raw as *mut _)
                .to_result()
                .expect("failed to destroy nvptx compiler");
        }
    }
}

#[derive(Debug)]
pub struct CompilerFailure {
    pub error: NvptxError,
    handle: nvptx_compiler_sys::nvPTXCompilerHandle,
}

impl Drop for CompilerFailure {
    fn drop(&mut self) {
        unsafe {
            nvptx_compiler_sys::nvPTXCompilerDestroy(&mut self.handle as *mut _)
                .to_result()
                .expect("failed to destroy nvptx compiler failure");
        }
    }
}

impl CompilerFailure {
    pub fn error_log(&self) -> NvptxResult<String> {
        let mut size = MaybeUninit::uninit();
        unsafe {
            nvptx_compiler_sys::nvPTXCompilerGetErrorLogSize(self.handle, size.as_mut_ptr())
                .to_result()?;
            let size = size.assume_init();
            let mut vec = Vec::with_capacity(size);
            #[allow(clippy::unnecessary_cast)]
            nvptx_compiler_sys::nvPTXCompilerGetErrorLog(
                self.handle,
                vec.as_mut_ptr() as *mut c_char,
            )
            .to_result()?;
            vec.set_len(size);
            Ok(String::from_utf8_lossy(&vec).to_string())
        }
    }
}

/// The result of a compiled program
#[derive(Debug)]
pub struct CompiledProgram {
    pub cubin: Vec<u8>,
    handle: nvptx_compiler_sys::nvPTXCompilerHandle,
}

impl Drop for CompiledProgram {
    fn drop(&mut self) {
        unsafe {
            nvptx_compiler_sys::nvPTXCompilerDestroy(&mut self.handle as *mut _)
                .to_result()
                .expect("failed to destroy nvptx compiled program");
        }
    }
}

impl CompiledProgram {
    pub fn info_log(&self) -> NvptxResult<String> {
        let mut size = MaybeUninit::uninit();
        unsafe {
            nvptx_compiler_sys::nvPTXCompilerGetInfoLogSize(self.handle, size.as_mut_ptr())
                .to_result()?;
            let size = size.assume_init();
            let mut vec = Vec::with_capacity(size);
            #[allow(clippy::unnecessary_cast)]
            nvptx_compiler_sys::nvPTXCompilerGetInfoLog(
                self.handle,
                vec.as_mut_ptr() as *mut c_char,
            )
            .to_result()?;
            vec.set_len(size);
            Ok(String::from_utf8_lossy(&vec).to_string())
        }
    }
}
