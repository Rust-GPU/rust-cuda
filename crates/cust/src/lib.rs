//! Safe, Fast, and user-friendly wrapper around the CUDA Driver API.
//!
//! # Low level CUDA interop
//!
//! Because additions to CUDA and libraries that use CUDA are everchanging, this library
//! provides unsafe functions for retrieving and setting handles to raw CUDA objects.
//! This allows advanced users to embed libraries that rely on CUDA, such as OptiX.
//!
//! # CUDA Terminology
//!
//! ## Devices and Hosts
//!
//! This crate and its documentation uses the terms "device" and "host" frequently, so it's worth
//! explaining them in more detail. A device refers to a CUDA-capable GPU or similar device and its
//! associated external memory space. The host is the CPU and its associated memory space. Data
//! must be transferred from host memory to device memory before the device can use it for
//! computations, and the results must then be transferred back to host memory.
//!
//! ## Contexts, Modules, Streams and Functions
//!
//! A CUDA context is akin to a process on the host - it contains all of the state for working with
//! a device, all memory allocations, etc. Each context is associated with a single device.
//!
//! A Module is similar to a shared-object library - it is a piece of compiled code which exports
//! functions and global values. Functions can be loaded from modules and launched on a device as
//! one might load a function from a shared-object file and call it. Functions are also known as
//! kernels and the two terms will be used interchangeably.
//!
//! A Stream is akin to a thread - asynchronous work such as kernel execution can be queued into a
//! stream. Work within a single stream will execute sequentially in the order that it was
//! submitted, and may interleave with work from other streams.
//!
//! ## Grids, Blocks and Threads
//!
//! CUDA devices typically execute kernel functions on many threads in parallel. These threads can
//! be grouped into thread blocks, which share an area of fast hardware memory known as shared
//! memory. Thread blocks can be one-, two-, or three-dimensional, which is helpful when working
//! with multi-dimensional data such as images. Thread blocks are then grouped into grids, which
//! can also be one-, two-, or three-dimensional.
//!
//! CUDA devices often contain multiple separate processors. Each processor is capable of excuting
//! many threads simultaneously, but they must be from the same thread block. Thus, it is important
//! to ensure that the grid size is large enough to provide work for all processors. On the other
//! hand, if the thread blocks are too small each processor will be under-utilized and the
//! code will be unable to make effective use of shared memory.
//!
//! # Usage
//!
//! Before using cust, you must install the CUDA development libraries for your system. Version
//! 9.0 or newer is required. You must also have a CUDA-capable GPU installed with the appropriate
//! drivers.
//!
//! Cust will try to find the CUDA libraries automatically, if it is unable to find it, you can set
//! `CUDA_LIBRARY_PATH` to some path manually.

#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod device;
pub mod error;
pub mod event;
pub mod external;
pub mod function;
// WIP
pub mod context;
#[allow(warnings)]
mod graph;
pub mod link;
pub mod memory;
pub mod module;
pub mod prelude;
pub mod stream;
// WIP
mod surface;
mod texture;
pub mod util;

pub use cust_derive::DeviceCopy;

use crate::context::{Context, ContextFlags};
use crate::device::Device;
use crate::error::{CudaResult, ToResult};
use bitflags::bitflags;
use cust_raw::driver_sys::{cuDriverGetVersion, cuInit};

bitflags! {
    /// Bit flags for initializing the CUDA driver. Currently, no flags are defined,
    /// so `CudaFlags::empty()` is the only valid value.
    pub struct CudaFlags: u32 {
        // We need to give bitflags at least one constant.
        #[doc(hidden)]
        const _ZERO = 0;
    }
}

/// Initialize the CUDA Driver API.
///
/// This must be called before any other custa function is called. Typically, this
/// should be at the start of your program. All other functions will fail unless the API is
/// initialized first.
///
/// The `flags` parameter is used to configure the CUDA API. Currently no flags are defined, so
/// it must be `CudaFlags::empty()`.
pub fn init(flags: CudaFlags) -> CudaResult<()> {
    unsafe { cuInit(flags.bits()).to_result() }
}

/// Shortcut for initializing the CUDA Driver API and creating a CUDA context with default settings
/// for the first device.
///
/// **You must keep this context alive while you do further operations or you will get an InvalidContext
/// error**. e.g. using `let _ctx = quick_init()?;`.
///
/// This is useful for testing or just setting up a basic CUDA context quickly. Users with more
/// complex needs (multiple devices, custom flags, etc.) should use `init` and create their own
/// context.
#[must_use = "The CUDA Context must be kept alive or errors will be issued for any CUDA function that is run"]
pub fn quick_init() -> CudaResult<Context> {
    init(CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let ctx = Context::new(device)?;
    ctx.set_flags(ContextFlags::SCHED_AUTO)?;
    Ok(ctx)
}

/// Struct representing the CUDA API version number.
#[derive(Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Copy, Clone)]
pub struct CudaApiVersion {
    version: i32,
}
impl CudaApiVersion {
    /// Returns the latest CUDA version supported by the CUDA driver.
    pub fn get() -> CudaResult<CudaApiVersion> {
        unsafe {
            let mut version: i32 = 0;
            cuDriverGetVersion(&mut version as *mut i32).to_result()?;
            Ok(CudaApiVersion { version })
        }
    }

    /// Return the major version number - eg. the 9 in version 9.2
    #[inline]
    pub fn major(self) -> i32 {
        self.version / 1000
    }

    /// Return the minor version number - eg. the 2 in version 9.2
    #[inline]
    pub fn minor(self) -> i32 {
        (self.version % 1000) / 10
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_api_version() {
        let version = CudaApiVersion { version: 9020 };
        assert_eq!(version.major(), 9);
        assert_eq!(version.minor(), 2);
    }

    #[test]
    fn test_init_twice() {
        init(CudaFlags::empty()).unwrap();
        init(CudaFlags::empty()).unwrap();
    }
}

// Fake module with a private trait used to prevent outside code from implementing certain traits.
pub(crate) mod private {
    pub trait Sealed {}
}
