//! CUDA Context handling.
//!
//! # New vs Legacy contexts
//!
//! The CUDA Driver API has two main ways of creating contexts. The "legacy" (legacy
//! meaning this is what the de-facto way of doing it in cust was) context handling,
//! and the "new" primary context handling. In the legacy way of handling contexts,
//! a thread could posess multiple contexts inside of a stack, and users would explicitly
//! create entire new contexts and set them as the current context at the top of the stack.
//!
//! This is great for control, but it causes a myriad of issues when trying to interoperate
//! with runtime API based libraries such as cuBLAS or cuFFT. Explicitly making and destroying
//! contexts causes a lot of problems with the runtime API because the runtime API will implicitly
//! use any context the driver API set as current. This sometimes causes segfaults and odd
//! behavior that is overall hard to manage if trying to use other CUDA libraries.
//!
//! The "new" primary context handling uses the same handling as the Runtime API. Instead
//! of context stacks, only a single context exists for every device, and this context
//! is reference-counted. Users can retain a handle to the primary context, increasing the
//! reference count, and release the context once they are done using it. Because this is
//! the same handling that the Runtime API uses, it is directly compatible with libraries
//! such as cuBLAS.
//!
//! Primary contexts also simplify the context API greatly, making new contexts on the same
//! device will just use the same context. This means there is no need for unowned contexts
//! when using multithreading. Users can simply make new contexts for every thread with no concern
//! that the context will be prematurely destroyed.
//!
//! So overall, we reccomend everyone use the new primary context handling, and avoid
//! the old legacy handling. Doing so will make your use of cust more compatible with
//! libraries like cuBLAS or cuFFT, as well as avoid potentially confusing context-based bugs.
//!
//! Primary contexts are the default in cust, you can use the old legacy context handling
//! with the [`legacy`] module.

use std::{
    mem::{self, MaybeUninit, transmute},
    ptr,
};

use cust_raw::driver_sys;

/// Legacy context handling.
pub mod legacy;

use crate::context::legacy::StreamPriorityRange;
use crate::{
    CudaApiVersion,
    device::Device,
    error::{CudaResult, DropResult, ToResult},
    private::Sealed,
};

pub trait ContextHandle: Sealed {
    fn get_inner(&self) -> driver_sys::CUcontext;
}
impl Sealed for Context {}
impl ContextHandle for Context {
    fn get_inner(&self) -> driver_sys::CUcontext {
        self.inner
    }
}

/// This enumeration represents configuration settings for devices which share hardware resources
/// between L1 cache and shared memory.
///
/// Note that this is only a preference - the driver will use the requested configuration if
/// possible, but it is free to choose a different configuration if required to execute functions.
///
/// See
/// [CurrentContext::get_cache_config](struct.CurrentContext.html#method.get_cache_config) and
/// [CurrentContext::set_cache_config](struct.CurrentContext.html#method.set_cache_config) to get
/// and set the cache config for the current context.
#[repr(u32)]
#[non_exhaustive]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum CacheConfig {
    /// No preference for shared memory or L1 (default)
    PreferNone = 0,
    /// Prefer larger shared memory and smaller L1 cache
    PreferShared = 1,
    /// Prefer larger L1 cache and smaller shared memory
    PreferL1 = 2,
    /// Prefer equal-sized L1 cache and shared memory
    PreferEqual = 3,
}

/// This enumeration represents the limited resources which can be accessed through
/// [CurrentContext::get_resource_limit](struct.CurrentContext.html#method.get_resource_limit) and
/// [CurrentContext::set_resource_limit](struct.CurrentContext.html#method.set_resource_limit).
#[repr(u32)]
#[non_exhaustive]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum ResourceLimit {
    /// The size in bytes of each GPU thread stack
    StackSize = 0,
    /// The size in bytes of the FIFO used by the `printf()` device system call.
    PrintfFifoSize = 1,
    /// The size in bytes of the heap used by the `malloc()` and `free()` device system calls.
    ///
    /// Note that this is used for memory allocated within a kernel launch; it is not related to the
    /// device memory allocated by the host.
    MallocHeapSize = 2,
    /// The maximum nesting depth of a grid at which a thread can safely call
    /// `cudaDeviceSynchronize()` to wait on child grid launches to complete.
    DeviceRuntimeSynchronizeDepth = 3,
    /// The maximum number of outstanding device runtime launches that can be made from the current
    /// context.
    DeviceRuntimePendingLaunchCount = 4,
    /// L2 cache fetch granularity
    MaxL2FetchGranularity = 5,
}

/// This enumeration represents the options for configuring the shared memory bank size.
///
/// See
/// [CurrentContext::get_shared_memory_config](struct.CurrentContext.html#method.get_shared_memory_config) and
/// [CurrentContext::set_shared_memory_config](struct.CurrentContext.html#method.set_shared_memory_config) to get
/// and set the cache config for the current context.
#[repr(u32)]
#[non_exhaustive]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum SharedMemoryConfig {
    /// Set shared-memory bank size to the default.
    DefaultBankSize = 0,
    /// Set shared-memory bank width to four bytes
    FourByteBankSize = 1,
    /// Set shared-memory bank width to eight bytes
    EightByteBankSize = 2,
}

bitflags::bitflags! {
    /// Bit flags for initializing the CUDA context.
    ///
    /// If you're not sure which flags to use, `MAP_HOST | SCHED_AUTO` is a good default.
    pub struct ContextFlags: u32 {
        /// Instructs CUDA to actively spin when waiting for results from the GPU. This can decrease
        /// latency when waiting for the GPU, but may lower the performance of other CPU threads
        /// if they are performing work in parallel with the CUDA thread.
        const SCHED_SPIN = 0x01;

        /// Instructs CUDA to yield its thread when waiting for results from the GPU. This can
        /// increase latency when waiting for the GPU, but can increase the performance of CPU
        /// threads performing work in parallel with the GPU.
        const SCHED_YIELD = 0x02;

        /// Instructs CUDA to block the CPU thread on a synchronization primitive when waiting for
        /// the GPU to finish work.
        const SCHED_BLOCKING_SYNC = 0x04;

        /// Instructs CUDA to automatically choose whether to yield to other OS threads while waiting
        /// for the GPU, or to spin the OS thread. This is the default.
        const SCHED_AUTO = 0x00;

        /// Instructs CUDA to support mapped pinned allocations. This flag must be set in order to
        /// use page-locked memory (see [LockedBuffer](../memory/struct.LockedBuffer.html])).
        const MAP_HOST = 0x08;

        /// Instruct CUDA not to reduce local memory after resizing local memory for a kernel. This
        /// can prevent thrashing by local memory allocations when launching many kernels with high
        /// local memory usage at the cost of potentially increased memory usage.
        const LMEM_RESIZE_TO_MAX = 0x10;
    }
}

#[derive(Debug)]
pub struct Context {
    inner: driver_sys::CUcontext,
    device: driver_sys::CUdevice,
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Clone for Context {
    fn clone(&self) -> Self {
        // because we already retained a context on this device successfully (self), it is
        // exceedingly rare that this function would fail, therefore a silent panic
        // is mostly okay
        Self::new(Device {
            device: self.device,
        })
        .expect("Failed to clone context")
    }
}

impl Context {
    /// Retains the primary context for this device and makes it current, incrementing the internal reference cycle
    /// that CUDA keeps track of. There is only one primary context associated with a device, multiple
    /// calls to this function with the same device will return the same internal context.
    ///
    /// This will **NOT** push the context to the stack, primary contexts do not interoperate
    /// with the context stack.
    pub fn new(device: Device) -> CudaResult<Self> {
        let mut inner = MaybeUninit::uninit();
        unsafe {
            driver_sys::cuDevicePrimaryCtxRetain(inner.as_mut_ptr(), device.as_raw())
                .to_result()?;
            let inner = inner.assume_init();
            driver_sys::cuCtxSetCurrent(inner).to_result()?;
            Ok(Self {
                inner,
                device: device.as_raw(),
            })
        }
    }

    /// Resets the primary context associated with the device, freeing all allocations created
    /// inside of the context. You must make sure that nothing else is using the context or using
    /// CUDA on the device in general. For this reason, it is usually highly advised to not use
    /// this function.
    ///
    /// # Safety
    ///
    /// Nothing else should be using the primary context for this device, otherwise,
    /// spurious errors or segfaults will occur.
    pub unsafe fn reset(device: &Device) -> CudaResult<()> {
        unsafe { driver_sys::cuDevicePrimaryCtxReset(device.as_raw()).to_result() }
    }

    /// Sets the flags for the device context, these flags will apply to any user of the primary
    /// context associated with this device.
    pub fn set_flags(&self, flags: ContextFlags) -> CudaResult<()> {
        unsafe { driver_sys::cuDevicePrimaryCtxSetFlags(self.device, flags.bits()).to_result() }
    }

    /// Returns the raw handle to this context.
    pub fn as_raw(&self) -> driver_sys::CUcontext {
        self.inner
    }

    /// Get the API version used to create this context.
    ///
    /// This is not necessarily the latest version supported by the driver.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::{Context, ContextFlags};
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// cust::init(cust::CudaFlags::empty())?;
    /// let device = Device::get_device(0)?;
    /// let context = Context::new(device)?;
    /// let version = context.get_api_version()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_api_version(&self) -> CudaResult<CudaApiVersion> {
        unsafe {
            let mut api_version = 0u32;
            driver_sys::cuCtxGetApiVersion(self.inner, &mut api_version as *mut u32).to_result()?;
            Ok(CudaApiVersion {
                version: api_version as i32,
            })
        }
    }

    /// Destroy a `Context`, returning an error.
    ///
    /// Destroying a context can return errors from previous asynchronous work. This function
    /// destroys the given context and returns the error and the un-destroyed context on failure.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::{Context, ContextFlags};
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::new(device)?;
    /// match Context::drop(context) {
    ///     Ok(()) => println!("Successfully destroyed"),
    ///     Err((e, ctx)) => {
    ///         println!("Failed to destroy context: {:?}", e);
    ///         // Do something with ctx
    ///     },
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn drop(mut ctx: Context) -> DropResult<Context> {
        if ctx.inner.is_null() {
            return Ok(());
        }

        unsafe {
            let inner = mem::replace(&mut ctx.inner, ptr::null_mut());
            match driver_sys::cuDevicePrimaryCtxRelease(ctx.device).to_result() {
                Ok(()) => {
                    mem::forget(ctx);
                    Ok(())
                }
                Err(e) => Err((
                    e,
                    Context {
                        inner,
                        device: ctx.device,
                    },
                )),
            }
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if self.inner.is_null() {
            return;
        }

        unsafe {
            self.inner = ptr::null_mut();
            let _ = driver_sys::cuDevicePrimaryCtxRelease(self.device);
        }
    }
}

/// Type representing the context being currently used.
#[derive(Debug)]
pub struct CurrentContext;
impl CurrentContext {
    /// Returns the preferred cache configuration for the current context.
    ///
    /// On devices where the L1 cache and shared memory use the same hardware resources, this
    /// function returns the preferred cache configuration for the current context. For devices
    /// where the size of the L1 cache and shared memory are fixed, this will always return
    /// `CacheConfig::PreferNone`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::{ Context, ContextFlags, CurrentContext };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::new(device)?;
    /// let cache_config = CurrentContext::get_cache_config()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_cache_config() -> CudaResult<CacheConfig> {
        unsafe {
            let mut config = CacheConfig::PreferNone;
            driver_sys::cuCtxGetCacheConfig(
                &mut config as *mut CacheConfig as *mut driver_sys::CUfunc_cache,
            )
            .to_result()?;
            Ok(config)
        }
    }

    /// Return the device ID for the current context.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::{ Context, ContextFlags, CurrentContext };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::new(device)?;
    /// let device = CurrentContext::get_device()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_device() -> CudaResult<Device> {
        unsafe {
            let mut device = Device { device: 0 };
            driver_sys::cuCtxGetDevice(&mut device.device as *mut driver_sys::CUdevice)
                .to_result()?;
            Ok(device)
        }
    }

    /// Return the context flags for the current context.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::{ Context, ContextFlags, CurrentContext };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::new(device)?;
    /// let flags = CurrentContext::get_flags()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_flags() -> CudaResult<ContextFlags> {
        unsafe {
            let mut flags = 0u32;
            driver_sys::cuCtxGetFlags(&mut flags as *mut u32).to_result()?;
            Ok(ContextFlags::from_bits_truncate(flags))
        }
    }

    /// Return resource limits for the current context.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::{ Context, ContextFlags, CurrentContext, ResourceLimit };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::new(device)?;
    /// let stack_size = CurrentContext::get_resource_limit(ResourceLimit::StackSize)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_resource_limit(resource: ResourceLimit) -> CudaResult<usize> {
        unsafe {
            let mut limit: usize = 0;
            driver_sys::cuCtxGetLimit(
                &mut limit as *mut usize,
                transmute::<ResourceLimit, driver_sys::CUlimit_enum>(resource),
            )
            .to_result()?;
            Ok(limit)
        }
    }

    /// Return resource limits for the current context.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::{ Context, ContextFlags, CurrentContext, ResourceLimit };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::new(device)?;
    /// let shared_mem_config = CurrentContext::get_shared_memory_config()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_shared_memory_config() -> CudaResult<SharedMemoryConfig> {
        unsafe {
            let mut cfg = SharedMemoryConfig::DefaultBankSize;
            driver_sys::cuCtxGetSharedMemConfig(
                &mut cfg as *mut SharedMemoryConfig as *mut driver_sys::CUsharedconfig,
            )
            .to_result()?;
            Ok(cfg)
        }
    }

    /// Return the least and greatest stream priorities.
    ///
    /// If the program attempts to create a stream with a priority outside of this range, it will be
    /// automatically clamped to within the valid range. If the device does not support stream
    /// priorities, the returned range will contain zeroes.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::{ Context, ContextFlags, CurrentContext};
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::new(device)?;
    /// let priority_range = CurrentContext::get_stream_priority_range()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_stream_priority_range() -> CudaResult<StreamPriorityRange> {
        unsafe {
            let mut range = StreamPriorityRange {
                least: 0,
                greatest: 0,
            };
            driver_sys::cuCtxGetStreamPriorityRange(
                &mut range.least as *mut i32,
                &mut range.greatest as *mut i32,
            )
            .to_result()?;
            Ok(range)
        }
    }

    /// Sets the preferred cache configuration for the current context.
    ///
    /// On devices where L1 cache and shared memory use the same hardware resources, this sets the
    /// preferred cache configuration for the current context. This is only a preference. The
    /// driver will use the requested configuration if possible, but is free to choose a different
    /// configuration if required to execute the function.
    ///
    /// This setting does nothing on devices where the size of the L1 cache and shared memory are
    /// fixed.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::{ Context, ContextFlags, CurrentContext, CacheConfig };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::new(device)?;
    /// CurrentContext::set_cache_config(CacheConfig::PreferL1)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_cache_config(cfg: CacheConfig) -> CudaResult<()> {
        unsafe {
            driver_sys::cuCtxSetCacheConfig(
                transmute::<CacheConfig, driver_sys::CUfunc_cache_enum>(cfg),
            )
            .to_result()
        }
    }

    /// Sets a requested resource limit for the current context.
    ///
    /// Note that this is only a request; the driver is free to modify the requested
    /// value to meet hardware requirements. Each limit has some specific restrictions.
    ///
    ///   * `StackSize`: Controls the stack size in bytes for each GPU thread
    ///   * `PrintfFifoSize`: Controls the size in bytes of the FIFO used by the
    ///     `printf()` device system call. This cannot be changed after a kernel has
    ///     been launched which uses the `printf()` function.
    ///   * `MallocHeapSize`: Controls the size in bytes of the heap used by the
    ///     `malloc()` and `free()` device system calls. This cannot be changed aftr a
    ///     kernel has been launched which uses the `malloc()` and `free()` system
    ///     calls.
    ///   * `DeviceRuntimeSyncDepth`: Controls the maximum nesting depth of a grid at
    ///     which a thread can safely call `cudaDeviceSynchronize()`. This cannot be
    ///     changed after a kernel has been launched which uses the device runtime. When
    ///     setting this limit, keep in mind that additional levels of sync depth
    ///     require the driver to reserve large amounts of device memory which can no
    ///     longer be used for device allocations.
    ///   * `DeviceRuntimePendingLaunchCount`: Controls the maximum number of
    ///     outstanding device runtime launches that can be made from the current
    ///     context. A grid is outstanding from the point of the launch up until the
    ///     grid is known to have completed. Keep in mind that increasing this limit
    ///     will require the driver to reserve larger amounts of device memory which can
    ///     no longer be used for device allocations.
    ///   * `MaxL2FetchGranularity`: Controls the L2 fetch granularity. This is purely a
    ///     performance hint and it can be ignored or clamped depending on the platform.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::{ Context, ContextFlags, CurrentContext, ResourceLimit };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::new(device)?;
    /// CurrentContext::set_resource_limit(ResourceLimit::StackSize, 2048)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_resource_limit(resource: ResourceLimit, limit: usize) -> CudaResult<()> {
        unsafe {
            driver_sys::cuCtxSetLimit(
                transmute::<ResourceLimit, driver_sys::CUlimit_enum>(resource),
                limit,
            )
            .to_result()?;
            Ok(())
        }
    }

    /// Sets the preferred shared memory configuration for the current context.
    ///
    /// On devices with configurable shared memory banks, this function will set the context's
    /// shared memory bank size which is used for subsequent kernel launches.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::{ Context, ContextFlags, CurrentContext, SharedMemoryConfig };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::new(device)?;
    /// CurrentContext::set_shared_memory_config(SharedMemoryConfig::DefaultBankSize)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_shared_memory_config(cfg: SharedMemoryConfig) -> CudaResult<()> {
        unsafe {
            driver_sys::cuCtxSetSharedMemConfig(transmute::<
                SharedMemoryConfig,
                driver_sys::CUsharedconfig_enum,
            >(cfg))
            .to_result()
        }
    }

    /// Set the given context as the current context for this thread.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::{ Context, ContextFlags, CurrentContext };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::new(device)?;
    /// CurrentContext::set_current(&context)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_current<C: ContextHandle>(c: &C) -> CudaResult<()> {
        unsafe {
            driver_sys::cuCtxSetCurrent(c.get_inner()).to_result()?;
            Ok(())
        }
    }

    /// Block to wait for a context's tasks to complete.
    pub fn synchronize() -> CudaResult<()> {
        unsafe {
            driver_sys::cuCtxSynchronize().to_result()?;
            Ok(())
        }
    }
}
