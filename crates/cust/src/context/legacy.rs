//! Legacy (old) context management which preceded primary contexts.
//!
//! # CUDA context management
//!
//! Most CUDA functions require a context. A CUDA context is analogous to a CPU process - it's
//! an isolated container for all runtime state, including configuration settings and the
//! device/unified/page-locked memory allocations. Each context has a separate memory space, and
//! pointers from one context do not work in another. Each context is associated with a single
//! device. Although it is possible to have multiple contexts associated with a single device, this
//! is strongly discouraged as it can cause a significant loss of performance.
//!
//! CUDA keeps a thread-local stack of contexts which the programmer can push to or pop from.
//! The top context in that stack is known as the "current" context and it is used in most CUDA
//! API calls. One context can be safely made current in multiple CPU threads.
//!
//! # Safety
//!
//! The CUDA context management API does not fit easily into Rust's safety guarantees.
//!
//! The thread-local stack (as well as the fact that any context can be on the stack for any number
//! of threads) means there is no clear owner for a CUDA context, but it still has to be cleaned up.
//! Also, the fact that a context can be current to multiple threads at once means that there can be
//! multiple implicit references to a context which are not controlled by Rust.
//!
//! cust handles ownership by providing an owning [`Context`](struct.Context.html) struct and
//! a non-owning [`UnownedContext`](struct.UnownedContext.html). When the `Context` is dropped, the
//! backing context is destroyed. The context could be current on other threads, though. In this
//! case, the context is still destroyed, and attempts to access the context on other threads will
//! fail with an error. This is (mostly) safe, if a bit inconvenient. It's only mostly safe because
//! other threads could be accessing that context while the destructor is running on this thread,
//! which could result in undefined behavior.
//!
//! In short, Rust's thread-safety guarantees cannot fully protect use of the context management
//! functions. The programmer must ensure that no other OS threads are using the `Context` when it
//! is dropped.
//!
//! # Examples
//!
//! For most commmon uses (one device, one OS thread) it should suffice to create a single context:
//!
//! ```
//! use cust::device::Device;
//! use cust::context::legacy::{Context, ContextFlags};
//! # use std::error::Error;
//! # fn main () -> Result<(), Box<dyn Error>> {
//!
//! cust::init(cust::CudaFlags::empty())?;
//! let device = Device::get_device(0)?;
//! let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
//! // call cust functions which use the context
//!
//! // The context will be destroyed when dropped or it falls out of scope.
//! drop(context);
//! # Ok(())
//! # }
//! ```
//!
//! If you have multiple OS threads that each submit work to the same device, you can get a handle
//! to the single context and pass it to each thread.
//!
//! ```
//! # use cust::context::legacy::{Context, ContextFlags, CurrentContext};
//! # use cust::device::Device;
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! # cust::init(cust::CudaFlags::empty())?;
//! # let device = Device::get_device(0)?;
//! // As before
//! let context =
//!     Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
//! let mut join_handles = vec![];
//!
//! for _ in 0..4 {
//!     let unowned = context.get_unowned();
//!     let join_handle = std::thread::spawn(move || {
//!         CurrentContext::set_current(&unowned).unwrap();
//!         // Call cust functions which use the context
//!     });
//!     join_handles.push(join_handle);
//! }
//! // We must ensure that the other threads are not using the context when it's destroyed.
//! for handle in join_handles {
//!     handle.join().unwrap();
//! }
//! // Now it's safe to drop the context.
//! drop(context);
//! # Ok(())
//! # }
//! ```
//!
//! If you have multiple devices, each device needs its own context.
//!
//! ```
//! # use cust::device::Device;
//! # use cust::context::legacy::{Context, ContextStack, ContextFlags, CurrentContext};
//! # use std::error::Error;
//! #
//! # fn main() -> Result<(), Box<dyn Error>> {
//! # cust::init(cust::CudaFlags::empty())?;
//! // Create and pop contexts for each device
//! let mut contexts = vec![];
//! for device in Device::devices()? {
//!     let device = device?;
//!     let ctx =
//!         Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
//!     ContextStack::pop()?;
//!     contexts.push(ctx);
//! }
//! CurrentContext::set_current(&contexts[0])?;
//!
//! // Call cust functions which will use the context
//!
//! # Ok(())
//! # }
//! ```

use std::mem;
use std::mem::transmute;
use std::ptr;

use cust_raw::driver_sys;
use cust_raw::driver_sys::CUcontext;

use crate::CudaApiVersion;
use crate::context::ContextHandle;
use crate::device::Device;
use crate::error::{CudaResult, DropResult, ToResult};
use crate::private::Sealed;

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

/// Owned handle to a CUDA context.
///
/// The context will be destroyed when this goes out of scope. If this is the current context on
/// the current OS thread, the next context on the stack (if any) will be made current. Note that
/// the context will be destroyed even if other threads are still using it. Attempts to access the
/// destroyed context from another thread will return an error.
#[derive(Debug)]
pub struct Context {
    inner: CUcontext,
}
impl Context {
    /// Create a CUDA context for the given device.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::legacy::{Context, ContextFlags};
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// cust::init(cust::CudaFlags::empty())?;
    /// let device = Device::get_device(0)?;
    /// let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create_and_push(flags: ContextFlags, device: Device) -> CudaResult<Context> {
        unsafe {
            // CUDA only provides a create-and-push operation, but that makes it hard to provide
            // lifetime guarantees so we create-and-push, then pop, then the programmer has to
            // push again.
            let mut ctx: CUcontext = ptr::null_mut();
            driver_sys::cuCtxCreate(
                &mut ctx as *mut CUcontext,
                #[cfg(cuCtxCreate_v4)]
                &mut driver_sys::CUctxCreateParams::default(),
                flags.bits(),
                device.as_raw(),
            )
            .to_result()?;
            Ok(Context { inner: ctx })
        }
    }

    /// Get the API version used to create this context.
    ///
    /// This is not necessarily the latest version supported by the driver.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::legacy::{Context, ContextFlags};
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// cust::init(cust::CudaFlags::empty())?;
    /// let device = Device::get_device(0)?;
    /// let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
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

    /// Returns an non-owning handle to this context.
    ///
    /// This is useful for sharing a single context between threads (though see the module-level
    /// documentation for safety details!).
    ///
    /// # Example
    ////*  */
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::legacy::{Context, ContextFlags};
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
    /// let unowned = context.get_unowned();
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_unowned(&self) -> UnownedContext {
        UnownedContext { inner: self.inner }
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
    /// # use cust::context::legacy::{Context, ContextFlags};
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
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
            match driver_sys::cuCtxDestroy(inner).to_result() {
                Ok(()) => {
                    mem::forget(ctx);
                    Ok(())
                }
                Err(e) => Err((e, Context { inner })),
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
            let inner = mem::replace(&mut self.inner, ptr::null_mut());
            let _ = driver_sys::cuCtxDestroy(inner);
        }
    }
}

impl Sealed for Context {}
impl ContextHandle for Context {
    fn get_inner(&self) -> CUcontext {
        self.inner
    }
}
impl Sealed for UnownedContext {}
impl ContextHandle for UnownedContext {
    fn get_inner(&self) -> CUcontext {
        self.inner
    }
}

/// Non-owning handle to a CUDA context.
#[derive(Debug, Clone)]
pub struct UnownedContext {
    inner: CUcontext,
}
unsafe impl Send for UnownedContext {}
unsafe impl Sync for UnownedContext {}
impl UnownedContext {
    /// Get the API version used to create this context.
    ///
    /// This is not necessarily the latest version supported by the driver.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::legacy::{Context, ContextFlags};
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
    /// let unowned = context.get_unowned();
    /// let version = unowned.get_api_version()?;
    /// #
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
}

/// Type used to represent the thread-local context stack.
#[derive(Debug)]
pub struct ContextStack;
impl ContextStack {
    /// Pop the current context off the stack and return the handle. That context may then be made
    /// current again (perhaps on a different CPU thread) by calling [push](#method.push).
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::legacy ::{Context, ContextFlags, ContextStack};
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// # let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
    /// let unowned = ContextStack::pop()?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    pub fn pop() -> CudaResult<UnownedContext> {
        unsafe {
            let mut ctx: CUcontext = ptr::null_mut();
            driver_sys::cuCtxPopCurrent(&mut ctx as *mut CUcontext).to_result()?;
            Ok(UnownedContext { inner: ctx })
        }
    }

    /// Push the given context to the top of the stack
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::legacy::{Context, ContextFlags, ContextStack};
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// # let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
    /// let unowned = ContextStack::pop()?;
    /// ContextStack::push(&unowned)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn push<C: ContextHandle>(ctx: &C) -> CudaResult<()> {
        unsafe {
            driver_sys::cuCtxPushCurrent(ctx.get_inner()).to_result()?;
            Ok(())
        }
    }
}

/// Struct representing a range of stream priorities.
///
/// By convention, lower numbers imply greater priorities. The range of meaningful stream priorities
/// is given by `[greatest, least]` - that is (numerically), `greatest <= least`.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct StreamPriorityRange {
    /// The least stream priority
    pub least: i32,
    /// The greatest stream priority
    pub greatest: i32,
}

/// Type representing the top context in the thread-local stack.
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
    /// # use cust::context::legacy::{ Context, ContextFlags, CurrentContext };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
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
    /// # use cust::context::legacy::{ Context, ContextFlags, CurrentContext };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
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
    /// # use cust::context::legacy::{ Context, ContextFlags, CurrentContext };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
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
    /// # use cust::context::legacy::{ Context, ContextFlags, CurrentContext, ResourceLimit };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
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
    /// # use cust::context::legacy::{ Context, ContextFlags, CurrentContext, ResourceLimit };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
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
    /// # use cust::context::legacy::{ Context, ContextFlags, CurrentContext};
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
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
    /// # use cust::context::legacy::{ Context, ContextFlags, CurrentContext, CacheConfig };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
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
    /// * `StackSize`: Controls the stack size in bytes for each GPU thread
    /// * `PrintfFifoSize`: Controls the size in bytes of the FIFO used by the
    ///   `printf()` device system call. This cannot be changed after a kernel has been
    ///   launched which uses the `printf()` function.
    /// * `MallocHeapSize`: Controls the size in bytes of the heap used by the
    ///   `malloc()` and `free()` device system calls. This cannot be changed aftr a
    ///   kernel has been launched which uses the `malloc()` and `free()` system calls.
    /// * `DeviceRuntimeSyncDepth`: Controls the maximum nesting depth of a grid at
    ///   which a thread can safely call `cudaDeviceSynchronize()`. This cannot be
    ///   changed after a kernel has been launched which uses the device runtime. When
    ///   setting this limit, keep in mind that additional levels of sync depth require
    ///   the driver to reserve large amounts of device memory which can no longer be
    ///   used for device allocations.
    /// * `DeviceRuntimePendingLaunchCount`: Controls the maximum number of outstanding
    ///   device runtime launches that can be made from the current context. A grid is
    ///   outstanding from the point of the launch up until the grid is known to have
    ///   completed. Keep in mind that increasing this limit will require the driver to
    ///   reserve larger amounts of device memory which can no longer be used for device
    ///   allocations.
    /// * `MaxL2FetchGranularity`: Controls the L2 fetch granularity. This is purely a
    ///   performance hint and it can be ignored or clamped depending on the platform.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::legacy::{ Context, ContextFlags, CurrentContext, ResourceLimit };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
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
    /// # use cust::context::legacy::{ Context, ContextFlags, CurrentContext, SharedMemoryConfig };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
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

    /// Returns a non-owning handle to the current context.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::legacy::{ Context, ContextFlags, CurrentContext };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
    /// let unowned = CurrentContext::get_current()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_current() -> CudaResult<UnownedContext> {
        unsafe {
            let mut ctx: CUcontext = ptr::null_mut();
            driver_sys::cuCtxGetCurrent(&mut ctx as *mut CUcontext).to_result()?;
            Ok(UnownedContext { inner: ctx })
        }
    }

    /// Set the given context as the current context for this thread.
    ///
    /// If there is no context set for this thread, this pushes the given context onto the stack.
    /// If there is a context set for this thread, this replaces the top context on the stack with
    /// the given context.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::legacy::{ Context, ContextFlags, CurrentContext };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::CudaFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
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
