//! Streams of work for the device to perform.
//!
//! In CUDA, most work is performed asynchronously. Even tasks such as memory copying can be
//! scheduled by the host and performed when ready. Scheduling this work is done using a Stream.
//!
//! A stream is required for all asynchronous tasks in CUDA, such as kernel launches and
//! asynchronous memory copying. Each task in a stream is performed in the order it was scheduled,
//! and tasks within a stream cannot overlap. Tasks scheduled in multiple streams may interleave or
//! execute concurrently. Sequencing between multiple streams can be achieved using events, which
//! are not currently supported by cust. Finally, the host can wait for all work scheduled in
//! a stream to be completed.

use std::ffi::c_void;
use std::mem;
use std::panic;
use std::ptr;

use cust_raw::driver_sys;
use cust_raw::driver_sys::{CUstream, cudaError_enum};

use crate::error::{CudaResult, DropResult, ToResult};
use crate::event::Event;
use crate::function::{BlockSize, Function, GridSize};

bitflags::bitflags! {
    /// Bit flags for configuring a CUDA Stream.
    pub struct StreamFlags: u32 {
        /// No flags set.
        const DEFAULT = 0x00;

        /// This stream does not synchronize with the NULL stream.
        ///
        /// **Note: this flag has been temporarily disabled because of [soundness concerns](https://github.com/Rust-GPU/rust-cuda/issues/15)**
        ///
        /// Note that the name is chosen to correspond to CUDA documentation, but is nevertheless
        /// misleading. All work within a single stream is ordered and asynchronous regardless
        /// of whether this flag is set. All streams in cust may execute work concurrently,
        /// regardless of the flag. However, for legacy reasons, CUDA has a notion of a NULL stream,
        /// which is used as the default when no other stream is provided. Work on other streams
        /// may not be executed concurrently with work on the NULL stream unless this flag is set.
        /// Since cust does not provide access to the NULL stream, this flag has no effect in
        /// most circumstances. However, it is recommended to use it anyway, as some other crate
        /// in this binary may be using the NULL stream directly.
        const NON_BLOCKING = 0x01;
    }
}

bitflags::bitflags! {
    /// Bit flags for configuring a CUDA Stream waiting on an CUDA Event.
    ///
    /// Current versions of CUDA support only the default flag.
    pub struct StreamWaitEventFlags: u32 {
        /// No flags set.
        const DEFAULT = 0x0;
    }
}

/// A stream of work for the device to perform.
///
/// See the module-level documentation for more information.
#[derive(Debug)]
pub struct Stream {
    inner: CUstream,
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    /// Create a new stream with the given flags and optional priority.
    ///
    /// By convention, `priority` follows a convention where lower numbers represent greater
    /// priorities. That is, work in a stream with a lower priority number may pre-empt work in
    /// a stream with a higher priority number. `Context::get_stream_priority_range` can be used
    /// to get the range of valid priority values; if priority is set outside that range, it will
    /// be automatically clamped to the lowest or highest number in the range.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cust::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use cust::stream::{Stream, StreamFlags};
    ///
    /// // With default priority
    /// let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    ///
    /// // With specific priority
    /// let priority = Stream::new(StreamFlags::NON_BLOCKING, 1i32.into())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(mut flags: StreamFlags, priority: Option<i32>) -> CudaResult<Self> {
        // NOTE(RDambrosio016): See https://github.com/Rust-GPU/rust-cuda/issues/15
        flags.remove(StreamFlags::NON_BLOCKING);
        unsafe {
            let mut stream = Stream {
                inner: ptr::null_mut(),
            };
            driver_sys::cuStreamCreateWithPriority(
                &mut stream.inner as *mut CUstream,
                flags.bits(),
                priority.unwrap_or(0),
            )
            .to_result()?;
            Ok(stream)
        }
    }

    /// Return the flags which were used to create this stream.
    pub fn get_flags(&self) -> CudaResult<StreamFlags> {
        unsafe {
            let mut bits = 0u32;
            driver_sys::cuStreamGetFlags(self.inner, &mut bits as *mut u32).to_result()?;
            Ok(StreamFlags::from_bits_truncate(bits))
        }
    }

    /// Return the priority of this stream.
    ///
    /// If this stream was created without a priority, returns the default priority.
    /// If the stream was created with a priority outside the valid range, returns the clamped
    /// priority.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cust::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use cust::stream::{Stream, StreamFlags};
    ///
    /// let stream = Stream::new(StreamFlags::NON_BLOCKING, 1i32.into())?;
    /// println!("{}", stream.get_priority()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_priority(&self) -> CudaResult<i32> {
        unsafe {
            let mut priority = 0i32;
            driver_sys::cuStreamGetPriority(self.inner, &mut priority as *mut i32).to_result()?;
            Ok(priority)
        }
    }

    /// Add a callback to a stream.
    ///
    /// The callback will be executed after all previously queued
    /// items in the stream have been completed. Subsequently queued
    /// items will not execute until the callback is finished.
    ///
    /// Callbacks must not make any CUDA API calls.
    ///
    /// The callback will be passed a `CudaResult<()>` indicating the
    /// current state of the device with `Ok(())` denoting normal operation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cust::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use cust::stream::{Stream, StreamFlags};
    ///
    /// let stream = Stream::new(StreamFlags::NON_BLOCKING, 1i32.into())?;
    ///
    /// // ... queue up some work on the stream
    ///
    /// stream.add_callback(Box::new(|status| {
    ///     println!("Device status is {:?}", status);
    /// }));
    ///
    /// // ... queue up some more work on the stream
    /// # Ok(())
    /// # }
    pub fn add_callback<T>(&self, callback: Box<T>) -> CudaResult<()>
    where
        T: FnOnce(CudaResult<()>) + Send,
    {
        unsafe {
            driver_sys::cuStreamAddCallback(
                self.inner,
                Some(callback_wrapper::<T>),
                Box::into_raw(callback) as *mut c_void,
                0,
            )
            .to_result()
        }
    }

    /// Wait until a stream's tasks are completed.
    ///
    /// Waits until the device has completed all operations scheduled for this stream.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cust::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use cust::stream::{Stream, StreamFlags};
    ///
    /// let stream = Stream::new(StreamFlags::NON_BLOCKING, 1i32.into())?;
    ///
    /// // ... queue up some work on the stream
    ///
    /// // Wait for the work to be completed.
    /// stream.synchronize()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn synchronize(&self) -> CudaResult<()> {
        unsafe { driver_sys::cuStreamSynchronize(self.inner).to_result() }
    }

    /// Make the stream wait on an event.
    ///
    /// All future work submitted to the stream will wait for the event to
    /// complete. Synchronization is performed on the device, if possible. The
    /// event may originate from different context or device than the stream.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::quick_init;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _context = quick_init()?;
    /// use cust::stream::{Stream, StreamFlags, StreamWaitEventFlags};
    /// use cust::event::{Event, EventFlags};
    ///
    /// let stream_0 = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    /// let stream_1 = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    /// let event = Event::new(EventFlags::DEFAULT)?;
    ///
    /// // do some work on stream_0 ...
    ///
    /// // record an event
    /// event.record(&stream_0)?;
    ///
    /// // wait until the work on stream_0 is finished before continuing stream_1
    /// stream_1.wait_event(&event, StreamWaitEventFlags::DEFAULT)?;
    /// # Ok(())
    /// }
    /// ```
    pub fn wait_event(&self, event: &Event, flags: StreamWaitEventFlags) -> CudaResult<()> {
        unsafe {
            driver_sys::cuStreamWaitEvent(self.inner, event.as_inner(), flags.bits()).to_result()
        }
    }

    // Hidden implementation detail function. Highly unsafe. Use the `launch!` macro instead.
    #[doc(hidden)]
    pub unsafe fn launch<G, B>(
        &self,
        func: &Function,
        grid_size: G,
        block_size: B,
        shared_mem_bytes: u32,
        args: &[*mut c_void],
    ) -> CudaResult<()>
    where
        G: Into<GridSize>,
        B: Into<BlockSize>,
    {
        let grid_size: GridSize = grid_size.into();
        let block_size: BlockSize = block_size.into();

        let to_u32 = |i: usize| i.try_into().expect("launch size must fit in a `u32`");
        unsafe {
            driver_sys::cuLaunchKernel(
                func.to_raw(),
                to_u32(grid_size.x),
                to_u32(grid_size.y),
                to_u32(grid_size.z),
                to_u32(block_size.x),
                to_u32(block_size.y),
                to_u32(block_size.z),
                shared_mem_bytes,
                self.inner,
                args.as_ptr() as *mut _,
                ptr::null_mut(),
            )
        }
        .to_result()
    }

    // Get the inner `CUstream` from the `Stream`. If you use this handle elsewhere,
    // make sure not to use it after the stream has been dropped. Or ManuallyDrop the struct to be safe.
    pub fn as_inner(&self) -> CUstream {
        self.inner
    }

    /// Destroy a `Stream`, returning an error.
    ///
    /// Destroying a stream can return errors from previous asynchronous work. This function
    /// destroys the given stream and returns the error and the un-destroyed stream on failure.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use cust::stream::{Stream, StreamFlags};
    ///
    /// let stream = Stream::new(StreamFlags::NON_BLOCKING, 1i32.into())?;
    /// match Stream::drop(stream) {
    ///     Ok(()) => println!("Successfully destroyed"),
    ///     Err((e, stream)) => {
    ///         println!("Failed to destroy stream: {:?}", e);
    ///         // Do something with stream
    ///     },
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn drop(mut stream: Stream) -> DropResult<Stream> {
        if stream.inner.is_null() {
            return Ok(());
        }

        unsafe {
            let inner = mem::replace(&mut stream.inner, ptr::null_mut());
            match driver_sys::cuStreamDestroy(inner).to_result() {
                Ok(()) => {
                    mem::forget(stream);
                    Ok(())
                }
                Err(e) => Err((e, Stream { inner })),
            }
        }
    }
}
impl Drop for Stream {
    fn drop(&mut self) {
        if self.inner.is_null() {
            return;
        }

        unsafe {
            let inner = mem::replace(&mut self.inner, ptr::null_mut());

            let _ = driver_sys::cuStreamDestroy(inner);
        }
    }
}
unsafe extern "C" fn callback_wrapper<T>(
    _stream: CUstream,
    status: cudaError_enum,
    callback: *mut c_void,
) where
    T: FnOnce(CudaResult<()>) + Send,
{
    // Stop panics from unwinding across the FFI
    let _ = panic::catch_unwind(|| {
        let callback: Box<T> = unsafe { Box::from_raw(callback as *mut T) };
        callback(status.to_result());
    });
}
