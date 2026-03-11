use std::mem::{self, ManuallyDrop, align_of, size_of, transmute};
use std::ops::{Deref, DerefMut};

#[cfg(feature = "bytemuck")]
pub use bytemuck;
#[cfg(feature = "bytemuck")]
use bytemuck::{Pod, PodCastError, Zeroable};
use cust_raw::driver_sys;

use crate::error::{CudaResult, DropResult, ToResult};
use crate::memory::device::{AsyncCopyDestination, CopyDestination, DeviceSlice};
use crate::memory::malloc::{cuda_free, cuda_malloc};
use crate::memory::{DeviceCopy, cuda_malloc_async};
use crate::memory::{DevicePointer, cuda_free_async};
use crate::stream::Stream;

/// Fixed-size device-side buffer. Provides basic access to device memory.
#[derive(Debug)]
#[repr(C)]
pub struct DeviceBuffer<T: DeviceCopy> {
    buf: DevicePointer<T>,
    len: usize,
}

unsafe impl<T: Send + DeviceCopy> Send for DeviceBuffer<T> {}
unsafe impl<T: Sync + DeviceCopy> Sync for DeviceBuffer<T> {}

impl<T: DeviceCopy> DeviceBuffer<T> {
    /// Allocate a new device buffer large enough to hold `size` `T`'s, but without
    /// initializing the contents.
    ///
    /// # Errors
    ///
    /// If the allocation fails, returns the error from CUDA. If `size` is large enough that
    /// `size * mem::sizeof::<T>()` overflows usize, then returns InvalidMemoryAllocation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the contents of the buffer are initialized before reading from
    /// the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let mut buffer = unsafe { DeviceBuffer::uninitialized(5).unwrap() };
    /// buffer.copy_from(&[0u64, 1, 2, 3, 4]).unwrap();
    /// ```
    pub unsafe fn uninitialized(size: usize) -> CudaResult<Self> {
        let ptr = if size > 0 && size_of::<T>() > 0 {
            unsafe { cuda_malloc(size)? }
        } else {
            // FIXME (AL): Do we /really/ want to allow creating an invalid buffer?
            DevicePointer::null()
        };
        Ok(DeviceBuffer {
            buf: ptr,
            len: size,
        })
    }

    /// Allocates device memory asynchronously on a stream, without initializing it.
    ///
    /// This doesn't actually allocate if `T` is zero sized.
    ///
    /// # Safety
    ///
    /// The allocated memory retains all of the unsafety of [`DeviceBuffer::uninitialized`], with
    /// the additional consideration that the memory cannot be used until it is actually allocated
    /// on the stream. This means proper stream ordering semantics must be followed, such as
    /// only enqueing kernel launches that use the memory AFTER the allocation call.
    ///
    /// You can synchronize the stream to ensure the memory allocation operation is complete.
    pub unsafe fn uninitialized_async(size: usize, stream: &Stream) -> CudaResult<Self> {
        let ptr = if size > 0 && size_of::<T>() > 0 {
            unsafe { cuda_malloc_async(stream, size)? }
        } else {
            DevicePointer::null()
        };
        Ok(DeviceBuffer {
            buf: ptr,
            len: size,
        })
    }

    /// Enqueues an operation to free the memory backed by this [`DeviceBuffer`] on a
    /// particular stream. The stream will free the allocation as soon as it reaches
    /// the operation in the stream. You can ensure the memory is freed by synchronizing
    /// the stream.
    ///
    /// This function uses internal memory pool semantics. Async allocations will reserve memory
    /// in the default memory pool in the stream, and async frees will release the memory back to the pool
    /// for further use by async allocations.
    ///
    /// The memory inside of the pool is all freed back to the OS once the stream is synchronized unless
    /// a custom pool is configured to not do so.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::{memory::*, stream::*};
    /// let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    /// let mut host_vals = [1, 2, 3];
    /// unsafe {
    ///     let mut allocated = DeviceBuffer::from_slice_async(&[4u8, 5, 6], &stream)?;
    ///     allocated.async_copy_to(&mut host_vals, &stream)?;
    ///     allocated.drop_async(&stream)?;
    /// }
    /// // ensure all async ops are done before trying to access the value
    /// stream.synchronize()?;
    /// assert_eq!(host_vals, [4, 5, 6]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn drop_async(self, stream: &Stream) -> CudaResult<()> {
        if self.buf.is_null() {
            return Ok(());
        }
        // make sure we dont run the normal destructor, otherwise a double drop will happen
        let me = ManuallyDrop::new(self);
        // SAFETY: we consume the box so its not possible to use the box past its drop point unless
        // you keep around a pointer, but in that case, we cannot guarantee safety.
        unsafe { cuda_free_async(stream, me.buf) }
    }

    /// Creates a `DeviceBuffer<T>` directly from the raw components of another device
    /// buffer.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren't checked:
    ///
    ///   * `ptr` needs to have been previously allocated via `DeviceBuffer` or
    ///     [`cuda_malloc`](fn.cuda_malloc.html).
    ///   * `ptr`'s `T` needs to have the same size and alignment as it was allocated
    ///     with.
    ///   * `capacity` needs to be the capacity that the pointer was allocated with.
    ///
    /// Violating these may cause problems like corrupting the CUDA driver's internal
    /// data structures.
    ///
    /// The ownership of `ptr` is effectively transferred to the `DeviceBuffer<T>` which
    /// may then deallocate, reallocate or change the contents of memory pointed to by
    /// the pointer at will. Ensure that nothing else uses the pointer after calling
    /// this function.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use std::mem;
    /// use cust::memory::*;
    ///
    /// let mut buffer = DeviceBuffer::from_slice(&[0u64; 5]).unwrap();
    /// let ptr = buffer.as_device_ptr();
    /// let size = buffer.len();
    ///
    /// mem::forget(buffer);
    ///
    /// let buffer = unsafe { DeviceBuffer::from_raw_parts(ptr, size) };
    /// ```
    pub unsafe fn from_raw_parts(ptr: DevicePointer<T>, capacity: usize) -> DeviceBuffer<T> {
        DeviceBuffer {
            buf: ptr,
            len: capacity,
        }
    }

    /// Destroy a `DeviceBuffer`, returning an error.
    ///
    /// Deallocating device memory can return errors from previous asynchronous work. This function
    /// destroys the given buffer and returns the error and the un-destroyed buffer on failure.
    ///
    /// # Example
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let x = DeviceBuffer::from_slice(&[10, 20, 30]).unwrap();
    /// match DeviceBuffer::drop(x) {
    ///     Ok(()) => println!("Successfully destroyed"),
    ///     Err((e, buf)) => {
    ///         println!("Failed to destroy buffer: {:?}", e);
    ///         // Do something with buf
    ///     },
    /// }
    /// ```
    pub fn drop(mut dev_buf: DeviceBuffer<T>) -> DropResult<DeviceBuffer<T>> {
        if dev_buf.buf.is_null() {
            return Ok(());
        }

        if dev_buf.len > 0 && size_of::<T>() > 0 {
            let capacity = dev_buf.len;
            let ptr = mem::replace(&mut dev_buf.buf, DevicePointer::null());
            unsafe {
                match cuda_free(ptr) {
                    Ok(()) => {
                        mem::forget(dev_buf);
                        Ok(())
                    }
                    Err(e) => Err((e, DeviceBuffer::from_raw_parts(ptr, capacity))),
                }
            }
        } else {
            Ok(())
        }
    }
}

#[cfg(feature = "bytemuck")]
impl<T: DeviceCopy + Zeroable> DeviceBuffer<T> {
    /// Allocate device memory and fill it with zeroes (`0u8`).
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let mut zero = DeviceBuffer::zeroed(4).unwrap();
    /// let mut values = [1u8, 2, 3, 4];
    /// zero.copy_to(&mut values).unwrap();
    /// assert_eq!(values, [0; 4]);
    /// ```
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    pub fn zeroed(size: usize) -> CudaResult<Self> {
        unsafe {
            let new_buf = DeviceBuffer::uninitialized(size)?;
            if size_of::<T>() != 0 {
                driver_sys::cuMemsetD8(new_buf.as_device_ptr().as_raw(), 0, size_of::<T>() * size)
                    .to_result()?;
            }
            Ok(new_buf)
        }
    }

    /// Allocates device memory asynchronously and asynchronously fills it with zeroes
    /// (`0u8`).
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Safety
    ///
    /// This method enqueues two operations on the stream: An async allocation and an
    /// async memset. Because of this, you must ensure that:
    ///   - The memory is not used in any way before it is actually allocated on the
    ///     stream. You can ensure this happens by synchronizing the stream explicitly
    ///     or using events.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::{memory::*, stream::*};
    /// let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    /// let mut values = [1u8, 2, 3, 4];
    /// unsafe {
    ///     let mut zero = DeviceBuffer::zeroed_async(4, &stream)?;
    ///     zero.async_copy_to(&mut values, &stream)?;
    ///     zero.drop_async(&stream)?;
    /// }
    /// stream.synchronize()?;
    /// assert_eq!(values, [0; 4]);
    /// # Ok(())
    /// # }
    /// ```
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    pub unsafe fn zeroed_async(size: usize, stream: &Stream) -> CudaResult<Self> {
        let new_buf = unsafe { DeviceBuffer::uninitialized_async(size, stream)? };
        if size_of::<T>() != 0 {
            unsafe {
                driver_sys::cuMemsetD8Async(
                    new_buf.as_device_ptr().as_raw(),
                    0,
                    size_of::<T>() * size,
                    stream.as_inner(),
                )
            }
            .to_result()?;
        }
        Ok(new_buf)
    }
}

#[cfg(feature = "bytemuck")]
fn casting_went_wrong(src: &str, err: PodCastError) -> ! {
    panic!("{src}>{err:?}");
}

#[cfg(feature = "bytemuck")]
impl<A: DeviceCopy + Pod> DeviceBuffer<A> {
    /// Same as [`DeviceBuffer::try_cast`] but panics if the cast fails.
    ///
    /// # Panics
    ///
    /// See [`DeviceBuffer::try_cast`].
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    pub fn cast<B: Pod + DeviceCopy>(self) -> DeviceBuffer<B> {
        match Self::try_cast(self) {
            Ok(b) => b,
            Err(e) => casting_went_wrong("cast", e),
        }
    }

    /// Tries to convert a [`DeviceBuffer`] of type `A` to a [`DeviceBuffer`] of type
    /// `B`. Returning an error if it failed.
    ///
    /// The length of the buffer after the conversion may have changed.
    ///
    /// # Failure
    ///
    ///   - If the target type has a greater alignment requirement.
    ///   - If the target element type is a different size and the output buffer
    ///     wouldn't have a whole number of elements. Such as `3` x [`u16`] -> `1.5` x
    ///     [`u32`].
    ///   - If either type is a ZST (but not both).
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    pub fn try_cast<B: Pod + DeviceCopy>(self) -> Result<DeviceBuffer<B>, PodCastError> {
        if align_of::<B>() > align_of::<A>()
            && !(self.buf.as_raw() as usize).is_multiple_of(align_of::<B>())
        {
            Err(PodCastError::TargetAlignmentGreaterAndInputNotAligned)
        } else if size_of::<B>() == size_of::<A>() {
            // SAFETY: we made sure sizes were compatible, and DeviceBuffer is repr(C)
            Ok(unsafe { transmute::<DeviceBuffer<A>, DeviceBuffer<B>>(self) })
        } else if size_of::<A>() == 0 || size_of::<B>() == 0 {
            Err(PodCastError::SizeMismatch)
        } else if (size_of::<A>() * self.len).is_multiple_of(size_of::<B>()) {
            let new_len = (size_of::<A>() * self.len) / size_of::<B>();
            let ret = Ok(DeviceBuffer {
                buf: self.buf.cast(),
                len: new_len,
            });
            std::mem::forget(self);
            ret
        } else {
            Err(PodCastError::OutputSliceWouldHaveSlop)
        }
    }
}

impl<T: DeviceCopy> DeviceBuffer<T> {
    /// Allocate a new device buffer of the same size as `slice`, initialized with a clone of
    /// the data in `slice`.
    ///
    /// # Errors
    ///
    /// If the allocation fails, returns the error from CUDA.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let values = [0u64; 5];
    /// let mut buffer = DeviceBuffer::from_slice(&values).unwrap();
    /// ```
    pub fn from_slice(slice: &[T]) -> CudaResult<Self> {
        unsafe {
            let mut uninit = DeviceBuffer::uninitialized(slice.len())?;
            uninit.copy_from(slice)?;
            Ok(uninit)
        }
    }

    /// Asynchronously allocate a new buffer of the same size as `slice`, initialized
    /// with a clone of the data in `slice`.
    ///
    /// # Safety
    ///
    /// For why this function is unsafe, see [AsyncCopyDestination](trait.AsyncCopyDestination.html)
    ///
    /// # Errors
    ///
    /// If the allocation fails, returns the error from CUDA.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// use cust::stream::{Stream, StreamFlags};
    ///
    /// let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
    /// let values = [0u64; 5];
    /// unsafe {
    ///     let mut buffer = DeviceBuffer::from_slice_async(&values, &stream).unwrap();
    ///     stream.synchronize();
    ///     // Perform some operation on the buffer
    /// }
    /// ```
    pub unsafe fn from_slice_async(slice: &[T], stream: &Stream) -> CudaResult<Self> {
        unsafe {
            let mut uninit = DeviceBuffer::uninitialized_async(slice.len(), stream)?;
            uninit.async_copy_from(slice, stream)?;
            Ok(uninit)
        }
    }

    /// Explicitly creates a [`DeviceSlice`] from this buffer.
    pub fn as_slice(&self) -> &DeviceSlice<T> {
        self
    }
}

impl<T: DeviceCopy> Deref for DeviceBuffer<T> {
    type Target = DeviceSlice<T>;

    fn deref(&self) -> &DeviceSlice<T> {
        unsafe { DeviceSlice::from_raw_parts(self.buf, self.len) }
    }
}

impl<T: DeviceCopy> DerefMut for DeviceBuffer<T> {
    fn deref_mut(&mut self) -> &mut DeviceSlice<T> {
        unsafe { DeviceSlice::from_raw_parts_mut(self.buf, self.len) }
    }
}

impl<T: DeviceCopy> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.buf.is_null() {
            return;
        }

        if self.len > 0 && size_of::<T>() > 0 {
            let ptr = mem::replace(&mut self.buf, DevicePointer::null());
            unsafe {
                let _ = cuda_free(ptr);
            }
        }
        self.len = 0;
    }
}

#[cfg(test)]
mod test_device_buffer {
    use super::*;
    use crate::stream::{Stream, StreamFlags};

    #[expect(dead_code)]
    #[derive(Clone, Copy, Debug)]
    struct ZeroSizedType;
    unsafe impl DeviceCopy for ZeroSizedType {}

    #[test]
    fn test_from_slice_drop() {
        let _context = crate::quick_init().unwrap();
        let buf = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        drop(buf);
    }

    #[test]
    fn test_copy_to_from_device() {
        let _context = crate::quick_init().unwrap();
        let start = [0u64, 1, 2, 3, 4, 5];
        let mut end = [0u64, 0, 0, 0, 0, 0];
        let buf = DeviceBuffer::from_slice(&start).unwrap();
        buf.copy_to(&mut end).unwrap();
        assert_eq!(start, end);
    }

    #[test]
    fn test_async_copy_to_from_device() {
        let _context = crate::quick_init().unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        let start = [0u64, 1, 2, 3, 4, 5];
        let mut end = [0u64, 0, 0, 0, 0, 0];
        unsafe {
            let buf = DeviceBuffer::from_slice_async(&start, &stream).unwrap();
            buf.async_copy_to(&mut end, &stream).unwrap();
        }
        stream.synchronize().unwrap();
        assert_eq!(start, end);
    }

    #[test]
    #[should_panic]
    fn test_copy_to_d2h_wrong_size() {
        let _context = crate::quick_init().unwrap();
        let buf = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        let mut end = [0u64, 1, 2, 3, 4];
        let _ = buf.copy_to(&mut end);
    }

    #[test]
    #[should_panic]
    fn test_async_copy_to_d2h_wrong_size() {
        let _context = crate::quick_init().unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        unsafe {
            let buf = DeviceBuffer::from_slice_async(&[0u64, 1, 2, 3, 4, 5], &stream).unwrap();
            let mut end = [0u64, 1, 2, 3, 4];
            let _ = buf.async_copy_to(&mut end, &stream);
        }
    }

    #[test]
    #[should_panic]
    fn test_copy_from_h2d_wrong_size() {
        let _context = crate::quick_init().unwrap();
        let start = [0u64, 1, 2, 3, 4];
        let mut buf = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        let _ = buf.copy_from(&start);
    }

    #[test]
    #[should_panic]
    fn test_async_copy_from_h2d_wrong_size() {
        let _context = crate::quick_init().unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        let start = [0u64, 1, 2, 3, 4];
        unsafe {
            let mut buf = DeviceBuffer::from_slice_async(&[0u64, 1, 2, 3, 4, 5], &stream).unwrap();
            let _ = buf.async_copy_from(&start, &stream);
        }
    }

    #[test]
    #[should_panic]
    fn test_copy_to_d2d_wrong_size() {
        let _context = crate::quick_init().unwrap();
        let buf = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        let mut end = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4]).unwrap();
        let _ = buf.copy_to(&mut end);
    }

    #[test]
    #[should_panic]
    fn test_async_copy_to_d2d_wrong_size() {
        let _context = crate::quick_init().unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        unsafe {
            let buf = DeviceBuffer::from_slice_async(&[0u64, 1, 2, 3, 4, 5], &stream).unwrap();
            let mut end = DeviceBuffer::from_slice_async(&[0u64, 1, 2, 3, 4], &stream).unwrap();
            let _ = buf.async_copy_to(&mut end, &stream);
        }
    }

    #[test]
    #[should_panic]
    fn test_copy_from_d2d_wrong_size() {
        let _context = crate::quick_init().unwrap();
        let mut buf = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        let start = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4]).unwrap();
        let _ = buf.copy_from(&start);
    }

    #[test]
    #[should_panic]
    fn test_async_copy_from_d2d_wrong_size() {
        let _context = crate::quick_init().unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        unsafe {
            let mut buf = DeviceBuffer::from_slice_async(&[0u64, 1, 2, 3, 4, 5], &stream).unwrap();
            let start = DeviceBuffer::from_slice_async(&[0u64, 1, 2, 3, 4], &stream).unwrap();
            let _ = buf.async_copy_from(&start, &stream);
        }
    }
}
