use std::borrow::{Borrow, BorrowMut};
use std::cmp::Ordering;
use std::convert::{AsMut, AsRef};
use std::fmt::{self, Display, Pointer};
use std::hash::{Hash, Hasher};
use std::mem;
use std::ops::{Deref, DerefMut};
use std::ptr;
use std::slice;

use cust_raw::driver_sys;

use super::DeviceCopy;
use crate::device::Device;
#[allow(unused_imports)]
use crate::device::DeviceAttribute;
use crate::error::*;
use crate::memory::malloc::{cuda_free_unified, cuda_malloc_unified};
use crate::memory::UnifiedPointer;
use crate::prelude::Stream;

/// A pointer type for heap-allocation in CUDA unified memory.
///
/// See the [`module-level documentation`](../memory/index.html) for more information on unified
/// memory. Should behave equivalently to `std::boxed::Box`, except that the allocated memory can be
/// seamlessly shared between host and device.
#[derive(Debug)]
pub struct UnifiedBox<T: DeviceCopy> {
    pub(crate) ptr: UnifiedPointer<T>,
}

unsafe impl<T: Send + DeviceCopy> Send for UnifiedBox<T> {}
unsafe impl<T: Sync + DeviceCopy> Sync for UnifiedBox<T> {}

impl<T: DeviceCopy> UnifiedBox<T> {
    /// Allocate unified memory and place val into it.
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Errors
    ///
    /// If a CUDA error occurs, returns that error.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let five = UnifiedBox::new(5).unwrap();
    /// ```
    pub fn new(val: T) -> CudaResult<Self> {
        if mem::size_of::<T>() == 0 {
            Ok(UnifiedBox {
                ptr: UnifiedPointer::null(),
            })
        } else {
            let mut ubox = unsafe { UnifiedBox::uninitialized()? };
            *ubox = val;
            Ok(ubox)
        }
    }

    /// Allocate unified memory without initializing it.
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Safety
    ///
    /// Since the backing memory is not initialized, this function is not safe. The caller must
    /// ensure that the backing memory is set to a valid value before it is read, else undefined
    /// behavior may occur.
    ///
    /// # Errors
    ///
    /// If a CUDA error occurs, returns that error.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let mut five = unsafe{ UnifiedBox::uninitialized().unwrap() };
    /// *five = 5u64;
    /// ```
    pub unsafe fn uninitialized() -> CudaResult<Self> {
        if mem::size_of::<T>() == 0 {
            Ok(UnifiedBox {
                ptr: UnifiedPointer::null(),
            })
        } else {
            let ptr = cuda_malloc_unified(1)?;
            Ok(UnifiedBox { ptr })
        }
    }

    /// Constructs a UnifiedBox from a raw pointer.
    ///
    /// After calling this function, the raw pointer and the memory it points to is owned by the
    /// UnifiedBox. The UnifiedBox destructor will free the allocated memory, but will not call the destructor
    /// of `T`. This function may accept any pointer produced by the `cuMemAllocManaged` CUDA API
    /// call.
    ///
    /// # Safety
    ///
    /// This function is unsafe because improper use may lead to memory problems. For example, a
    /// double free may occur if this function is called twice on the same pointer, or a segfault
    /// may occur if the pointer is not one returned by the appropriate API call.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let x = UnifiedBox::new(5).unwrap();
    /// let ptr = UnifiedBox::into_unified(x).as_raw_mut();
    /// let x = unsafe { UnifiedBox::from_raw(ptr) };
    /// ```
    pub unsafe fn from_raw(ptr: *mut T) -> Self {
        UnifiedBox {
            ptr: UnifiedPointer::wrap(ptr),
        }
    }

    /// Constructs a UnifiedBox from a UnifiedPointer.
    ///
    /// After calling this function, the pointer and the memory it points to is owned by the
    /// UnifiedBox. The UnifiedBox destructor will free the allocated memory, but will not call the destructor
    /// of `T`. This function may accept any pointer produced by the `cuMemAllocManaged` CUDA API
    /// call, such as one taken from `UnifiedBox::into_unified`.
    ///
    /// # Safety
    ///
    /// This function is unsafe because improper use may lead to memory problems. For example, a
    /// double free may occur if this function is called twice on the same pointer, or a segfault
    /// may occur if the pointer is not one returned by the appropriate API call.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let x = UnifiedBox::new(5).unwrap();
    /// let ptr = UnifiedBox::into_unified(x);
    /// let x = unsafe { UnifiedBox::from_unified(ptr) };
    /// ```
    pub unsafe fn from_unified(ptr: UnifiedPointer<T>) -> Self {
        UnifiedBox { ptr }
    }

    /// Consumes the UnifiedBox, returning the wrapped UnifiedPointer.
    ///
    /// After calling this function, the caller is responsible for the memory previously managed by
    /// the UnifiedBox. In particular, the caller should properly destroy T and deallocate the memory.
    /// The easiest way to do so is to create a new UnifiedBox using the `UnifiedBox::from_unified` function.
    ///
    /// Note: This is an associated function, which means that you have to all it as
    /// `UnifiedBox::into_unified(b)` instead of `b.into_unified()` This is so that there is no conflict with
    /// a method on the inner type.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let x = UnifiedBox::new(5).unwrap();
    /// let ptr = UnifiedBox::into_unified(x);
    /// # unsafe { UnifiedBox::from_unified(ptr) };
    /// ```
    #[allow(clippy::wrong_self_convention)]
    pub fn into_unified(mut b: UnifiedBox<T>) -> UnifiedPointer<T> {
        let ptr = mem::replace(&mut b.ptr, UnifiedPointer::null());
        mem::forget(b);
        ptr
    }

    /// Returns the contained unified pointer without consuming the box.
    ///
    /// This is useful for passing the box to a kernel launch.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let mut x = UnifiedBox::new(5).unwrap();
    /// let ptr = x.as_unified_ptr();
    /// println!("{:p}", ptr);
    /// ```
    pub fn as_unified_ptr(&self) -> UnifiedPointer<T> {
        self.ptr
    }

    /// Consumes and leaks the UnifiedBox, returning a mutable reference, &'a mut T. Note that the type T
    /// must outlive the chosen lifetime 'a. If the type has only static references, or none at all,
    /// this may be chosen to be 'static.
    ///
    /// This is mainly useful for data that lives for the remainder of the program's life. Dropping
    /// the returned reference will cause a memory leak. If this is not acceptable, the reference
    /// should be wrapped with the UnifiedBox::from_raw function to produce a new UnifiedBox. This UnifiedBox can then
    /// be dropped, which will properly destroy T and release the allocated memory.
    ///
    /// Note: This is an associated function, which means that you have to all it as
    /// `UnifiedBox::leak(b)` instead of `b.leak()` This is so that there is no conflict with
    /// a method on the inner type.
    pub fn leak<'a>(b: UnifiedBox<T>) -> &'a mut T
    where
        T: 'a,
    {
        unsafe { &mut *UnifiedBox::into_unified(b).as_raw_mut() }
    }

    /// Destroy a `UnifiedBox`, returning an error.
    ///
    /// Deallocating unified memory can return errors from previous asynchronous work. This function
    /// destroys the given box and returns the error and the un-destroyed box on failure.
    ///
    /// # Example
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let x = UnifiedBox::new(5).unwrap();
    /// match UnifiedBox::drop(x) {
    ///     Ok(()) => println!("Successfully destroyed"),
    ///     Err((e, uni_box)) => {
    ///         println!("Failed to destroy box: {:?}", e);
    ///         // Do something with uni_box
    ///     },
    /// }
    /// ```
    pub fn drop(mut uni_box: UnifiedBox<T>) -> DropResult<UnifiedBox<T>> {
        if uni_box.ptr.is_null() {
            return Ok(());
        }

        let ptr = mem::replace(&mut uni_box.ptr, UnifiedPointer::null());
        unsafe {
            match cuda_free_unified(ptr) {
                Ok(()) => {
                    mem::forget(uni_box);
                    Ok(())
                }
                Err(e) => Err((e, UnifiedBox { ptr })),
            }
        }
    }
}
impl<T: DeviceCopy> Drop for UnifiedBox<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let ptr = mem::replace(&mut self.ptr, UnifiedPointer::null());
            unsafe {
                let _ = cuda_free_unified(ptr);
            }
        }
    }
}

impl<T: DeviceCopy> Borrow<T> for UnifiedBox<T> {
    fn borrow(&self) -> &T {
        self
    }
}
impl<T: DeviceCopy> BorrowMut<T> for UnifiedBox<T> {
    fn borrow_mut(&mut self) -> &mut T {
        self
    }
}
impl<T: DeviceCopy> AsRef<T> for UnifiedBox<T> {
    fn as_ref(&self) -> &T {
        self
    }
}
impl<T: DeviceCopy> AsMut<T> for UnifiedBox<T> {
    fn as_mut(&mut self) -> &mut T {
        self
    }
}
impl<T: DeviceCopy> Deref for UnifiedBox<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.ptr.as_raw() }
    }
}
impl<T: DeviceCopy> DerefMut for UnifiedBox<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.ptr.as_raw_mut() }
    }
}
impl<T: Display + DeviceCopy> Display for UnifiedBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}
impl<T: DeviceCopy> Pointer for UnifiedBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, f)
    }
}
impl<T: DeviceCopy + PartialEq> PartialEq for UnifiedBox<T> {
    fn eq(&self, other: &UnifiedBox<T>) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}
impl<T: DeviceCopy + Eq> Eq for UnifiedBox<T> {}
impl<T: DeviceCopy + PartialOrd> PartialOrd for UnifiedBox<T> {
    fn partial_cmp(&self, other: &UnifiedBox<T>) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
    fn lt(&self, other: &UnifiedBox<T>) -> bool {
        PartialOrd::lt(&**self, &**other)
    }
    fn le(&self, other: &UnifiedBox<T>) -> bool {
        PartialOrd::le(&**self, &**other)
    }
    fn ge(&self, other: &UnifiedBox<T>) -> bool {
        PartialOrd::ge(&**self, &**other)
    }
    fn gt(&self, other: &UnifiedBox<T>) -> bool {
        PartialOrd::gt(&**self, &**other)
    }
}
impl<T: DeviceCopy + Ord> Ord for UnifiedBox<T> {
    fn cmp(&self, other: &UnifiedBox<T>) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}
impl<T: DeviceCopy + Hash> Hash for UnifiedBox<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

/// Fixed-size buffer in unified memory.
///
/// See the [`module-level documentation`](../memory/index.html) for more details on unified memory.
#[derive(Debug)]
#[repr(C)]
pub struct UnifiedBuffer<T: DeviceCopy> {
    buf: UnifiedPointer<T>,
    capacity: usize,
}
impl<T: DeviceCopy + Clone> UnifiedBuffer<T> {
    /// Allocate a new unified buffer large enough to hold `size` `T`'s and initialized with
    /// clones of `value`.
    ///
    /// # Errors
    ///
    /// If the allocation fails, returns the error from CUDA. If `size` is large enough that
    /// `size * mem::sizeof::<T>()` overflows usize, then returns InvalidMemoryAllocation.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let mut buffer = UnifiedBuffer::new(&0u64, 5).unwrap();
    /// buffer[0] = 1;
    /// ```
    pub fn new(value: &T, size: usize) -> CudaResult<Self> {
        unsafe {
            let mut uninit = UnifiedBuffer::uninitialized(size)?;
            for x in 0..size {
                *uninit.get_unchecked_mut(x) = *value;
            }
            Ok(uninit)
        }
    }

    /// Allocate a new unified buffer of the same size as `slice`, initialized with a clone of
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
    /// let mut buffer = UnifiedBuffer::from_slice(&values).unwrap();
    /// buffer[0] = 1;
    /// ```
    pub fn from_slice(slice: &[T]) -> CudaResult<Self> {
        unsafe {
            let mut uninit = UnifiedBuffer::uninitialized(slice.len())?;
            for (i, x) in slice.iter().enumerate() {
                *uninit.get_unchecked_mut(i) = *x;
            }
            Ok(uninit)
        }
    }
}
impl<T: DeviceCopy> UnifiedBuffer<T> {
    /// Allocate a new unified buffer large enough to hold `size` `T`'s, but without
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
    /// let mut buffer = unsafe { UnifiedBuffer::uninitialized(5).unwrap() };
    /// for i in buffer.iter_mut() {
    ///     *i = 0u64;
    /// }
    /// ```
    pub unsafe fn uninitialized(size: usize) -> CudaResult<Self> {
        let ptr = if size > 0 && mem::size_of::<T>() > 0 {
            cuda_malloc_unified(size)?
        } else {
            UnifiedPointer::wrap(ptr::NonNull::dangling().as_ptr())
        };
        Ok(UnifiedBuffer {
            buf: ptr,
            capacity: size,
        })
    }

    /// Extracts a slice containing the entire buffer.
    ///
    /// Equivalent to `&s[..]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let buffer = UnifiedBuffer::new(&0u64, 5).unwrap();
    /// let sum : u64 = buffer.as_slice().iter().sum();
    /// ```
    pub fn as_slice(&self) -> &[T] {
        self
    }

    /// Extracts a mutable slice of the entire buffer.
    ///
    /// Equivalent to `&mut s[..]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let mut buffer = UnifiedBuffer::new(&0u64, 5).unwrap();
    /// for i in buffer.as_mut_slice() {
    ///     *i = 12u64;
    /// }
    /// ```
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }

    /// Returns a `UnifiedPointer<T>` to the buffer.
    ///
    /// The caller must ensure that the buffer outlives the returned pointer, or it will end up
    /// pointing to garbage.
    ///
    /// Modifying the buffer is guaranteed not to cause its buffer to be reallocated, so pointers
    /// cannot be invalidated in that manner, but other types may be added in the future which can
    /// reallocate.
    pub fn as_unified_ptr(&mut self) -> UnifiedPointer<T> {
        self.buf
    }

    /// Creates a `UnifiedBuffer<T>` directly from the raw components of another unified
    /// buffer.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren't checked:
    ///
    ///   * `ptr` needs to have been previously allocated via `UnifiedBuffer` or
    ///     [`cuda_malloc_unified`](fn.cuda_malloc_unified.html).
    ///   * `ptr`'s `T` needs to have the same size and alignment as it was allocated
    ///     with.
    ///   * `capacity` needs to be the capacity that the pointer was allocated with.
    ///
    /// Violating these may cause problems like corrupting the CUDA driver's internal
    /// data structures.
    ///
    /// The ownership of `ptr` is effectively transferred to the `UnifiedBuffer<T>`
    /// which may then deallocate, reallocate or change the contents of memory pointed
    /// to by the pointer at will. Ensure that nothing else uses the pointer after
    /// calling this function.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use std::mem;
    /// use cust::memory::*;
    ///
    /// let mut buffer = UnifiedBuffer::new(&0u64, 5).unwrap();
    /// let ptr = buffer.as_unified_ptr();
    /// let size = buffer.len();
    ///
    /// mem::forget(buffer);
    ///
    /// let buffer = unsafe { UnifiedBuffer::from_raw_parts(ptr, size) };
    /// ```
    pub unsafe fn from_raw_parts(ptr: UnifiedPointer<T>, capacity: usize) -> UnifiedBuffer<T> {
        UnifiedBuffer { buf: ptr, capacity }
    }

    /// Destroy a `UnifiedBuffer`, returning an error.
    ///
    /// Deallocating unified memory can return errors from previous asynchronous work. This function
    /// destroys the given buffer and returns the error and the un-destroyed buffer on failure.
    ///
    /// # Example
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let x = UnifiedBuffer::from_slice(&[10u32, 20, 30]).unwrap();
    /// match UnifiedBuffer::drop(x) {
    ///     Ok(()) => println!("Successfully destroyed"),
    ///     Err((e, buf)) => {
    ///         println!("Failed to destroy buffer: {:?}", e);
    ///         // Do something with buf
    ///     },
    /// }
    /// ```
    pub fn drop(mut uni_buf: UnifiedBuffer<T>) -> DropResult<UnifiedBuffer<T>> {
        if uni_buf.buf.is_null() {
            return Ok(());
        }

        if uni_buf.capacity > 0 && mem::size_of::<T>() > 0 {
            let capacity = uni_buf.capacity;
            let ptr = mem::replace(&mut uni_buf.buf, UnifiedPointer::null());
            unsafe {
                match cuda_free_unified(ptr) {
                    Ok(()) => {
                        mem::forget(uni_buf);
                        Ok(())
                    }
                    Err(e) => Err((e, UnifiedBuffer::from_raw_parts(ptr, capacity))),
                }
            }
        } else {
            Ok(())
        }
    }
}

impl<T: DeviceCopy> AsRef<[T]> for UnifiedBuffer<T> {
    fn as_ref(&self) -> &[T] {
        self
    }
}
impl<T: DeviceCopy> AsMut<[T]> for UnifiedBuffer<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}
impl<T: DeviceCopy> Deref for UnifiedBuffer<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe {
            let p = self.buf.as_raw();
            slice::from_raw_parts(p, self.capacity)
        }
    }
}
impl<T: DeviceCopy> DerefMut for UnifiedBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe {
            let ptr = self.buf.as_raw_mut();
            slice::from_raw_parts_mut(ptr, self.capacity)
        }
    }
}
impl<T: DeviceCopy> Drop for UnifiedBuffer<T> {
    fn drop(&mut self) {
        if self.buf.is_null() {
            return;
        }

        if self.capacity > 0 && mem::size_of::<T>() > 0 {
            unsafe {
                let ptr = mem::replace(&mut self.buf, UnifiedPointer::null());
                let _ = cuda_free_unified(ptr);
            }
        }
        self.capacity = 0;
    }
}

/// Functions for advising the driver about certain uses of unified memory. Such as advising the driver
/// to prefetch memory or to treat memory as read-mostly.
///
/// Note that none of the following APIs are required for correctness and/or safety, any use of the memory
/// will be valid no matter the use of the following functions. However, such uses may be very inefficient and/or
/// have increased memory consumption.
pub trait MemoryAdvise<T: DeviceCopy>: private::Sealed {
    fn as_slice(&self) -> &[T];

    // prefetch is documented as only being able to return Success, InvalidValue, or InvalidDevice.
    // None of which should ever happen because Streams, Devices, and unified buffers are always valid.
    // So we don't return a CUDA result.

    /// Advises the driver to enqueue an operation on the stream to prefetch the memory to the CPU.
    /// This will cause the driver to fetch the data back to the CPU as soon as the operation is reached
    /// on the stream.
    ///
    /// The CPU must have the attribute [`DeviceAttribute::ConcurrentManagedAccess`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _context = cust::quick_init().unwrap();
    /// # use cust::prelude::*;
    /// use cust::memory::*;
    /// let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    /// let x = UnifiedBuffer::from_slice(&[10u32, 20, 30])?;
    /// x.prefetch_to_host(&stream);
    /// stream.synchronize()?;
    /// # Ok(())
    /// # }
    /// ```
    fn prefetch_to_host(&self, stream: &Stream) -> CudaResult<()> {
        let slice = self.as_slice();
        let mem_size = std::mem::size_of_val(slice);

        unsafe {
            let id = -1; // -1 is CU_DEVICE_CPU
            driver_sys::cuMemPrefetchAsync(
                slice.as_ptr() as driver_sys::CUdeviceptr,
                mem_size,
                #[cfg(cuMemPrefetchAsync_v2)]
                driver_sys::CUmemLocation {
                    type_: driver_sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                    id,
                },
                #[cfg(not(cuMemPrefetchAsync_v2))]
                id,
                #[cfg(cuMemPrefetchAsync_v2)]
                0, // flags for future use, must be 0 as of CUDA 13.0
                stream.as_inner(),
            )
            .to_result()?;
        }
        Ok(())
    }

    /// Advises the driver to enqueue an operation on the stream to prefetch the memory to a certain GPU.
    /// This will cause the driver to fetch the data to the specified device as soon as the operation
    /// is reached on the stream.
    ///
    /// The device must have the attribute [`DeviceAttribute::ConcurrentManagedAccess`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _context = cust::quick_init().unwrap();
    /// # use cust::prelude::*;
    /// use cust::memory::*;
    /// let device = Device::get_device(0)?;
    /// let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    /// let x = UnifiedBuffer::from_slice(&[10u32, 20, 30])?;
    /// x.prefetch_to_device(&stream, &device);
    /// stream.synchronize()?;
    /// # Ok(())
    /// # }
    /// ```
    fn prefetch_to_device(&self, stream: &Stream, device: &Device) -> CudaResult<()> {
        let slice = self.as_slice();
        let mem_size = std::mem::size_of_val(slice);

        unsafe {
            let id = device.as_raw();
            driver_sys::cuMemPrefetchAsync(
                slice.as_ptr() as driver_sys::CUdeviceptr,
                mem_size,
                #[cfg(cuMemPrefetchAsync_v2)]
                driver_sys::CUmemLocation {
                    type_: driver_sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                    id,
                },
                #[cfg(not(cuMemPrefetchAsync_v2))]
                id,
                #[cfg(cuMemPrefetchAsync_v2)]
                0, // flags for future use, must be 0 as of CUDA 13.0
                stream.as_inner(),
            )
            .to_result()?;
        }
        Ok(())
    }

    /// Advises the driver that this memory range is mostly going to be read to, and occasionally written to.
    ///
    /// Any read accesses from any processor will create a read-only copy of at least the accessed pages in that processor's memory.
    ///
    /// Additionally, when prefetching, a read-only copy of the data will be created on the destination processor. If any processor
    /// attempts to write to this data, all copies of the corresponding page will be invalidated except for the one where the write occurred.
    ///
    /// For a page to be read-duplicated, the accessing processor must have a non-zero value for [`DeviceAttribute::ConcurrentManagedAccess`].
    /// Additionally, if a context is created on a device that does not have [`DeviceAttribute::ConcurrentManagedAccess`], then read-duplication
    /// will not occur until all such contexts are destroyed.
    fn advise_read_mostly(&self, read_mostly: bool) -> CudaResult<()> {
        let slice = self.as_slice();
        let mem_size = std::mem::size_of_val(slice);

        let advice = if read_mostly {
            driver_sys::CUmem_advise::CU_MEM_ADVISE_SET_READ_MOSTLY
        } else {
            driver_sys::CUmem_advise::CU_MEM_ADVISE_UNSET_READ_MOSTLY
        };

        unsafe {
            let id = 0;
            driver_sys::cuMemAdvise(
                slice.as_ptr() as driver_sys::CUdeviceptr,
                mem_size,
                advice,
                #[cfg(cuMemAdvise_v2)]
                driver_sys::CUmemLocation {
                    type_: driver_sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                    id,
                },
                #[cfg(not(cuMemAdvise_v2))]
                id,
            )
            .to_result()?;
        }
        Ok(())
    }

    /// Advises the driver as to the preferred device for this memory range. Either
    /// a device with `Some(device)` or the CPU with `None`. If the device is a GPU,
    /// it must have [`DeviceAttribute::ConcurrentManagedAccess`].
    ///
    /// Setting the preferred location does not cause the data to be migrated to that location immediately.
    /// It instead guides the migration policy when a fault occurs on the memory region. If the data is already in
    /// its preferred location and the faulting processor can establish a mapping without requiring the data to be migrated,
    /// then data migration will be avoided. On the other hand, if the data is not there or a mapping cannot be established,
    /// then it will be migrated to the accessing processor.
    ///
    /// Having a preferred location can override the page thrash detection and resolution logic in the unified memory driver.
    /// Normally if a page is detected to be constantly thrashing between processors, the page may eventually be pinned to
    /// host memory by the driver. But if the preferred location is set as device memory, then the page will continue
    /// to thrash indefinitely.
    ///
    /// If [`advise_read_mostly`](Self::advise_read_mostly) is set on this memory region or a subset of it, then the policies
    /// associated with that device will override the policies of this advice.
    ///
    /// This advice does not prevent the use of [`prefetch_to_host`](Self::prefetch_to_host) or [`prefetch_to_device`](Self::prefetch_to_device).
    fn preferred_location(&self, preferred_location: Option<Device>) -> CudaResult<()> {
        let slice = self.as_slice();
        let mem_size = std::mem::size_of_val(slice);

        unsafe {
            let id = preferred_location.map(|d| d.as_raw()).unwrap_or(-1); // -1 is CU_DEVICE_CPU
            driver_sys::cuMemAdvise(
                slice.as_ptr() as driver_sys::CUdeviceptr,
                mem_size,
                driver_sys::CUmem_advise::CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                #[cfg(cuMemAdvise_v2)]
                driver_sys::CUmemLocation {
                    type_: driver_sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                    id,
                },
                #[cfg(not(cuMemAdvise_v2))]
                id,
            )
            .to_result()?;
        }
        Ok(())
    }

    /// Undoes the most recent changes by [`preferred_location`](Self::preferred_location).
    fn unset_preferred_location(&self) -> CudaResult<()> {
        let slice = self.as_slice();
        let mem_size = std::mem::size_of_val(slice);

        unsafe {
            let id = 0;
            driver_sys::cuMemAdvise(
                slice.as_ptr() as driver_sys::CUdeviceptr,
                mem_size,
                driver_sys::CUmem_advise::CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION,
                #[cfg(cuMemAdvise_v2)]
                driver_sys::CUmemLocation {
                    type_: driver_sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                    id,
                },
                #[cfg(not(cuMemAdvise_v2))]
                id,
            )
            .to_result()?;
        }
        Ok(())
    }
}

impl<T: DeviceCopy> MemoryAdvise<T> for UnifiedBox<T> {
    fn as_slice(&self) -> &[T] {
        // SAFETY: unified pointers are valid on the CPU
        unsafe { std::slice::from_raw_parts(self.as_unified_ptr().as_raw(), 1) }
    }
}

impl<T: DeviceCopy> MemoryAdvise<T> for UnifiedBuffer<T> {
    fn as_slice(&self) -> &[T] {
        self
    }
}

mod private {
    pub trait Sealed {}
    impl<T: super::DeviceCopy> Sealed for super::UnifiedBox<T> {}
    impl<T: super::DeviceCopy> Sealed for super::UnifiedBuffer<T> {}
}

#[cfg(test)]
mod test_unified_box {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    struct ZeroSizedType;
    unsafe impl DeviceCopy for ZeroSizedType {}

    #[test]
    fn test_allocate_and_free() {
        let _context = crate::quick_init().unwrap();
        let mut x = UnifiedBox::new(5u64).unwrap();
        *x = 10;
        assert_eq!(10, *x);
        drop(x);
    }

    #[test]
    fn test_allocates_for_non_zst() {
        let _context = crate::quick_init().unwrap();
        let x = UnifiedBox::new(5u64).unwrap();
        let ptr = UnifiedBox::into_unified(x);
        assert!(!ptr.is_null());
        let _ = unsafe { UnifiedBox::from_unified(ptr) };
    }

    #[test]
    fn test_doesnt_allocate_for_zero_sized_type() {
        let _context = crate::quick_init().unwrap();
        let x = UnifiedBox::new(ZeroSizedType).unwrap();
        let ptr = UnifiedBox::into_unified(x);
        assert!(ptr.is_null());
        let _ = unsafe { UnifiedBox::from_unified(ptr) };
    }

    #[test]
    fn test_into_from_unified() {
        let _context = crate::quick_init().unwrap();
        let x = UnifiedBox::new(5u64).unwrap();
        let ptr = UnifiedBox::into_unified(x);
        let _ = unsafe { UnifiedBox::from_unified(ptr) };
    }

    #[test]
    fn test_equality() {
        let _context = crate::quick_init().unwrap();
        let x = UnifiedBox::new(5u64).unwrap();
        let y = UnifiedBox::new(5u64).unwrap();
        let z = UnifiedBox::new(0u64).unwrap();
        assert_eq!(x, y);
        assert!(x != z);
    }

    #[test]
    fn test_ordering() {
        let _context = crate::quick_init().unwrap();
        let x = UnifiedBox::new(1u64).unwrap();
        let y = UnifiedBox::new(2u64).unwrap();

        assert!(x < y);
    }
}
#[cfg(test)]
mod test_unified_buffer {
    use super::*;
    use std::mem;

    #[derive(Clone, Copy, Debug)]
    struct ZeroSizedType;
    unsafe impl DeviceCopy for ZeroSizedType {}

    #[test]
    fn test_new() {
        let _context = crate::quick_init().unwrap();
        let val = 0u64;
        let mut buffer = UnifiedBuffer::new(&val, 5).unwrap();
        buffer[0] = 1;
    }

    #[test]
    fn test_from_slice() {
        let _context = crate::quick_init().unwrap();
        let values = [0u64; 10];
        let mut buffer = UnifiedBuffer::from_slice(&values).unwrap();
        for i in buffer[0..3].iter_mut() {
            *i = 10;
        }
    }

    #[test]
    fn from_raw_parts() {
        let _context = crate::quick_init().unwrap();
        let mut buffer = UnifiedBuffer::new(&0u64, 5).unwrap();
        buffer[2] = 1;
        let ptr = buffer.as_unified_ptr();
        let len = buffer.len();
        mem::forget(buffer);

        let buffer = unsafe { UnifiedBuffer::from_raw_parts(ptr, len) };
        assert_eq!(&[0u64, 0, 1, 0, 0], buffer.as_slice());
        drop(buffer);
    }

    #[test]
    fn zero_length_buffer() {
        let _context = crate::quick_init().unwrap();
        let buffer = UnifiedBuffer::new(&0u64, 0).unwrap();
        drop(buffer);
    }

    #[test]
    fn zero_size_type() {
        let _context = crate::quick_init().unwrap();
        let buffer = UnifiedBuffer::new(&ZeroSizedType, 10).unwrap();
        drop(buffer);
    }

    #[test]
    fn overflows_usize() {
        let _context = crate::quick_init().unwrap();
        let err = UnifiedBuffer::new(&0u64, ::std::usize::MAX - 1).unwrap_err();
        assert_eq!(CudaError::InvalidMemoryAllocation, err);
    }

    #[test]
    fn test_unified_pointer_implements_traits_safely() {
        let _context = crate::quick_init().unwrap();
        let x = UnifiedBox::new(5u64).unwrap();
        let y = UnifiedBox::new(0u64).unwrap();

        // If the impls dereference the pointer, this should segfault.
        let _ = Ord::cmp(&x.as_unified_ptr(), &y.as_unified_ptr());
        let _ = PartialOrd::partial_cmp(&x.as_unified_ptr(), &y.as_unified_ptr());
        let _ = PartialEq::eq(&x.as_unified_ptr(), &y.as_unified_ptr());

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&x.as_unified_ptr(), &mut hasher);

        let _ = format!("{:?}", x.as_unified_ptr());
        let _ = format!("{:p}", x.as_unified_ptr());
    }
}
