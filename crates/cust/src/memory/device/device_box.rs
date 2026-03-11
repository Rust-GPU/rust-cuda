use std::fmt::{self, Pointer};
use std::mem::{self, ManuallyDrop, MaybeUninit};
use std::os::raw::c_void;

use cust_raw::driver_sys;

use crate::error::{CudaResult, DropResult, ToResult};
use crate::memory::DevicePointer;
use crate::memory::device::AsyncCopyDestination;
use crate::memory::device::CopyDestination;
use crate::memory::malloc::{cuda_free, cuda_malloc};
use crate::memory::{DeviceCopy, cuda_free_async, cuda_malloc_async};
use crate::stream::Stream;

/// A pointer type for heap-allocation in CUDA device memory.
///
/// See the [`module-level documentation`](../memory/index.html) for more information on device memory.
#[derive(Debug)]
pub struct DeviceBox<T: DeviceCopy> {
    pub(crate) ptr: DevicePointer<T>,
}

unsafe impl<T: Send + DeviceCopy> Send for DeviceBox<T> {}
unsafe impl<T: Sync + DeviceCopy> Sync for DeviceBox<T> {}

impl<T: DeviceCopy> DeviceBox<T> {
    /// Allocate device memory and place val into it.
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Errors
    ///
    /// If a CUDA error occurs, return the error.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let five = DeviceBox::new(&5).unwrap();
    /// ```
    pub fn new(val: &T) -> CudaResult<Self> {
        let mut dev_box = unsafe { DeviceBox::uninitialized()? };
        dev_box.copy_from(val)?;
        Ok(dev_box)
    }

    /// Allocates device memory asynchronously and asynchronously copies `val` into it.
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// If the memory behind `val` is not page-locked (pinned), a staging buffer will be
    /// allocated using a worker thread. If you are going to be making many asynchronous
    /// copies, it is generally a good idea to keep the data as a
    /// [`crate::memory::LockedBuffer`] or [`crate::memory::LockedBox`]. This will
    /// ensure the driver does not have to allocate a staging buffer on its own.
    ///
    /// However, don't keep all of your data as page-locked, doing so might slow down
    /// the OS because it is unable to page out that memory to disk.
    ///
    /// # Safety
    ///
    /// This method enqueues two operations on the stream: An async allocation and an
    /// async memcpy. Because of this, you must ensure that:
    ///   - The memory is not used in any way before it is actually allocated on the
    ///     stream. You can ensure this happens by synchronizing the stream explicitly
    ///     or using events.
    ///   - `val` is still valid when the memory copy actually takes place.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::{memory::*, stream::*};
    /// let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    /// let mut host_val = 0;
    /// unsafe {
    ///     let mut allocated = DeviceBox::new_async(&5u8, &stream)?;
    ///     allocated.async_copy_to(&mut host_val, &stream)?;
    ///     allocated.drop_async(&stream)?;
    /// }
    /// // ensure all async ops are done before trying to access the value
    /// stream.synchronize()?;
    /// assert_eq!(host_val, 5);
    /// # Ok(())
    /// # }
    pub unsafe fn new_async(val: &T, stream: &Stream) -> CudaResult<Self> {
        unsafe {
            let mut dev_box = DeviceBox::uninitialized_async(stream)?;
            dev_box.async_copy_from(val, stream)?;
            Ok(dev_box)
        }
    }

    /// Enqueues an operation to free the memory backed by this [`DeviceBox`] on a
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
    /// let mut host_val = 0;
    /// unsafe {
    ///     let mut allocated = DeviceBox::new_async(&5u8, &stream)?;
    ///     allocated.async_copy_to(&mut host_val, &stream)?;
    ///     allocated.drop_async(&stream)?;
    /// }
    /// // ensure all async ops are done before trying to access the value
    /// stream.synchronize()?;
    /// assert_eq!(host_val, 5);
    /// # Ok(())
    /// # }
    pub fn drop_async(self, stream: &Stream) -> CudaResult<()> {
        if self.ptr.is_null() {
            return Ok(());
        }
        // make sure we dont run the normal destructor, otherwise a double drop will happen
        let me = ManuallyDrop::new(self);
        // SAFETY: we consume the box so its not possible to use the box past its drop point unless
        // you keep around a pointer, but in that case, we cannot guarantee safety.
        unsafe { cuda_free_async(stream, me.ptr) }
    }

    /// Read the data back from the GPU into host memory.
    pub fn as_host_value(&self) -> CudaResult<T> {
        let mut val = MaybeUninit::uninit();
        // SAFETY: We do not read from the uninitialized reference.
        unsafe {
            self.copy_to(val.assume_init_mut())?;
            Ok(val.assume_init())
        }
    }
}

#[cfg(feature = "bytemuck")]
impl<T: DeviceCopy + bytemuck::Zeroable> DeviceBox<T> {
    /// Allocate device memory and fill it with zeroes (`0u8`).
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let mut zero = DeviceBox::zeroed().unwrap();
    /// let mut value = 5u64;
    /// zero.copy_to(&mut value).unwrap();
    /// assert_eq!(0, value);
    /// ```
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    pub fn zeroed() -> CudaResult<Self> {
        unsafe {
            let new_box = DeviceBox::uninitialized()?;
            if mem::size_of::<T>() != 0 {
                driver_sys::cuMemsetD8(new_box.as_device_ptr().as_raw(), 0, mem::size_of::<T>())
                    .to_result()?;
            }
            Ok(new_box)
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
    /// let mut value = 5u64;
    /// unsafe {
    ///     let mut zero = DeviceBox::zeroed_async(&stream)?;
    ///     zero.async_copy_to(&mut value, &stream)?;
    ///     zero.drop_async(&stream)?;
    /// }
    /// stream.synchronize()?;
    /// assert_eq!(value, 0);
    /// # Ok(())
    /// # }
    /// ```
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    pub unsafe fn zeroed_async(stream: &Stream) -> CudaResult<Self> {
        let new_box = unsafe { DeviceBox::uninitialized_async(stream)? };
        if mem::size_of::<T>() != 0 {
            unsafe {
                driver_sys::cuMemsetD8Async(
                    new_box.as_device_ptr().as_raw(),
                    0,
                    mem::size_of::<T>(),
                    stream.as_inner(),
                )
            }
            .to_result()?;
        }
        Ok(new_box)
    }
}

impl<T: DeviceCopy> DeviceBox<T> {
    /// Allocate device memory, but do not initialize it.
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Safety
    ///
    /// Since the backing memory is not initialized, this function is not safe. The caller must
    /// ensure that the backing memory is set to a valid value before it is read, else undefined
    /// behavior may occur.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let mut five = unsafe { DeviceBox::uninitialized().unwrap() };
    /// five.copy_from(&5u64).unwrap();
    /// ```
    pub unsafe fn uninitialized() -> CudaResult<Self> {
        if mem::size_of::<T>() == 0 {
            Ok(DeviceBox {
                ptr: DevicePointer::null(),
            })
        } else {
            let ptr = unsafe { cuda_malloc(1)? };
            Ok(DeviceBox { ptr })
        }
    }

    /// Allocates device memory asynchronously on a stream, without initializing it.
    ///
    /// This doesn't actually allocate if `T` is zero sized.
    ///
    /// # Safety
    ///
    /// The allocated memory retains all of the unsafety of [`DeviceBox::uninitialized`], with
    /// the additional consideration that the memory cannot be used until it is actually allocated
    /// on the stream. This means proper stream ordering semantics must be followed, such as
    /// only enqueing kernel launches that use the memory AFTER the allocation call.
    ///
    /// You can synchronize the stream to ensure the memory allocation operation is complete.
    pub unsafe fn uninitialized_async(stream: &Stream) -> CudaResult<Self> {
        if mem::size_of::<T>() == 0 {
            Ok(DeviceBox {
                ptr: DevicePointer::null(),
            })
        } else {
            let ptr = unsafe { cuda_malloc_async(stream, 1)? };
            Ok(DeviceBox { ptr })
        }
    }

    /// Constructs a DeviceBox from a raw pointer.
    ///
    /// After calling this function, the raw pointer and the memory it points to is owned by the
    /// DeviceBox. The DeviceBox destructor will free the allocated memory, but will not call the destructor
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
    /// let x = DeviceBox::new(&5).unwrap();
    /// let ptr = DeviceBox::into_device(x).as_raw();
    /// let x: DeviceBox<i32> = unsafe { DeviceBox::from_raw(ptr) };
    /// ```
    pub unsafe fn from_raw(ptr: driver_sys::CUdeviceptr) -> Self {
        DeviceBox {
            ptr: DevicePointer::from_raw(ptr),
        }
    }

    /// Constructs a DeviceBox from a DevicePointer.
    ///
    /// After calling this function, the pointer and the memory it points to is owned by the
    /// DeviceBox. The DeviceBox destructor will free the allocated memory, but will not call the destructor
    /// of `T`. This function may accept any pointer produced by the `cuMemAllocManaged` CUDA API
    /// call, such as one taken from `DeviceBox::into_device`.
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
    /// let x = DeviceBox::new(&5).unwrap();
    /// let ptr = DeviceBox::into_device(x);
    /// let x = unsafe { DeviceBox::from_device(ptr) };
    /// ```
    pub unsafe fn from_device(ptr: DevicePointer<T>) -> Self {
        DeviceBox { ptr }
    }

    /// Consumes the DeviceBox, returning the wrapped DevicePointer.
    ///
    /// After calling this function, the caller is responsible for the memory previously managed by
    /// the DeviceBox. In particular, the caller should properly destroy T and deallocate the memory.
    /// The easiest way to do so is to create a new DeviceBox using the `DeviceBox::from_device` function.
    ///
    /// Note: This is an associated function, which means that you have to all it as
    /// `DeviceBox::into_device(b)` instead of `b.into_device()` This is so that there is no conflict with
    /// a method on the inner type.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let x = DeviceBox::new(&5).unwrap();
    /// let ptr = DeviceBox::into_device(x);
    /// # unsafe { DeviceBox::from_device(ptr) };
    /// ```
    #[allow(clippy::wrong_self_convention)]
    pub fn into_device(mut b: DeviceBox<T>) -> DevicePointer<T> {
        let ptr = mem::replace(&mut b.ptr, DevicePointer::null());
        mem::forget(b);
        ptr
    }

    /// Returns the contained device pointer without consuming the box.
    ///
    /// This is useful for passing the box to a kernel launch.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let mut x = DeviceBox::new(&5).unwrap();
    /// let ptr = x.as_device_ptr();
    /// println!("{:p}", ptr);
    /// ```
    pub fn as_device_ptr(&self) -> DevicePointer<T> {
        self.ptr
    }

    /// Destroy a `DeviceBox`, returning an error.
    ///
    /// Deallocating device memory can return errors from previous asynchronous work. This function
    /// destroys the given box and returns the error and the un-destroyed box on failure.
    ///
    /// # Example
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let x = DeviceBox::new(&5).unwrap();
    /// match DeviceBox::drop(x) {
    ///     Ok(()) => println!("Successfully destroyed"),
    ///     Err((e, dev_box)) => {
    ///         println!("Failed to destroy box: {:?}", e);
    ///         // Do something with dev_box
    ///     },
    /// }
    /// ```
    pub fn drop(mut dev_box: DeviceBox<T>) -> DropResult<DeviceBox<T>> {
        if dev_box.ptr.is_null() {
            return Ok(());
        }

        let ptr = mem::replace(&mut dev_box.ptr, DevicePointer::null());
        unsafe {
            match cuda_free(ptr) {
                Ok(()) => {
                    mem::forget(dev_box);
                    Ok(())
                }
                Err(e) => Err((e, DeviceBox { ptr })),
            }
        }
    }
}
impl<T: DeviceCopy> Drop for DeviceBox<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }

        let ptr = mem::replace(&mut self.ptr, DevicePointer::null());
        unsafe {
            let _ = cuda_free(ptr);
        }
    }
}

impl<T: DeviceCopy> Pointer for DeviceBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ptr = self.ptr.as_raw() as *const c_void;
        fmt::Pointer::fmt(&ptr, f)
    }
}

impl<T: DeviceCopy> crate::private::Sealed for DeviceBox<T> {}
impl<T: DeviceCopy> CopyDestination<T> for DeviceBox<T> {
    fn copy_from(&mut self, val: &T) -> CudaResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                driver_sys::cuMemcpyHtoD(self.ptr.as_raw(), val as *const T as *const c_void, size)
                    .to_result()?
            }
        }
        Ok(())
    }

    fn copy_to(&self, val: &mut T) -> CudaResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                driver_sys::cuMemcpyDtoH(val as *const T as *mut c_void, self.ptr.as_raw(), size)
                    .to_result()?
            }
        }
        Ok(())
    }
}
impl<T: DeviceCopy> CopyDestination<DeviceBox<T>> for DeviceBox<T> {
    fn copy_from(&mut self, val: &DeviceBox<T>) -> CudaResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                driver_sys::cuMemcpyDtoD(self.ptr.as_raw(), val.ptr.as_raw(), size).to_result()?
            }
        }
        Ok(())
    }

    fn copy_to(&self, val: &mut DeviceBox<T>) -> CudaResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                driver_sys::cuMemcpyDtoD(val.ptr.as_raw(), self.ptr.as_raw(), size).to_result()?
            }
        }
        Ok(())
    }
}
impl<T: DeviceCopy> AsyncCopyDestination<T> for DeviceBox<T> {
    unsafe fn async_copy_from(&mut self, val: &T, stream: &Stream) -> CudaResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                driver_sys::cuMemcpyHtoDAsync(
                    self.ptr.as_raw(),
                    val as *const _ as *const c_void,
                    size,
                    stream.as_inner(),
                )
            }
            .to_result()?
        }
        Ok(())
    }

    unsafe fn async_copy_to(&self, val: &mut T, stream: &Stream) -> CudaResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                driver_sys::cuMemcpyDtoHAsync(
                    val as *mut _ as *mut c_void,
                    self.ptr.as_raw(),
                    size,
                    stream.as_inner(),
                )
            }
            .to_result()?
        }
        Ok(())
    }
}
impl<T: DeviceCopy> AsyncCopyDestination<DeviceBox<T>> for DeviceBox<T> {
    unsafe fn async_copy_from(&mut self, val: &DeviceBox<T>, stream: &Stream) -> CudaResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                driver_sys::cuMemcpyDtoDAsync(
                    self.ptr.as_raw(),
                    val.ptr.as_raw(),
                    size,
                    stream.as_inner(),
                )
            }
            .to_result()?
        }
        Ok(())
    }

    unsafe fn async_copy_to(&self, val: &mut DeviceBox<T>, stream: &Stream) -> CudaResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                driver_sys::cuMemcpyDtoDAsync(
                    val.ptr.as_raw(),
                    self.ptr.as_raw(),
                    size,
                    stream.as_inner(),
                )
            }
            .to_result()?
        }
        Ok(())
    }
}

#[cfg(test)]
mod test_device_box {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    struct ZeroSizedType;
    unsafe impl DeviceCopy for ZeroSizedType {}

    #[test]
    fn test_allocate_and_free_device_box() {
        let _context = crate::quick_init().unwrap();
        let x = DeviceBox::new(&5u64).unwrap();
        drop(x);
    }

    #[test]
    fn test_device_box_allocates_for_non_zst() {
        let _context = crate::quick_init().unwrap();
        let x = DeviceBox::new(&5u64).unwrap();
        let ptr = DeviceBox::into_device(x);
        assert!(!ptr.is_null());
        let _ = unsafe { DeviceBox::from_device(ptr) };
    }

    #[test]
    fn test_device_box_doesnt_allocate_for_zero_sized_type() {
        let _context = crate::quick_init().unwrap();
        let x = DeviceBox::new(&ZeroSizedType).unwrap();
        let ptr = DeviceBox::into_device(x);
        assert!(ptr.is_null());
        let _ = unsafe { DeviceBox::from_device(ptr) };
    }

    #[test]
    fn test_into_from_device() {
        let _context = crate::quick_init().unwrap();
        let x = DeviceBox::new(&5u64).unwrap();
        let ptr = DeviceBox::into_device(x);
        let _ = unsafe { DeviceBox::from_device(ptr) };
    }

    #[test]
    fn test_copy_host_to_device() {
        let _context = crate::quick_init().unwrap();
        let y = 5u64;
        let mut x = DeviceBox::new(&0u64).unwrap();
        x.copy_from(&y).unwrap();
        let mut z = 10u64;
        x.copy_to(&mut z).unwrap();
        assert_eq!(y, z);
    }

    #[test]
    fn test_copy_device_to_host() {
        let _context = crate::quick_init().unwrap();
        let x = DeviceBox::new(&5u64).unwrap();
        let mut y = 0u64;
        x.copy_to(&mut y).unwrap();
        assert_eq!(5, y);
    }

    #[test]
    fn test_copy_device_to_device() {
        let _context = crate::quick_init().unwrap();
        let x = DeviceBox::new(&5u64).unwrap();
        let mut y = DeviceBox::new(&0u64).unwrap();
        let mut z = DeviceBox::new(&0u64).unwrap();
        x.copy_to(&mut y).unwrap();
        z.copy_from(&y).unwrap();

        let mut h = 0u64;
        z.copy_to(&mut h).unwrap();
        assert_eq!(5, h);
    }

    #[test]
    fn test_device_pointer_implements_traits_safely() {
        let _context = crate::quick_init().unwrap();
        let x = DeviceBox::new(&5u64).unwrap();
        let y = DeviceBox::new(&0u64).unwrap();

        // If the impls dereference the pointer, this should segfault.
        let _ = Ord::cmp(&x.as_device_ptr(), &y.as_device_ptr());
        let _ = PartialOrd::partial_cmp(&x.as_device_ptr(), &y.as_device_ptr());
        let _ = PartialEq::eq(&x.as_device_ptr(), &y.as_device_ptr());

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&x.as_device_ptr(), &mut hasher);

        let _ = format!("{:?}", x.as_device_ptr());
        let _ = format!("{:p}", x.as_device_ptr());
    }
}
