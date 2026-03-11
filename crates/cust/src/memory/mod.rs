//! Access to CUDA's memory allocation and transfer functions.
//!
//! The memory module provides a safe wrapper around CUDA's memory allocation and transfer functions.
//! This includes access to device memory, unified memory, and page-locked host memory.
//!
//! # Device Memory
//!
//! Device memory is just what it sounds like - memory allocated on the device. Device memory
//! cannot be accessed from the host directly, but data can be copied to and from the device.
//! cust exposes device memory through the [`DeviceBox`](struct.DeviceBox.html) and
//! [`DeviceBuffer`](struct.DeviceBuffer.html) structures. Pointers to device memory are
//! represented by [`DevicePointer`](struct.DevicePointer.html), while slices in device memory are
//! represented by [`DeviceSlice`](struct.DeviceSlice.html).
//!
//! # Unified Memory
//!
//! Unified memory is a memory allocation which can be read from and written to by both the host
//! and the device. When the host (or device) attempts to access a page of unified memory, it is
//! seamlessly transferred from host RAM to device RAM or vice versa. The programmer may also
//! choose to explicitly prefetch data to one side or another. cust exposes unified memory through the
//! [`UnifiedBox`](struct.UnifiedBox.html) and [`UnifiedBuffer`](struct.UnifiedBuffer.html)
//! structures, and pointers to unified memory are represented by
//! [`UnifiedPointer`](struct.UnifiedPointer.html). Since unified memory is accessible to the host,
//! slices in unified memory are represented by normal Rust slices.
//!
//! Unified memory is generally easier to use than device memory, but there are drawbacks. It is
//! possible to allocate more memory than is available on the card, and this can result in very slow
//! paging behavior. Additionally, it can require careful use of prefetching to achieve optimum
//! performance. Finally, unified memory is not supported on some older systems.
//!
//! ## Warning
//!
//! ⚠️ **On certain systems/OSes/GPUs, accessing Unified memory from the CPU while the GPU is currently
//! using it (e.g. before stream synchronization) will cause a Page Error/Segfault. For this reason,
//! we strongly suggest to treat unified memory as exclusive to the GPU while it is being used by a kernel** ⚠️
//!
//! This is not considered Undefined Behavior because the behavior is always "either works, or yields a page error/segfault",
//! doing this will never corrupt memory or cause other undesireable behavior.
//!
//! # Page-locked Host Memory
//!
//! Page-locked memory is memory that the operating system has locked into physical RAM, and will
//! not page out to disk. When copying data from the process' memory space to the device, the CUDA
//! driver needs to first copy the data to a page-locked region of host memory, then initiate a DMA
//! transfer to copy the data to the device itself. Likewise, when transferring from device to host,
//! the driver copies the data into page-locked host memory then into the normal memory space. This
//! extra copy can be eliminated if the data is loaded or generated directly into page-locked
//! memory. cust exposes page-locked memory through the
//! [`LockedBuffer`](struct.LockedBuffer.html) struct.
//!
//! For example, if the programmer needs to read an array of bytes from disk and transfer it to the
//! device, it would be best to create a `LockedBuffer`, load the bytes directly into the
//! `LockedBuffer`, and then copy them to a `DeviceBuffer`. If the bytes are in a `Vec<u8>`, there
//! would be no advantage to using a `LockedBuffer`.
//!
//! However, since the OS cannot page out page-locked memory, excessive use can slow down the entire
//! system (including other processes) as physical RAM is tied up.  Therefore, page-locked memory
//! should be used sparingly.
//!
//! # FFI Information
//!
//! The internal representations of `DevicePointer<T>` and `UnifiedPointer<T>` are guaranteed to be
//! the same as `*mut T` and they can be safely passed through an FFI boundary to code expecting
//! raw pointers (though keep in mind that device-only pointers cannot be dereferenced on the CPU).
//! This is important when launching kernels written in C.
//!
//! As with regular Rust, all other types (eg. `DeviceBuffer` or `UnifiedBox`) are not FFI-safe.
//! Their internal representations are not guaranteed to be anything in particular, and are not
//! guaranteed to be the same in different versions of cust. If you need to pass them through
//! an FFI boundary, you must convert them to FFI-safe primitives yourself. For example, with
//! `UnifiedBuffer`, use the `as_unified_ptr()` and `len()` functions to get the primitives, and
//! `mem::forget()` the Buffer so that it isn't dropped. Again, as with regular Rust, the caller is
//! responsible for reconstructing the `UnifiedBuffer` using `from_raw_parts()` and dropping it to
//! ensure that the memory allocation is safely cleaned up.

pub mod array;

mod device;
mod locked;
mod malloc;
mod pointer;
mod unified;

pub use self::device::*;
pub use self::locked::*;
pub use self::malloc::*;
pub use self::pointer::*;
pub use self::unified::*;

use crate::error::*;

#[cfg(feature = "bytemuck")]
pub use bytemuck;

pub use crate::DeviceCopy;
pub use cust_core::_hidden::DeviceCopy;

use std::ffi::c_void;

use cust_raw::driver_sys;

/// A trait describing a generic buffer that can be accessed from the GPU. This could be either a [`UnifiedBuffer`]
/// or a regular [`DeviceBuffer`].
#[allow(clippy::len_without_is_empty)]
pub trait GpuBuffer<T: DeviceCopy>: private::Sealed {
    fn as_device_ptr(&self) -> DevicePointer<T>;
    fn len(&self) -> usize;
}

impl<T: DeviceCopy> GpuBuffer<T> for DeviceBuffer<T> {
    fn as_device_ptr(&self) -> DevicePointer<T> {
        self.as_slice().as_device_ptr()
    }

    fn len(&self) -> usize {
        (**self).len()
    }
}

impl<T: DeviceCopy> GpuBuffer<T> for UnifiedBuffer<T> {
    fn as_device_ptr(&self) -> DevicePointer<T> {
        DevicePointer::from_raw(self.as_ptr() as u64)
    }

    fn len(&self) -> usize {
        (**self).len()
    }
}

/// A trait describing a generic pointer that can be accessed from the GPU. This could be either a [`UnifiedBox`]
/// or a regular [`DeviceBox`].
pub trait GpuBox<T: DeviceCopy>: private::Sealed {
    fn as_device_ptr(&self) -> DevicePointer<T>;
}

impl<T: DeviceCopy> GpuBox<T> for DeviceBox<T> {
    fn as_device_ptr(&self) -> DevicePointer<T> {
        self.ptr
    }
}

impl<T: DeviceCopy> GpuBox<T> for UnifiedBox<T> {
    fn as_device_ptr(&self) -> DevicePointer<T> {
        DevicePointer::from_raw(self.ptr.as_raw() as u64)
    }
}

/// A trait describing a region of memory on the device with a base pointer and
/// a size, used to be generic over DeviceBox, DeviceBuffer, DeviceVariable etc.
pub trait DeviceMemory {
    /// Get the raw cuda device pointer
    fn as_raw_ptr(&self) -> driver_sys::CUdeviceptr;

    /// Get the size of the memory region in bytes
    fn size_in_bytes(&self) -> usize;
}

impl<T: DeviceCopy> DeviceMemory for DeviceBox<T> {
    fn as_raw_ptr(&self) -> driver_sys::CUdeviceptr {
        self.as_device_ptr().as_raw()
    }

    fn size_in_bytes(&self) -> usize {
        std::mem::size_of::<T>()
    }
}

impl<T: DeviceCopy> DeviceMemory for DeviceVariable<T> {
    fn as_raw_ptr(&self) -> driver_sys::CUdeviceptr {
        self.as_device_ptr().as_raw()
    }

    fn size_in_bytes(&self) -> usize {
        std::mem::size_of::<T>()
    }
}

impl<T: DeviceCopy> DeviceMemory for DeviceBuffer<T> {
    fn as_raw_ptr(&self) -> driver_sys::CUdeviceptr {
        self.as_device_ptr().as_raw()
    }

    fn size_in_bytes(&self) -> usize {
        std::mem::size_of::<T>() * self.len()
    }
}

impl<T: DeviceCopy> DeviceMemory for DeviceSlice<T> {
    fn as_raw_ptr(&self) -> driver_sys::CUdeviceptr {
        self.as_device_ptr().as_raw()
    }

    fn size_in_bytes(&self) -> usize {
        std::mem::size_of::<T>() * self.len()
    }
}

mod private {
    use super::{DeviceBox, DeviceBuffer, DeviceCopy, UnifiedBox, UnifiedBuffer};

    pub trait Sealed {}
    impl<T: DeviceCopy> Sealed for UnifiedBuffer<T> {}
    impl<T: DeviceCopy> Sealed for DeviceBuffer<T> {}
    impl<T: DeviceCopy> Sealed for UnifiedBox<T> {}
    impl<T: DeviceCopy> Sealed for DeviceBox<T> {}
}

/// Simple wrapper over cuMemcpyHtoD
#[allow(clippy::missing_safety_doc)]
pub unsafe fn memcpy_htod(
    d_ptr: driver_sys::CUdeviceptr,
    src_ptr: *const c_void,
    size: usize,
) -> CudaResult<()> {
    unsafe { driver_sys::cuMemcpyHtoD(d_ptr, src_ptr, size).to_result()? };
    Ok(())
}

/// Simple wrapper over cuMemcpyDtoH
#[allow(clippy::missing_safety_doc)]
pub unsafe fn memcpy_dtoh(
    d_ptr: *mut c_void,
    src_ptr: driver_sys::CUdeviceptr,
    size: usize,
) -> CudaResult<()> {
    unsafe { driver_sys::cuMemcpyDtoH(d_ptr, src_ptr, size).to_result()? };
    Ok(())
}

/// Similar to `cudaMemcpy2D` with `HostToDevice` copy type.
///
/// `dpitch`/`spitch` is bytes between the start of two rows.
/// `width` is the number of *elements* (not bytes) in a row.
/// `height` is the total number of rows (not bytes).
///
/// # Examples
///
/// ```
/// # let _context = cust::quick_init().unwrap();
/// # fn foo() -> Result<(), cust::error::CudaError> {
/// use cust::memory::*;
/// unsafe {
///     // Allocate space for a 3x3 matrix of f32s
///     let (device_buffer, pitch) = cuda_malloc_pitched::<f32>(3, 3)?;
///
///     let src_array: [f32; 9] = [
///         1.0, 2.0, 3.0,
///         4.0, 5.0, 6.0,
///         7.0, 8.0, 9.0];
///
///     memcpy_2d_htod(
///         device_buffer,
///         pitch,
///         src_array.as_slice().as_ptr(),
///         3*std::mem::size_of::<f32>(),
///         3,
///         3
///     )?;
///
///     let mut dst_array = [0.0f32; 9];
///
///     memcpy_2d_dtoh(
///         dst_array.as_mut_slice().as_mut_ptr(),
///         3*std::mem::size_of::<f32>(),
///         device_buffer,
///         pitch,
///         3,
///         3
///     )?;
///
///     assert_eq!(dst_array, src_array);
///     cuda_free(device_buffer)?;
/// }
/// # Ok(())
/// # }
/// # foo().unwrap();
/// ```
#[allow(clippy::missing_safety_doc)]
pub unsafe fn memcpy_2d_htod<T: DeviceCopy>(
    dst: DevicePointer<T>,
    dpitch: usize,
    src: *const T,
    spitch: usize,
    width: usize,
    height: usize,
) -> CudaResult<()> {
    use cust_raw::driver_sys::CUmemorytype;

    let width_in_bytes = width
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(CudaError::InvalidMemoryAllocation)?;

    let pcopy = driver_sys::CUDA_MEMCPY2D_st {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
        srcHost: src as *const c_void,
        srcDevice: 0,                                             // Ignored
        srcArray: std::ptr::null_mut::<driver_sys::CUarray_st>(), // Ignored
        srcPitch: spitch,
        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut::<c_void>(), // Ignored
        dstDevice: dst.as_raw(),
        dstArray: std::ptr::null_mut::<driver_sys::CUarray_st>(), // Ignored
        dstPitch: dpitch,
        WidthInBytes: width_in_bytes,
        Height: height,
    };

    unsafe { driver_sys::cuMemcpy2D(&pcopy).to_result()? };
    Ok(())
}

/// Similar to `cudaMemcpy2D` with `DeviceToHost` copy type.
///
/// `dpitch`/`spitch` is bytes between the start of two rows.
/// `width` is the number of *elements* (not bytes) in a row.
/// `height` is the total number of rows (not bytes).
///
/// # Examples
///
/// ```
/// # let _context = cust::quick_init().unwrap();
/// # fn foo() -> Result<(), cust::error::CudaError> {
/// use cust::memory::*;
/// unsafe {
///     // Allocate space for a 3x3 matrix of f32s
///     let (device_buffer, pitch) = cuda_malloc_pitched::<f32>(3, 3)?;
///
///     let src_array: [f32; 9] = [
///         1.0, 2.0, 3.0,
///         4.0, 5.0, 6.0,
///         7.0, 8.0, 9.0];
///
///     memcpy_2d_htod(
///         device_buffer,
///         pitch,
///         src_array.as_slice().as_ptr(),
///         3*std::mem::size_of::<f32>(),
///         3,
///         3
///     )?;
///
///     let mut dst_array = [0.0f32; 9];
///
///     memcpy_2d_dtoh(
///         dst_array.as_mut_slice().as_mut_ptr(),
///         3*std::mem::size_of::<f32>(),
///         device_buffer,
///         pitch,
///         3,
///         3
///     )?;
///
///     assert_eq!(dst_array, src_array);
///     cuda_free(device_buffer)?;
/// }
/// # Ok(())
/// # }
/// # foo().unwrap();
/// ```
#[allow(clippy::missing_safety_doc)]
pub unsafe fn memcpy_2d_dtoh<T: DeviceCopy>(
    dst: *mut T,
    dpitch: usize,
    src: DevicePointer<T>,
    spitch: usize,
    width: usize,
    height: usize,
) -> CudaResult<()> {
    use cust_raw::driver_sys::CUmemorytype;

    let width_in_bytes = width
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(CudaError::InvalidMemoryAllocation)?;

    let pcopy = driver_sys::CUDA_MEMCPY2D_st {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
        srcHost: std::ptr::null_mut::<c_void>(), // Ignored
        srcDevice: src.as_raw(),
        srcArray: std::ptr::null_mut::<driver_sys::CUarray_st>(), // Ignored
        srcPitch: spitch,
        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
        dstHost: dst as *mut c_void,
        dstDevice: 0,                                             // Ignored
        dstArray: std::ptr::null_mut::<driver_sys::CUarray_st>(), // Ignored
        dstPitch: dpitch,
        WidthInBytes: width_in_bytes,
        Height: height,
    };

    unsafe { driver_sys::cuMemcpy2D(&pcopy).to_result()? };
    Ok(())
}

/// Get the current free and total memory.
///
/// Returns in `.1` the total amount of memory available to the the current context.
/// Returns in `.0` the amount of memory on the device that is free according to
/// the OS. CUDA is not guaranteed to be able to allocate all of the memory that
/// the OS reports as free.
pub fn mem_get_info() -> CudaResult<(usize, usize)> {
    let mut mem_free = 0;
    let mut mem_total = 0;
    unsafe {
        driver_sys::cuMemGetInfo(&mut mem_free, &mut mem_total).to_result()?;
    }
    Ok((mem_free, mem_total))
}
