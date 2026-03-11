use std::fmt::{self, Debug, Formatter};
use std::marker::PhantomData;
use std::ops::{
    Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};
use std::os::raw::c_void;
use std::ptr::{slice_from_raw_parts, slice_from_raw_parts_mut};
use std::slice;

#[cfg(feature = "bytemuck")]
use bytemuck::{Pod, Zeroable};
use cust_raw::driver_sys;

use crate::error::{CudaResult, ToResult};
use crate::memory::DevicePointer;
use crate::memory::device::AsyncCopyDestination;
use crate::memory::device::{CopyDestination, DeviceBuffer};
use crate::memory::{DeviceCopy, DeviceMemory};
use crate::stream::Stream;

/// Fixed-size device-side slice.
#[repr(transparent)]
pub struct DeviceSlice<T: DeviceCopy> {
    _phantom: PhantomData<T>,
    slice: [()],
}

unsafe impl<T: Send + DeviceCopy> Send for DeviceSlice<T> {}
unsafe impl<T: Sync + DeviceCopy> Sync for DeviceSlice<T> {}

impl<T: DeviceCopy> Debug for DeviceSlice<T> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        formatter
            .debug_struct("DeviceSlice")
            .field("ptr", &self.as_device_ptr().as_ptr())
            .field("len", &self.len())
            .finish()
    }
}

// This works by faking a regular slice out of the device raw-pointer and the length and transmuting
// I have no idea if this is safe or not. Probably not, though I can't imagine how the compiler
// could possibly know that the pointer is not de-referenceable. I'm banking that we get proper
// Dynamicaly-sized Types before the compiler authors break this assumption.
impl<T: DeviceCopy> DeviceSlice<T> {
    /// Returns the number of elements in the slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let a = DeviceBuffer::from_slice(&[1, 2, 3]).unwrap();
    /// assert_eq!(a.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// Returns `true` if the slice has a length of 0.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let a : DeviceBuffer<u64> = unsafe { DeviceBuffer::uninitialized(0).unwrap() };
    /// assert!(a.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return a raw device-pointer to the slice's buffer.
    ///
    /// The caller must ensure that the slice outlives the pointer this function returns, or else
    /// it will end up pointing to garbage. The caller must also ensure that the pointer is not
    /// dereferenced by the CPU.
    ///
    /// Examples:
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let a = DeviceBuffer::from_slice(&[1, 2, 3]).unwrap();
    /// println!("{:p}", a.as_slice().as_device_ptr());
    /// ```
    pub fn as_device_ptr(&self) -> DevicePointer<T> {
        DevicePointer::from_raw(self as *const _ as *const () as usize as u64)
    }

    pub fn as_host_vec(&self) -> CudaResult<Vec<T>> {
        let mut vec = Vec::with_capacity(self.len());
        // SAFETY: The slice points to uninitialized memory, but we only write to it. Once it is
        // written, all values are valid, so we can (and must) change the length of the vector.
        unsafe {
            self.copy_to(slice::from_raw_parts_mut(vec.as_mut_ptr(), self.len()))?;
            vec.set_len(self.len())
        }
        Ok(vec)
    }

    /* TODO (AL): keep these?
    /// Divides one DeviceSlice into two at a given index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding the index `mid` itself) and
    /// the second will contain all indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// # Panics
    ///
    /// Panics if `min > len`.
    ///
    /// Examples:
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let buf = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
    /// let (left, right) = buf.split_at(3);
    /// let mut left_host = [0u64, 0, 0];
    /// let mut right_host = [0u64, 0, 0];
    /// left.copy_to(&mut left_host).unwrap();
    /// right.copy_to(&mut right_host).unwrap();
    /// assert_eq!([0u64, 1, 2], left_host);
    /// assert_eq!([3u64, 4, 5], right_host);
    /// ```
    pub fn split_at(&self, mid: usize) -> (&DeviceSlice<T>, &DeviceSlice<T>) {
        let (left, right) = self.0.split_at(mid);
        unsafe {
            (
                DeviceSlice::from_slice(left),
                DeviceSlice::from_slice(right),
            )
        }
    }

    /// Divides one mutable DeviceSlice into two at a given index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding the index `mid` itself) and
    /// the second will contain all indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// # Panics
    ///
    /// Panics if `min > len`.
    ///
    /// Examples:
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let mut buf = DeviceBuffer::from_slice(&[0u64, 0, 0, 0, 0, 0]).unwrap();
    ///
    /// {
    ///     let (left, right) = buf.split_at_mut(3);
    ///     let left_host = [0u64, 1, 2];
    ///     let right_host = [3u64, 4, 5];
    ///     left.copy_from(&left_host).unwrap();
    ///     right.copy_from(&right_host).unwrap();
    /// }
    ///
    /// let mut host_full = [0u64; 6];
    /// buf.copy_to(&mut host_full).unwrap();
    /// assert_eq!([0u64, 1, 2, 3, 4, 5], host_full);
    /// ```
    pub fn split_at_mut(&mut self, mid: usize) -> (&mut DeviceSlice<T>, &mut DeviceSlice<T>) {
        let (left, right) = self.0.split_at_mut(mid);
        unsafe {
            (
                DeviceSlice::from_slice_mut(left),
                DeviceSlice::from_slice_mut(right),
            )
        }
    }
    */

    /// Forms a slice from a `DevicePointer` and a length.
    ///
    /// The `len` argument is the number of _elements_, not the number of bytes.
    ///
    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given pointer is valid for `len`
    /// elements, nor whether the lifetime inferred is a suitable lifetime for the returned slice.
    ///
    /// # Caveat
    ///
    /// The lifetime for the returned slice is inferred from its usage. To prevent accidental misuse,
    /// it's suggested to tie the lifetime to whatever source lifetime is safe in the context, such
    /// as by providing a helper function taking the lifetime of a host value for the slice or
    /// by explicit annotation.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let mut x = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
    /// // Manually slice the buffer (this is not recommended!)
    /// let ptr = unsafe { x.as_device_ptr().offset(1) };
    /// let slice = unsafe { DeviceSlice::from_raw_parts(ptr, 2) };
    /// let mut host_buf = [0u64, 0];
    /// slice.copy_to(&mut host_buf).unwrap();
    /// assert_eq!([1u64, 2], host_buf);
    /// ```
    #[allow(clippy::needless_pass_by_value)]
    pub unsafe fn from_raw_parts<'a>(ptr: DevicePointer<T>, len: usize) -> &'a DeviceSlice<T> {
        unsafe { &*(slice_from_raw_parts(ptr.as_ptr(), len) as *const DeviceSlice<T>) }
    }

    /// Performs the same functionality as `from_raw_parts`, except that a
    /// mutable slice is returned.
    ///
    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given pointer is valid for `len`
    /// elements, nor whether the lifetime inferred is a suitable lifetime for the returned slice.
    ///
    /// This function is unsafe as there is no guarantee that the given pointer is valid for `len`
    /// elements, not whether the lifetime inferred is a suitable lifetime for the returned slice,
    /// as well as not being able to provide a non-aliasing guarantee of the returned
    /// mutable slice. `data` must be non-null and aligned even for zero-length
    /// slices as with `from_raw_parts`.
    ///
    /// See the documentation of `from_raw_parts` for more details.
    pub unsafe fn from_raw_parts_mut<'a>(
        ptr: DevicePointer<T>,
        len: usize,
    ) -> &'a mut DeviceSlice<T> {
        unsafe { &mut *(slice_from_raw_parts_mut(ptr.as_mut_ptr(), len) as *mut DeviceSlice<T>) }
    }
}

#[cfg(feature = "bytemuck")]
impl<T: DeviceCopy + Pod> DeviceSlice<T> {
    // NOTE(RDambrosio016): async memsets kind of blur the line between safe and unsafe, the only
    // unsafe thing i can imagine could happen is someone allocs a buffer, launches an async memset, then
    // tries to read back the value. However, it is unclear whether this is actually UB. Even if the
    // reads get jumbled into the writes, well, we know this type is Pod, so any byte value is fine for it.
    // So currently these functions are unsafe, but we may want to reevaluate this in the future.

    /// Sets the memory range of this buffer to contiguous `8-bit` values of `value`.
    ///
    /// In total it will set `sizeof<T> * len` values of `value` contiguously.
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    pub fn set_8(&mut self, value: u8) -> CudaResult<()> {
        if self.size_in_bytes() == 0 {
            return Ok(());
        }

        // SAFETY: We know T can hold any value because it is `Pod`, and
        // sub-byte alignment isn't a thing so we know the alignment is right.
        unsafe {
            driver_sys::cuMemsetD8(self.as_raw_ptr(), value, self.size_in_bytes()).to_result()
        }
    }

    /// Sets the memory range of this buffer to contiguous `8-bit` values of `value` asynchronously.
    ///
    /// In total it will set `sizeof<T> * len` values of `value` contiguously.
    ///
    /// # Safety
    ///
    /// This operation is async so it does not complete immediately, it uses stream-ordering semantics.
    /// Therefore you should not read/write from/to the memory range until the operation is complete.
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    pub unsafe fn set_8_async(&mut self, value: u8, stream: &Stream) -> CudaResult<()> {
        if self.size_in_bytes() == 0 {
            return Ok(());
        }

        unsafe {
            driver_sys::cuMemsetD8Async(
                self.as_raw_ptr(),
                value,
                self.size_in_bytes(),
                stream.as_inner(),
            )
        }
        .to_result()
    }

    /// Sets the memory range of this buffer to contiguous `16-bit` values of `value`.
    ///
    /// In total it will set `(sizeof<T> / 2) * len` values of `value` contiguously.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `self.ptr % 2 != 0` (the pointer is not aligned to at least 2 bytes).
    /// - `(size_of::<T>() * self.len) % 2 != 0` (the data size is not a multiple of 2 bytes)
    #[track_caller]
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    pub fn set_16(&mut self, value: u16) -> CudaResult<()> {
        let data_len = self.size_in_bytes();
        assert_eq!(
            data_len % 2,
            0,
            "Buffer length is not a multiple of 2 bytes!"
        );
        assert_eq!(
            self.as_raw_ptr() % 2,
            0,
            "Buffer pointer is not aligned to at least 2 bytes!"
        );
        unsafe { driver_sys::cuMemsetD16(self.as_raw_ptr(), value, data_len / 2).to_result() }
    }

    /// Sets the memory range of this buffer to contiguous `16-bit` values of `value` asynchronously.
    ///
    /// In total it will set `(sizeof<T> / 2) * len` values of `value` contiguously.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `self.ptr % 2 != 0` (the pointer is not aligned to at least 2 bytes).
    /// - `(size_of::<T>() * self.len) % 2 != 0` (the data size is not a multiple of 2 bytes)
    ///
    /// # Safety
    ///
    /// This operation is async so it does not complete immediately, it uses stream-ordering semantics.
    /// Therefore you should not read/write from/to the memory range until the operation is complete.
    #[track_caller]
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    pub unsafe fn set_16_async(&mut self, value: u16, stream: &Stream) -> CudaResult<()> {
        let data_len = self.size_in_bytes();
        assert_eq!(
            data_len % 2,
            0,
            "Buffer length is not a multiple of 2 bytes!"
        );
        assert_eq!(
            self.as_raw_ptr() % 2,
            0,
            "Buffer pointer is not aligned to at least 2 bytes!"
        );
        unsafe {
            driver_sys::cuMemsetD16Async(self.as_raw_ptr(), value, data_len / 2, stream.as_inner())
        }
        .to_result()
    }

    /// Sets the memory range of this buffer to contiguous `32-bit` values of `value`.
    ///
    /// In total it will set `(sizeof<T> / 4) * len` values of `value` contiguously.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `self.ptr % 4 != 0` (the pointer is not aligned to at least 4 bytes).
    /// - `(size_of::<T>() * self.len) % 4 != 0` (the data size is not a multiple of 4 bytes)
    #[track_caller]
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    pub fn set_32(&mut self, value: u32) -> CudaResult<()> {
        let data_len = self.size_in_bytes();
        assert_eq!(
            data_len % 4,
            0,
            "Buffer length is not a multiple of 4 bytes!"
        );
        assert_eq!(
            self.as_raw_ptr() % 4,
            0,
            "Buffer pointer is not aligned to at least 4 bytes!"
        );
        unsafe { driver_sys::cuMemsetD32(self.as_raw_ptr(), value, data_len / 4).to_result() }
    }

    /// Sets the memory range of this buffer to contiguous `32-bit` values of `value` asynchronously.
    ///
    /// In total it will set `(sizeof<T> / 4) * len` values of `value` contiguously.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `self.ptr % 4 != 0` (the pointer is not aligned to at least 4 bytes).
    /// - `(size_of::<T>() * self.len) % 4 != 0` (the data size is not a multiple of 4 bytes)
    ///
    /// # Safety
    ///
    /// This operation is async so it does not complete immediately, it uses stream-ordering semantics.
    /// Therefore you should not read/write from/to the memory range until the operation is complete.
    #[track_caller]
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    pub unsafe fn set_32_async(&mut self, value: u32, stream: &Stream) -> CudaResult<()> {
        let data_len = self.size_in_bytes();
        assert_eq!(
            data_len % 4,
            0,
            "Buffer length is not a multiple of 4 bytes!"
        );
        assert_eq!(
            self.as_raw_ptr() % 4,
            0,
            "Buffer pointer is not aligned to at least 4 bytes!"
        );
        unsafe {
            driver_sys::cuMemsetD32Async(self.as_raw_ptr(), value, data_len / 4, stream.as_inner())
        }
        .to_result()
    }
}

#[cfg(feature = "bytemuck")]
impl<T: DeviceCopy + Zeroable> DeviceSlice<T> {
    /// Sets this slice's data to zero.
    pub fn set_zero(&mut self) -> CudaResult<()> {
        // SAFETY: this is fine because Zeroable guarantees a zero byte-pattern is safe
        // for this type. And a slice of bytes can represent any type.
        let erased = unsafe {
            DeviceSlice::from_raw_parts_mut(self.as_device_ptr().cast::<u8>(), self.size_in_bytes())
        };
        erased.set_8(0)
    }

    /// Sets this slice's data to zero asynchronously.
    ///
    /// # Safety
    ///
    /// This operation is async so it does not complete immediately, it uses stream-ordering semantics.
    /// Therefore you should not read/write from/to the memory range until the operation is complete.
    pub unsafe fn set_zero_async(&mut self, stream: &Stream) -> CudaResult<()> {
        // SAFETY: this is fine because Zeroable guarantees a zero byte-pattern is safe
        // for this type. And a slice of bytes can represent any type.
        unsafe {
            let erased = DeviceSlice::from_raw_parts_mut(
                self.as_device_ptr().cast::<u8>(),
                self.size_in_bytes(),
            );
            erased.set_8_async(0, stream)
        }
    }
}

pub trait DeviceSliceIndex<T: DeviceCopy> {
    /// Indexes into this slice without checking if it is in-bounds.
    ///
    /// # Safety
    ///
    /// The range must be in-bounds of the slice.
    unsafe fn get_unchecked(self, slice: &DeviceSlice<T>) -> &DeviceSlice<T>;
    fn index(self, slice: &DeviceSlice<T>) -> &DeviceSlice<T>;

    /// Indexes into this slice without checking if it is in-bounds.
    ///
    /// # Safety
    ///
    /// The range must be in-bounds of the slice.
    unsafe fn get_unchecked_mut(self, slice: &mut DeviceSlice<T>) -> &mut DeviceSlice<T>;
    fn index_mut(self, slice: &mut DeviceSlice<T>) -> &mut DeviceSlice<T>;
}

#[inline(never)]
#[cold]
#[track_caller]
fn slice_start_index_len_fail(index: usize, len: usize) -> ! {
    panic!("range start index {index} out of range for slice of length {len}");
}

#[inline(never)]
#[cold]
#[track_caller]
fn slice_end_index_len_fail(index: usize, len: usize) -> ! {
    panic!("range end index {index} out of range for slice of length {len}");
}

#[inline(never)]
#[cold]
#[track_caller]
fn slice_index_order_fail(index: usize, end: usize) -> ! {
    panic!("slice index starts at {index} but ends at {end}");
}

#[inline(never)]
#[cold]
#[track_caller]
fn slice_end_index_overflow_fail() -> ! {
    panic!("attempted to index slice up to maximum usize");
}

impl<T: DeviceCopy> DeviceSliceIndex<T> for usize {
    unsafe fn get_unchecked(self, slice: &DeviceSlice<T>) -> &DeviceSlice<T> {
        unsafe { (self..self + 1).get_unchecked(slice) }
    }
    fn index(self, slice: &DeviceSlice<T>) -> &DeviceSlice<T> {
        slice.index(self..self + 1)
    }

    unsafe fn get_unchecked_mut(self, slice: &mut DeviceSlice<T>) -> &mut DeviceSlice<T> {
        unsafe { (self..self + 1).get_unchecked_mut(slice) }
    }
    fn index_mut(self, slice: &mut DeviceSlice<T>) -> &mut DeviceSlice<T> {
        slice.index_mut(self..self + 1)
    }
}

impl<T: DeviceCopy> DeviceSliceIndex<T> for Range<usize> {
    unsafe fn get_unchecked(self, slice: &DeviceSlice<T>) -> &DeviceSlice<T> {
        unsafe {
            DeviceSlice::from_raw_parts(
                slice.as_device_ptr().add(self.start),
                self.end - self.start,
            )
        }
    }
    fn index(self, slice: &DeviceSlice<T>) -> &DeviceSlice<T> {
        if self.start > self.end {
            slice_index_order_fail(self.start, self.end);
        } else if self.end > slice.len() {
            slice_end_index_len_fail(self.end, slice.len());
        }
        // SAFETY: `self` is checked to be valid and in bounds above.
        unsafe { self.get_unchecked(slice) }
    }

    unsafe fn get_unchecked_mut(self, slice: &mut DeviceSlice<T>) -> &mut DeviceSlice<T> {
        unsafe {
            DeviceSlice::from_raw_parts_mut(
                slice.as_device_ptr().add(self.start),
                self.end - self.start,
            )
        }
    }
    fn index_mut(self, slice: &mut DeviceSlice<T>) -> &mut DeviceSlice<T> {
        if self.start > self.end {
            slice_index_order_fail(self.start, self.end);
        } else if self.end > slice.len() {
            slice_end_index_len_fail(self.end, slice.len());
        }
        // SAFETY: `self` is checked to be valid and in bounds above.
        unsafe { self.get_unchecked_mut(slice) }
    }
}

impl<T: DeviceCopy> DeviceSliceIndex<T> for RangeTo<usize> {
    unsafe fn get_unchecked(self, slice: &DeviceSlice<T>) -> &DeviceSlice<T> {
        unsafe { (0..self.end).get_unchecked(slice) }
    }
    fn index(self, slice: &DeviceSlice<T>) -> &DeviceSlice<T> {
        (0..self.end).index(slice)
    }

    unsafe fn get_unchecked_mut(self, slice: &mut DeviceSlice<T>) -> &mut DeviceSlice<T> {
        unsafe { (0..self.end).get_unchecked_mut(slice) }
    }
    fn index_mut(self, slice: &mut DeviceSlice<T>) -> &mut DeviceSlice<T> {
        (0..self.end).index_mut(slice)
    }
}

impl<T: DeviceCopy> DeviceSliceIndex<T> for RangeFrom<usize> {
    unsafe fn get_unchecked(self, slice: &DeviceSlice<T>) -> &DeviceSlice<T> {
        unsafe { (self.start..slice.len()).get_unchecked(slice) }
    }
    fn index(self, slice: &DeviceSlice<T>) -> &DeviceSlice<T> {
        if self.start > slice.len() {
            slice_start_index_len_fail(self.start, slice.len());
        }
        // SAFETY: `self` is checked to be valid and in bounds above.
        unsafe { self.get_unchecked(slice) }
    }

    unsafe fn get_unchecked_mut(self, slice: &mut DeviceSlice<T>) -> &mut DeviceSlice<T> {
        unsafe { (self.start..slice.len()).get_unchecked_mut(slice) }
    }
    fn index_mut(self, slice: &mut DeviceSlice<T>) -> &mut DeviceSlice<T> {
        if self.start > slice.len() {
            slice_start_index_len_fail(self.start, slice.len());
        }
        // SAFETY: `self` is checked to be valid and in bounds above.
        unsafe { self.get_unchecked_mut(slice) }
    }
}

impl<T: DeviceCopy> DeviceSliceIndex<T> for RangeFull {
    unsafe fn get_unchecked(self, slice: &DeviceSlice<T>) -> &DeviceSlice<T> {
        slice
    }
    fn index(self, slice: &DeviceSlice<T>) -> &DeviceSlice<T> {
        slice
    }

    unsafe fn get_unchecked_mut(self, slice: &mut DeviceSlice<T>) -> &mut DeviceSlice<T> {
        slice
    }
    fn index_mut(self, slice: &mut DeviceSlice<T>) -> &mut DeviceSlice<T> {
        slice
    }
}

fn into_slice_range(range: RangeInclusive<usize>) -> Range<usize> {
    let exclusive_end = range.end() + 1;
    let start = if range.is_empty() {
        exclusive_end
    } else {
        *range.start()
    };
    start..exclusive_end
}

impl<T: DeviceCopy> DeviceSliceIndex<T> for RangeInclusive<usize> {
    unsafe fn get_unchecked(self, slice: &DeviceSlice<T>) -> &DeviceSlice<T> {
        unsafe { into_slice_range(self).get_unchecked(slice) }
    }
    fn index(self, slice: &DeviceSlice<T>) -> &DeviceSlice<T> {
        if *self.end() == usize::MAX {
            slice_end_index_overflow_fail();
        }
        into_slice_range(self).index(slice)
    }

    unsafe fn get_unchecked_mut(self, slice: &mut DeviceSlice<T>) -> &mut DeviceSlice<T> {
        unsafe { into_slice_range(self).get_unchecked_mut(slice) }
    }
    fn index_mut(self, slice: &mut DeviceSlice<T>) -> &mut DeviceSlice<T> {
        if *self.end() == usize::MAX {
            slice_end_index_overflow_fail();
        }
        into_slice_range(self).index_mut(slice)
    }
}

impl<T: DeviceCopy> DeviceSliceIndex<T> for RangeToInclusive<usize> {
    unsafe fn get_unchecked(self, slice: &DeviceSlice<T>) -> &DeviceSlice<T> {
        unsafe { (0..=self.end).get_unchecked(slice) }
    }
    fn index(self, slice: &DeviceSlice<T>) -> &DeviceSlice<T> {
        (0..=self.end).index(slice)
    }

    unsafe fn get_unchecked_mut(self, slice: &mut DeviceSlice<T>) -> &mut DeviceSlice<T> {
        unsafe { (0..=self.end).get_unchecked_mut(slice) }
    }
    fn index_mut(self, slice: &mut DeviceSlice<T>) -> &mut DeviceSlice<T> {
        (0..=self.end).index_mut(slice)
    }
}

impl<T: DeviceCopy, Idx: DeviceSliceIndex<T>> Index<Idx> for DeviceSlice<T> {
    type Output = DeviceSlice<T>;

    fn index(&self, index: Idx) -> &DeviceSlice<T> {
        index.index(self)
    }
}

impl<T: DeviceCopy, Idx: DeviceSliceIndex<T>> IndexMut<Idx> for DeviceSlice<T> {
    fn index_mut(&mut self, index: Idx) -> &mut DeviceSlice<T> {
        index.index_mut(self)
    }
}

impl<T: DeviceCopy> crate::private::Sealed for DeviceSlice<T> {}
impl<T: DeviceCopy, I: AsRef<[T]> + AsMut<[T]> + ?Sized> CopyDestination<I> for DeviceSlice<T> {
    fn copy_from(&mut self, val: &I) -> CudaResult<()> {
        let val = val.as_ref();
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = self.size_in_bytes();
        if size != 0 {
            unsafe {
                driver_sys::cuMemcpyHtoD(self.as_raw_ptr(), val.as_ptr() as *const c_void, size)
                    .to_result()?
            }
        }
        Ok(())
    }

    fn copy_to(&self, val: &mut I) -> CudaResult<()> {
        let val = val.as_mut();
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = self.size_in_bytes();
        if size != 0 {
            unsafe {
                driver_sys::cuMemcpyDtoH(val.as_mut_ptr() as *mut c_void, self.as_raw_ptr(), size)
                    .to_result()?
            }
        }
        Ok(())
    }
}
impl<T: DeviceCopy> CopyDestination<DeviceSlice<T>> for DeviceSlice<T> {
    fn copy_from(&mut self, val: &DeviceSlice<T>) -> CudaResult<()> {
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = self.size_in_bytes();
        if size != 0 {
            unsafe {
                driver_sys::cuMemcpyDtoD(self.as_raw_ptr(), val.as_raw_ptr(), size).to_result()?
            }
        }
        Ok(())
    }

    fn copy_to(&self, val: &mut DeviceSlice<T>) -> CudaResult<()> {
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = self.size_in_bytes();
        if size != 0 {
            unsafe {
                driver_sys::cuMemcpyDtoD(val.as_raw_ptr(), self.as_raw_ptr(), size).to_result()?
            }
        }
        Ok(())
    }
}
impl<T: DeviceCopy> CopyDestination<DeviceBuffer<T>> for DeviceSlice<T> {
    fn copy_from(&mut self, val: &DeviceBuffer<T>) -> CudaResult<()> {
        self.copy_from(val as &DeviceSlice<T>)
    }

    fn copy_to(&self, val: &mut DeviceBuffer<T>) -> CudaResult<()> {
        self.copy_to(val as &mut DeviceSlice<T>)
    }
}
impl<T: DeviceCopy, I: AsRef<[T]> + AsMut<[T]> + ?Sized> AsyncCopyDestination<I>
    for DeviceSlice<T>
{
    unsafe fn async_copy_from(&mut self, val: &I, stream: &Stream) -> CudaResult<()> {
        let val = val.as_ref();
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = self.size_in_bytes();
        if size != 0 {
            unsafe {
                driver_sys::cuMemcpyHtoDAsync(
                    self.as_raw_ptr(),
                    val.as_ptr() as *const c_void,
                    size,
                    stream.as_inner(),
                )
            }
            .to_result()?
        }
        Ok(())
    }

    unsafe fn async_copy_to(&self, val: &mut I, stream: &Stream) -> CudaResult<()> {
        let val = val.as_mut();
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = self.size_in_bytes();
        if size != 0 {
            unsafe {
                driver_sys::cuMemcpyDtoHAsync(
                    val.as_mut_ptr() as *mut c_void,
                    self.as_raw_ptr(),
                    size,
                    stream.as_inner(),
                )
            }
            .to_result()?
        }
        Ok(())
    }
}
impl<T: DeviceCopy> AsyncCopyDestination<DeviceSlice<T>> for DeviceSlice<T> {
    unsafe fn async_copy_from(&mut self, val: &DeviceSlice<T>, stream: &Stream) -> CudaResult<()> {
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = self.size_in_bytes();
        if size != 0 {
            unsafe {
                driver_sys::cuMemcpyDtoDAsync(
                    self.as_raw_ptr(),
                    val.as_raw_ptr(),
                    size,
                    stream.as_inner(),
                )
            }
            .to_result()?
        }
        Ok(())
    }

    unsafe fn async_copy_to(&self, val: &mut DeviceSlice<T>, stream: &Stream) -> CudaResult<()> {
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = self.size_in_bytes();
        if size != 0 {
            unsafe {
                driver_sys::cuMemcpyDtoDAsync(
                    val.as_raw_ptr(),
                    self.as_raw_ptr(),
                    size,
                    stream.as_inner(),
                )
            }
            .to_result()?
        }
        Ok(())
    }
}
impl<T: DeviceCopy> AsyncCopyDestination<DeviceBuffer<T>> for DeviceSlice<T> {
    unsafe fn async_copy_from(&mut self, val: &DeviceBuffer<T>, stream: &Stream) -> CudaResult<()> {
        unsafe { self.async_copy_from(val as &DeviceSlice<T>, stream) }
    }

    unsafe fn async_copy_to(&self, val: &mut DeviceBuffer<T>, stream: &Stream) -> CudaResult<()> {
        unsafe { self.async_copy_to(val as &mut DeviceSlice<T>, stream) }
    }
}
