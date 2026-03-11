use crate::error::*;
use crate::memory::DeviceCopy;
use crate::memory::malloc::{cuda_free_locked, cuda_malloc_locked};
use std::mem;
use std::ops;
use std::ptr;
use std::slice;

/// Fixed-size host-side buffer in page-locked memory.
///
/// See the [`module-level documentation`](../memory/index.html) for more details on page-locked
/// memory.
#[derive(Debug)]
pub struct LockedBuffer<T: DeviceCopy> {
    buf: *mut T,
    capacity: usize,
}

unsafe impl<T: Send + DeviceCopy> Send for LockedBuffer<T> {}
unsafe impl<T: Sync + DeviceCopy> Sync for LockedBuffer<T> {}

impl<T: DeviceCopy + Clone> LockedBuffer<T> {
    /// Allocate a new page-locked buffer large enough to hold `size` `T`'s and initialized with
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
    /// let mut buffer = LockedBuffer::new(&0u64, 5).unwrap();
    /// buffer[0] = 1;
    /// ```
    pub fn new(value: &T, size: usize) -> CudaResult<Self> {
        unsafe {
            let mut uninit = LockedBuffer::uninitialized(size)?;
            for x in 0..size {
                *uninit.get_unchecked_mut(x) = *value;
            }
            Ok(uninit)
        }
    }

    /// Allocate a new page-locked buffer of the same size as `slice`, initialized with a clone of
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
    /// let mut buffer = LockedBuffer::from_slice(&values).unwrap();
    /// buffer[0] = 1;
    /// ```
    pub fn from_slice(slice: &[T]) -> CudaResult<Self> {
        unsafe {
            let mut uninit = LockedBuffer::uninitialized(slice.len())?;
            for (i, x) in slice.iter().enumerate() {
                *uninit.get_unchecked_mut(i) = *x;
            }
            Ok(uninit)
        }
    }
}
impl<T: DeviceCopy> LockedBuffer<T> {
    /// Allocate a new page-locked buffer large enough to hold `size` `T`'s, but without
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
    /// let mut buffer = unsafe { LockedBuffer::uninitialized(5).unwrap() };
    /// for i in buffer.iter_mut() {
    ///     *i = 0u64;
    /// }
    /// ```
    pub unsafe fn uninitialized(size: usize) -> CudaResult<Self> {
        let ptr: *mut T = if size > 0 && mem::size_of::<T>() > 0 {
            unsafe { cuda_malloc_locked(size)? }
        } else {
            ptr::NonNull::dangling().as_ptr()
        };
        Ok(LockedBuffer {
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
    /// let buffer = LockedBuffer::new(&0u64, 5).unwrap();
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
    /// let mut buffer = LockedBuffer::new(&0u64, 5).unwrap();
    /// for i in buffer.as_mut_slice() {
    ///     *i = 12u64;
    /// }
    /// ```
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }

    /// Creates a `LockedBuffer<T>` directly from the raw components of another locked
    /// buffer.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren't checked:
    ///
    ///   * `ptr` needs to have been previously allocated via `LockedBuffer` or
    ///     [`cuda_malloc_locked`](fn.cuda_malloc_locked.html).
    ///   * `ptr`'s `T` needs to have the same size and alignment as it was allocated
    ///     with.
    ///   * `capacity` needs to be the capacity that the pointer was allocated with.
    ///
    /// Violating these may cause problems like corrupting the CUDA driver's internal
    /// data structures.
    ///
    /// The ownership of `ptr` is effectively transferred to the `LockedBuffer<T>` which
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
    /// let mut buffer = LockedBuffer::new(&0u64, 5).unwrap();
    /// let ptr = buffer.as_mut_ptr();
    /// let size = buffer.len();
    ///
    /// mem::forget(buffer);
    ///
    /// let buffer = unsafe { LockedBuffer::from_raw_parts(ptr, size) };
    /// ```
    pub unsafe fn from_raw_parts(ptr: *mut T, size: usize) -> LockedBuffer<T> {
        LockedBuffer {
            buf: ptr,
            capacity: size,
        }
    }

    /// Destroy a `LockedBuffer`, returning an error.
    ///
    /// Deallocating page-locked memory can return errors from previous asynchronous work. This function
    /// destroys the given buffer and returns the error and the un-destroyed buffer on failure.
    ///
    /// # Example
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let x = LockedBuffer::new(&0u64, 5).unwrap();
    /// match LockedBuffer::drop(x) {
    ///     Ok(()) => println!("Successfully destroyed"),
    ///     Err((e, buf)) => {
    ///         println!("Failed to destroy buffer: {:?}", e);
    ///         // Do something with buf
    ///     },
    /// }
    /// ```
    pub fn drop(mut buf: LockedBuffer<T>) -> DropResult<LockedBuffer<T>> {
        if buf.buf.is_null() {
            return Ok(());
        }

        if buf.capacity > 0 && mem::size_of::<T>() > 0 {
            let capacity = buf.capacity;
            let ptr = mem::replace(&mut buf.buf, ptr::null_mut());
            unsafe {
                match cuda_free_locked(ptr) {
                    Ok(()) => {
                        mem::forget(buf);
                        Ok(())
                    }
                    Err(e) => Err((e, LockedBuffer::from_raw_parts(ptr, capacity))),
                }
            }
        } else {
            Ok(())
        }
    }
}

impl<T: DeviceCopy> AsRef<[T]> for LockedBuffer<T> {
    fn as_ref(&self) -> &[T] {
        self
    }
}
impl<T: DeviceCopy> AsMut<[T]> for LockedBuffer<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}
impl<T: DeviceCopy> ops::Deref for LockedBuffer<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe {
            let p = self.buf;
            slice::from_raw_parts(p, self.capacity)
        }
    }
}
impl<T: DeviceCopy> ops::DerefMut for LockedBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe {
            let ptr = self.buf;
            slice::from_raw_parts_mut(ptr, self.capacity)
        }
    }
}
impl<T: DeviceCopy> Drop for LockedBuffer<T> {
    fn drop(&mut self) {
        if self.buf.is_null() {
            return;
        }

        if self.capacity > 0 && mem::size_of::<T>() > 0 {
            unsafe {
                let _ = cuda_free_locked(self.buf);
            }
        }
        self.capacity = 0;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::mem;

    #[derive(Clone, Copy, Debug)]
    struct ZeroSizedType;
    unsafe impl DeviceCopy for ZeroSizedType {}

    #[test]
    fn test_new() {
        let _context = crate::quick_init().unwrap();
        let val = 0u64;
        let mut buffer = LockedBuffer::new(&val, 5).unwrap();
        buffer[0] = 1;
    }

    #[test]
    fn test_from_slice() {
        let _context = crate::quick_init().unwrap();
        let values = [0u64; 10];
        let mut buffer = LockedBuffer::from_slice(&values).unwrap();
        for i in buffer[0..3].iter_mut() {
            *i = 10;
        }
    }

    #[test]
    fn from_raw_parts() {
        let _context = crate::quick_init().unwrap();
        let mut buffer = LockedBuffer::new(&0u64, 5).unwrap();
        buffer[2] = 1;
        let ptr = buffer.as_mut_ptr();
        let len = buffer.len();
        mem::forget(buffer);

        let buffer = unsafe { LockedBuffer::from_raw_parts(ptr, len) };
        assert_eq!(&[0u64, 0, 1, 0, 0], buffer.as_slice());
        drop(buffer);
    }

    #[test]
    fn zero_length_buffer() {
        let _context = crate::quick_init().unwrap();
        let buffer = LockedBuffer::new(&0u64, 0).unwrap();
        drop(buffer);
    }

    #[test]
    fn zero_size_type() {
        let _context = crate::quick_init().unwrap();
        let buffer = LockedBuffer::new(&ZeroSizedType, 10).unwrap();
        drop(buffer);
    }

    #[test]
    fn overflows_usize() {
        let _context = crate::quick_init().unwrap();
        let err = LockedBuffer::new(&0u64, ::std::usize::MAX - 1).unwrap_err();
        assert_eq!(CudaError::InvalidMemoryAllocation, err);
    }

    #[test]
    fn test_allocate_correct_size() {
        let _context = crate::quick_init().unwrap();

        // Placeholder - read out available system memory here
        let allocation_size = 1;
        unsafe {
            // Test if allocation fails with an out-of-memory error
            let _buffer = LockedBuffer::<u64>::uninitialized(allocation_size).unwrap();
        }
    }
}
