use std::{
    mem::{self, ManuallyDrop},
    ops::{Deref, DerefMut},
    ptr::null_mut,
};

use crate::{
    error::CudaResult,
    memory::{DeviceCopy, cuda_free_locked, cuda_malloc_locked},
};

/// Page-locked box in host memory.
///
/// # Page-locked memory
///
/// Modern OSes have the ability to page out memory pages to
/// storage areas such as the disk to reduce RAM usage. However, this
/// causes issues with certain things such as asynchronous memory
/// copies because the memory may get paged out to disk while
/// the driver is staging a DMA transfer. For this reason, CUDA
/// provides functions for allocating page-locked memory that is pinned
/// to RAM.
///
/// This memory is generally only used as a staging buffer for memory
/// copies to the GPU. Significant use of pinned memory can reduce
/// overall performance as the OS is forced to page out more memory
/// to disk.
#[derive(Debug)]
pub struct LockedBox<T: DeviceCopy> {
    pub(crate) ptr: *mut T,
}

unsafe impl<T: Send + DeviceCopy> Send for LockedBox<T> {}
unsafe impl<T: Sync + DeviceCopy> Sync for LockedBox<T> {}

impl<T: DeviceCopy> LockedBox<T> {
    /// Creates an uninitialized [`LockedBox`]. The contents must
    /// not be read until the box is written to.
    ///
    /// # Safety
    ///
    /// The memory inside of the box is uninitialized, it must not be
    /// read in any way.
    pub unsafe fn uninitialized() -> CudaResult<Self> {
        if mem::size_of::<T>() == 0 {
            Ok(LockedBox { ptr: null_mut() })
        } else {
            let ptr = unsafe { cuda_malloc_locked(1)? };
            Ok(LockedBox { ptr })
        }
    }

    /// Allocates page-locked memory and copies `val` into it.
    ///
    /// # Example
    ///
    /// ```
    /// use cust::memory::LockedBox;
    ///
    ///
    pub fn new(val: &T) -> CudaResult<Self> {
        unsafe {
            let mut uninit = Self::uninitialized()?;
            *uninit = *val;
            Ok(uninit)
        }
    }

    /// Consumes this box, returning the underlying allocation pointer. The
    /// backing memory will not be freed, it is up to the user to free it with either
    /// [`cuda_free_locked`] or [`LockedBox::from_raw`].
    pub fn into_raw(self) -> *mut T {
        ManuallyDrop::new(self).ptr
    }

    /// Creates a [`LockedBox`] from a raw pointer, taking ownership of the data backed behind
    /// the pointer.
    ///
    /// # Safety
    ///
    /// The pointer must be a valid pinned memory allocation as obtained from [`LockedBox::into_raw`] or [`cuda_malloc_locked`].
    pub unsafe fn from_raw(ptr: *mut T) -> Self {
        Self { ptr }
    }

    /// Returns the underlying pointer backing this [`LockedBox`] without consuming the box,
    /// meaning that the box will still free the memory once out of scope. It is up to the user
    /// to make sure they don't free the pointer.
    pub fn as_raw(&self) -> *mut T {
        self.ptr
    }
}

impl<T: DeviceCopy> Drop for LockedBox<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }

        unsafe {
            let _ = cuda_free_locked(self.ptr);
        }
    }
}

impl<T: DeviceCopy> Deref for LockedBox<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.ptr }
    }
}

impl<T: DeviceCopy> DerefMut for LockedBox<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.ptr }
    }
}
