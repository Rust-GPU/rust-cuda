use crate::error::CudaResult;
use crate::memory::DeviceCopy;
use crate::memory::device::CopyDestination;
use crate::memory::{DeviceBox, DevicePointer};
use std::ops::{Deref, DerefMut};

/// Wrapper around a variable on the host and a [`DeviceBox`] holding the
/// variable on the device, allowing for easy synchronization and storage.
#[derive(Debug)]
pub struct DeviceVariable<T: DeviceCopy> {
    mem: DeviceBox<T>,
    var: T,
}

impl<T: DeviceCopy> DeviceVariable<T> {
    /// Create a new `DeviceVariable` wrapping `var`.
    ///
    /// Allocates storage on the device and copies `var` to the device.
    pub fn new(var: T) -> CudaResult<Self> {
        let mem = DeviceBox::new(&var)?;
        Ok(Self { mem, var })
    }

    /// Copy the host copy of the variable to the device
    pub fn copy_htod(&mut self) -> CudaResult<()> {
        self.mem.copy_from(&self.var)
    }

    /// Copy the device copy of the variable to the host
    pub fn copy_dtoh(&mut self) -> CudaResult<()> {
        self.mem.copy_to(&mut self.var)
    }

    pub fn as_device_ptr(&self) -> DevicePointer<T> {
        self.mem.as_device_ptr()
    }
}

impl<T: DeviceCopy> Deref for DeviceVariable<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.var
    }
}

impl<T: DeviceCopy> DerefMut for DeviceVariable<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.var
    }
}
