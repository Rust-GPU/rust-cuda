use std::{
    mem::{ManuallyDrop, MaybeUninit},
    os::raw::c_ulonglong,
};

use cust_raw::driver_sys::{
    CUDA_RESOURCE_DESC, CUsurfObject, cuSurfObjectCreate, cuSurfObjectDestroy,
    cuSurfObjectGetResourceDesc,
};

use crate::{
    error::{CudaResult, ToResult},
    memory::array::ArrayObject,
    texture::{ResourceDescriptor, ResourceDescriptorFlags, ResourceType},
};

pub struct Surface {
    _destroy_array_on_drop: bool,
    handle: CUsurfObject,
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            // drop the descriptor, which causes the array inside it to be dropped too
            if false {
                let res = self.resource_desc();
                if let Ok(res) = res {
                    let _ = ManuallyDrop::into_inner(res);
                }
            }

            let _ = cuSurfObjectDestroy(self.handle);
        }
    }
}

pub type SurfaceHandle = c_ulonglong;

impl Surface {
    /// The opaque handle to this surface on the gpu. This is used for passing to a kernel.
    pub fn handle(&self) -> SurfaceHandle {
        self.handle
    }

    pub fn new(resource_desc: ResourceDescriptor) -> CudaResult<Self> {
        let raw = resource_desc.into_raw();
        unsafe {
            let mut uninit = MaybeUninit::<CUsurfObject>::uninit();
            cuSurfObjectCreate(uninit.as_mut_ptr(), &raw as *const _).to_result()?;
            Ok(Self {
                handle: uninit.assume_init(),
                _destroy_array_on_drop: true,
            })
        }
    }

    pub fn from_array(array: ArrayObject) -> CudaResult<Self> {
        let resource_desc = ResourceDescriptor {
            flags: ResourceDescriptorFlags::empty(),
            ty: ResourceType::Array { array },
        };
        Self::new(resource_desc)
    }

    pub fn into_array(mut self) -> CudaResult<Option<ArrayObject>> {
        let desc = unsafe { ManuallyDrop::take(&mut self.resource_desc()?) };
        self._destroy_array_on_drop = false;
        Ok(match desc.ty {
            ResourceType::Array { array } => Some(array),
        })
    }

    // see Texture::resource_desc on why this is unsafe and private and returns a manuallydrop
    unsafe fn resource_desc(&mut self) -> CudaResult<ManuallyDrop<ResourceDescriptor>> {
        let raw = {
            let mut uninit = MaybeUninit::<CUDA_RESOURCE_DESC>::uninit();
            unsafe {
                cuSurfObjectGetResourceDesc(uninit.as_mut_ptr(), self.handle).to_result()?;
                uninit.assume_init()
            }
        };
        Ok(ManuallyDrop::new(ResourceDescriptor::from_raw(raw)))
    }
}
