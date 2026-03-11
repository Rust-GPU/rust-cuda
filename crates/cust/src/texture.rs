use std::mem::ManuallyDrop;
use std::mem::MaybeUninit;
use std::mem::transmute;
use std::os::raw::c_ulonglong;
use std::os::raw::{c_float, c_uint};
use std::ptr;

use cust_raw::driver_sys;
use cust_raw::driver_sys::{
    CUDA_RESOURCE_DESC, CUDA_RESOURCE_DESC_st__bindgen_ty_1,
    CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1, CUDA_RESOURCE_VIEW_DESC, CUDA_TEXTURE_DESC,
    CUresourcetype, CUtexObject, cuTexObjectCreate, cuTexObjectDestroy, cuTexObjectGetResourceDesc,
};

use crate::error::CudaResult;
use crate::error::ToResult;
use crate::memory::array::ArrayDescriptor;
use crate::memory::array::ArrayFormat;
use crate::memory::array::ArrayObject;

/// How a texture should behave if it's adressed with out of bounds indices.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureAdressingMode {
    /// Wraps around for adresses that are out of bounds.
    Wrap = 0,
    /// Clamps to the edges of the texture for adresses that are out of bounds.
    Clamp = 1,
    /// Mirrors the texture for adresses that are out of bounds.
    Mirror = 2,
    /// Uses the border color for adresses that are out of bounds.
    Border = 3,
}

/// The filtering mode to be used when fetching from the texture.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureFilterMode {
    Point = 0,
    Linear = 1,
}

bitflags::bitflags! {
    /// Flags which modify the behavior of CUDA texture creation.
    #[derive(Default, Debug, Clone, Copy)]
    pub struct TextureDescriptorFlags: c_uint {
        /// Suppresses the default behavior of having the texture promote data to floating point data in the range
        /// of [0, 1]. This flag does nothing if the texture is a texture of `u32`s.
        const READ_AS_INTEGER = driver_sys::CU_TRSF_READ_AS_INTEGER;
        /// Suppresses the default behavior of having the texture coordinates range from [0, Dim], where Dim is the
        /// width or height of the CUDA array. Instead, the texture coordinates [0, 1] reference the entire array.
        /// This flag must be set if a mipmapped array is being used.
        const NORMALIZED_COORDINATES = driver_sys::CU_TRSF_NORMALIZED_COORDINATES;
        /// Disables any trilinear filtering optimizations. Trilinear optimizations improve texture filtering performance
        /// by allowing bilinear filtering on textures in scenarios where it can closely approximate the expected results.
        const DISABLE_TRILINEAR_OPTIMIZATION = 0x20; // cuda-sys doesnt have this for some reason?
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TextureDescriptor {
    /// The adressing mode for each dimension of the texture data.
    pub adress_modes: [TextureAdressingMode; 3],
    /// The filtering mode to be used when fetching from the texture.
    pub filter_mode: TextureFilterMode,
    /// Any flags to modify the texture creation.
    pub flags: TextureDescriptorFlags,
    /// The maximum anisotropy ratio for anisotropic filtering. This will be clamped to `[1.0, 16.0]`.
    pub max_anisotropy: c_uint,
    /// The filter mode used when the calculated mipmap level lies between two defined mipmap levels.
    pub mipmap_filter_mode: TextureFilterMode,
    /// The offset to be applied to the calculated mipmap level.
    pub mipmap_level_bias: c_float,
    /// The lower end of the mipmap level range to clamp access to.
    pub min_mipmap_level_clamp: c_float,
    /// The upper end of the mipmap level range to clamp access to.
    pub max_mipmap_level_clamp: c_float,
    /// The border color of the texture.
    pub border_color: [c_float; 4],
}

impl Default for TextureDescriptor {
    fn default() -> Self {
        Self {
            adress_modes: [TextureAdressingMode::Clamp; 3],
            filter_mode: TextureFilterMode::Point,
            flags: TextureDescriptorFlags::empty(),
            max_anisotropy: 1,
            mipmap_filter_mode: TextureFilterMode::Point,
            mipmap_level_bias: 0.0,
            min_mipmap_level_clamp: 0.0,
            max_mipmap_level_clamp: 0.0,
            border_color: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

impl TextureDescriptor {
    pub fn to_raw(self) -> CUDA_TEXTURE_DESC {
        let TextureDescriptor {
            adress_modes,
            filter_mode,
            flags,
            max_anisotropy,
            mipmap_filter_mode,
            mipmap_level_bias,
            min_mipmap_level_clamp,
            max_mipmap_level_clamp,
            border_color,
        } = self;
        CUDA_TEXTURE_DESC {
            addressMode: unsafe {
                transmute::<[TextureAdressingMode; 3], [driver_sys::CUaddress_mode_enum; 3]>(
                    adress_modes,
                )
            },
            filterMode: unsafe {
                transmute::<TextureFilterMode, driver_sys::CUfilter_mode_enum>(filter_mode)
            },
            flags: flags.bits(),
            maxAnisotropy: max_anisotropy,
            mipmapFilterMode: unsafe {
                transmute::<TextureFilterMode, driver_sys::CUfilter_mode_enum>(mipmap_filter_mode)
            },
            mipmapLevelBias: mipmap_level_bias,
            minMipmapLevelClamp: min_mipmap_level_clamp,
            maxMipmapLevelClamp: max_mipmap_level_clamp,
            borderColor: border_color,
            reserved: [0; 12],
        }
    }
}

/// Specifies how the data in the CUDA array/mipmapped array should be interpreted for the texture. This could incur a change in the
/// size of the texture data.
///
/// If the format is a block compressed format, then the underlying array must have a base of format [`ArrayFormat::U32`] with 2 or 4 channels depending
/// on the compressed format. ex. BC1 and BC4 require the CUDA array to have a format of [`ArrayFormat::U32`] with 2 channels. The other BC formats require
/// the resource to have the same format but with 4 channels.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceViewFormat {
    /// No resource view format (use underlying resource format)
    None = 0,
    /// 1 channel unsigned 8-bit integers
    U8x1 = 1,
    /// 2 channel unsigned 8-bit integers
    U8x2 = 2,
    /// 4 channel unsigned 8-bit integers
    U8x4 = 3,
    /// 1 channel signed 8-bit integers
    I8x1 = 4,
    /// 2 channel signed 8-bit integers
    I8x2 = 5,
    /// 4 channel signed 8-bit integers
    I8x4 = 6,
    /// 1 channel unsigned 16-bit integers
    U16x1 = 7,
    /// 2 channel unsigned 16-bit integers
    U16x2 = 8,
    /// 4 channel unsigned 16-bit integers
    U16x4 = 9,
    /// 1 channel signed 16-bit integers
    I16x1 = 10,
    /// 2 channel signed 16-bit integers
    I16x2 = 11,
    /// 4 channel signed 16-bit integers
    I16x4 = 12,
    /// 1 channel unsigned 32-bit integers
    U32x1 = 13,
    /// 2 channel unsigned 32-bit integers
    U32x2 = 14,
    /// 4 channel unsigned 32-bit integers
    U32x4 = 15,
    /// 1 channel signed 32-bit integers
    I32x1 = 16,
    /// 2 channel signed 32-bit integers
    I32x2 = 17,
    /// 4 channel signed 32-bit integers
    I32x4 = 18,
    /// 1 channel 16-bit floating point
    F16x1 = 19,
    /// 2 channel 16-bit floating point
    F16x2 = 20,
    /// 4 channel 16-bit floating point
    F16x4 = 21,
    /// 1 channel 32-bit floating point
    F32x1 = 22,
    /// 2 channel 32-bit floating point
    F32x2 = 23,
    /// 4 channel 32-bit floating point
    F32x4 = 24,
    /// Block compressed 1
    BC1 = 25,
    /// Block compressed 2
    BC2 = 26,
    /// Block compressed 3
    BC3 = 27,
    /// Block compressed 4 unsigned
    BC4U = 28,
    /// Block compressed 4 signed
    BC4S = 29,
    /// Block compressed 5 unsigned
    BC5U = 30,
    /// Block compressed 5 signed
    BC5S = 31,
    /// Block compressed 6 unsigned half-float
    BC6HU = 32,
    /// Block compressed 6 signed half-float
    BC6HS = 33,
    /// Block compressed 7
    BC7 = 34,
}

impl ResourceViewFormat {
    pub fn from_array_format(format: ArrayFormat, num_channels: c_uint) -> Self {
        // i spent more time on this macro than it would have taken me to just write the matches out
        // but thats kind of the essence of automation
        macro_rules! format_impl {
            ($num_channels:ident, $original:ident, $($res:ident),*) => {{
                if format == ArrayFormat::$original {
                    let res = [$(ResourceViewFormat::$res),*];
                    return match $num_channels {
                        1 => res[0],
                        2 => res[1],
                        4 => res[2],
                        _ => unreachable!("num_channels must be 1, 2, or 4")
                    };
                }
            }}
        }

        format_impl!(num_channels, U8, U8x1, U8x2, U8x4);
        format_impl!(num_channels, U16, U16x1, U16x2, U16x4);
        format_impl!(num_channels, U32, U32x1, U32x2, U32x4);
        format_impl!(num_channels, I8, I8x1, I8x2, I8x4);
        format_impl!(num_channels, I16, I16x1, I16x2, I16x4);
        format_impl!(num_channels, I32, I32x1, I32x2, I32x4);
        format_impl!(num_channels, F32, F32x1, F32x2, F32x4);
        assert_ne!(
            format,
            ArrayFormat::F64,
            "CUDA Does not have 64 bit float textures, you can instead use int textures with 2 channels then cast the ints to a double in the kernel"
        );
        unreachable!()
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ResourceViewDescriptor {
    /// The format of the resource view.
    pub format: ResourceViewFormat,
    /// The new width of the texture data. If this is a compressed format this must be 4x the original width.
    /// Otherwise, it must be equal to that of the original resource.
    pub width: usize,
    /// The new height of the texture data. If this is a compressed format this must be 4x the original height.
    /// Otherwise, it must be equal to that of the original resource.
    pub height: usize,
    /// The new depth of the texture data. If this is a compressed format this must be 4x the original depth.
    /// Otherwise, it must be equal to that of the original resource.
    pub depth: usize,
    /// The most detailed mipmap level. This will be the new level zero. For non-mipmapped resources this must be `0`.
    /// This value will be relative to [`TextureDescriptor::min_mipmap_level_clamp`] and [`TextureDescriptor::max_mipmap_level_clamp`]. Ex.
    /// if the first mipmap level is `2` and the min level clamp is `1.2`, then the actual min mipmap level clamp will be `3.2`.
    pub first_mipmap_level: c_uint,
    /// The least detailed mipmap level. This must be `0` for non-mipmapped resources.
    pub last_mipmap_level: c_uint,
    /// The first layer index for layered textures. This must be `0` for non-layered resources.
    pub first_layer: c_uint,
    /// The last layer index for layered textures. This must be `0` for non-layered resources.
    pub last_layer: c_uint,
}

impl ResourceViewDescriptor {
    pub fn from_array_desc(desc: &ArrayDescriptor) -> Self {
        Self {
            format: ResourceViewFormat::from_array_format(desc.format(), desc.num_channels()),
            width: desc.width(),
            height: desc.height(),
            depth: desc.depth(),
            first_mipmap_level: 0,
            last_mipmap_level: 0,
            first_layer: 0,
            last_layer: 0,
        }
    }

    pub fn to_raw(self) -> CUDA_RESOURCE_VIEW_DESC {
        let ResourceViewDescriptor {
            format,
            width,
            height,
            depth,
            first_mipmap_level,
            last_mipmap_level,
            first_layer,
            last_layer,
        } = self;

        CUDA_RESOURCE_VIEW_DESC {
            format: unsafe {
                transmute::<ResourceViewFormat, driver_sys::CUresourceViewFormat_enum>(format)
            },
            width,
            height,
            depth,
            firstMipmapLevel: first_mipmap_level,
            lastMipmapLevel: last_mipmap_level,
            firstLayer: first_layer,
            lastLayer: last_layer,
            reserved: [0; 16],
        }
    }
}

bitflags::bitflags! {
    /// Flags for a resource descriptor. Currently empty.
    #[derive(Default, Debug)]
    pub struct ResourceDescriptorFlags: c_uint {
        #[doc(hidden)]
        const _ZERO = 0;
    }
}

#[non_exhaustive]
#[derive(Debug)]
pub enum ResourceType {
    Array { array: ArrayObject },
    // TODO: validate the soundness of linear and pitch2, they require some pointer to memory, but
    // it might be possible to cause unsoundness by allocating some type then allocating a texture, and reading back
    // the texture to host memory. Causing GPU UB is probably fine, but using that to cause host UB is not acceptable.

    // Linear {
    //     format: ArrayFormat,
    //     num_channels: u32,
    //     size: usize,
    // },
    // Pitch2d {
    //     format: ArrayFormat,
    //     num_channels: u32,
    //     width: usize,
    //     height: usize,
    //     pitch_in_bytes: usize,
    // },
}

#[derive(Debug)]
pub struct ResourceDescriptor {
    pub flags: ResourceDescriptorFlags,
    pub ty: ResourceType,
}

impl ResourceDescriptor {
    pub fn into_raw(self) -> CUDA_RESOURCE_DESC {
        let ty = match self.ty {
            ResourceType::Array { .. } => CUresourcetype::CU_RESOURCE_TYPE_ARRAY,
            // ResourceType::Linear { .. } => CUresourcetype::CU_RESOURCE_TYPE_LINEAR,
            // ResourceType::Pitch2d { .. } => CUresourcetype::CU_RESOURCE_TYPE_PITCH2D,
        };

        // we can't just use `array.handle`, this will cause the array object to call `Drop` and destroy the
        // array prematurely, which will yield a status access violation when we try to create the texture object
        // so we need to essentially leak the array into just a handle.
        let res = match self.ty {
            ResourceType::Array { array } => CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
                array: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
                    hArray: array.into_raw(),
                },
            },
            // ResourceType::Linear { format, num_channels, size }
        };

        CUDA_RESOURCE_DESC {
            resType: ty,
            flags: self.flags.bits(),
            res,
        }
    }

    // TODO: evaluate if its possible to cause UB by making a raw descriptor with an invalid array handle.
    pub(crate) fn from_raw(raw: CUDA_RESOURCE_DESC) -> Self {
        match raw.resType {
            driver_sys::CUresourcetype_enum::CU_RESOURCE_TYPE_ARRAY => Self {
                flags: ResourceDescriptorFlags::from_bits(raw.flags)
                    .expect("invalid resource descriptor flags"),
                ty: ResourceType::Array {
                    array: ArrayObject {
                        handle: unsafe { raw.res.array.hArray },
                    },
                },
            },
            _ => panic!("Unsupported resource descriptor"),
        }
    }
}

#[derive(Debug)]
pub struct Texture {
    // needed to tell the destructor if it should drop the array if we havent
    // used into_array. TODO: figure out a good way to deal with array ownership issues.
    _destroy_array_on_destruct: bool,
    handle: CUtexObject,
}

impl Drop for Texture {
    fn drop(&mut self) {
        unsafe {
            // drop the descriptor, which causes the array inside it to be dropped too
            if false {
                let res = self.resource_desc();
                if let Ok(res) = res {
                    let _ = ManuallyDrop::into_inner(res);
                }
            }

            let _ = cuTexObjectDestroy(self.handle);
        }
    }
}

pub type TextureHandle = c_ulonglong;

impl Texture {
    /// The opaque handle to this texture on the gpu. This is used for passing to a kernel.
    pub fn handle(&self) -> TextureHandle {
        self.handle
    }

    pub fn new(
        resource_desc: ResourceDescriptor,
        texture_desc: TextureDescriptor,
        resource_view_desc: Option<ResourceViewDescriptor>,
    ) -> CudaResult<Self> {
        let handle = unsafe {
            let mut uninit = MaybeUninit::<CUtexObject>::uninit();
            let resource_view_desc =
                if let Some(x) = resource_view_desc.map(|x| Box::new(x.to_raw())) {
                    Box::into_raw(x)
                } else {
                    ptr::null_mut()
                };

            let resource_desc = &resource_desc.into_raw();
            let texture_desc = &texture_desc.to_raw();

            cuTexObjectCreate(
                uninit.as_mut_ptr(),
                resource_desc as *const _,
                texture_desc as *const _,
                resource_view_desc as *const _,
            )
            .to_result()?;
            if !resource_view_desc.is_null() {
                let _ = Box::from_raw(resource_view_desc);
            }
            uninit.assume_init()
        };
        Ok(Self {
            handle,
            _destroy_array_on_destruct: true,
        })
    }

    pub fn from_array(array: ArrayObject) -> CudaResult<Self> {
        let resource_desc = ResourceDescriptor {
            flags: ResourceDescriptorFlags::empty(),
            ty: ResourceType::Array { array },
        };
        Self::new(resource_desc, Default::default(), None)
    }

    pub fn into_array(mut self) -> CudaResult<Option<ArrayObject>> {
        let desc = unsafe { ManuallyDrop::take(&mut self.resource_desc()?) };
        self._destroy_array_on_destruct = false;
        Ok(match desc.ty {
            ResourceType::Array { array } => Some(array),
        })
    }

    // pub fn array(&mut self) -> CudaResult<Option<&ArrayObject>> {
    //     let desc = self.resource_desc()?;
    //     Ok(match desc.ty {
    //         ResourceType::Array { array } => Some(array),
    //     })
    // }

    // this function returns a ManuallyDrop because dropping the descriptor will cause the underlying
    // array to be dropped, which will cause UB or undesired consequences.
    unsafe fn resource_desc(&mut self) -> CudaResult<ManuallyDrop<ResourceDescriptor>> {
        let raw = {
            let mut uninit = MaybeUninit::<CUDA_RESOURCE_DESC>::uninit();
            unsafe {
                cuTexObjectGetResourceDesc(uninit.as_mut_ptr(), self.handle).to_result()?;
                uninit.assume_init()
            }
        };
        Ok(ManuallyDrop::new(ResourceDescriptor::from_raw(raw)))
    }

    // pub fn resource_view_desc(&self) -> CudaResult<ResourceViewDescriptor> {
    //     let raw = unsafe {
    //         let ptr = ptr::null_mut();
    //         cuTexObjectGetResourceViewDesc(ptr, self.handle).to_result()?;
    //         *ptr
    //     };
    //     Ok(ResourceViewDescriptor::)
    // }
}
