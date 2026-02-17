pub(crate) mod driver_types_sys;
mod error;
pub mod sys;

pub use error::*;

use core::ffi::c_void;
use core::mem::MaybeUninit;
use sys as cuda;

bitflags::bitflags! {
    /// Bit flags for configuring a CUDA Stream.
    pub struct StreamFlags: u32 {
        /// No flags set.
        const DEFAULT = 0x00;

        /// This stream does not synchronize with the NULL stream.
        ///
        /// Note that the name is chosen to correspond to CUDA documentation, but is nevertheless
        /// misleading. All work within a single stream is ordered and asynchronous regardless
        /// of whether this flag is set. All streams in cust may execute work concurrently,
        /// regardless of the flag. However, for legacy reasons, CUDA has a notion of a NULL stream,
        /// which is used as the default when no other stream is provided. Work on other streams
        /// may not be executed concurrently with work on the NULL stream unless this flag is set.
        /// Since cust does not provide access to the NULL stream, this flag has no effect in
        /// most circumstances. However, it is recommended to use it anyway, as some other crate
        /// in this binary may be using the NULL stream directly.
        const NON_BLOCKING = 0x01;
    }
}

#[derive(Debug)]
pub struct Stream {
    raw: cuda::cudaStream_t,
}

impl Stream {
    /// Creates a new stream with flags.
    pub fn new(flags: StreamFlags) -> CudaResult<Self> {
        let mut stream = MaybeUninit::uninit();

        unsafe {
            cuda::cudaStreamCreateWithFlags(stream.as_mut_ptr(), flags.bits).to_result()?;
            Ok(Self {
                raw: stream.assume_init(),
            })
        }
    }

    #[doc(hidden)]
    pub fn launch(&self, param_buf: *mut c_void) -> CudaResult<()> {
        unsafe { cuda::cudaLaunchDeviceV2(param_buf, self.raw).to_result() }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            cuda::cudaStreamDestroy(self.raw);
        }
    }
}

#[macro_export]
macro_rules! launch {
    ($func:ident<<<$grid_dim:expr, $block_dim:expr, $smem_size:expr, $stream:ident>>>($($param:expr),* $(,)?)) => {{
        let grid_dim = ::$crate::rt::GridDim::from($grid_dim);
        let block_dim = ::$crate::rt::BlockDim::from($block_dim);
        let mut buf = ::$crate::rt::sys::cudaGetParameterBufferV2(
            &$func as *const _ as *const ::core::ffi::c_void,
            ::$crate::rt::sys::dim3 {
                x: grid_dim.x,
                y: grid_dim.y,
                z: grid_dim.z
            },
            ::$crate::rt::sys::dim3 {
                x: block_dim.x,
                y: block_dim.y,
                z: block_dim.z
            },
            $smem_size
        ) as *mut u8;
        unsafe {
            let mut offset = 0;
            $(
                let param = $param;
                let size = ::core::mem::size_of_val(&param)
                let mut buf_idx = (offset as f32 / size as f32).ceil() as usize + 1;
                offset = buf_idx * size;
                let ptr = &param as *const _ as *const u8;
                let dst = buf.add(offset);
                ::core::ptr::copy_nonoverlapping(&param as *const _ as *const u8, dst, size);
            )*
        }
        if false {
            $func($($param),*);
        }
        $stream.launch(buf as *mut ::core::ffi::c_void).to_result()
    }};
}

/// Dimensions of a grid, or the number of thread blocks in a kernel launch.
///
/// Each component of a `GridSize` must be at least 1. The maximum size depends on your device's
/// compute capability, but maximums of `x = (2^31)-1, y = 65535, z = 65535` are common. Launching
/// a kernel with a grid size greater than these limits will cause an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridSize {
    /// Width of grid in blocks
    pub x: usize,
    /// Height of grid in blocks
    pub y: usize,
    /// Depth of grid in blocks
    pub z: usize,
}
impl GridSize {
    /// Create a one-dimensional grid of `x` blocks
    #[inline]
    pub fn x(x: usize) -> GridSize {
        GridSize { x, y: 1, z: 1 }
    }

    /// Create a two-dimensional grid of `x * y` blocks
    #[inline]
    pub fn xy(x: usize, y: usize) -> GridSize {
        GridSize { x, y, z: 1 }
    }

    /// Create a three-dimensional grid of `x * y * z` blocks
    #[inline]
    pub fn xyz(x: usize, y: usize, z: usize) -> GridSize {
        GridSize { x, y, z }
    }
}
impl From<usize> for GridSize {
    fn from(x: usize) -> GridSize {
        GridSize::x(x)
    }
}
impl From<(usize, usize)> for GridSize {
    fn from((x, y): (usize, usize)) -> GridSize {
        GridSize::xy(x, y)
    }
}
impl From<(usize, usize, usize)> for GridSize {
    fn from((x, y, z): (usize, usize, usize)) -> GridSize {
        GridSize::xyz(x, y, z)
    }
}
impl From<&GridSize> for GridSize {
    fn from(other: &GridSize) -> GridSize {
        other.clone()
    }
}
impl From<glam::USizeVec2> for GridSize {
    fn from(vec: glam::USizeVec2) -> Self {
        GridSize::xy(vec.x, vec.y)
    }
}
impl From<glam::USizeVec3> for GridSize {
    fn from(vec: glam::USizeVec3) -> Self {
        GridSize::xyz(vec.x, vec.y, vec.z)
    }
}

/// Dimensions of a thread block, or the number of threads in a block.
///
/// Each component of a `BlockSize` must be at least 1. The maximum size depends on your device's
/// compute capability, but maximums of `x = 1024, y = 1024, z = 64` are common. In addition, the
/// limit on total number of threads in a block (`x * y * z`) is also defined by the compute
/// capability, typically 1024. Launching a kernel with a block size greater than these limits will
/// cause an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockSize {
    /// X dimension of each thread block
    pub x: usize,
    /// Y dimension of each thread block
    pub y: usize,
    /// Z dimension of each thread block
    pub z: usize,
}
impl BlockSize {
    /// Create a one-dimensional block of `x` threads
    #[inline]
    pub fn x(x: usize) -> BlockSize {
        BlockSize { x, y: 1, z: 1 }
    }

    /// Create a two-dimensional block of `x * y` threads
    #[inline]
    pub fn xy(x: usize, y: usize) -> BlockSize {
        BlockSize { x, y, z: 1 }
    }

    /// Create a three-dimensional block of `x * y * z` threads
    #[inline]
    pub fn xyz(x: usize, y: usize, z: usize) -> BlockSize {
        BlockSize { x, y, z }
    }
}
impl From<usize> for BlockSize {
    fn from(x: usize) -> BlockSize {
        BlockSize::x(x)
    }
}
impl From<(usize, usize)> for BlockSize {
    fn from((x, y): (usize, usize)) -> BlockSize {
        BlockSize::xy(x, y)
    }
}
impl From<(usize, usize, usize)> for BlockSize {
    fn from((x, y, z): (usize, usize, usize)) -> BlockSize {
        BlockSize::xyz(x, y, z)
    }
}
impl From<&BlockSize> for BlockSize {
    fn from(other: &BlockSize) -> BlockSize {
        other.clone()
    }
}
impl From<glam::USizeVec2> for BlockSize {
    fn from(vec: glam::USizeVec2) -> Self {
        BlockSize::xy(vec.x, vec.y)
    }
}
impl From<glam::USizeVec3> for BlockSize {
    fn from(vec: glam::USizeVec3) -> Self {
        BlockSize::xyz(vec.x, vec.y, vec.z)
    }
}
