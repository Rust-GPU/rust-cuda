//! Functions and types for working with CUDA kernels.

use std::marker::PhantomData;
use std::mem::{MaybeUninit, transmute};

use cust_raw::driver_sys;
use cust_raw::driver_sys::CUfunction;

use crate::context::{CacheConfig, SharedMemoryConfig};
use crate::error::{CudaResult, ToResult};
use crate::module::Module;

/// Dimensions of a grid, or the number of thread blocks in a kernel launch.
///
/// Each component of a `GridSize` must be at least 1. The maximum size depends on your device's
/// compute capability, but maximums of `x = (2^31)-1, y = 65535, z = 65535` are common. Launching
/// a kernel with a grid size greater than these limits will cause an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        *other
    }
}
#[cfg(feature = "vek")]
impl From<vek::Vec2<usize>> for GridSize {
    fn from(vec: vek::Vec2<usize>) -> Self {
        GridSize::xy(vec.x, vec.y)
    }
}
#[cfg(feature = "vek")]
impl From<vek::Vec3<usize>> for GridSize {
    fn from(vec: vek::Vec3<usize>) -> Self {
        GridSize::xyz(vec.x, vec.y, vec.z)
    }
}

#[cfg(feature = "glam")]
impl From<glam::USizeVec2> for GridSize {
    fn from(vec: glam::USizeVec2) -> Self {
        GridSize::xy(vec.x, vec.y)
    }
}
#[cfg(feature = "glam")]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        *other
    }
}
#[cfg(feature = "vek")]
impl From<vek::Vec2<usize>> for BlockSize {
    fn from(vec: vek::Vec2<usize>) -> Self {
        BlockSize::xy(vec.x, vec.y)
    }
}
#[cfg(feature = "vek")]
impl From<vek::Vec3<usize>> for BlockSize {
    fn from(vec: vek::Vec3<usize>) -> Self {
        BlockSize::xyz(vec.x, vec.y, vec.z)
    }
}

#[cfg(feature = "glam")]
impl From<glam::USizeVec2> for BlockSize {
    fn from(vec: glam::USizeVec2) -> Self {
        BlockSize::xy(vec.x, vec.y)
    }
}
#[cfg(feature = "glam")]
impl From<glam::USizeVec3> for BlockSize {
    fn from(vec: glam::USizeVec3) -> Self {
        BlockSize::xyz(vec.x, vec.y, vec.z)
    }
}

/// All supported function attributes for [Function::get_attribute](struct.Function.html#method.get_attribute)
#[repr(u32)]
#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum FunctionAttribute {
    /// The maximum number of threads per block, beyond which a launch would fail. This depends on
    /// both the function and the device.
    MaxThreadsPerBlock = 0,

    /// The size in bytes of the statically-allocated shared memory required by this function.
    SharedMemorySizeBytes = 1,

    /// The size in bytes of the constant memory required by this function
    ConstSizeBytes = 2,

    /// The size in bytes of local memory used by each thread of this function
    LocalSizeBytes = 3,

    /// The number of registers used by each thread of this function
    NumRegisters = 4,

    /// The PTX virtual architecture version for which the function was compiled. This value is the
    /// major PTX version * 10 + the minor PTX version, so version 1.3 would return the value 13.
    PtxVersion = 5,

    /// The binary architecture version for which the function was compiled. Encoded the same way as
    /// PtxVersion.
    BinaryVersion = 6,

    /// The attribute to indicate whether the function has been compiled with user specified
    /// option "-Xptxas --dlcm=ca" set.
    CacheModeCa = 7,
}

/// Handle to a global kernel function.
#[derive(Debug)]
pub struct Function<'a> {
    inner: CUfunction,
    module: PhantomData<&'a Module>,
}

unsafe impl Send for Function<'_> {}
unsafe impl Sync for Function<'_> {}

impl Function<'_> {
    pub(crate) fn new(inner: CUfunction, _module: &Module) -> Function<'_> {
        Function {
            inner,
            module: PhantomData,
        }
    }

    /// Returns information about a function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cust::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// # use cust::module::Module;
    /// # use std::ffi::CString;
    /// # let ptx = CString::new(include_str!("../resources/add.ptx"))?;
    /// # let module = Module::load_from_string(&ptx)?;
    /// use cust::function::FunctionAttribute;
    /// let function = module.get_function("sum")?;
    /// let shared_memory = function.get_attribute(FunctionAttribute::SharedMemorySizeBytes)?;
    /// println!("This function uses {} bytes of shared memory", shared_memory);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_attribute(&self, attr: FunctionAttribute) -> CudaResult<i32> {
        unsafe {
            let mut val = 0i32;
            driver_sys::cuFuncGetAttribute(
                &mut val as *mut i32,
                // This should be safe, as the repr and values of FunctionAttribute should match.
                ::std::mem::transmute::<FunctionAttribute, driver_sys::CUfunction_attribute_enum>(
                    attr,
                ),
                self.inner,
            )
            .to_result()?;
            Ok(val)
        }
    }

    /// Sets the preferred cache configuration for this function.
    ///
    /// On devices where L1 cache and shared memory use the same hardware resources, this sets the
    /// preferred cache configuration for this function. This is only a preference. The
    /// driver will use the requested configuration if possible, but is free to choose a different
    /// configuration if required to execute the function. This setting will override the
    /// context-wide setting.
    ///
    /// This setting does nothing on devices where the size of the L1 cache and shared memory are
    /// fixed.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// # use cust::module::Module;
    /// # use std::ffi::CString;
    /// # let ptx = CString::new(include_str!("../resources/add.ptx"))?;
    /// # let module = Module::load_from_string(&ptx)?;
    /// use cust::context::CacheConfig;
    /// let mut function = module.get_function("sum")?;
    /// function.set_cache_config(CacheConfig::PreferL1)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_cache_config(&mut self, config: CacheConfig) -> CudaResult<()> {
        unsafe {
            driver_sys::cuFuncSetCacheConfig(
                self.inner,
                transmute::<CacheConfig, driver_sys::CUfunc_cache_enum>(config),
            )
            .to_result()
        }
    }

    /// Sets the preferred shared memory configuration for this function.
    ///
    /// On devices with configurable shared memory banks, this function will set this function's
    /// shared memory bank size which is used for subsequent launches of this function. If not set,
    /// the context-wide setting will be used instead.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// # use cust::module::Module;
    /// # use std::ffi::CString;
    /// # let ptx = CString::new(include_str!("../resources/add.ptx"))?;
    /// # let module = Module::load_from_string(&ptx)?;
    /// use cust::context::SharedMemoryConfig;
    /// let mut function = module.get_function("sum")?;
    /// function.set_shared_memory_config(SharedMemoryConfig::EightByteBankSize)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_shared_memory_config(&mut self, cfg: SharedMemoryConfig) -> CudaResult<()> {
        unsafe {
            driver_sys::cuFuncSetSharedMemConfig(
                self.inner,
                transmute::<SharedMemoryConfig, driver_sys::CUsharedconfig_enum>(cfg),
            )
            .to_result()
        }
    }

    /// Retrieves a raw handle to this function.
    pub fn to_raw(&self) -> CUfunction {
        self.inner
    }

    // occupancy ----

    /// The amount of dynamic shared memory available per block when launching `blocks` on
    /// a streaming multiprocessor.
    pub fn available_dynamic_shared_memory_per_block(
        &self,
        blocks: GridSize,
        block_size: BlockSize,
    ) -> CudaResult<usize> {
        let num_blocks = blocks.x * blocks.y * blocks.z;
        let total_block_size = block_size.x * block_size.y * block_size.z;

        let mut result = MaybeUninit::uninit();
        unsafe {
            driver_sys::cuOccupancyAvailableDynamicSMemPerBlock(
                result.as_mut_ptr(),
                self.to_raw(),
                num_blocks as i32,
                total_block_size as i32,
            )
            .to_result()?;
            Ok(result.assume_init())
        }
    }

    /// The maximum number of active blocks per streaming multiprocessor when this function
    /// is launched with a specific `block_size` with some amount of dynamic shared memory.
    pub fn max_active_blocks_per_multiprocessor(
        &self,
        block_size: BlockSize,
        dynamic_smem_size: usize,
    ) -> CudaResult<u32> {
        let total_block_size = block_size.x * block_size.y * block_size.z;

        let mut num_blocks = MaybeUninit::uninit();
        unsafe {
            driver_sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(
                num_blocks.as_mut_ptr(),
                self.to_raw(),
                total_block_size as i32,
                dynamic_smem_size,
            )
            .to_result()?;
            Ok(num_blocks.assume_init() as u32)
        }
    }

    // TODO(RDambrosio016): Figure out a way to safely wrap a rust closure to pass it to cuda for blockSizeToDynamicSMemSize.
    // It is an issue because we need to prevent unwinding but the no-unwinding wrapper cannot capture the function from its scope.

    /// Returns a reasonable block and grid size to achieve the maximum capacity for the launch (the max number
    /// of active warps with the fewest blocks per multiprocessor).
    ///
    /// # Params
    ///
    /// `dynamic_smem_size` is the amount of dynamic shared memory required by this function. We currently do not expose
    /// a way of determining this dynamically based on block size due to safety concerns.
    ///
    /// `block_size_limit` is the maximum block size that this function is designed to handle. if this is `0` CUDA will use the maximum
    /// block size permitted by the device/function instead.
    ///
    /// Note: all panics by `dynamic_smem_size` will be ignored and the function will instead use `0`.
    pub fn suggested_launch_configuration(
        &self,
        dynamic_smem_size: usize,
        block_size_limit: BlockSize,
    ) -> CudaResult<(usize, usize)> {
        let mut min_grid_size = MaybeUninit::uninit();
        let mut block_size = MaybeUninit::uninit();

        let total_block_size_limit = block_size_limit.x * block_size_limit.y * block_size_limit.z;

        unsafe {
            driver_sys::cuOccupancyMaxPotentialBlockSize(
                min_grid_size.as_mut_ptr(),
                block_size.as_mut_ptr(),
                self.to_raw(),
                None,
                dynamic_smem_size,
                total_block_size_limit as i32,
            )
            .to_result()?;
            Ok((
                min_grid_size.assume_init() as usize,
                block_size.assume_init() as usize,
            ))
        }
    }
}

/// Launch a kernel function asynchronously.
///
/// # Syntax:
///
/// The format of this macro is designed to resemble the triple-chevron syntax used to launch
/// kernels in CUDA C. There are two forms available:
///
/// ```ignore
/// let result = launch!(module.function_name<<<grid, block, shared_memory_size, stream>>>(parameter1, parameter2...));
/// ```
///
/// This will load a kernel called `function_name` from the module `module` and launch it with
/// the given grid/block size on the given stream. Unlike in CUDA C, the shared memory size and
/// stream parameters are not optional. The shared memory size is a number of bytes per thread for
/// dynamic shared memory (Note that this uses `extern __shared__ int x[]` in CUDA C, not the
/// fixed-length arrays created by `__shared__ int x[64]`. This will usually be zero.).
/// `stream` must be the name of a [`Stream`](stream/struct.Stream.html) value.
/// `grid` can be any value which implements [`Into<GridSize>`](function/struct.GridSize.html) (such as
/// `u32` values, tuples of up to three `u32` values, and GridSize structures) and likewise `block`
/// can be any value that implements [`Into<BlockSize>`](function/struct.BlockSize.html).
///
/// NOTE: due to some limitations of Rust's macro system, `module` and `stream` must be local
/// variable names. Paths or function calls will not work.
///
/// The second form is similar:
///
/// ```ignore
/// let result = launch!(function<<<grid, block, shared_memory_size, stream>>>(parameter1, parameter2...));
/// ```
///
/// In this variant, the `function` parameter must be a variable. Use this form to avoid looking up
/// the kernel function for each call.
///
/// # Safety
///
/// Launching kernels must be done in an `unsafe` block. Calling a kernel is similar to calling a
/// foreign-language function, as the kernel itself could be written in C or unsafe Rust. The kernel
/// must accept the same number and type of parameters that are passed to the `launch!` macro. The
/// kernel must not write invalid data (for example, invalid enums) into areas of memory that can
/// be copied back to the host. The programmer must ensure that the host does not access device or
/// unified memory that the kernel could write to until after calling `stream.synchronize()`.
///
/// # Examples
///
/// ```
/// # #[macro_use]
/// # use cust::*;
/// # use std::error::Error;
/// use cust::memory::*;
/// use cust::module::Module;
/// use cust::stream::*;
/// use std::ffi::CString;
///
/// # fn main() -> Result<(), Box<dyn Error>> {
///
/// // Set up the context, load the module, and create a stream to run kernels in.
/// let _ctx = cust::quick_init()?;
/// let ptx = CString::new(include_str!("../resources/add.ptx"))?;
/// let module = Module::load_from_string(&ptx)?;
/// let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
///
/// // Create buffers for data
/// let mut in_x = DeviceBuffer::from_slice(&[1.0f32; 10])?;
/// let mut in_y = DeviceBuffer::from_slice(&[2.0f32; 10])?;
/// let mut out_1 = DeviceBuffer::from_slice(&[0.0f32; 10])?;
/// let mut out_2 = DeviceBuffer::from_slice(&[0.0f32; 10])?;
///
/// // This kernel adds each element in `in_x` and `in_y` and writes the result into `out`.
/// unsafe {
///     // Launch the kernel with one block of one thread, no dynamic shared memory on `stream`.
///     let result = launch!(module.sum<<<1, 1, 0, stream>>>(
///         in_x.as_device_ptr(),
///         in_y.as_device_ptr(),
///         out_1.as_device_ptr(),
///         out_1.len()
///     ));
///     // `launch!` returns an error in case anything went wrong with the launch itself, but
///     // kernel launches are asynchronous so errors caused by the kernel (eg. invalid memory
///     // access) will show up later at some other CUDA API call (probably at `synchronize()`
///     // below).
///     result?;
///
///     // Launch the kernel again using the `function` form:
///     let sum = module.get_function("sum")?;
///     // Launch with 1x1x1 (1) blocks of 10x1x1 (10) threads, to show that you can use tuples to
///     // configure grid and block size.
///     let result = launch!(sum<<<(1, 1, 1), (10, 1, 1), 0, stream>>>(
///         in_x.as_device_ptr(),
///         in_y.as_device_ptr(),
///         out_2.as_device_ptr(),
///         out_2.len()
///     ));
///     result?;
/// }
///
/// // Kernel launches are asynchronous, so we wait for the kernels to finish executing.
/// stream.synchronize()?;
///
/// // Copy the results back to host memory
/// let mut out_host = [0.0f32; 20];
/// out_1.copy_to(&mut out_host[0..10])?;
/// out_2.copy_to(&mut out_host[10..20])?;
///
/// for x in out_host.iter() {
///     assert_eq!(3.0, *x);
/// }
/// # Ok(())
/// # }
/// ```
///
#[macro_export]
macro_rules! launch {
    ($module:ident . $function:ident <<<$grid:expr, $block:expr, $shared:expr, $stream:ident>>>( $( $arg:expr),* $(,)?)) => {
        {
            let function = $module.get_function(stringify!($function));
            match function {
                Ok(f) => launch!(f<<<$grid, $block, $shared, $stream>>>( $($arg),* ) ),
                Err(e) => Err(e),
            }
        }
    };
    ($function:ident <<<$grid:expr, $block:expr, $shared:expr, $stream:ident>>>( $( $arg:expr),* $(,)?)) => {
        {
            fn assert_impl_devicecopy<T: $crate::memory::DeviceCopy>(_val: T) {}
            if false {
                $(
                    assert_impl_devicecopy($arg);
                )*
            };

            $stream.launch(&$function, $grid, $block, $shared,
                &[
                    $(
                        &$arg as *const _ as *mut ::std::ffi::c_void,
                    )*
                ]
            )
        }
    };
}
