//! Functions and types for working with CUDA modules.

use std::ffi::{c_void, CStr, CString};
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::os::raw::c_uint;
use std::path::Path;
use std::ptr;

use cust_raw::driver_sys;

use crate::error::{CudaResult, DropResult, ToResult};
use crate::function::Function;
use crate::memory::{CopyDestination, DeviceCopy, DevicePointer};

/// A compiled CUDA module, loaded into a context.
#[derive(Debug)]
pub struct Module {
    inner: driver_sys::CUmodule,
}

unsafe impl Send for Module {}
unsafe impl Sync for Module {}

/// The possible optimization levels when JIT compiling a PTX module. `O4` by default (most optimized).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptLevel {
    O0 = 0,
    O1 = 1,
    O2 = 2,
    O3 = 3,
    O4 = 4,
}

/// How to handle cases where a loaded module's data does not contain an exact match for the
/// specified architecture.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JitFallback {
    /// Prefer to compile PTX if present if an exact binary match is not found.
    PreferPtx = 0,
    /// Prefer to fall back to a compatible binary code match if exact match is not found.
    /// This means the driver may pick binary code for `7.0` if your device is `7.2` for example.
    PreferCompatibleBinary = 1,
}

/// Different options that could be applied when loading a module.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModuleJitOption {
    /// Specifies the maximum amount of registers any compiled PTX is allowed to use.
    MaxRegisters(u32),
    /// Specifies the optimization level for the JIT compiler.
    OptLevel(OptLevel),
    /// Determines the PTX target from the current context's architecture. Cannot be combined with
    /// [`ModuleJitOption::Target`].
    DetermineTargetFromContext,
    /// Specifies the target for the JIT compiler. Cannot be combined with [`ModuleJitOption::DetermineTargetFromContext`].
    Target(driver_sys::CUjit_target),
    /// Specifies how to handle cases where a loaded module's data does not have an exact match for the specified
    /// architecture.
    Fallback(JitFallback),
    /// Generates debug info in the compiled binary.
    GenenerateDebugInfo(bool),
    /// Generates line info in the compiled binary.
    GenerateLineInfo(bool),
}

impl ModuleJitOption {
    pub fn into_raw(opts: &[Self]) -> (Vec<driver_sys::CUjit_option>, Vec<*mut c_void>) {
        // And here we stumble across one of the most horrific things i have ever seen in my entire
        // journey of working with many parts of CUDA. As a background, CUDA usually wants an array
        // of pointers to values when it takes void**, after all, this is what is expected by anyone.
        // However, there is a SINGLE exception in the entire driver API, and that is cuModuleLoadDataEx,
        // it actually wants you to pass values by value instead of by ref if they fit into pointer length.
        // Therefore something like MaxRegisters should be passed as `u32 as usize as *mut c_void`.
        // This is completely undocumented. I initially brought this up to an nvidia developer,
        // who eventually was able to figure out this issue, currently it appears to be labeled "not a bug",
        // however this will likely be changed in the future, or at least get documented better. (hopefully)
        let mut raw_opts = Vec::with_capacity(opts.len());
        let mut raw_vals = Vec::with_capacity(opts.len());

        for opt in opts {
            match opt {
                Self::MaxRegisters(regs) => {
                    raw_opts.push(driver_sys::CUjit_option::CU_JIT_MAX_REGISTERS);
                    raw_vals.push(*regs as usize as *mut c_void);
                }
                Self::OptLevel(level) => {
                    raw_opts.push(driver_sys::CUjit_option::CU_JIT_OPTIMIZATION_LEVEL);
                    raw_vals.push(*level as usize as *mut c_void);
                }
                Self::DetermineTargetFromContext => {
                    raw_opts.push(driver_sys::CUjit_option::CU_JIT_TARGET_FROM_CUCONTEXT);
                }
                Self::Target(target) => {
                    raw_opts.push(driver_sys::CUjit_option::CU_JIT_TARGET);
                    raw_vals.push(*target as usize as *mut c_void);
                }
                Self::Fallback(fallback) => {
                    raw_opts.push(driver_sys::CUjit_option::CU_JIT_FALLBACK_STRATEGY);
                    raw_vals.push(*fallback as usize as *mut c_void);
                }
                Self::GenenerateDebugInfo(gen) => {
                    raw_opts.push(driver_sys::CUjit_option::CU_JIT_GENERATE_DEBUG_INFO);
                    raw_vals.push(*gen as usize as *mut c_void);
                }
                Self::GenerateLineInfo(gen) => {
                    raw_opts.push(driver_sys::CUjit_option::CU_JIT_GENERATE_LINE_INFO);
                    raw_vals.push(*gen as usize as *mut c_void)
                }
            }
        }
        (raw_opts, raw_vals)
    }
}

#[cfg(unix)]
fn path_to_bytes<P: AsRef<Path>>(path: P) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    path.as_ref().as_os_str().as_bytes().to_vec()
}

#[cfg(not(unix))]
fn path_to_bytes<P: AsRef<Path>>(path: P) -> Vec<u8> {
    path.as_ref().to_string_lossy().to_string().into_bytes()
}

impl Module {
    /// Load a module from the given path into the current context.
    ///
    /// The given path should be either a cubin file, a ptx file, or a fatbin file such as
    /// those produced by `nvcc`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use cust::module::Module;
    /// use std::ffi::CString;
    ///
    /// let module = Module::from_file("./resources/add.ptx")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> CudaResult<Module> {
        unsafe {
            let mut bytes = path_to_bytes(path);
            if !bytes.contains(&0) {
                bytes.push(0);
            }
            let mut module = Module {
                inner: ptr::null_mut(),
            };
            driver_sys::cuModuleLoad(
                &mut module.inner as *mut driver_sys::CUmodule,
                bytes.as_ptr() as *const _,
            )
            .to_result()?;
            Ok(module)
        }
    }

    /// Creates a new module by loading a fatbin (fat binary) file.
    ///
    /// Fatbinary files are files that contain multiple ptx or cubin files. The driver will choose already-built
    /// cubin if it is present, and otherwise JIT compile any PTX in the file to cubin.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use cust::module::Module;
    /// let fatbin_bytes = std::fs::read("./resources/add.fatbin")?;
    /// // will return InvalidSource if the fatbin does not contain any compatible code, meaning, either
    /// // cubin compiled for the same device architecture OR PTX that can be JITted into valid code.
    /// let module = Module::from_fatbin(&fatbin_bytes, &[])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_fatbin<T: AsRef<[u8]>>(
        bytes: T,
        options: &[ModuleJitOption],
    ) -> CudaResult<Module> {
        // fatbins can be loaded just like cubins, we just use different methods so it's explicit.
        // please don't use from_cubin for fatbins, that is pure chaos and ferris will come to your house
        Self::from_cubin(bytes, options)
    }

    /// Creates a new module by loading a cubin (CUDA Binary) file.
    ///
    /// Cubins are architecture/compute-capability specific files generated as the final step of the CUDA compilation
    /// process. They cannot be interchanged across compute capabilities unlike PTX (to some degree). You can create one
    /// using the PTX compiler APIs, the cust [`Linker`](crate::link::Linker), or nvcc (`nvcc a.ptx --cubin -arch=sm_XX`).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use cust::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use cust::module::Module;
    /// let cubin_bytes = std::fs::read("./resources/add.cubin")?;
    /// // will return InvalidSource if the cubin arch doesn't match the context's device arch!
    /// let module = Module::from_cubin(&cubin_bytes, &[])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_cubin<T: AsRef<[u8]>>(bytes: T, options: &[ModuleJitOption]) -> CudaResult<Module> {
        // it is very unclear whether cuda wants or doesn't want a null terminator. The method works
        // whether you have one or not. So for safety we just add one. In theory you can figure out the
        // length of an ELF image without a null terminator. But the docs are confusing, so we add one just
        // to be sure.
        let mut bytes = bytes.as_ref().to_vec();
        bytes.push(0);
        // SAFETY: the image is known to be dereferenceable
        unsafe { Self::load_module(bytes.as_ptr() as *const c_void, options) }
    }

    unsafe fn load_module(image: *const c_void, options: &[ModuleJitOption]) -> CudaResult<Module> {
        let mut module = Module {
            inner: ptr::null_mut(),
        };
        let (mut options, mut option_values) = ModuleJitOption::into_raw(options);
        driver_sys::cuModuleLoadDataEx(
            &mut module.inner as *mut driver_sys::CUmodule,
            image,
            options.len() as c_uint,
            options.as_mut_ptr(),
            option_values.as_mut_ptr(),
        )
        .to_result()?;
        Ok(module)
    }

    /// Creates a new module from a [`CStr`] pointing to PTX code.
    ///
    /// The driver will JIT the PTX into arch-specific cubin or pick already-cached cubin if available.
    pub fn from_ptx_cstr(cstr: &CStr, options: &[ModuleJitOption]) -> CudaResult<Module> {
        // SAFETY: the image is known to be dereferenceable
        unsafe { Self::load_module(cstr.as_ptr() as *const c_void, options) }
    }

    /// Creates a new module from a PTX string, allocating an intermediate buffer for the [`CString`].
    ///
    /// The driver will JIT the PTX into arch-specific cubin or pick already-cached cubin if available.
    ///
    /// # Panics
    ///
    /// Panics if `string` contains a nul.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use cust::module::Module;
    /// let ptx = std::fs::read_to_string("./resources/add.ptx")?;
    /// let module = Module::from_ptx(&ptx, &[])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_ptx<T: AsRef<str>>(string: T, options: &[ModuleJitOption]) -> CudaResult<Module> {
        let cstr = CString::new(string.as_ref())
            .expect("string given to Module::from_str contained nul bytes");
        Self::from_ptx_cstr(cstr.as_c_str(), options)
    }

    /// Load a module from a normal (rust) string, implicitly making it into
    /// a cstring.
    #[deprecated(
        since = "0.3.0",
        note = "from_str was too generic of a name, use from_ptx instead, passing an empty slice of options (usually)"
    )]
    #[allow(clippy::should_implement_trait)]
    pub fn from_str<T: AsRef<str>>(string: T) -> CudaResult<Module> {
        let cstr = CString::new(string.as_ref())
            .expect("string given to Module::from_str contained nul bytes");
        #[allow(deprecated)]
        Self::load_from_string(cstr.as_c_str())
    }

    /// Load a module from a CStr.
    ///
    /// This is useful in combination with `include_str!`, to include the device code into the
    /// compiled executable.
    ///
    /// The given CStr must contain the bytes of a cubin file, a ptx file or a fatbin file such as
    /// those produced by `nvcc`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use cust::module::Module;
    /// use std::ffi::CString;
    ///
    /// let image = CString::new(include_str!("../resources/add.ptx"))?;
    /// let module = Module::load_from_string(&image)?;
    /// # Ok(())
    /// # }
    /// ```
    #[deprecated(
        since = "0.3.0",
        note = "load_from_string was an inconsistent name with inconsistent params, use from_ptx/from_ptx_cstr, passing
    an empty slice of options (usually)
    "
    )]
    pub fn load_from_string(image: &CStr) -> CudaResult<Module> {
        unsafe {
            let mut module = Module {
                inner: ptr::null_mut(),
            };
            driver_sys::cuModuleLoadData(
                &mut module.inner as *mut driver_sys::CUmodule,
                image.as_ptr() as *const c_void,
            )
            .to_result()?;
            Ok(module)
        }
    }

    /// Get a reference to a global symbol, which can then be copied to/from.
    ///
    /// # Panics:
    ///
    /// This function panics if the size of the symbol is not the same as the `mem::sizeof<T>()`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cust::*;
    /// # use cust::memory::CopyDestination;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use cust::module::Module;
    /// use std::ffi::CString;
    ///
    /// let ptx = CString::new(include_str!("../resources/add.ptx"))?;
    /// let module = Module::load_from_string(&ptx)?;
    /// let name = CString::new("my_constant")?;
    /// let symbol = module.get_global::<u32>(&name)?;
    /// let mut host_const = 0;
    /// symbol.copy_to(&mut host_const)?;
    /// assert_eq!(314, host_const);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_global<'a, T: DeviceCopy>(&'a self, name: &CStr) -> CudaResult<Symbol<'a, T>> {
        unsafe {
            let mut ptr: DevicePointer<T> = DevicePointer::null();
            let mut size: usize = 0;

            driver_sys::cuModuleGetGlobal(
                &mut ptr as *mut DevicePointer<T> as *mut driver_sys::CUdeviceptr,
                &mut size as *mut usize,
                self.inner,
                name.as_ptr(),
            )
            .to_result()?;
            assert_eq!(size, mem::size_of::<T>());
            Ok(Symbol {
                ptr,
                module: PhantomData,
            })
        }
    }

    /// Get a reference to a kernel function which can then be launched.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cust::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use cust::module::Module;
    /// use std::ffi::CString;
    ///
    /// let ptx = CString::new(include_str!("../resources/add.ptx"))?;
    /// let module = Module::load_from_string(&ptx)?;
    /// let function = module.get_function("sum")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_function<T: AsRef<str>>(&'_ self, name: T) -> CudaResult<Function<'_>> {
        unsafe {
            let name = name.as_ref();
            let cstr = CString::new(name).expect("Argument to get_function had a nul");
            let mut func: driver_sys::CUfunction = ptr::null_mut();

            driver_sys::cuModuleGetFunction(
                &mut func as *mut driver_sys::CUfunction,
                self.inner,
                cstr.as_ptr(),
            )
            .to_result()?;
            Ok(Function::new(func, self))
        }
    }

    /// Destroy a `Module`, returning an error.
    ///
    /// Destroying a module can return errors from previous asynchronous work. This function
    /// destroys the given module and returns the error and the un-destroyed module on failure.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use cust::module::Module;
    /// use std::ffi::CString;
    ///
    /// let ptx = CString::new(include_str!("../resources/add.ptx"))?;
    /// let module = Module::load_from_string(&ptx)?;
    /// match Module::drop(module) {
    ///     Ok(()) => println!("Successfully destroyed"),
    ///     Err((e, module)) => {
    ///         println!("Failed to destroy module: {:?}", e);
    ///         // Do something with module
    ///     },
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn drop(mut module: Module) -> DropResult<Module> {
        if module.inner.is_null() {
            return Ok(());
        }

        unsafe {
            let inner = mem::replace(&mut module.inner, ptr::null_mut());
            match driver_sys::cuModuleUnload(inner).to_result() {
                Ok(()) => {
                    mem::forget(module);
                    Ok(())
                }
                Err(e) => Err((e, Module { inner })),
            }
        }
    }
}
impl Drop for Module {
    fn drop(&mut self) {
        if self.inner.is_null() {
            return;
        }
        unsafe {
            // No choice but to panic if this fails...
            let module = mem::replace(&mut self.inner, ptr::null_mut());
            let _ = driver_sys::cuModuleUnload(module);
        }
    }
}

/// Handle to a symbol defined within a CUDA module.
#[derive(Debug)]
pub struct Symbol<'a, T: DeviceCopy> {
    ptr: DevicePointer<T>,
    module: PhantomData<&'a Module>,
}
impl<T: DeviceCopy> crate::private::Sealed for Symbol<'_, T> {}
impl<T: DeviceCopy> fmt::Pointer for Symbol<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, f)
    }
}
impl<T: DeviceCopy> CopyDestination<T> for Symbol<'_, T> {
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
