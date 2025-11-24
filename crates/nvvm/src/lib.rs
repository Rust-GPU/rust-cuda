//! High level safe bindings to the NVVM compiler (libnvvm) for writing CUDA GPU kernels with a subset of LLVM IR.

use std::{
    ffi::{CStr, CString},
    fmt::Display,
    mem::MaybeUninit,
    ptr::null_mut,
    str::FromStr,
};

use strum::IntoEnumIterator;

use cust_raw::nvvm_sys;

pub use cust_raw::nvvm_sys::LIBDEVICE_BITCODE;

/// Get the major and minor NVVM IR version.
pub fn ir_version() -> (i32, i32) {
    unsafe {
        let mut major_ir = MaybeUninit::uninit();
        let mut minor_ir = MaybeUninit::uninit();
        let mut major_dbg = MaybeUninit::uninit();
        let mut minor_dbg = MaybeUninit::uninit();
        // according to the docs this can't fail
        let _ = nvvm_sys::nvvmIRVersion(
            major_ir.as_mut_ptr(),
            minor_ir.as_mut_ptr(),
            major_dbg.as_mut_ptr(),
            minor_dbg.as_mut_ptr(),
        );
        (major_ir.assume_init(), minor_ir.assume_init())
    }
}

/// Get the major and minor NVVM debug metadata version.
pub fn dbg_version() -> (i32, i32) {
    unsafe {
        let mut major_ir = MaybeUninit::uninit();
        let mut minor_ir = MaybeUninit::uninit();
        let mut major_dbg = MaybeUninit::uninit();
        let mut minor_dbg = MaybeUninit::uninit();
        // according to the docs this can't fail
        let _ = nvvm_sys::nvvmIRVersion(
            major_ir.as_mut_ptr(),
            minor_ir.as_mut_ptr(),
            major_dbg.as_mut_ptr(),
            minor_dbg.as_mut_ptr(),
        );
        (major_dbg.assume_init(), minor_dbg.assume_init())
    }
}

/// Get the major and minor NVVM version.
pub fn nvvm_version() -> (i32, i32) {
    unsafe {
        let mut major = MaybeUninit::uninit();
        let mut minor = MaybeUninit::uninit();
        // according to the docs this can't fail
        let _ = nvvm_sys::nvvmVersion(major.as_mut_ptr(), minor.as_mut_ptr());
        (major.assume_init(), minor.assume_init())
    }
}

/// Rust version of `nvvmResult`.
/// - `NVVM_SUCCESS` isn't covered because this type only covers the error cases, due to Rust
///   having `Result` where the success case is separate from the error cases.
/// - `NVVM_ERROR_INVALID_PROGRAM` isn't covered because it's not possible to get an invalid
///   program handle through this safe api.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvvmError {
    /// The NVVM compiler ran out of memory.
    OutOfMemory,
    /// The program could not be created for an unspecified reason.
    ProgramCreationFailure,
    IrVersionMismatch,
    InvalidInput,
    /// The IR given to the program was invalid. Getting the compiler
    /// log should yield more info.
    InvalidIr,
    /// A compile option given to the compiler was invalid.
    InvalidOption,
    /// The program has no modules OR all modules are lazy modules.
    NoModuleInProgram,
    /// Compilation failed because of bad IR or other reasons. Getting the compiler
    /// log should yield more info.
    CompilationError,
}

impl Display for NvvmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            let ptr = nvvm_sys::nvvmGetErrorString(self.to_raw());
            f.write_str(&CStr::from_ptr(ptr).to_string_lossy())
        }
    }
}

impl NvvmError {
    fn to_raw(self) -> nvvm_sys::nvvmResult {
        match self {
            NvvmError::CompilationError => nvvm_sys::nvvmResult::NVVM_ERROR_COMPILATION,
            NvvmError::OutOfMemory => nvvm_sys::nvvmResult::NVVM_ERROR_OUT_OF_MEMORY,
            NvvmError::ProgramCreationFailure => {
                nvvm_sys::nvvmResult::NVVM_ERROR_PROGRAM_CREATION_FAILURE
            }
            NvvmError::IrVersionMismatch => nvvm_sys::nvvmResult::NVVM_ERROR_IR_VERSION_MISMATCH,
            NvvmError::InvalidOption => nvvm_sys::nvvmResult::NVVM_ERROR_INVALID_OPTION,
            NvvmError::InvalidInput => nvvm_sys::nvvmResult::NVVM_ERROR_INVALID_INPUT,
            NvvmError::InvalidIr => nvvm_sys::nvvmResult::NVVM_ERROR_INVALID_IR,
            NvvmError::NoModuleInProgram => nvvm_sys::nvvmResult::NVVM_ERROR_NO_MODULE_IN_PROGRAM,
        }
    }

    fn from_raw(result: nvvm_sys::nvvmResult) -> Self {
        use NvvmError::*;
        match result {
            nvvm_sys::nvvmResult::NVVM_ERROR_COMPILATION => CompilationError,
            nvvm_sys::nvvmResult::NVVM_ERROR_OUT_OF_MEMORY => OutOfMemory,
            nvvm_sys::nvvmResult::NVVM_ERROR_PROGRAM_CREATION_FAILURE => ProgramCreationFailure,
            nvvm_sys::nvvmResult::NVVM_ERROR_IR_VERSION_MISMATCH => IrVersionMismatch,
            nvvm_sys::nvvmResult::NVVM_ERROR_INVALID_OPTION => InvalidOption,
            nvvm_sys::nvvmResult::NVVM_ERROR_INVALID_INPUT => InvalidInput,
            nvvm_sys::nvvmResult::NVVM_ERROR_INVALID_IR => InvalidIr,
            nvvm_sys::nvvmResult::NVVM_ERROR_NO_MODULE_IN_PROGRAM => NoModuleInProgram,
            nvvm_sys::nvvmResult::NVVM_SUCCESS => panic!(),
            _ => unreachable!(),
        }
    }
}

trait ToNvvmResult {
    fn to_result(self) -> Result<(), NvvmError>;
}

impl ToNvvmResult for nvvm_sys::nvvmResult {
    fn to_result(self) -> Result<(), NvvmError> {
        let err = match self {
            nvvm_sys::nvvmResult::NVVM_SUCCESS => return Ok(()),
            _ => NvvmError::from_raw(self),
        };
        Err(err)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvvmOption {
    /// Generate debug info, valid only with an opt-level of `0` (`-g`).
    GenDebugInfo,
    /// Generate line number info (`-generate-line-info`).
    GenLineInfo,
    /// Whether to disable optimizations (opt level 0).
    NoOpts,
    /// The NVVM arch to use.
    Arch(NvvmArch),
    /// Whether to flush denormal values to zero when performing single-precision
    /// floating point operations. False by default.
    Ftz,
    /// Whether to use a fast approximation for sqrt instead of
    /// IEEE round-to-nearest mode for single-precision float square root.
    FastSqrt,
    /// Whether to use a fast approximation for div and reciprocal instead of
    /// IEEE round-to-nearest mode for single-precision float division.
    FastDiv,
    /// Whether to enable FMA contraction.
    NoFmaContraction,
}

impl Display for NvvmOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let res = match self {
            Self::GenDebugInfo => "-g",
            Self::GenLineInfo => "-generate-line-info",
            Self::NoOpts => "-opt=0",
            Self::Arch(arch) => return f.write_str(&format!("-arch={arch}")),
            Self::Ftz => "-ftz=1",
            Self::FastSqrt => "-prec-sqrt=0",
            Self::FastDiv => "-prec-div=0",
            Self::NoFmaContraction => "-fma=0",
        };
        f.write_str(res)
    }
}

impl FromStr for NvvmOption {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        Ok(match s {
            "-g" => Self::GenDebugInfo,
            "-generate-line-info" => Self::GenLineInfo,
            _ if s.starts_with("-opt=") => {
                let slice = &s[5..];
                if slice == "0" {
                    Self::NoOpts
                } else if slice == "3" {
                    // implied
                    return Err("-opt=3 is the default".to_string());
                } else {
                    return Err(format!("unknown -opt value: {slice}"));
                }
            }
            _ if s.starts_with("-ftz=") => {
                let slice = &s[5..];
                if slice == "1" {
                    Self::Ftz
                } else if slice == "0" {
                    // implied
                    return Err("-ftz=0 is the default".to_string());
                } else {
                    return Err(format!("unknown -ftz value: {slice}"));
                }
            }
            _ if s.starts_with("-prec-sqrt=") => {
                let slice = &s[11..];
                if slice == "0" {
                    Self::FastSqrt
                } else if slice == "1" {
                    // implied
                    return Err("-prec-sqrt=1 is the default".to_string());
                } else {
                    return Err(format!("unknown -prec-sqrt value: {slice}"));
                }
            }
            _ if s.starts_with("-prec-div=") => {
                let slice = &s[10..];
                if slice == "0" {
                    Self::FastDiv
                } else if slice == "1" {
                    // implied
                    return Err("-prec-div=1 is the default".to_string());
                } else {
                    return Err(format!("unknown -prec-div value: {slice}"));
                }
            }
            _ if s.starts_with("-fma=") => {
                let slice = &s[5..];
                if slice == "0" {
                    Self::NoFmaContraction
                } else if slice == "1" {
                    // implied
                    return Err("-fma=1 is the default".to_string());
                } else {
                    return Err(format!("unknown -fma value: {slice}"));
                }
            }
            _ if s.starts_with("-arch=") => {
                let slice = &s[6..];
                match NvvmArch::from_str(slice) {
                    Ok(arch) => Self::Arch(arch),
                    Err(_) => return Err(format!("unknown -arch value: {slice}")),
                }
            }
            _ => return Err(format!("unknown option: {s}")),
        })
    }
}

/// Nvvm architecture.
///
/// The following table indicates which `compute_*` values are supported by which CUDA versions.
///
/// ```text
/// -----------------------------------------------------------------------------
///             | Supported `compute_*` values (written vertically)
/// -----------------------------------------------------------------------------
/// CUDA        |                                 1 1 1 1 1 1
/// Toolkit     | 5 5 5 6 6 6 7 7 7 7 8 8 8 8 8 9 0 0 0 1 2 2
/// version     | 0 2 3 0 1 2 0 2 3 5 0 6 7 8 9 0 0 1 3 0 0 1
/// -----------------------------------------------------------------------------
/// 12.[01].0   | b b b b b b b b b b b b - - b b - - - - - -
/// 12.2.0      | b b b b b b b b b b b b - - b a - - - - - -
/// 12.[3456].0 | b b b b b b b b b b b b b - b a - - - - - -
/// 12.8.0      | b b b b b b b b b b b b b - b a a a - - a -
/// 12.9.0      | b b b b b b b b - b b b b - b a f f f - f f
/// 13.0.0      | - - - - - - - - - b b b b b b a f - f f f f
/// -----------------------------------------------------------------------------  
/// Legend:
/// - 'b': baseline features only
/// - 'a': baseline + architecture-specific features
/// - 'f': baseline + architecture-specific + family-specific features
///
/// Note: there was no 12.7 release.
/// ```
///
/// For example, CUDA 12.9.0 supports `compute_89`, `compute_90{,a}`, `compute_100{,a,f}`.
///
/// This information is from "PTX Compiler APIs" documents under
/// <https://developer.nvidia.com/cuda-toolkit-archive>, e.g.
/// <https://docs.nvidia.com/cuda/archive/13.0.0/ptx-compiler-api/index.html>. (Adjust the version
/// in that URL as necessary.) Specifically, the `compute-*` values allowed with the `--gpu-name`
/// option.
///
/// # Example
///
/// ```
/// // The default value is `NvvmArch::Compute75`.
/// # use nvvm::NvvmArch;
/// assert_eq!(NvvmArch::default(), NvvmArch::Compute75);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, strum::EnumIter)]
pub enum NvvmArch {
    Compute50,
    Compute52,
    Compute53,
    Compute60,
    Compute61,
    Compute62,
    Compute70,
    Compute72,
    Compute73,
    /// This default value of 7.5 corresponds to Turing and later devices. We default to this
    /// because it is the minimum supported by CUDA 13.0 while being in the middle of the range
    /// supported by CUDA 12.x.
    // WARNING: If you change the default, consider updating the `--target-arch` values used for
    // compiletests in `ci_linux.yml` and `.github/workflows/ci_{linux,windows}.yml`.
    #[default]
    Compute75,
    Compute80,
    Compute86,
    Compute87,
    Compute88,
    Compute89,
    Compute90,
    Compute90a,
    Compute100,
    Compute100f,
    Compute100a,
    Compute101,
    Compute101f,
    Compute101a,
    Compute103,
    Compute103f,
    Compute103a,
    Compute110,
    Compute110f,
    Compute110a,
    Compute120,
    Compute120f,
    Compute120a,
    Compute121,
    Compute121f,
    Compute121a,
}

impl Display for NvvmArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.target_feature())
    }
}

impl FromStr for NvvmArch {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "compute_50" => NvvmArch::Compute50,
            "compute_52" => NvvmArch::Compute52,
            "compute_53" => NvvmArch::Compute53,
            "compute_60" => NvvmArch::Compute60,
            "compute_61" => NvvmArch::Compute61,
            "compute_62" => NvvmArch::Compute62,
            "compute_70" => NvvmArch::Compute70,
            "compute_72" => NvvmArch::Compute72,
            "compute_73" => NvvmArch::Compute73,
            "compute_75" => NvvmArch::Compute75,
            "compute_80" => NvvmArch::Compute80,
            "compute_86" => NvvmArch::Compute86,
            "compute_87" => NvvmArch::Compute87,
            "compute_88" => NvvmArch::Compute88,
            "compute_89" => NvvmArch::Compute89,
            "compute_90" => NvvmArch::Compute90,
            "compute_90a" => NvvmArch::Compute90a,
            "compute_100" => NvvmArch::Compute100,
            "compute_100f" => NvvmArch::Compute100f,
            "compute_100a" => NvvmArch::Compute100a,
            "compute_101" => NvvmArch::Compute101,
            "compute_101f" => NvvmArch::Compute101f,
            "compute_101a" => NvvmArch::Compute101a,
            "compute_103" => NvvmArch::Compute103,
            "compute_103f" => NvvmArch::Compute103f,
            "compute_103a" => NvvmArch::Compute103a,
            "compute_110" => NvvmArch::Compute110,
            "compute_110f" => NvvmArch::Compute110f,
            "compute_110a" => NvvmArch::Compute110a,
            "compute_120" => NvvmArch::Compute120,
            "compute_120f" => NvvmArch::Compute120f,
            "compute_120a" => NvvmArch::Compute120a,
            "compute_121" => NvvmArch::Compute121,
            "compute_121f" => NvvmArch::Compute121f,
            "compute_121a" => NvvmArch::Compute121a,
            _ => return Err("unknown compile target"),
        })
    }
}

impl NvvmArch {
    /// Get the numeric capability value (e.g., 90 for `Compute90` or `Compute90a`).
    pub fn capability_value(&self) -> u32 {
        match self {
            Self::Compute50 => 50,
            Self::Compute52 => 52,
            Self::Compute53 => 53,
            Self::Compute60 => 60,
            Self::Compute61 => 61,
            Self::Compute62 => 62,
            Self::Compute70 => 70,
            Self::Compute72 => 72,
            Self::Compute73 => 73,
            Self::Compute75 => 75,
            Self::Compute80 => 80,
            Self::Compute86 => 86,
            Self::Compute87 => 87,
            Self::Compute88 => 88,
            Self::Compute89 => 89,
            Self::Compute90 => 90,
            Self::Compute90a => 90,
            Self::Compute100 => 100,
            Self::Compute100f => 100,
            Self::Compute100a => 100,
            Self::Compute101 => 101,
            Self::Compute101f => 101,
            Self::Compute101a => 101,
            Self::Compute103 => 103,
            Self::Compute103f => 103,
            Self::Compute103a => 103,
            Self::Compute110 => 110,
            Self::Compute110f => 110,
            Self::Compute110a => 110,
            Self::Compute120 => 120,
            Self::Compute120f => 120,
            Self::Compute120a => 120,
            Self::Compute121 => 121,
            Self::Compute121f => 121,
            Self::Compute121a => 121,
        }
    }

    /// Get the major version number (e.g., 7 for Compute70)
    pub fn major_version(&self) -> u32 {
        self.capability_value() / 10
    }

    /// Get the minor version number (e.g., 5 for Compute75)
    pub fn minor_version(&self) -> u32 {
        self.capability_value() % 10
    }

    /// Get the target feature string (e.g., "compute_50" for `Compute50`, "compute_90a" for
    /// `Compute90a`).
    pub fn target_feature(&self) -> &'static str {
        match self {
            Self::Compute50 => "compute_50",
            Self::Compute52 => "compute_52",
            Self::Compute53 => "compute_53",
            Self::Compute60 => "compute_60",
            Self::Compute61 => "compute_61",
            Self::Compute62 => "compute_62",
            Self::Compute70 => "compute_70",
            Self::Compute72 => "compute_72",
            Self::Compute73 => "compute_73",
            Self::Compute75 => "compute_75",
            Self::Compute80 => "compute_80",
            Self::Compute86 => "compute_86",
            Self::Compute87 => "compute_87",
            Self::Compute88 => "compute_88",
            Self::Compute89 => "compute_89",
            Self::Compute90 => "compute_90",
            Self::Compute90a => "compute_90a",
            Self::Compute100 => "compute_100",
            Self::Compute100f => "compute_100f",
            Self::Compute100a => "compute_100a",
            Self::Compute101 => "compute_101",
            Self::Compute101f => "compute_101f",
            Self::Compute101a => "compute_101a",
            Self::Compute103 => "compute_103",
            Self::Compute103f => "compute_103f",
            Self::Compute103a => "compute_103a",
            Self::Compute110 => "compute_110",
            Self::Compute110f => "compute_110f",
            Self::Compute110a => "compute_110a",
            Self::Compute120 => "compute_120",
            Self::Compute120f => "compute_120f",
            Self::Compute120a => "compute_120a",
            Self::Compute121 => "compute_121",
            Self::Compute121f => "compute_121f",
            Self::Compute121a => "compute_121a",
        }
    }

    /// Gets all target features supported by this compilation target. This effectively answers
    /// the question "for a given compilation target, what architectural features can be used?"
    /// E.g. the "compute_90" compilation target includes features from "compute_80" and earlier.
    /// This set of features does not change over time.
    ///
    /// Note that this is different to the question "for a given compilation target, what devices
    /// can the generated PTX code run on?" E.g. PTX code compiled for the "compute_90" compilation
    /// target can run on devices with compute capability 9.0 and later. This set of devices will
    /// expand over time, as new devices are released.
    ///
    /// # Examples
    ///
    /// ```
    /// use nvvm::NvvmArch::*;
    /// let features = Compute61.all_target_features();
    /// assert_eq!(
    ///     features,
    ///     vec![Compute50, Compute52, Compute53, Compute60, Compute61]
    /// );
    /// ```
    ///
    /// # External resources
    ///
    /// For more details on family and architecture-specific features, see:
    /// <https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/>
    pub fn all_target_features(&self) -> Vec<Self> {
        // All lower-or-equal baseline features are included.
        let included_baseline = |arch: &NvvmArch| {
            arch.is_base_variant() && arch.capability_value() <= self.capability_value()
        };

        // All lower-or-equal-with-same-major-version family features are included.
        let included_family = |arch: &NvvmArch| {
            arch.is_family_variant()
                && arch.major_version() == self.major_version()
                && arch.minor_version() <= self.minor_version()
        };

        if self.is_architecture_variant() {
            // Architecture-specific ('a' suffix) features include:
            // - all lower-or-equal baseline features
            // - all lower-or-equal-with-same-major-version family features
            // - itself
            NvvmArch::iter()
                .filter(|arch| included_baseline(arch) || included_family(arch) || arch == self)
                .collect()
        } else if self.is_family_variant() {
            // Family-specific ('f' suffix) features include:
            // - all lower-or-equal baseline features
            // - all lower-or-equal-with-same-major-version family features
            NvvmArch::iter()
                .filter(|arch| included_baseline(arch) || included_family(arch))
                .collect()
        } else {
            // Baseline (no suffix) features include:
            // - all lower-or-equal baseline features
            NvvmArch::iter().filter(included_baseline).collect()
        }
    }

    /// Check if this architecture is a base variant (no suffix)
    pub fn is_base_variant(&self) -> bool {
        !self
            .target_feature()
            .ends_with(|c| char::is_ascii_alphabetic(&c))
    }

    /// Check if this architecture is a family-specific variant (f suffix)
    /// Family-specific features are supported across devices within the same major compute capability
    pub fn is_family_variant(&self) -> bool {
        self.target_feature().ends_with('f')
    }

    /// Check if this architecture is an architecture-specific variant (a suffix)
    /// Architecture-specific features are locked to that exact compute capability only
    pub fn is_architecture_variant(&self) -> bool {
        self.target_feature().ends_with('a')
    }

    /// Get the base architecture for this variant (strips f/a suffix if present)
    pub fn base_architecture(&self) -> Self {
        match self {
            // Already base variants
            Self::Compute50
            | Self::Compute52
            | Self::Compute53
            | Self::Compute60
            | Self::Compute61
            | Self::Compute62
            | Self::Compute70
            | Self::Compute72
            | Self::Compute73
            | Self::Compute75
            | Self::Compute80
            | Self::Compute86
            | Self::Compute87
            | Self::Compute88
            | Self::Compute89
            | Self::Compute90
            | Self::Compute100
            | Self::Compute101
            | Self::Compute103
            | Self::Compute110
            | Self::Compute120
            | Self::Compute121 => *self,

            // Family-specific variants
            Self::Compute100f => Self::Compute100,
            Self::Compute101f => Self::Compute101,
            Self::Compute103f => Self::Compute103,
            Self::Compute110f => Self::Compute110,
            Self::Compute120f => Self::Compute120,
            Self::Compute121f => Self::Compute121,

            // Architecture-specific variants
            Self::Compute90a => Self::Compute90,
            Self::Compute100a => Self::Compute100,
            Self::Compute101a => Self::Compute101,
            Self::Compute103a => Self::Compute103,
            Self::Compute110a => Self::Compute110,
            Self::Compute120a => Self::Compute120,
            Self::Compute121a => Self::Compute121,
        }
    }

    /// Get all available variants for the same base architecture (including the base)
    pub fn get_variants(&self) -> Vec<Self> {
        let base = self.base_architecture();
        let base_value = base.capability_value();

        NvvmArch::iter()
            .filter(|arch| arch.capability_value() == base_value)
            .collect()
    }

    /// Get all available variants for a given capability value
    pub fn variants_for_capability(capability: u32) -> Vec<Self> {
        NvvmArch::iter()
            .filter(|arch| arch.capability_value() == capability)
            .collect()
    }
}

pub struct NvvmProgram {
    raw: nvvm_sys::nvvmProgram,
}

impl Drop for NvvmProgram {
    fn drop(&mut self) {
        unsafe {
            nvvm_sys::nvvmDestroyProgram(&mut self.raw as *mut _)
                .to_result()
                .expect("failed to destroy nvvm program");
        }
    }
}

impl NvvmProgram {
    /// Make a new NVVM program.
    pub fn new() -> Result<Self, NvvmError> {
        unsafe {
            let mut raw = MaybeUninit::uninit();
            nvvm_sys::nvvmCreateProgram(raw.as_mut_ptr()).to_result()?;
            Ok(Self {
                raw: raw.assume_init(),
            })
        }
    }

    /// Compile this program into PTX assembly bytes (they *should* be ascii per the PTX ISA ref but they are returned as bytes to be safe).
    ///
    pub fn compile(&self, options: &[NvvmOption]) -> Result<Vec<u8>, NvvmError> {
        unsafe {
            let options = options.iter().map(|x| format!("{x}\0")).collect::<Vec<_>>();
            let mut options_ptr = options
                .iter()
                .map(|x| x.as_ptr().cast())
                .collect::<Vec<_>>();

            nvvm_sys::nvvmCompileProgram(self.raw, options.len() as i32, options_ptr.as_mut_ptr())
                .to_result()?;
            let mut size = 0;
            nvvm_sys::nvvmGetCompiledResultSize(self.raw, &mut size as *mut usize as *mut _)
                .to_result()?;
            let mut buf: Vec<u8> = Vec::with_capacity(size);
            nvvm_sys::nvvmGetCompiledResult(self.raw, buf.as_mut_ptr().cast()).to_result()?;
            buf.set_len(size);
            // 𝖇𝖆𝖓𝖎𝖘𝖍 𝖙𝖍𝖞 𝖓𝖚𝖑
            buf.pop();
            Ok(buf)
        }
    }

    /// Add a bitcode module to this nvvm program.
    pub fn add_module(&self, bitcode: &[u8], name: String) -> Result<(), NvvmError> {
        unsafe {
            let cstring = CString::new(name).expect("module name with nul");
            nvvm_sys::nvvmAddModuleToProgram(
                self.raw,
                bitcode.as_ptr().cast(),
                bitcode.len(),
                cstring.as_ptr(),
            )
            .to_result()
        }
    }

    /// Add a bitcode module lazily to this nvvm program. This means that a symbol in this module
    /// is only loaded if it is used by a previous module. According to libnvvm docs, this also
    /// makes the symbols internal to the NVVM IR module, allowing for further optimizations.
    ///
    /// **Do not feed LLVM IR to this method, [`add_module`](Self::add_module) seems to allow it for now, but
    /// it yields an empty ptx file if given to this method**
    pub fn add_lazy_module(&self, bitcode: &[u8], name: String) -> Result<(), NvvmError> {
        unsafe {
            let cstring = CString::new(name).expect("module name with nul");
            nvvm_sys::nvvmLazyAddModuleToProgram(
                self.raw,
                bitcode.as_ptr().cast(),
                bitcode.len(),
                cstring.as_ptr(),
            )
            .to_result()
        }
    }

    /// Get the compiler/verifier log message. This includes any errors that may have happened during compilation
    /// or during verification as well as any warnings. If you are having trouble with your program yielding a
    /// compilation error, looking at this log *after* attempting compilation should help.
    ///
    /// Returns `None` if the log is empty and automatically strips off the nul at the end of the log.
    pub fn compiler_log(&self) -> Result<Option<String>, NvvmError> {
        unsafe {
            let mut size = MaybeUninit::uninit();
            nvvm_sys::nvvmGetProgramLogSize(self.raw, size.as_mut_ptr()).to_result()?;
            let size = size.assume_init();
            let mut buf: Vec<u8> = Vec::with_capacity(size);
            nvvm_sys::nvvmGetProgramLog(self.raw, buf.as_mut_ptr().cast()).to_result()?;
            buf.set_len(size);
            // 𝖇𝖆𝖓𝖎𝖘𝖍 𝖙𝖍𝖞 𝖓𝖚𝖑
            buf.pop();
            let string = String::from_utf8(buf).expect("nvvm compiler log was not utf8");
            Ok(Some(string).filter(|s| !s.is_empty()))
        }
    }

    /// Verify the program without actually compiling it. In the case of invalid IR, you can find
    /// more detailed error info by calling [`compiler_log`](Self::compiler_log).
    pub fn verify(&self) -> Result<(), NvvmError> {
        unsafe { nvvm_sys::nvvmVerifyProgram(self.raw, 0, null_mut()).to_result() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;
    use NvvmArch::*;

    #[test]
    fn nvvm_arch_capability_value() {
        assert_eq!(Compute50.capability_value(), 50);
        assert_eq!(Compute52.capability_value(), 52);
        assert_eq!(Compute53.capability_value(), 53);
        assert_eq!(Compute60.capability_value(), 60);
        assert_eq!(Compute61.capability_value(), 61);
        assert_eq!(Compute62.capability_value(), 62);
        assert_eq!(Compute70.capability_value(), 70);
        assert_eq!(Compute72.capability_value(), 72);
        assert_eq!(Compute73.capability_value(), 73);
        assert_eq!(Compute75.capability_value(), 75);
        assert_eq!(Compute80.capability_value(), 80);
        assert_eq!(Compute86.capability_value(), 86);
        assert_eq!(Compute87.capability_value(), 87);
        assert_eq!(Compute88.capability_value(), 88);
        assert_eq!(Compute89.capability_value(), 89);
        assert_eq!(Compute90.capability_value(), 90);
        assert_eq!(Compute90a.capability_value(), 90);
        assert_eq!(Compute100.capability_value(), 100);
        assert_eq!(Compute100f.capability_value(), 100);
        assert_eq!(Compute100a.capability_value(), 100);
        assert_eq!(Compute101.capability_value(), 101);
        assert_eq!(Compute101f.capability_value(), 101);
        assert_eq!(Compute101a.capability_value(), 101);
        assert_eq!(Compute103.capability_value(), 103);
        assert_eq!(Compute103f.capability_value(), 103);
        assert_eq!(Compute103a.capability_value(), 103);
        assert_eq!(Compute110.capability_value(), 110);
        assert_eq!(Compute110f.capability_value(), 110);
        assert_eq!(Compute110a.capability_value(), 110);
        assert_eq!(Compute120.capability_value(), 120);
        assert_eq!(Compute120f.capability_value(), 120);
        assert_eq!(Compute120a.capability_value(), 120);
    }

    #[test]
    fn nvvm_arch_major_minor_version() {
        // Test major/minor version extraction
        assert_eq!(Compute53.major_version(), 5);
        assert_eq!(Compute53.minor_version(), 3);

        assert_eq!(Compute70.major_version(), 7);
        assert_eq!(Compute70.minor_version(), 0);

        assert_eq!(Compute121.major_version(), 12);
        assert_eq!(Compute121.minor_version(), 1);

        // Suffixes don't affect version numbers
        assert_eq!(Compute100f.major_version(), 10);
        assert_eq!(Compute100f.minor_version(), 0);

        assert_eq!(Compute90a.major_version(), 9);
        assert_eq!(Compute90a.minor_version(), 0);
    }

    #[test]
    fn nvvm_arch_target_feature() {
        // Test baseline features
        assert_eq!(Compute50.target_feature(), "compute_50");
        assert_eq!(Compute61.target_feature(), "compute_61");
        assert_eq!(Compute90.target_feature(), "compute_90");
        assert_eq!(Compute100.target_feature(), "compute_100");
        assert_eq!(Compute120.target_feature(), "compute_120");

        // Test family-specfic ('f') features
        assert_eq!(Compute100f.target_feature(), "compute_100f");
        assert_eq!(Compute101f.target_feature(), "compute_101f");
        assert_eq!(Compute103f.target_feature(), "compute_103f");
        assert_eq!(Compute120f.target_feature(), "compute_120f");
        assert_eq!(Compute121f.target_feature(), "compute_121f");

        // Test architecture-specific ('a') features
        assert_eq!(Compute90a.target_feature(), "compute_90a");
        assert_eq!(Compute100a.target_feature(), "compute_100a");
        assert_eq!(Compute101a.target_feature(), "compute_101a");
        assert_eq!(Compute103a.target_feature(), "compute_103a");
        assert_eq!(Compute120a.target_feature(), "compute_120a");
        assert_eq!(Compute121a.target_feature(), "compute_121a");
    }

    #[test]
    fn nvvm_arch_all_target_features() {
        assert_eq!(Compute50.all_target_features(), vec![Compute50]);

        assert_eq!(
            Compute70.all_target_features(),
            vec![Compute50, Compute52, Compute53, Compute60, Compute61, Compute62, Compute70]
        );

        assert_eq!(
            Compute90.all_target_features(),
            vec![
                Compute50, Compute52, Compute53, Compute60, Compute61, Compute62, Compute70,
                Compute72, Compute73, Compute75, Compute80, Compute86, Compute87, Compute88,
                Compute89, Compute90,
            ]
        );

        assert_eq!(
            Compute90a.all_target_features(),
            vec![
                Compute50, Compute52, Compute53, Compute60, Compute61, Compute62, Compute70,
                Compute72, Compute73, Compute75, Compute80, Compute86, Compute87, Compute88,
                Compute89, Compute90, Compute90a,
            ]
        );

        assert_eq!(
            Compute100a.all_target_features(),
            vec![
                Compute50,
                Compute52,
                Compute53,
                Compute60,
                Compute61,
                Compute62,
                Compute70,
                Compute72,
                Compute73,
                Compute75,
                Compute80,
                Compute86,
                Compute87,
                Compute88,
                Compute89,
                Compute90,
                Compute100,
                Compute100f,
                Compute100a,
            ]
        );

        assert_eq!(
            Compute100f.all_target_features(),
            vec![
                Compute50,
                Compute52,
                Compute53,
                Compute60,
                Compute61,
                Compute62,
                Compute70,
                Compute72,
                Compute73,
                Compute75,
                Compute80,
                Compute86,
                Compute87,
                Compute88,
                Compute89,
                Compute90,
                Compute100,
                Compute100f,
            ]
        );

        assert_eq!(
            Compute101a.all_target_features(),
            vec![
                Compute50,
                Compute52,
                Compute53,
                Compute60,
                Compute61,
                Compute62,
                Compute70,
                Compute72,
                Compute73,
                Compute75,
                Compute80,
                Compute86,
                Compute87,
                Compute88,
                Compute89,
                Compute90,
                Compute100,
                Compute100f,
                Compute101,
                Compute101f,
                Compute101a,
            ]
        );

        assert_eq!(
            Compute101f.all_target_features(),
            vec![
                Compute50,
                Compute52,
                Compute53,
                Compute60,
                Compute61,
                Compute62,
                Compute70,
                Compute72,
                Compute73,
                Compute75,
                Compute80,
                Compute86,
                Compute87,
                Compute88,
                Compute89,
                Compute90,
                Compute100,
                Compute100f,
                Compute101,
                Compute101f,
            ]
        );

        assert_eq!(
            Compute120.all_target_features(),
            vec![
                Compute50, Compute52, Compute53, Compute60, Compute61, Compute62, Compute70,
                Compute72, Compute73, Compute75, Compute80, Compute86, Compute87, Compute88,
                Compute89, Compute90, Compute100, Compute101, Compute103, Compute110, Compute120,
            ]
        );

        assert_eq!(
            Compute120f.all_target_features(),
            vec![
                Compute50,
                Compute52,
                Compute53,
                Compute60,
                Compute61,
                Compute62,
                Compute70,
                Compute72,
                Compute73,
                Compute75,
                Compute80,
                Compute86,
                Compute87,
                Compute88,
                Compute89,
                Compute90,
                Compute100,
                Compute101,
                Compute103,
                Compute110,
                Compute120,
                Compute120f,
            ]
        );

        assert_eq!(
            Compute120a.all_target_features(),
            vec![
                Compute50,
                Compute52,
                Compute53,
                Compute60,
                Compute61,
                Compute62,
                Compute70,
                Compute72,
                Compute73,
                Compute75,
                Compute80,
                Compute86,
                Compute87,
                Compute88,
                Compute89,
                Compute90,
                Compute100,
                Compute101,
                Compute103,
                Compute110,
                Compute120,
                Compute120f,
                Compute120a,
            ]
        );
    }

    #[test]
    fn options_parse_correctly() {
        use NvvmOption::{self, *};

        let ok = |opt, val| assert_eq!(NvvmOption::from_str(opt), Ok(val));
        let err = |opt, s: &str| assert_eq!(NvvmOption::from_str(opt), Err(s.to_string()));

        ok("-arch=compute_50", Arch(Compute50));
        ok("-arch=compute_52", Arch(Compute52));
        ok("-arch=compute_53", Arch(Compute53));
        ok("-arch=compute_60", Arch(Compute60));
        ok("-arch=compute_61", Arch(Compute61));
        ok("-arch=compute_62", Arch(Compute62));
        ok("-arch=compute_70", Arch(Compute70));
        ok("-arch=compute_72", Arch(Compute72));
        ok("-arch=compute_73", Arch(Compute73));
        ok("-arch=compute_75", Arch(Compute75));
        ok("-arch=compute_80", Arch(Compute80));
        ok("-arch=compute_86", Arch(Compute86));
        ok("-arch=compute_87", Arch(Compute87));
        ok("-arch=compute_88", Arch(Compute88));
        ok("-arch=compute_89", Arch(Compute89));
        ok("-arch=compute_90", Arch(Compute90));
        ok("-arch=compute_90a", Arch(Compute90a));
        ok("-arch=compute_100", Arch(Compute100));
        ok("-arch=compute_100f", Arch(Compute100f));
        ok("-arch=compute_100a", Arch(Compute100a));
        ok("-arch=compute_101", Arch(Compute101));
        ok("-arch=compute_101f", Arch(Compute101f));
        ok("-arch=compute_101a", Arch(Compute101a));
        ok("-arch=compute_110", Arch(Compute110));
        ok("-arch=compute_110f", Arch(Compute110f));
        ok("-arch=compute_110a", Arch(Compute110a));
        ok("-arch=compute_120", Arch(Compute120));
        ok("-arch=compute_120f", Arch(Compute120f));
        ok("-arch=compute_120a", Arch(Compute120a));
        ok("-arch=compute_121", Arch(Compute121));
        ok("-arch=compute_121f", Arch(Compute121f));
        ok("-arch=compute_121a", Arch(Compute121a));
        ok("-fma=0", NoFmaContraction);
        ok("-ftz=1", Ftz);
        ok("-g", GenDebugInfo);
        ok("-generate-line-info", GenLineInfo);
        ok("-opt=0", NoOpts);
        ok("-prec-div=0", FastDiv);
        ok("-prec-sqrt=0", FastSqrt);

        err("blah", "unknown option: blah");
        err("-aardvark", "unknown option: -aardvark");
        err("-arch=compute75", "unknown -arch value: compute75");
        err("-arch=compute_10", "unknown -arch value: compute_10");
        err("-arch=compute_100x", "unknown -arch value: compute_100x");
        err("-opt=3", "-opt=3 is the default");
        err("-opt=99", "unknown -opt value: 99");
    }

    #[test]
    fn nvvm_arch_variant_checks() {
        // Base variants
        assert!(Compute90.is_base_variant());
        assert!(Compute120.is_base_variant());
        assert!(!Compute90.is_family_variant());
        assert!(!Compute90.is_architecture_variant());

        // Family-specific variants
        assert!(Compute120f.is_family_variant());
        assert!(!Compute120f.is_base_variant());
        assert!(!Compute120f.is_architecture_variant());

        // Architecture-specific variants
        assert!(Compute90a.is_architecture_variant());
        assert!(Compute120a.is_architecture_variant());
        assert!(!Compute90a.is_base_variant());
        assert!(!Compute90a.is_family_variant());
    }

    #[test]
    fn nvvm_arch_base_architecture() {
        // Base variants return themselves
        assert_eq!(Compute90.base_architecture(), Compute90);
        assert_eq!(Compute120.base_architecture(), Compute120);

        // Family-specific variants return base
        assert_eq!(Compute120f.base_architecture(), Compute120);
        assert_eq!(Compute101f.base_architecture(), Compute101);

        // Architecture variants return base
        assert_eq!(Compute90a.base_architecture(), Compute90);
        assert_eq!(Compute120a.base_architecture(), Compute120);
    }

    #[test]
    fn nvvm_arch_get_variants() {
        // Architecture with only base variant
        let compute80_variants = Compute80.get_variants();
        assert_eq!(compute80_variants, vec![Compute80]);

        // Architecture with architecture and base variants
        assert_eq!(Compute90.get_variants(), vec![Compute90, Compute90a]);

        // Architecture with all three variants
        let expected120 = vec![Compute120, Compute120f, Compute120a];
        assert_eq!(Compute120.get_variants(), expected120);
        assert_eq!(Compute120f.get_variants(), expected120);
        assert_eq!(Compute120a.get_variants(), expected120);
    }

    #[test]
    fn nvvm_arch_variants_for_capability() {
        // Capability with single variant
        assert_eq!(NvvmArch::variants_for_capability(75), vec![Compute75]);

        // Capability with multiple variants
        assert_eq!(
            NvvmArch::variants_for_capability(101),
            vec![Compute101, Compute101f, Compute101a]
        );

        // Non-existent capability
        assert!(NvvmArch::variants_for_capability(999).is_empty());
    }
}
