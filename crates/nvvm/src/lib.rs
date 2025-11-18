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
        // according to the docs this cant fail
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
        // according to the docs this cant fail
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
        // according to the docs this cant fail
        let _ = nvvm_sys::nvvmVersion(major.as_mut_ptr(), minor.as_mut_ptr());
        (major.assume_init(), minor.assume_init())
    }
}

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
    // InvalidProgram isnt handled because its not possible
    // to get an invalid program handle through this safe api
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
                if !slice.starts_with("compute_") {
                    return Err(format!("unknown -arch value: {slice}"));
                }
                let arch_num = &slice[8..];
                let arch = match arch_num {
                    "35" => NvvmArch::Compute35,
                    "37" => NvvmArch::Compute37,
                    "50" => NvvmArch::Compute50,
                    "52" => NvvmArch::Compute52,
                    "53" => NvvmArch::Compute53,
                    "60" => NvvmArch::Compute60,
                    "61" => NvvmArch::Compute61,
                    "62" => NvvmArch::Compute62,
                    "70" => NvvmArch::Compute70,
                    "72" => NvvmArch::Compute72,
                    "75" => NvvmArch::Compute75,
                    "80" => NvvmArch::Compute80,
                    "86" => NvvmArch::Compute86,
                    "87" => NvvmArch::Compute87,
                    "89" => NvvmArch::Compute89,
                    "90" => NvvmArch::Compute90,
                    "90a" => NvvmArch::Compute90a,
                    "100" => NvvmArch::Compute100,
                    "100f" => NvvmArch::Compute100f,
                    "100a" => NvvmArch::Compute100a,
                    "101" => NvvmArch::Compute101,
                    "101f" => NvvmArch::Compute101f,
                    "101a" => NvvmArch::Compute101a,
                    "103" => NvvmArch::Compute103,
                    "103f" => NvvmArch::Compute103f,
                    "103a" => NvvmArch::Compute103a,
                    "120" => NvvmArch::Compute120,
                    "120f" => NvvmArch::Compute120f,
                    "120a" => NvvmArch::Compute120a,
                    "121" => NvvmArch::Compute121,
                    "121f" => NvvmArch::Compute121f,
                    "121a" => NvvmArch::Compute121a,
                    _ => return Err(format!("unknown -arch=compute_NN value: {arch_num}")),
                };
                Self::Arch(arch)
            }
            _ => return Err(format!("unknown option: {s}")),
        })
    }
}

/// Nvvm architecture, default is `Compute52`
#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::EnumIter)]
pub enum NvvmArch {
    Compute35,
    Compute37,
    Compute50,
    Compute52,
    Compute53,
    Compute60,
    Compute61,
    Compute62,
    Compute70,
    Compute72,
    Compute75,
    Compute80,
    Compute86,
    Compute87,
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
    Compute120,
    Compute120f,
    Compute120a,
    Compute121,
    Compute121f,
    Compute121a,
}

impl Display for NvvmArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let raw = format!("{self:?}").to_ascii_lowercase();
        // Handle architectures with suffixes (e.g., Compute90a -> compute_90a)
        if let Some(pos) = raw.find(|c: char| c.is_ascii_digit()) {
            let (prefix, rest) = raw.split_at(pos);
            write!(f, "{prefix}_{rest}")
        } else {
            // Fallback for unexpected format
            f.write_str(&raw)
        }
    }
}

impl Default for NvvmArch {
    fn default() -> Self {
        Self::Compute52
    }
}

impl NvvmArch {
    /// Get the numeric capability value (e.g., 35 for Compute35)
    pub fn capability_value(&self) -> u32 {
        match self {
            Self::Compute35 => 35,
            Self::Compute37 => 37,
            Self::Compute50 => 50,
            Self::Compute52 => 52,
            Self::Compute53 => 53,
            Self::Compute60 => 60,
            Self::Compute61 => 61,
            Self::Compute62 => 62,
            Self::Compute70 => 70,
            Self::Compute72 => 72,
            Self::Compute75 => 75,
            Self::Compute80 => 80,
            Self::Compute86 => 86,
            Self::Compute87 => 87,
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

    /// Get the target feature string (e.g., "compute_35" for Compute35, "compute_90a" for Compute90a)
    pub fn target_feature(&self) -> String {
        match self {
            Self::Compute35 => "compute_35".to_string(),
            Self::Compute37 => "compute_37".to_string(),
            Self::Compute50 => "compute_50".to_string(),
            Self::Compute52 => "compute_52".to_string(),
            Self::Compute53 => "compute_53".to_string(),
            Self::Compute60 => "compute_60".to_string(),
            Self::Compute61 => "compute_61".to_string(),
            Self::Compute62 => "compute_62".to_string(),
            Self::Compute70 => "compute_70".to_string(),
            Self::Compute72 => "compute_72".to_string(),
            Self::Compute75 => "compute_75".to_string(),
            Self::Compute80 => "compute_80".to_string(),
            Self::Compute86 => "compute_86".to_string(),
            Self::Compute87 => "compute_87".to_string(),
            Self::Compute89 => "compute_89".to_string(),
            Self::Compute90 => "compute_90".to_string(),
            Self::Compute90a => "compute_90a".to_string(),
            Self::Compute100 => "compute_100".to_string(),
            Self::Compute100f => "compute_100f".to_string(),
            Self::Compute100a => "compute_100a".to_string(),
            Self::Compute101 => "compute_101".to_string(),
            Self::Compute101f => "compute_101f".to_string(),
            Self::Compute101a => "compute_101a".to_string(),
            Self::Compute103 => "compute_103".to_string(),
            Self::Compute103f => "compute_103f".to_string(),
            Self::Compute103a => "compute_103a".to_string(),
            Self::Compute120 => "compute_120".to_string(),
            Self::Compute120f => "compute_120f".to_string(),
            Self::Compute120a => "compute_120a".to_string(),
            Self::Compute121 => "compute_121".to_string(),
            Self::Compute121f => "compute_121f".to_string(),
            Self::Compute121a => "compute_121a".to_string(),
        }
    }

    /// Get all target features up to and including this architecture.
    ///
    /// # PTX Forward-Compatibility Rules (per NVIDIA documentation):
    ///
    /// - **No suffix** (compute_XX): PTX is forward-compatible across all future architectures.
    ///   Example: compute_70 runs on CC 7.0, 8.x, 9.x, 10.x, 12.x, and all future GPUs.
    ///
    /// - **Family-specific 'f' suffix** (compute_XXf): Forward-compatible within the same major
    ///   version family. Supports devices with same major CC and equal or higher minor CC.
    ///   Example: compute_100f runs on CC 10.0, 10.3, and future 10.x devices, but NOT on 11.x.
    ///
    /// - **Architecture-specific 'a' suffix** (compute_XXa): The code only runs on GPUs of that
    ///   specific CC and no others. No forward or backward compatibility whatsoever.
    ///   These features are primarily related to Tensor Core programming.
    ///   Example: compute_100a ONLY runs on CC 10.0, not on 10.3, 10.1, 9.0, or any other version.
    ///
    /// For more details on family and architecture-specific features, see:
    /// <https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/>
    pub fn all_target_features(&self) -> Vec<String> {
        let mut features: Vec<String> = if self.is_architecture_variant() {
            // 'a' variants: include all available instructions for the architecture
            // This means: all base variants up to same version, all 'f' variants with same major and <= minor, plus itself
            let base_features: Vec<String> = NvvmArch::iter()
                .filter(|arch| {
                    arch.is_base_variant() && arch.capability_value() <= self.capability_value()
                })
                .map(|arch| arch.target_feature())
                .collect();

            let family_features: Vec<String> = NvvmArch::iter()
                .filter(|arch| {
                    arch.is_family_variant()
                        && arch.major_version() == self.major_version()
                        && arch.minor_version() <= self.minor_version()
                })
                .map(|arch| arch.target_feature())
                .collect();

            base_features
                .into_iter()
                .chain(family_features)
                .chain(std::iter::once(self.target_feature()))
                .collect()
        } else if self.is_family_variant() {
            // 'f' variants: same major version with equal or higher minor version
            NvvmArch::iter()
                .filter(|arch| {
                    // Include base variants with same major and >= minor version
                    arch.is_base_variant()
                        && arch.major_version() == self.major_version()
                        && arch.minor_version() >= self.minor_version()
                })
                .map(|arch| arch.target_feature())
                .chain(std::iter::once(self.target_feature())) // Add the 'f' variant itself
                .collect()
        } else {
            // Base variants: all base architectures from lower or equal versions
            NvvmArch::iter()
                .filter(|arch| {
                    arch.is_base_variant() && arch.capability_value() <= self.capability_value()
                })
                .map(|arch| arch.target_feature())
                .collect()
        };

        features.sort();
        features
    }

    /// Create an iterator over all architectures from Compute35 up to and including this one
    pub fn iter_up_to(&self) -> impl Iterator<Item = Self> {
        let current = self.capability_value();
        NvvmArch::iter().filter(move |arch| arch.capability_value() <= current)
    }

    /// Check if this architecture is a base variant (no suffix)
    pub fn is_base_variant(&self) -> bool {
        let feature = self.target_feature();
        // A base variant doesn't end with any letter suffix
        !feature
            .chars()
            .last()
            .is_some_and(|c| c.is_ascii_alphabetic())
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
            Self::Compute35
            | Self::Compute37
            | Self::Compute50
            | Self::Compute52
            | Self::Compute53
            | Self::Compute60
            | Self::Compute61
            | Self::Compute62
            | Self::Compute70
            | Self::Compute72
            | Self::Compute75
            | Self::Compute80
            | Self::Compute86
            | Self::Compute87
            | Self::Compute89
            | Self::Compute90
            | Self::Compute100
            | Self::Compute101
            | Self::Compute103
            | Self::Compute120
            | Self::Compute121 => *self,

            // Family-specific variants
            Self::Compute100f => Self::Compute100,
            Self::Compute101f => Self::Compute101,
            Self::Compute103f => Self::Compute103,
            Self::Compute120f => Self::Compute120,
            Self::Compute121f => Self::Compute121,

            // Architecture-specific variants
            Self::Compute90a => Self::Compute90,
            Self::Compute100a => Self::Compute100,
            Self::Compute101a => Self::Compute101,
            Self::Compute103a => Self::Compute103,
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
    use std::str::FromStr;

    #[test]
    fn nvvm_arch_capability_value() {
        use crate::NvvmArch;

        assert_eq!(NvvmArch::Compute35.capability_value(), 35);
        assert_eq!(NvvmArch::Compute37.capability_value(), 37);
        assert_eq!(NvvmArch::Compute50.capability_value(), 50);
        assert_eq!(NvvmArch::Compute52.capability_value(), 52);
        assert_eq!(NvvmArch::Compute53.capability_value(), 53);
        assert_eq!(NvvmArch::Compute60.capability_value(), 60);
        assert_eq!(NvvmArch::Compute61.capability_value(), 61);
        assert_eq!(NvvmArch::Compute62.capability_value(), 62);
        assert_eq!(NvvmArch::Compute70.capability_value(), 70);
        assert_eq!(NvvmArch::Compute72.capability_value(), 72);
        assert_eq!(NvvmArch::Compute75.capability_value(), 75);
        assert_eq!(NvvmArch::Compute80.capability_value(), 80);
        assert_eq!(NvvmArch::Compute86.capability_value(), 86);
        assert_eq!(NvvmArch::Compute87.capability_value(), 87);
        assert_eq!(NvvmArch::Compute89.capability_value(), 89);
        assert_eq!(NvvmArch::Compute90.capability_value(), 90);
    }

    #[test]
    fn nvvm_arch_major_minor_version() {
        use crate::NvvmArch;

        // Test major/minor version extraction
        assert_eq!(NvvmArch::Compute35.major_version(), 3);
        assert_eq!(NvvmArch::Compute35.minor_version(), 5);

        assert_eq!(NvvmArch::Compute70.major_version(), 7);
        assert_eq!(NvvmArch::Compute70.minor_version(), 0);

        assert_eq!(NvvmArch::Compute121.major_version(), 12);
        assert_eq!(NvvmArch::Compute121.minor_version(), 1);

        // Suffixes don't affect version numbers
        assert_eq!(NvvmArch::Compute100f.major_version(), 10);
        assert_eq!(NvvmArch::Compute100f.minor_version(), 0);

        assert_eq!(NvvmArch::Compute90a.major_version(), 9);
        assert_eq!(NvvmArch::Compute90a.minor_version(), 0);
    }

    #[test]
    fn nvvm_arch_target_feature_format_base_variants() {
        use crate::NvvmArch;

        // Test base variants format
        assert_eq!(NvvmArch::Compute35.target_feature(), "compute_35");
        assert_eq!(NvvmArch::Compute61.target_feature(), "compute_61");
        assert_eq!(NvvmArch::Compute90.target_feature(), "compute_90");
        assert_eq!(NvvmArch::Compute100.target_feature(), "compute_100");
        assert_eq!(NvvmArch::Compute120.target_feature(), "compute_120");
    }

    #[test]
    fn nvvm_arch_target_feature_format_family_variants() {
        use crate::NvvmArch;

        // Test family ('f') variants format
        assert_eq!(NvvmArch::Compute100f.target_feature(), "compute_100f");
        assert_eq!(NvvmArch::Compute101f.target_feature(), "compute_101f");
        assert_eq!(NvvmArch::Compute103f.target_feature(), "compute_103f");
        assert_eq!(NvvmArch::Compute120f.target_feature(), "compute_120f");
        assert_eq!(NvvmArch::Compute121f.target_feature(), "compute_121f");
    }

    #[test]
    fn nvvm_arch_target_feature_format_architecture_variants() {
        use crate::NvvmArch;

        // Test architecture ('a') variants format
        assert_eq!(NvvmArch::Compute90a.target_feature(), "compute_90a");
        assert_eq!(NvvmArch::Compute100a.target_feature(), "compute_100a");
        assert_eq!(NvvmArch::Compute101a.target_feature(), "compute_101a");
        assert_eq!(NvvmArch::Compute103a.target_feature(), "compute_103a");
        assert_eq!(NvvmArch::Compute120a.target_feature(), "compute_120a");
        assert_eq!(NvvmArch::Compute121a.target_feature(), "compute_121a");
    }

    #[test]
    fn nvvm_arch_all_target_features_includes_lower_capabilities() {
        use crate::NvvmArch;

        // Compute35 only includes itself
        let compute35_features = NvvmArch::Compute35.all_target_features();
        assert_eq!(compute35_features, vec!["compute_35"]);

        // Compute50 includes all lower base capabilities
        let compute50_features = NvvmArch::Compute50.all_target_features();
        assert_eq!(
            compute50_features,
            vec!["compute_35", "compute_37", "compute_50"]
        );

        // Compute61 includes all lower base capabilities
        let compute61_features = NvvmArch::Compute61.all_target_features();
        assert_eq!(
            compute61_features,
            vec![
                "compute_35",
                "compute_37",
                "compute_50",
                "compute_52",
                "compute_53",
                "compute_60",
                "compute_61"
            ]
        );

        // Test 'a' variant - includes all available instructions for the architecture
        // This means: all base variants up to same version, all 'f' variants with same major and <= minor, plus itself
        let compute90a_features = NvvmArch::Compute90a.all_target_features();
        // Should include all base up to 90
        assert!(compute90a_features.contains(&"compute_35".to_string()));
        assert!(compute90a_features.contains(&"compute_90".to_string()));
        // Should include the 'a' variant itself
        assert!(compute90a_features.contains(&"compute_90a".to_string()));
        // Should NOT include any 'f' variants (90 has no 'f' variants)

        // Test compute100a - should include base variants, and 100f
        let compute100a_features = NvvmArch::Compute100a.all_target_features();
        // Should include all base up to 100
        assert!(compute100a_features.contains(&"compute_90".to_string()));
        assert!(compute100a_features.contains(&"compute_100".to_string()));
        // Should include 100f (same major, <= minor)
        assert!(compute100a_features.contains(&"compute_100f".to_string()));
        // Should NOT include 101f or 103f (higher minor)
        assert!(!compute100a_features.contains(&"compute_101f".to_string()));
        assert!(!compute100a_features.contains(&"compute_103f".to_string()));
        // Should include itself
        assert!(compute100a_features.contains(&"compute_100a".to_string()));

        // Test compute101a
        let compute101a_features = NvvmArch::Compute101a.all_target_features();
        // Should include all base up to 101
        assert!(compute101a_features.contains(&"compute_100".to_string()));
        assert!(compute101a_features.contains(&"compute_101".to_string()));
        // Should include 100f and 101f (same major, <= minor)
        assert!(compute101a_features.contains(&"compute_100f".to_string()));
        assert!(compute101a_features.contains(&"compute_101f".to_string()));
        // Should NOT include 103f (higher minor)
        assert!(!compute101a_features.contains(&"compute_103f".to_string()));
        // Should include itself
        assert!(compute101a_features.contains(&"compute_101a".to_string()));

        // Test 'f' variant - includes same major version with >= minor
        let compute120f_features = NvvmArch::Compute120f.all_target_features();
        assert!(compute120f_features.contains(&"compute_120".to_string()));
        assert!(compute120f_features.contains(&"compute_121".to_string())); // Higher minor included
        assert!(compute120f_features.contains(&"compute_120f".to_string())); // Self included
        assert!(!compute120f_features.contains(&"compute_120a".to_string())); // No 'a' variants
        assert!(!compute120f_features.contains(&"compute_121f".to_string())); // No other 'f' variants
        assert!(!compute120f_features.contains(&"compute_121a".to_string())); // No 'a' variants
                                                                              // Should NOT include different major versions
        assert!(!compute120f_features.contains(&"compute_100".to_string()));
        assert!(!compute120f_features.contains(&"compute_90".to_string()));

        // Test 'f' variant with 100f
        let compute100f_features = NvvmArch::Compute100f.all_target_features();
        assert!(compute100f_features.contains(&"compute_100".to_string())); // Same version base
        assert!(compute100f_features.contains(&"compute_101".to_string())); // Higher minor
        assert!(compute100f_features.contains(&"compute_103".to_string())); // Higher minor
        assert!(compute100f_features.contains(&"compute_100f".to_string())); // Self
        assert!(!compute100f_features.contains(&"compute_101f".to_string())); // No other 'f' variants
        assert!(!compute100f_features.contains(&"compute_90".to_string())); // Different major

        // Test 'f' variant with 101f
        let compute101f_features = NvvmArch::Compute101f.all_target_features();
        assert!(!compute101f_features.contains(&"compute_100".to_string())); // Lower minor NOT included
        assert!(compute101f_features.contains(&"compute_101".to_string())); // Same version base
        assert!(compute101f_features.contains(&"compute_103".to_string())); // Higher minor included
        assert!(compute101f_features.contains(&"compute_101f".to_string())); // Self
        assert!(!compute101f_features.contains(&"compute_101a".to_string())); // No 'a' variants

        // Compute90 includes lower base capabilities
        let compute90_features = NvvmArch::Compute90.all_target_features();
        assert_eq!(
            compute90_features,
            vec![
                "compute_35",
                "compute_37",
                "compute_50",
                "compute_52",
                "compute_53",
                "compute_60",
                "compute_61",
                "compute_62",
                "compute_70",
                "compute_72",
                "compute_75",
                "compute_80",
                "compute_86",
                "compute_87",
                "compute_89",
                "compute_90"
            ]
        );
    }

    #[test]
    fn target_feature_synthesis_supports_conditional_compilation_patterns() {
        use crate::NvvmArch;

        // When targeting Compute61, should enable all lower capabilities
        let features = NvvmArch::Compute61.all_target_features();

        // Should enable compute_60 (for f64 atomics)
        assert!(features.contains(&"compute_60".to_string()));

        // Should enable compute_50 (for 64-bit integer atomics)
        assert!(features.contains(&"compute_50".to_string()));

        // Should enable compute_35 (baseline)
        assert!(features.contains(&"compute_35".to_string()));

        // Should enable the target itself
        assert!(features.contains(&"compute_61".to_string()));

        // Should NOT enable higher capabilities
        assert!(!features.contains(&"compute_62".to_string()));
        assert!(!features.contains(&"compute_70".to_string()));
    }

    #[test]
    fn target_feature_synthesis_enables_correct_cfg_patterns() {
        use crate::NvvmArch;

        // Test that targeting Compute70 enables appropriate cfg patterns
        let features = NvvmArch::Compute70.all_target_features();

        // These should all be true for compute_70 target
        let expected_enabled = [
            "compute_35",
            "compute_37",
            "compute_50",
            "compute_52",
            "compute_53",
            "compute_60",
            "compute_61",
            "compute_62",
            "compute_70",
        ];

        for feature in expected_enabled {
            assert!(
                features.contains(&feature.to_string()),
                "Compute70 should enable {} for cfg(target_feature = \"{}\")",
                feature,
                feature
            );
        }

        // These should NOT be enabled for compute_70 target
        let expected_disabled = ["compute_72", "compute_75", "compute_80", "compute_90"];

        for feature in expected_disabled {
            assert!(
                !features.contains(&feature.to_string()),
                "Compute70 should NOT enable {}",
                feature
            );
        }
    }

    #[test]
    fn nvvm_arch_iter_up_to_includes_only_lower_or_equal() {
        use crate::NvvmArch;

        // Compute35 only includes itself
        let archs: Vec<_> = NvvmArch::Compute35.iter_up_to().collect();
        assert_eq!(archs, vec![NvvmArch::Compute35]);

        // Compute52 includes all up to 52
        let archs: Vec<_> = NvvmArch::Compute52.iter_up_to().collect();
        assert_eq!(
            archs,
            vec![
                NvvmArch::Compute35,
                NvvmArch::Compute37,
                NvvmArch::Compute50,
                NvvmArch::Compute52,
            ]
        );

        // Compute75 includes all up to 75
        let archs: Vec<_> = NvvmArch::Compute75.iter_up_to().collect();
        assert_eq!(
            archs,
            vec![
                NvvmArch::Compute35,
                NvvmArch::Compute37,
                NvvmArch::Compute50,
                NvvmArch::Compute52,
                NvvmArch::Compute53,
                NvvmArch::Compute60,
                NvvmArch::Compute61,
                NvvmArch::Compute62,
                NvvmArch::Compute70,
                NvvmArch::Compute72,
                NvvmArch::Compute75,
            ]
        );
    }

    #[test]
    fn options_parse_correctly() {
        use crate::NvvmArch::*;
        use crate::NvvmOption::{self, *};

        let ok = |opt, val| assert_eq!(NvvmOption::from_str(opt), Ok(val));
        let err = |opt, s: &str| assert_eq!(NvvmOption::from_str(opt), Err(s.to_string()));

        ok("-arch=compute_35", Arch(Compute35));
        ok("-arch=compute_37", Arch(Compute37));
        ok("-arch=compute_50", Arch(Compute50));
        ok("-arch=compute_52", Arch(Compute52));
        ok("-arch=compute_53", Arch(Compute53));
        ok("-arch=compute_60", Arch(Compute60));
        ok("-arch=compute_61", Arch(Compute61));
        ok("-arch=compute_62", Arch(Compute62));
        ok("-arch=compute_70", Arch(Compute70));
        ok("-arch=compute_72", Arch(Compute72));
        ok("-arch=compute_75", Arch(Compute75));
        ok("-arch=compute_80", Arch(Compute80));
        ok("-arch=compute_86", Arch(Compute86));
        ok("-arch=compute_87", Arch(Compute87));
        ok("-arch=compute_89", Arch(Compute89));
        ok("-arch=compute_90", Arch(Compute90));
        ok("-arch=compute_90a", Arch(Compute90a));
        ok("-arch=compute_100", Arch(Compute100));
        ok("-arch=compute_100f", Arch(Compute100f));
        ok("-arch=compute_100a", Arch(Compute100a));
        ok("-arch=compute_101", Arch(Compute101));
        ok("-arch=compute_101f", Arch(Compute101f));
        ok("-arch=compute_101a", Arch(Compute101a));
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
        err("-arch=compute_10", "unknown -arch=compute_NN value: 10");
        err("-arch=compute_100x", "unknown -arch=compute_NN value: 100x");
        err("-opt=3", "-opt=3 is the default");
        err("-opt=99", "unknown -opt value: 99");
    }

    #[test]
    fn nvvm_arch_variant_checks() {
        use crate::NvvmArch;

        // Base variants
        assert!(NvvmArch::Compute90.is_base_variant());
        assert!(NvvmArch::Compute120.is_base_variant());
        assert!(!NvvmArch::Compute90.is_family_variant());
        assert!(!NvvmArch::Compute90.is_architecture_variant());

        // Family-specific variants
        assert!(NvvmArch::Compute120f.is_family_variant());
        assert!(!NvvmArch::Compute120f.is_base_variant());
        assert!(!NvvmArch::Compute120f.is_architecture_variant());

        // Architecture-specific variants
        assert!(NvvmArch::Compute90a.is_architecture_variant());
        assert!(NvvmArch::Compute120a.is_architecture_variant());
        assert!(!NvvmArch::Compute90a.is_base_variant());
        assert!(!NvvmArch::Compute90a.is_family_variant());
    }

    #[test]
    fn nvvm_arch_base_architecture() {
        use crate::NvvmArch;

        // Base variants return themselves
        assert_eq!(NvvmArch::Compute90.base_architecture(), NvvmArch::Compute90);
        assert_eq!(
            NvvmArch::Compute120.base_architecture(),
            NvvmArch::Compute120
        );

        // Floating-point variants return base
        assert_eq!(
            NvvmArch::Compute120f.base_architecture(),
            NvvmArch::Compute120
        );
        assert_eq!(
            NvvmArch::Compute101f.base_architecture(),
            NvvmArch::Compute101
        );

        // Architecture variants return base
        assert_eq!(
            NvvmArch::Compute90a.base_architecture(),
            NvvmArch::Compute90
        );
        assert_eq!(
            NvvmArch::Compute120a.base_architecture(),
            NvvmArch::Compute120
        );
    }

    #[test]
    fn nvvm_arch_get_variants() {
        use crate::NvvmArch;

        // Architecture with only base variant
        let compute80_variants = NvvmArch::Compute80.get_variants();
        assert_eq!(compute80_variants, vec![NvvmArch::Compute80]);

        // Architecture with architecture and base variants
        let mut compute90_variants = NvvmArch::Compute90.get_variants();
        compute90_variants.sort_by_key(|v| format!("{:?}", v));
        assert_eq!(
            compute90_variants,
            vec![NvvmArch::Compute90, NvvmArch::Compute90a]
        );

        // Architecture with all three variants
        let mut compute120_variants = NvvmArch::Compute120.get_variants();
        compute120_variants.sort_by_key(|v| format!("{:?}", v));
        assert_eq!(
            compute120_variants,
            vec![
                NvvmArch::Compute120,
                NvvmArch::Compute120a,
                NvvmArch::Compute120f
            ]
        );

        // Getting variants from a variant returns all variants
        let compute120f_variants = NvvmArch::Compute120f.get_variants();
        assert_eq!(compute120f_variants.len(), 3);
        assert!(compute120f_variants.contains(&NvvmArch::Compute120));
        assert!(compute120f_variants.contains(&NvvmArch::Compute120f));
        assert!(compute120f_variants.contains(&NvvmArch::Compute120a));
    }

    #[test]
    fn nvvm_arch_a_suffix_includes_all_available_instructions() {
        use crate::NvvmArch;

        // Test that 'a' suffix variants include all available instructions for the architecture
        // While they only RUN on exact CC, they enable all base and family features during compilation

        // Test Compute90a
        let features = NvvmArch::Compute90a.all_target_features();
        assert!(features.contains(&"compute_90a".to_string())); // Includes itself
        assert!(features.contains(&"compute_90".to_string())); // Includes base
        assert!(features.contains(&"compute_80".to_string())); // Includes lower versions
        assert!(!features.contains(&"compute_100".to_string())); // Does NOT include higher versions

        // Test Compute100a
        let features = NvvmArch::Compute100a.all_target_features();
        assert!(features.contains(&"compute_100a".to_string())); // Includes itself
        assert!(features.contains(&"compute_100".to_string())); // Includes base
        assert!(features.contains(&"compute_100f".to_string())); // Includes family variant
        assert!(features.contains(&"compute_90".to_string())); // Includes lower base versions
        assert!(!features.contains(&"compute_90a".to_string())); // Does NOT include other 'a' variants
        assert!(!features.contains(&"compute_101f".to_string())); // Does NOT include higher minor family variants

        // Test Compute120a
        let features = NvvmArch::Compute120a.all_target_features();
        assert!(features.contains(&"compute_120a".to_string())); // Includes itself
        assert!(features.contains(&"compute_120".to_string())); // Includes base
        assert!(features.contains(&"compute_120f".to_string())); // Includes family variant (same minor)
        assert!(features.contains(&"compute_100".to_string())); // Includes lower base versions
        assert!(!features.contains(&"compute_121f".to_string())); // Does NOT include higher minor family variants
    }

    #[test]
    fn nvvm_arch_variants_for_capability() {
        use crate::NvvmArch;

        // Capability with single variant
        let compute75_variants = NvvmArch::variants_for_capability(75);
        assert_eq!(compute75_variants, vec![NvvmArch::Compute75]);

        // Capability with multiple variants
        let mut compute101_variants = NvvmArch::variants_for_capability(101);
        compute101_variants.sort_by_key(|v| format!("{:?}", v));
        assert_eq!(
            compute101_variants,
            vec![
                NvvmArch::Compute101,
                NvvmArch::Compute101a,
                NvvmArch::Compute101f
            ]
        );

        // Non-existent capability
        let compute999_variants = NvvmArch::variants_for_capability(999);
        assert!(compute999_variants.is_empty());
    }
}
