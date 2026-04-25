use std::env;
use std::error;
use std::ffi;
use std::fs;
use std::iter;
use std::path;

const CUDA_ROOT_ENVS: &[&str] = &["CUDA_PATH", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR"];
const CUDA_LIBRARY_PATH_ENV: &str = "CUDA_LIBRARY_PATH";

/// Represents the CUDA SDK installation.
#[derive(Debug, Clone)]
pub struct CudaSdk {
    /// The root directory of the CUDA SDK installation, related paths
    /// and versions.
    cuda_root: path::PathBuf,
    cuda_include_paths: Vec<path::PathBuf>,
    cuda_library_paths: Vec<path::PathBuf>,
    driver_version: u32,
    runtime_version: u32,
    /// libNVVM related paths.
    nvvm_include_paths: Vec<path::PathBuf>,
    nvvm_library_paths: Vec<path::PathBuf>,
    libdevice_bitcode_path: path::PathBuf,
}

impl CudaSdk {
    /// Creates a new `CudaSdk` instance by locating the CUDA SDK installation
    /// and parsing versions from various header files.
    ///
    /// # Errors
    /// Returns an error if the CUDA SDK cannot be found or if the versions cannot be parsed.
    pub fn new() -> Result<Self, Box<dyn error::Error>> {
        let cuda_root = Self::find_cuda_root().ok_or("CUDA SDK cannot be found.")?;
        // Retrieve the CUDA related versions.
        let header_path = cuda_root.join("include").join("cuda.h");
        let header_content = fs::read_to_string(header_path)?;
        let driver_version = Self::parse_driver_version(header_content.as_str())?;
        let header_path = cuda_root.join("include").join("cuda_runtime_api.h");
        let header_content = fs::read_to_string(header_path)?;
        let runtime_version = Self::parse_runtime_version(header_content.as_str())?;
        // Retrieve the CUDA include paths and library paths.
        let cuda_include_paths = vec![cuda_root.join("include")];
        let cuda_library_paths = Self::find_cuda_library_dirs(cuda_root.as_path())?;
        // Retrieve the NVVM related paths.
        let nvvm_include_paths = Self::find_nvvm_include_dirs(cuda_root.as_path())?;
        let nvvm_library_paths = Self::find_nvvm_library_dirs(cuda_root.as_path())?;
        let libdevice_bitcode_path = cuda_root
            .join("nvvm")
            .join("libdevice")
            .join("libdevice.10.bc");
        if !libdevice_bitcode_path.is_file() {
            return Err(format!(
                "libdevice bitcode file not found: {}.",
                libdevice_bitcode_path.display()
            )
            .into());
        }

        Ok(Self {
            cuda_root,
            cuda_include_paths,
            cuda_library_paths,
            driver_version,
            runtime_version,
            nvvm_include_paths,
            nvvm_library_paths,
            libdevice_bitcode_path,
        })
    }

    /// Returns the root path of the CUDA SDK installation.
    pub fn cuda_root(&self) -> &path::Path {
        self.cuda_root.as_path()
    }

    /// Returns the full version of the CUDA SDK as an integer.
    /// For example, CUDA 11.8 is represented as 11080.
    pub fn driver_version(&self) -> u32 {
        self.driver_version
    }

    /// Returns the major version of the CUDA SDK.
    /// For example, for CUDA 11.8, this method returns 11.
    pub fn driver_version_major(&self) -> u32 {
        self.driver_version / 1000
    }

    /// Returns the minor version of the CUDA SDK.
    /// For example, for CUDA 11.8, this method returns 8.
    pub fn driver_version_minor(&self) -> u32 {
        self.driver_version / 10 % 100
    }

    /// Returns the CUDA runtime version which is defined in
    /// `cuda_runtime_api.h` file as: `#define CUDART_VERSION 12080`
    pub fn runtime_version(&self) -> u32 {
        self.runtime_version
    }

    pub fn cuda_include_paths(&self) -> &[path::PathBuf] {
        &self.cuda_include_paths
    }

    pub fn cuda_library_paths(&self) -> &[path::PathBuf] {
        &self.cuda_library_paths
    }

    pub fn nvvm_include_paths(&self) -> &[path::PathBuf] {
        &self.nvvm_include_paths
    }

    pub fn nvvm_library_paths(&self) -> &[path::PathBuf] {
        &self.nvvm_library_paths
    }

    pub fn libdevice_bitcode_path(&self) -> &path::Path {
        self.libdevice_bitcode_path.as_path()
    }

    pub fn related_cuda_envs(&self) -> Vec<String> {
        CUDA_ROOT_ENVS
            .iter()
            .map(|name| name.to_string())
            .chain(iter::once(CUDA_LIBRARY_PATH_ENV.to_string()))
            .collect::<Vec<_>>()
    }

    /// Attempts to locate the root directory of the CUDA SDK installation.
    ///
    /// Searches common environment variables and default installation paths.
    /// Returns `None` if no valid CUDA SDK installation is found.
    fn find_cuda_root() -> Option<path::PathBuf> {
        // Search through the common environment variables first.
        let p = CUDA_ROOT_ENVS
            .iter()
            .filter_map(|name| env::var(name).ok())
            .find(|s| Self::is_cuda_root_path(s.as_str()))
            .map(path::PathBuf::from);
        if p.is_some() {
            return p;
        }
        // Then default installation paths.
        if cfg!(target_os = "windows") {
            const CUDA_DEFAULT_PATHS: &[&str] = &[
                "C:/CUDA",
                "C:/Program Files/NVIDIA",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA",
            ];
            CUDA_DEFAULT_PATHS
                .iter()
                .flat_map(Self::subdirs)
                .find(|p| Self::is_cuda_root_path(p))
        } else {
            const CUDA_DEFAULT_PATHS: &[&str] = &["/usr/lib/cuda", "/usr/local/cuda", "/opt/cuda"];
            CUDA_DEFAULT_PATHS
                .iter()
                .find(|s| Self::is_cuda_root_path(s))
                .map(path::PathBuf::from)
        }
    }

    fn find_cuda_library_dirs(
        cuda_root: &path::Path,
    ) -> Result<Vec<path::PathBuf>, Box<dyn error::Error>> {
        let (target, triple) = Self::parse_target_triple()?;
        assert!(triple.len() >= 3, "Invalid target triple: {triple:?}");

        let search_dirs = match [triple[0].as_str(), triple[1].as_str(), triple[2].as_str()] {
            ["x86_64", "pc", "windows"] => {
                vec![cuda_root.join("lib").join("x64")]
            }
            [_, _, "windows"] => {
                panic!(
                    "Cannot support Windows architecture other than \
                    x86_64-pc-windows-*. target: {target}"
                );
            }
            [_, _, "linux"] => {
                vec![
                    cuda_root.join("lib"),
                    cuda_root.join("lib").join("stubs"),
                    cuda_root.join("lib64"),
                    cuda_root.join("lib64").join("stubs"),
                    cuda_root.join("targets").join("x86_64-linux").join("lib"),
                ]
            }
            [_, _, _] => {
                panic!("Unsupported target triple: {target}");
            }
        };
        let library_dirs = [Self::parse_cuda_library_path_env(), search_dirs].concat();
        let library_dirs = Self::normalize_dirpaths(library_dirs);
        Ok(library_dirs)
    }

    fn find_nvvm_include_dirs(
        cuda_root: &path::Path,
    ) -> Result<Vec<path::PathBuf>, Box<dyn error::Error>> {
        let search_dirs = vec![cuda_root.join("nvvm").join("include")];
        let include_dirs = Self::normalize_dirpaths(search_dirs);
        Ok(include_dirs)
    }

    fn find_nvvm_library_dirs(
        cuda_root: &path::Path,
    ) -> Result<Vec<path::PathBuf>, Box<dyn error::Error>> {
        // The bin paths are required to find the cicc compiler.
        let search_dirs = if cfg!(target_os = "windows") {
            vec![
                cuda_root.join("nvvm").join("bin"),
                cuda_root.join("nvvm").join("lib").join("x64"),
            ]
        } else {
            vec![
                cuda_root.join("nvvm").join("bin"),
                cuda_root.join("nvvm").join("lib64"),
                cuda_root.join("nvvm").join("lib"),
            ]
        };
        let library_dirs = Self::normalize_dirpaths(search_dirs);
        Ok(library_dirs)
    }

    fn parse_cuda_library_path_env() -> Vec<path::PathBuf> {
        // The location of the libcuda, libcudart, libcublas, etc. can be hardcoded with the
        // CUDA_LIBRARY_PATH environment variable.
        match env::var_os(CUDA_LIBRARY_PATH_ENV) {
            Some(v) => env::split_paths(v.as_os_str()).collect::<Vec<_>>(),
            None => vec![],
        }
    }

    /// Checks if the given path is a valid CUDA SDK installation by verifying
    /// the existence of the `cuda.h` header file in the `include` directory.
    fn is_cuda_root_path<P: AsRef<path::Path>>(path: P) -> bool {
        path.as_ref().join("include").join("cuda.h").is_file()
    }

    /// Parses the content of the `cuda.h` header file to extract the driver version.
    ///
    /// # Errors
    /// Returns an error if the `CUDA_VERSION` definition cannot be found or parsed.
    fn parse_driver_version(header_content: &str) -> Result<u32, Box<dyn error::Error>> {
        let version = header_content
            .lines()
            .find(|line| line.contains("#define CUDA_VERSION"))
            .and_then(|line| line.split_whitespace().last())
            .ok_or("Cannot find CUDA_VERSION from CUDA header file.")?;
        let version = version
            .parse::<u32>()
            .map_err(|_| format!("Cannot parse CUDA_VERSION as u32: '{version}'"))?;
        Ok(version)
    }

    /// Parses the content of the `cuda_runtime.h` header file to extract the runtime version.
    ///
    /// # Errors
    /// Returns an error if the `CUDART_VERSION` definition cannot be found or parsed.
    fn parse_runtime_version(header_content: &str) -> Result<u32, Box<dyn error::Error>> {
        let version = header_content
            .lines()
            .find(|line| line.contains("#define CUDART_VERSION"))
            .and_then(|line| line.split_whitespace().last())
            .ok_or("Cannot find CUDART_VERSION from cuda_runtime header file.")?;
        let version = version
            .parse::<u32>()
            .map_err(|_| format!("Cannot parse CUDART_VERSION as u32: '{version}'"))?;
        Ok(version)
    }

    fn parse_target_triple() -> Result<(String, Vec<String>), Box<dyn error::Error>> {
        let target = env::var("TARGET")
            .map_err(|_| "cargo did not set the TARGET environment variable as required.")?;

        // Targets use '-' separators. e.g. x86_64-pc-windows-msvc, x86_64-unknown-linux-gnu, etc.
        let triple = target
            .as_str()
            .split('-')
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        Ok((target, triple))
    }

    fn follow_symlink(p: &path::Path) -> Result<path::PathBuf, Box<dyn error::Error>> {
        let mut p = p.to_path_buf();
        while p.is_symlink() {
            p = p.read_link()?;
        }
        Ok(p)
    }

    fn path_dedup(paths: Vec<path::PathBuf>) -> Vec<path::PathBuf> {
        let mut seen = std::collections::HashSet::new();
        paths
            .into_iter()
            .filter(|p| seen.insert(p.clone()))
            .collect()
    }

    fn normalize_dirpaths(dirs: Vec<path::PathBuf>) -> Vec<path::PathBuf> {
        let dirs = dirs
            .into_iter()
            .filter(|d| d.exists())
            .filter_map(|d| Self::follow_symlink(d.as_path()).ok())
            .collect::<Vec<_>>();
        let dirs = Self::path_dedup(dirs);
        dirs.into_iter().filter(|d| d.is_dir()).collect()
    }

    fn subdirs<P>(p: P) -> Vec<path::PathBuf>
    where
        P: AsRef<path::Path>,
    {
        let p = p.as_ref();
        if !p.exists() || !p.is_dir() {
            return vec![];
        }

        let mut ret = Vec::new();
        let read_dir = match fs::read_dir(p) {
            Ok(d) => d,
            Err(_) => return vec![],
        };
        for entry in read_dir {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let subpath = entry.path();
            // Skip current and parent directories
            if subpath.file_name() == Some(ffi::OsStr::new("."))
                || subpath.file_name() == Some(ffi::OsStr::new(".."))
            {
                continue;
            }
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                ret.push(subpath);
            }
        }
        ret
    }
}
