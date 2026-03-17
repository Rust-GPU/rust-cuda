use std::env;
use std::error;
use std::fs;
use std::path;
use std::path::Path;

/// Represents the cuDNN SDK installation.
#[derive(Debug, Clone)]
pub struct CudnnSdk {
    /// cuDNN related paths and version numbers.
    cudnn_include_path: path::PathBuf,
    cudnn_version: [u32; 3],
}

impl CudnnSdk {
    /// Creates a new `cuDNN` instance by locating the cuDNN SDK installation
    /// and parsing its version from the `cudnn_version.h` header file.
    pub fn new() -> Result<Self, Box<dyn error::Error>> {
        // Retrieve the cuDNN include paths.
        let cudnn_include_path = Self::find_cudnn_include_dir()?;
        // Retrieve the cuDNN version.
        let header_path = cudnn_include_path.join("cudnn_version.h");
        let header_content = fs::read_to_string(header_path)?;
        let cudnn_version = Self::parse_cudnn_version(header_content.as_str())?;
        Ok(Self {
            cudnn_include_path,
            cudnn_version,
        })
    }

    pub fn cudnn_include_path(&self) -> &path::Path {
        self.cudnn_include_path.as_path()
    }

    /// Returns the full version of cuDNN as an integer.
    /// For example, cuDNN 9.8.0 is represented as 90800.
    pub fn cudnn_version(&self) -> u32 {
        let [major, minor, patch] = self.cudnn_version;
        major * 10000 + minor * 100 + patch
    }

    pub fn cudnn_version_major(&self) -> u32 {
        self.cudnn_version[0]
    }

    pub fn cudnn_version_minor(&self) -> u32 {
        self.cudnn_version[1]
    }

    pub fn cudnn_version_patch(&self) -> u32 {
        self.cudnn_version[2]
    }

    /// Checks if the given path is a valid cuDNN installation by verifying
    /// the existence of cuDNN header files.
    fn is_cudnn_include_path<P: AsRef<path::Path>>(path: P) -> bool {
        let p = path.as_ref();
        p.join("cudnn.h").is_file() && p.join("cudnn_version.h").is_file()
    }

    #[cfg(target_os = "windows")]
    fn is_vxy_dir_name(name: &str) -> bool {
        name.starts_with('v')
            && name[1..]
                .split('.')
                .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()))
    }

    #[cfg(target_os = "windows")]
    fn collect_windows_cudnn_include_paths(bases: &[&Path]) -> Vec<path::PathBuf> {
        let mut paths = Vec::new();

        for base in bases {
            if let Ok(entries) = fs::read_dir(base) {
                for entry in entries.flatten() {
                    if let Ok(file_type) = entry.file_type() {
                        if file_type.is_dir() {
                            let name = entry.file_name();
                            if let Some(name_str) = name.to_str() {
                                if Self::is_vxy_dir_name(name_str) {
                                    paths.push(base.join(name_str).join("include"));
                                }
                            }
                        }
                    }
                }
            }
        }

        paths
    }

    fn find_cudnn_include_dir() -> Result<path::PathBuf, Box<dyn error::Error>> {
        let cudnn_include_dir = env::var_os("CUDNN_INCLUDE_DIR");

        #[cfg(not(target_os = "windows"))]
        const CUDNN_DEFAULT_PATHS: &[&str] = &[
            "/usr/include",
            "/usr/local/include",
            // CUDA 13 seems to have moved the headers into arch-specific directories.
            "/usr/include/x86_64-linux-gnu",
            "/usr/include/aarch64-linux-gnu",
            "/usr/local/include/x86_64-linux-gnu",
            "/usr/local/include/aarch64-linux-gnu",
        ];

        #[cfg(not(target_os = "windows"))]
        let mut cudnn_paths: Vec<path::PathBuf> =
            CUDNN_DEFAULT_PATHS.iter().map(Path::new).map(path::PathBuf::from).collect();

        #[cfg(target_os = "windows")]
        let mut cudnn_paths: Vec<path::PathBuf> = {
            // Legacy standalone cuDNN installs following NVIDIA's documentation.
            let mut paths = vec![
                path::PathBuf::from("C:/Program Files/NVIDIA/CUDNN/v9.x/include"),
                path::PathBuf::from("C:/Program Files/NVIDIA/CUDNN/v8.x/include"),
            ];

            // Dynamically discover CUDA and cuDNN installs by matching vX.Y-style directories.
            let bases = [
                Path::new("C:/Program Files/NVIDIA/CUDNN"),
                Path::new("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"),
            ];

            paths.extend(Self::collect_windows_cudnn_include_paths(&bases));

            paths
        };

        if let Some(override_path) = &cudnn_include_dir {
            cudnn_paths.push(Path::new(override_path).to_path_buf());
        }

        cudnn_paths
            .iter()
            .find(|p| Self::is_cudnn_include_path(p))
            .map(|p| p.to_path_buf())
            .ok_or("Cannot find cuDNN include directory.".into())
    }

    fn parse_cudnn_version(header_content: &str) -> Result<[u32; 3], Box<dyn error::Error>> {
        let [major, minor, patch] = ["CUDNN_MAJOR", "CUDNN_MINOR", "CUDNN_PATCHLEVEL"]
            .into_iter()
            .map(|macro_name| {
                let version = header_content
                    .lines()
                    .find(|line| line.contains(format!("#define {macro_name}").as_str()))
                    .and_then(|line| line.split_whitespace().last())
                    .ok_or(format!("Cannot find {macro_name} from cuDNN header file.").as_str())?;
                version
                    .parse::<u32>()
                    .map_err(|_| format!("Cannot parse {macro_name} as u32: '{}'", version))
            })
            .collect::<Result<Vec<u32>, _>>()?
            .try_into()
            .map_err(|_| "Invalid cuDNN version length.")?;
        Ok([major, minor, patch])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "windows")]
    #[test]
    fn is_vxy_dir_name_accepts_valid_versions() {
        assert!(CudnnSdk::is_vxy_dir_name("v9.0"));
        assert!(CudnnSdk::is_vxy_dir_name("v10.6"));
        assert!(CudnnSdk::is_vxy_dir_name("v12.1"));
        assert!(CudnnSdk::is_vxy_dir_name("v13.0"));
        assert!(CudnnSdk::is_vxy_dir_name("v9.10.3"));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn is_vxy_dir_name_rejects_invalid_versions() {
        assert!(!CudnnSdk::is_vxy_dir_name("v"));
        assert!(!CudnnSdk::is_vxy_dir_name("v9."));
        assert!(!CudnnSdk::is_vxy_dir_name("v.9"));
        assert!(!CudnnSdk::is_vxy_dir_name("v9.a"));
        assert!(!CudnnSdk::is_vxy_dir_name("9.0"));
        assert!(!CudnnSdk::is_vxy_dir_name("vx.y"));
        assert!(!CudnnSdk::is_vxy_dir_name("random"));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn collect_windows_cudnn_include_paths_discovers_multiple_versions() {
        use std::fs::create_dir_all;

        let tmp_dir = env::temp_dir().join("cudnn_sdk_tests");
        if tmp_dir.exists() {
            fs::remove_dir_all(&tmp_dir).ok();
        }

        let cuda_base = tmp_dir.join("CUDA");
        let v10_6 = cuda_base.join("v10.6");
        let v12_1 = cuda_base.join("v12.1");
        let v13_0 = cuda_base.join("v13.0");

        for ver in [&v10_6, &v12_1, &v13_0] {
            let include_dir = ver.join("include");
            create_dir_all(&include_dir).unwrap();
            fs::write(include_dir.join("cudnn.h"), "// stub").unwrap();
            fs::write(include_dir.join("cudnn_version.h"), "// stub").unwrap();
        }

        // Also add some junk directories that should be ignored.
        create_dir_all(cuda_base.join("vbad")).unwrap();
        create_dir_all(cuda_base.join("not-a-version")).unwrap();

        let bases: [&Path; 1] = [&cuda_base];
        let mut paths = CudnnSdk::collect_windows_cudnn_include_paths(&bases);
        paths.sort();

        assert!(paths.contains(&v10_6.join("include")));
        assert!(paths.contains(&v12_1.join("include")));
        assert!(paths.contains(&v13_0.join("include")));
        assert_eq!(paths.len(), 3);
    }
}
