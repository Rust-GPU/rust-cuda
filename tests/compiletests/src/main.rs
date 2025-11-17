use clap::Parser;
use nvvm::NvvmArch;
use std::env;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Parser)]
#[command(bin_name = "cargo compiletest")]
struct Opt {
    /// Automatically update stderr/stdout files.
    #[arg(long)]
    bless: bool,

    /// The CUDA compute capability to target (e.g., compute_70, compute_80, compute_90).
    /// Can specify multiple architectures comma-separated.
    // WARNING: This should be kept in sync with the default on `CudaBuilder::arch`.
    #[arg(long, default_values_t = [NvvmArch::default()], value_delimiter = ',')]
    target_arch: Vec<NvvmArch>,

    /// Only run tests that match these filters.
    #[arg(name = "FILTER")]
    filters: Vec<String>,
}

impl Opt {
    pub fn architectures(&self) -> impl Iterator<Item = NvvmArch> + use<'_> {
        self.target_arch.iter().copied()
    }
}

const CUDA_TARGET: &str = "nvptx64-nvidia-cuda";

#[derive(Copy, Clone)]
enum DepKind {
    CudaLib,
    ProcMacro,
}

impl DepKind {
    fn prefix_and_extension(self) -> (&'static str, &'static str) {
        match self {
            Self::CudaLib => ("lib", "rlib"),
            Self::ProcMacro => (env::consts::DLL_PREFIX, env::consts::DLL_EXTENSION),
        }
    }

    fn target_dir_suffix(self, target: &str) -> String {
        match self {
            Self::CudaLib => format!("{target}/release/deps"),
            Self::ProcMacro => "release/deps".into(),
        }
    }
}

fn main() {
    let opt = Opt::parse();

    let tests_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = tests_dir.parent().unwrap().parent().unwrap().to_path_buf();
    let original_target_dir = workspace_root.join("target");
    let deps_target_dir = original_target_dir.join("compiletest-deps");
    let compiletest_build_dir = original_target_dir.join("compiletest-results");

    // Find the rustc_codegen_nvvm backend before changing directory
    let codegen_backend_path = find_rustc_codegen_nvvm(&workspace_root);

    // HACK(eddyb) force `compiletest` to pass `ui/...` relative paths to `rustc`,
    // which should always end up being the same regardless of the path that the
    // Rust CUDA repo is checked out at (among other things, this avoids hardcoded
    // `compiletest` limits being hit by e.g. users with slightly longer paths).
    std::env::set_current_dir(tests_dir).unwrap();
    let tests_dir = PathBuf::from("");

    let runner = Runner {
        opt,
        tests_dir,
        compiletest_build_dir,
        deps_target_dir,
        codegen_backend_path,
    };

    runner.run_mode("ui");
}

struct Runner {
    opt: Opt,
    tests_dir: PathBuf,
    compiletest_build_dir: PathBuf,
    deps_target_dir: PathBuf,
    codegen_backend_path: PathBuf,
}

impl Runner {
    /// Runs the given `mode` on the directory that matches that name, using the
    /// backend provided by `codegen_backend_path`.
    #[allow(clippy::string_add)]
    fn run_mode(&self, mode: &'static str) {
        /// RUSTFLAGS passed to all test files.
        fn test_rustc_flags(
            codegen_backend_path: &Path,
            deps: &TestDeps,
            indirect_deps_dirs: &[&Path],
            target_arch: &str,
        ) -> String {
            [
                &*rust_flags(codegen_backend_path, target_arch),
                &*indirect_deps_dirs
                    .iter()
                    .map(|dir| format!("-L dependency={}", dir.display()))
                    .fold(String::new(), |a, b| b + " " + &a),
                "--edition 2021",
                &*format!("--extern noprelude:core={}", deps.core.display()),
                &*format!(
                    "--extern noprelude:compiler_builtins={}",
                    deps.compiler_builtins.display()
                ),
                &*format!(
                    "--extern cuda_std_macros={}",
                    deps.cuda_std_macros.display()
                ),
                &*format!("--extern cuda_std={}", deps.cuda_std.display()),
                "--crate-type cdylib",
                "-Zunstable-options",
                "-Zcrate-attr=no_std",
                "-Zcrate-attr=feature(abi_ptx)",
                "-Zui-testing",
            ]
            .join(" ")
        }

        struct Variation {
            name: &'static str,
            extra_flags: &'static str,
        }
        const VARIATIONS: &[Variation] = &[Variation {
            name: "default",
            extra_flags: "",
        }];

        for (arch, variation) in self.opt.architectures().flat_map(|arch| {
            VARIATIONS
                .iter()
                .map(move |variation| (arch.target_feature(), variation))
        }) {
            // HACK(eddyb) in order to allow *some* tests to have separate output
            // in different testing variations (i.e. experimental features), while
            // keeping *most* of the tests unchanged, we make use of "stage IDs",
            // which offer `// only-S` and `// ignore-S` for any stage ID `S`.
            let stage_id = if variation.name == "default" {
                // Use the architecture name as the stage ID.
                arch.to_string()
            } else {
                // Include the variation name in the stage ID.
                format!("{}-{}", arch, variation.name)
            };

            println!("Testing arch: {stage_id}\n");

            let libs = build_deps(
                &self.deps_target_dir,
                &self.codegen_backend_path,
                CUDA_TARGET,
                arch,
            );
            let mut flags = test_rustc_flags(
                &self.codegen_backend_path,
                &libs,
                &[
                    &self
                        .deps_target_dir
                        .join(DepKind::CudaLib.target_dir_suffix(CUDA_TARGET)),
                    &self
                        .deps_target_dir
                        .join(DepKind::ProcMacro.target_dir_suffix(CUDA_TARGET)),
                ],
                arch,
            );
            flags += variation.extra_flags;

            let config = compiletest_rs::Config {
                stage_id,
                target_rustcflags: Some(flags),
                mode: mode.parse().expect("Invalid mode"),
                target: CUDA_TARGET.to_string(),
                src_base: self.tests_dir.join(mode),
                build_base: self.compiletest_build_dir.clone(),
                bless: self.opt.bless,
                filters: self.opt.filters.clone(),
                ..compiletest_rs::Config::default()
            };
            // FIXME(eddyb) do we need this? shouldn't `compiletest` be independent?
            config.clean_rmeta();

            // Set up CUDA environment
            setup_cuda_environment();

            compiletest_rs::run_tests(&config);
        }
    }
}

/// Runs the processes needed to build `cuda-std` & other deps.
fn build_deps(
    deps_target_dir: &Path,
    codegen_backend_path: &Path,
    target: &str,
    arch: &str,
) -> TestDeps {
    // Build compiletests-deps-helper using the same approach as cuda_builder
    let rustflags = vec![
        format!("-Zcodegen-backend={}", codegen_backend_path.display()),
        "-Zcrate-attr=feature(register_tool)".into(),
        "-Zcrate-attr=register_tool(nvvm_internal)".into(),
        "-Zcrate-attr=no_std".into(),
        "-Zcrate-attr=feature(abi_ptx)".into(),
        "-Zsaturating_float_casts=false".into(),
        "-Cembed-bitcode=no".into(),
        "-Cdebuginfo=0".into(),
        "-Coverflow-checks=off".into(),
        "-Copt-level=3".into(),
        "-Cpanic=abort".into(),
        "-Cno-redzone=yes".into(),
        format!("-Cllvm-args=-arch={} --override-libm", arch),
        format!("-Ctarget-feature=+{}", arch),
    ];

    let cargo_encoded_rustflags = rustflags.join("\x1f");

    std::process::Command::new("cargo")
        .args([
            "build",
            "--lib",
            "-p",
            "compiletests-deps-helper",
            "--release",
            "-Zbuild-std=core,alloc",
            "-Zbuild-std-features=panic_immediate_abort",
            &*format!("--target={target}"),
        ])
        .arg("--target-dir")
        .arg(deps_target_dir)
        .env("CARGO_ENCODED_RUSTFLAGS", cargo_encoded_rustflags)
        .env("CUDA_ARCH", "70")
        .stderr(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .status()
        .and_then(map_status_to_result)
        .unwrap();

    let compiler_builtins = find_lib(
        deps_target_dir,
        "compiler_builtins",
        DepKind::CudaLib,
        target,
    );
    let core = find_lib(deps_target_dir, "core", DepKind::CudaLib, target);
    let cuda_std = find_lib(deps_target_dir, "cuda_std", DepKind::CudaLib, target);
    let cuda_std_macros = find_lib(
        deps_target_dir,
        "cuda_std_macros",
        DepKind::ProcMacro,
        target,
    );

    let all_libs = [&compiler_builtins, &core, &cuda_std, &cuda_std_macros];
    if all_libs.iter().any(|r| r.is_err()) {
        // FIXME(eddyb) `missing_count` should always be `0` anyway.
        // FIXME(eddyb) use `--message-format=json-render-diagnostics` to
        // avoid caring about duplicates (or search within files at all).
        let missing_count = all_libs
            .iter()
            .filter(|r| matches!(r, Err(FindLibError::Missing)))
            .count();
        let duplicate_count = all_libs
            .iter()
            .filter(|r| matches!(r, Err(FindLibError::Duplicate)))
            .count();
        eprintln!(
            "warning: cleaning deps ({missing_count} missing libs, {duplicate_count} duplicated libs)"
        );
        clean_deps(deps_target_dir);
        build_deps(deps_target_dir, codegen_backend_path, target, arch)
    } else {
        TestDeps {
            core: core.ok().unwrap(),
            compiler_builtins: compiler_builtins.ok().unwrap(),
            cuda_std: cuda_std.ok().unwrap(),
            cuda_std_macros: cuda_std_macros.ok().unwrap(),
        }
    }
}

fn clean_deps(deps_target_dir: &Path) {
    std::process::Command::new("cargo")
        .arg("clean")
        .arg("--target-dir")
        .arg(deps_target_dir)
        .stderr(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .status()
        .and_then(map_status_to_result)
        .unwrap();
}

enum FindLibError {
    Missing,
    Duplicate,
}

/// Attempt find the rlib that matches `base`, if multiple rlibs are found then
/// a clean build is required and `Err(FindLibError::Duplicate)` is returned.
fn find_lib(
    deps_target_dir: &Path,
    base: impl AsRef<Path>,
    dep_kind: DepKind,
    target: &str,
) -> Result<PathBuf, FindLibError> {
    let base = base.as_ref();
    let (expected_prefix, expected_extension) = dep_kind.prefix_and_extension();
    let expected_name = format!("{}{}", expected_prefix, base.display());

    let dir = deps_target_dir.join(dep_kind.target_dir_suffix(target));

    let matching_paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            let name = {
                let name = path.file_stem();
                if name.is_none() {
                    return false;
                }
                name.unwrap()
            };

            let name_matches = name.to_str().unwrap().starts_with(&expected_name)
                && name.len() == expected_name.len() + 17   // we expect our name, '-', and then 16 hexadecimal digits
                && ends_with_dash_hash(name.to_str().unwrap());
            let extension_matches = path
                .extension()
                .is_some_and(|ext| ext == expected_extension);

            name_matches && extension_matches
        })
        .collect();

    match matching_paths.len() {
        0 => Err(FindLibError::Missing),
        1 => Ok(matching_paths.into_iter().next().unwrap()),
        _ => Err(FindLibError::Duplicate),
    }
}

/// Returns whether this string ends with a dash ('-'), followed by 16 lowercase hexadecimal characters
fn ends_with_dash_hash(s: &str) -> bool {
    let n = s.len();
    if n < 17 {
        return false;
    }
    let mut bytes = s.bytes().skip(n - 17);
    if bytes.next() != Some(b'-') {
        return false;
    }

    bytes.all(|b| b.is_ascii_hexdigit())
}

/// Paths to all of the library artifacts of dependencies needed to compile tests.
struct TestDeps {
    core: PathBuf,
    compiler_builtins: PathBuf,
    cuda_std: PathBuf,
    cuda_std_macros: PathBuf,
}

/// The RUSTFLAGS passed to all CUDA builds.
// FIXME(eddyb) expose most of these from `cuda-builder`.
fn rust_flags(codegen_backend_path: &Path, target_arch: &str) -> String {
    [
        &*format!("-Zcodegen-backend={}", codegen_backend_path.display()),
        // Ensure the codegen backend is emitted in `.d` files to force Cargo
        // to rebuild crates compiled with it when it changes (this used to be
        // the default until https://github.com/rust-lang/rust/pull/93969).
        "-Zbinary-dep-depinfo",
        "-Csymbol-mangling-version=v0",
        "-Zcrate-attr=feature(register_tool)",
        "-Zcrate-attr=register_tool(nvvm_internal)",
        // HACK(eddyb) this is the same configuration that we test with, and
        // ensures no unwanted surprises from e.g. `core` debug assertions.
        "-Coverflow-checks=off",
        "-Cdebug-assertions=off",
        // HACK(eddyb) we need this for `core::fmt::rt::Argument::new_*` calls
        // to *never* be inlined, so we can pattern-match the calls themselves.
        "-Zinline-mir=off",
        // HACK(eddyb) avoid ever reusing instantiations from `compiler_builtins`
        // which is special-cased to turn calls to functions that never return,
        // into aborts, and this applies to the panics of UB-checking helpers
        // (https://github.com/rust-lang/rust/pull/122580#issuecomment-3033026194)
        // but while upstream that only loses the panic message, for us it's even
        // worse, as we lose the chance to remove otherwise-dead `fmt::Arguments`.
        "-Zshare-generics=off",
        // NOTE(eddyb) flags copied from `cuda-builder` are all above this line.
        "-Cdebuginfo=2",
        "-Cembed-bitcode=no",
        &format!("-Ctarget-feature=+{target_arch}"),
        "-Cpanic=abort",
        "-Cno-redzone=yes",
        &format!("-Cllvm-args=-arch={target_arch}"),
        "-Cllvm-args=--override-libm",
    ]
    .join(" ")
}

/// Convenience function to map process failure to results in Rust.
fn map_status_to_result(status: std::process::ExitStatus) -> io::Result<()> {
    match status.success() {
        true => Ok(()),
        false => Err(io::Error::other(format!(
            "process terminated with non-zero code: {}",
            status.code().unwrap_or(0)
        ))),
    }
}

// https://github.com/rust-lang/cargo/blob/1857880b5124580c4aeb4e8bc5f1198f491d61b1/src/cargo/util/paths.rs#L29-L52
fn dylib_path_envvar() -> &'static str {
    if cfg!(windows) {
        "PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_FALLBACK_LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    }
}

fn dylib_path() -> Vec<PathBuf> {
    match env::var_os(dylib_path_envvar()) {
        Some(var) => env::split_paths(&var).collect(),
        None => Vec::new(),
    }
}

#[cfg(windows)]
fn setup_windows_dll_path(codegen_backend_path: &Path) {
    fn add_to_dylib_path(dir: &Path) {
        let lib_path_var = dylib_path_envvar();
        let existing_path = env::var(lib_path_var).unwrap_or_default();
        let separator = ";";

        let dir_str = dir.to_string_lossy();
        // Check if the directory is already in the path
        if !existing_path
            .split(separator)
            .any(|p| p == dir_str.as_ref())
        {
            let new_path = if existing_path.is_empty() {
                dir_str.to_string()
            } else {
                format!("{dir_str}{separator}{existing_path}")
            };
            env::set_var(lib_path_var, new_path);
        }
    }

    // Add the directory containing the codegen backend
    if let Some(dir) = codegen_backend_path.parent() {
        add_to_dylib_path(dir);
    }

    // Try to find LLVM directories and add them to PATH
    // Look for llvm-config to find LLVM installation
    let llvm_config_paths = vec![
        "llvm-config",
        "llvm-config-7",
        "llvm-config.exe",
        "llvm-config-7.exe",
    ];

    for llvm_config in &llvm_config_paths {
        if let Ok(output) = Command::new(llvm_config).arg("--bindir").output() {
            if output.status.success() {
                if let Ok(bindir) = String::from_utf8(output.stdout) {
                    let bindir = bindir.trim();
                    let bindir_path = Path::new(bindir);
                    if bindir_path.exists() {
                        add_to_dylib_path(bindir_path);
                        // Also add the lib directory which might contain DLLs
                        if let Some(parent) = bindir_path.parent() {
                            let libdir = parent.join("lib");
                            if libdir.exists() {
                                add_to_dylib_path(&libdir);
                            }
                        }
                    }
                }
                break;
            }
        }
    }

    // Also check common LLVM installation directories on Windows
    let common_llvm_paths = vec![
        "C:\\Program Files\\LLVM\\bin",
        "C:\\Program Files (x86)\\LLVM\\bin",
        "C:\\Tools\\LLVM\\bin",
        "C:\\llvm\\bin",
    ];

    for path in &common_llvm_paths {
        let path = Path::new(path);
        if path.exists() {
            add_to_dylib_path(path);
        }
    }
}

fn find_rustc_codegen_nvvm(workspace_root: &Path) -> PathBuf {
    let filename = format!(
        "{}rustc_codegen_nvvm{}",
        env::consts::DLL_PREFIX,
        env::consts::DLL_SUFFIX
    );

    // First check if it's already built
    let target_dir = workspace_root.join("target");
    let search_paths = vec![
        target_dir.join("debug").join(&filename),
        target_dir.join("release").join(&filename),
    ];

    for path in &search_paths {
        if path.is_file() {
            // On Windows, ensure the directory containing the DLL is in PATH
            // so that its dependencies can be found
            #[cfg(windows)]
            setup_windows_dll_path(path);

            return path.clone();
        }
    }

    // If not found, try to build it
    println!("Building rustc_codegen_nvvm...");
    let status = Command::new("cargo")
        .args(["build", "-p", "rustc_codegen_nvvm"])
        .current_dir(workspace_root)
        .status()
        .expect("Failed to execute cargo build");

    if !status.success() {
        panic!("Failed to build rustc_codegen_nvvm");
    }

    // Try to find it again after building
    for path in &search_paths {
        if path.is_file() {
            // On Windows, ensure the directory containing the DLL is in PATH
            #[cfg(windows)]
            setup_windows_dll_path(path);

            return path.clone();
        }
    }

    // Last resort: check library path
    for mut path in dylib_path() {
        path.push(&filename);
        if path.is_file() {
            return path;
        }
    }
    panic!("Could not find {filename} in library path or target directory");
}

fn setup_cuda_environment() {
    // Set library path to include CUDA NVVM libraries
    let lib_path_var = dylib_path_envvar();

    // Try to find CUDA installation
    let cuda_paths = vec![
        "/usr/local/cuda/nvvm/lib64",
        "/usr/local/cuda-12/nvvm/lib64",
        "/usr/local/cuda-11/nvvm/lib64",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\nvvm\\lib\\x64",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\nvvm\\lib\\x64",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\nvvm\\lib\\x64",
    ];

    let mut found_cuda_paths = Vec::new();

    // Check CUDA_PATH environment variable
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let nvvm_path = Path::new(&cuda_path).join("nvvm").join("lib64");
        if nvvm_path.exists() {
            found_cuda_paths.push(nvvm_path.to_string_lossy().to_string());
        }
        let nvvm_path_win = Path::new(&cuda_path).join("nvvm").join("lib").join("x64");
        if nvvm_path_win.exists() {
            found_cuda_paths.push(nvvm_path_win.to_string_lossy().to_string());
        }
    }

    // Check standard paths
    for path in &cuda_paths {
        if Path::new(path).exists() {
            found_cuda_paths.push(path.to_string());
        }
    }

    if !found_cuda_paths.is_empty() {
        let existing_path = env::var(lib_path_var).unwrap_or_default();
        let separator = if cfg!(windows) { ";" } else { ":" };

        let new_paths = found_cuda_paths.join(separator);
        let new_lib_path = if existing_path.is_empty() {
            new_paths
        } else {
            format!("{new_paths}{separator}{existing_path}")
        };

        env::set_var(lib_path_var, new_lib_path);
    }
}
