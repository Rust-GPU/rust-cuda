use std::{
    env,
    ffi::{OsStr, OsString},
    fmt::Display,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use curl::easy::Easy;
use tar::Archive;
use xz::read::XzDecoder;

static PREBUILT_LLVM_URL_LLVM7: &str =
    "https://github.com/rust-gpu/rustc_codegen_nvvm-llvm/releases/download/LLVM-7.1.0/";

fn main() {
    rustc_llvm_build(llvm19_enabled());
}

fn fail(s: &str) -> ! {
    println!("\n\n{s}\n\n");
    std::process::exit(1);
}

#[track_caller]
pub fn output(cmd: &mut Command) -> String {
    let output = match cmd.stderr(Stdio::inherit()).output() {
        Ok(status) => status,
        Err(e) => fail(&format!("failed to execute command: {cmd:?}\nerror: {e}")),
    };
    assert!(
        output.status.success(),
        "command did not execute successfully: {:?}\n\
    expected success, got: {}",
        cmd,
        output.status
    );

    String::from_utf8(output.stdout).unwrap()
}

fn llvm19_enabled() -> bool {
    tracked_env_var_os("CARGO_FEATURE_LLVM19").is_some()
}

fn required_major_llvm_version(llvm19_enabled: bool) -> u8 {
    if llvm19_enabled { 19 } else { 7 }
}

fn command_version(path: &Path) -> Option<String> {
    let output = Command::new(path).arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }

    Some(String::from_utf8(output.stdout).ok()?.trim().to_string())
}

fn llvm_major_version(path: &Path) -> Option<u8> {
    command_version(path)?
        .split(|ch: char| !ch.is_ascii_digit())
        .find(|segment| !segment.is_empty())?
        .parse()
        .ok()
}

fn llvm_version_matches(path: &Path, required_major: u8) -> bool {
    llvm_major_version(path) == Some(required_major)
}

fn sibling_llvm_tool(llvm_config: &Path, tool_prefix: &str) -> Option<PathBuf> {
    let file_name = llvm_config.file_name()?.to_str()?;
    let suffix = file_name.strip_prefix("llvm-config")?;
    Some(llvm_config.with_file_name(format!("{tool_prefix}{suffix}")))
}

fn target_to_llvm_prebuilt(target: &str) -> String {
    let base = match target {
        "x86_64-pc-windows-msvc" => "windows-x86_64",
        // NOTE(RDambrosio016): currently disabled because of weird issues with segfaults and building the C++ shim
        // "x86_64-unknown-linux-gnu" => "linux-x86_64",
        _ => panic!(
            "Unsupported target with no matching prebuilt LLVM: `{target}`, install LLVM and set LLVM_CONFIG"
        ),
    };
    format!("{base}.tar.xz")
}

fn find_llvm_config(target: &str, llvm19_enabled: bool) -> PathBuf {
    if llvm19_enabled {
        return find_llvm_config_llvm19();
    }

    find_llvm_config_llvm7(target)
}

fn find_llvm_config_llvm19() -> PathBuf {
    let required_major = required_major_llvm_version(true);
    let mut candidates = Vec::new();

    if let Some(path) = tracked_env_var_os("LLVM_CONFIG_19") {
        candidates.push(PathBuf::from(path));
    }

    candidates.push(PathBuf::from("llvm-config-19"));

    if let Some(cuda_home) = tracked_env_var_os("CUDA_HOME") {
        let cuda_home = PathBuf::from(cuda_home);
        candidates.push(cuda_home.join("nvvm").join("bin").join("llvm-config"));
        candidates.push(cuda_home.join("bin").join("llvm-config"));
    }

    for candidate in &candidates {
        if llvm_version_matches(candidate, required_major) {
            return candidate.clone();
        }
    }

    let tried = candidates
        .iter()
        .map(|candidate| format!("  - {}", candidate.display()))
        .collect::<Vec<_>>()
        .join("\n");

    fail(&format!(
        "LLVM 19 support is enabled, but no LLVM 19 toolchain was found.\n\
         Tried:\n{tried}\n\n\
         Set LLVM_CONFIG_19=/path/to/llvm-config from an LLVM 19 installation."
    ));
}

fn find_llvm_config_llvm7(target: &str) -> PathBuf {
    let required_major = required_major_llvm_version(false);
    // first, if LLVM_CONFIG is set then see if its llvm version if 7.x, if so, use that.
    let config_env = tracked_env_var_os("LLVM_CONFIG");
    // if LLVM_CONFIG is not set, try using llvm-config as a normal app in PATH.
    let path_to_try = config_env.unwrap_or_else(|| "llvm-config".into());

    // if USE_PREBUILT_LLVM is set to 1 then download prebuilt llvm without trying llvm-config
    if tracked_env_var_os("USE_PREBUILT_LLVM") != Some("1".into()) {
        let cmd = Command::new(&path_to_try).arg("--version").output();

        if let Ok(out) = cmd {
            let version = String::from_utf8(out.stdout).unwrap();
            if version.starts_with(&required_major.to_string()) {
                return PathBuf::from(path_to_try);
            }
            println!(
                "cargo:warning=Prebuilt llvm-config version does not start with {required_major}"
            );
        } else {
            println!("cargo:warning=Failed to run prebuilt llvm-config");
        }
    }

    // otherwise, download prebuilt LLVM.
    println!("cargo:warning=Downloading prebuilt LLVM");
    let mut url = tracked_env_var_os("PREBUILT_LLVM_URL")
        .map(|x| x.to_string_lossy().to_string())
        .unwrap_or_else(|| PREBUILT_LLVM_URL_LLVM7.to_string());

    let prebuilt_name = target_to_llvm_prebuilt(target);
    url = format!("{url}{prebuilt_name}");

    let out = env::var("OUT_DIR").expect("OUT_DIR was not set");
    let mut easy = Easy::new();

    easy.url(&url).unwrap();
    easy.follow_location(true).unwrap();
    let mut xz_encoded = Vec::with_capacity(20_000_000); // 20mb
    {
        let mut transfer = easy.transfer();
        transfer
            .write_function(|data| {
                xz_encoded.extend_from_slice(data);
                Ok(data.len())
            })
            .expect("Failed to download prebuilt LLVM");
        transfer
            .perform()
            .expect("Failed to download prebuilt LLVM");
    }

    let decompressor = XzDecoder::new(xz_encoded.as_slice());
    let mut ar = Archive::new(decompressor);

    ar.unpack(&out).expect("Failed to unpack LLVM to LLVM dir");
    let out_path = PathBuf::from(out).join(prebuilt_name.strip_suffix(".tar.xz").unwrap());

    println!("cargo:rerun-if-changed={}", out_path.display());

    out_path
        .join("bin")
        .join(format!("llvm-config{}", std::env::consts::EXE_SUFFIX))
}

fn find_llvm_as_llvm19(llvm_config: &Path) -> PathBuf {
    let required_major = required_major_llvm_version(true);
    let mut candidates = Vec::new();

    if let Some(path) = sibling_llvm_tool(llvm_config, "llvm-as") {
        candidates.push(path);
    }

    candidates.push(PathBuf::from("llvm-as-19"));
    candidates.push(PathBuf::from("llvm-as"));

    for candidate in &candidates {
        if llvm_version_matches(candidate, required_major) {
            return candidate.clone();
        }
    }

    let tried = candidates
        .iter()
        .map(|candidate| format!("  - {}", candidate.display()))
        .collect::<Vec<_>>()
        .join("\n");

    fail(&format!(
        "LLVM 19 support is enabled, but llvm-as 19 was not found.\n\
         Tried:\n{tried}"
    ));
}

fn detect_llvm_link() -> (&'static str, &'static str) {
    // Force the link mode we want, preferring static by default, but
    // possibly overridden by `configure --enable-llvm-link-shared`.
    if tracked_env_var_os("LLVM_LINK_SHARED").is_some() {
        ("dylib", "--link-shared")
    } else {
        ("static", "--link-static")
    }
}

pub fn tracked_env_var_os<K: AsRef<OsStr> + Display>(key: K) -> Option<OsString> {
    println!("cargo:rerun-if-env-changed={key}");
    env::var_os(key)
}

fn configure_libintrinsics(llvm_config: &Path, llvm19_enabled: bool) {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR was not set"));

    // Both paths share `libintrinsics.ll`. The LLVM 7 build consumes the checked-in
    // `libintrinsics.bc` (regenerate manually with `llvm-as-7` when the .ll changes).
    // The LLVM 19 build assembles the same .ll on the fly with `llvm-as-19`.
    build_helper::rerun_if_changed(Path::new("libintrinsics.ll"));

    if llvm19_enabled {
        let input = manifest_dir.join("libintrinsics.ll");
        let output = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR was not set"))
            .join("libintrinsics_v19.bc");
        let llvm_as = find_llvm_as_llvm19(llvm_config);

        let status = Command::new(&llvm_as)
            .arg(&input)
            .arg("-o")
            .arg(&output)
            .stderr(Stdio::inherit())
            .stdout(Stdio::inherit())
            .status()
            .unwrap_or_else(|err| {
                fail(&format!(
                    "failed to execute llvm-as for LLVM 19: {llvm_as:?}\nerror: {err}"
                ))
            });

        if !status.success() {
            fail(&format!(
                "llvm-as did not assemble {} successfully",
                input.display()
            ));
        }

        println!(
            "cargo:rustc-env=NVVM_LIBINTRINSICS_BC_PATH={}",
            output.display()
        );
    } else {
        build_helper::rerun_if_changed(Path::new("libintrinsics.bc"));
        println!(
            "cargo:rustc-env=NVVM_LIBINTRINSICS_BC_PATH={}",
            manifest_dir.join("libintrinsics.bc").display()
        );
    }
}

fn rustc_llvm_build(llvm19_enabled: bool) {
    let target = env::var("TARGET").expect("TARGET was not set");
    let llvm_config = find_llvm_config(&target, llvm19_enabled);
    let required_major = required_major_llvm_version(llvm19_enabled);

    configure_libintrinsics(&llvm_config, llvm19_enabled);

    let required_components = &["ipo", "bitreader", "bitwriter", "lto", "nvptx"];

    let components = output(Command::new(&llvm_config).arg("--components"));
    let mut components = components.split_whitespace().collect::<Vec<_>>();
    components.retain(|c| required_components.contains(c));

    for component in required_components {
        assert!(
            components.contains(component),
            "require llvm component {component} but wasn't found"
        );
    }

    for component in components.iter() {
        println!("cargo:rustc-cfg=llvm_component=\"{component}\"");
    }

    // Link in our own LLVM shims, compiled with the same flags as LLVM
    let mut cmd = Command::new(&llvm_config);
    cmd.arg("--cxxflags");
    let cxxflags = output(&mut cmd);
    let mut cfg = cc::Build::new();
    cfg.warnings(false);
    for flag in cxxflags.split_whitespace() {
        if flag.starts_with("-flto") {
            continue;
        }
        // ignore flags that aren't supported in gcc 8
        if flag == "-Wcovered-switch-default" {
            continue;
        }
        if flag == "-Wstring-conversion" {
            continue;
        }
        if flag == "-Werror=unguarded-availability-new" {
            continue;
        }

        cfg.flag(flag);
    }

    for component in &components {
        let mut flag = String::from("LLVM_COMPONENT_");
        flag.push_str(&component.to_uppercase());
        cfg.define(&flag, None);
    }

    let llvm_version_major = required_major.to_string();
    cfg.define("LLVM_VERSION_MAJOR", Some(llvm_version_major.as_str()));

    if tracked_env_var_os("LLVM_RUSTLLVM").is_some() {
        cfg.define("LLVM_RUSTLLVM", None);
    }

    build_helper::rerun_if_changed(Path::new("rustc_llvm_wrapper"));
    cfg.file("rustc_llvm_wrapper/RustWrapper.cpp")
        .file("rustc_llvm_wrapper/PassWrapper.cpp")
        .include("rustc_llvm_wrapper")
        .cpp(true)
        .cpp_link_stdlib(None) // we handle this below
        .compile("llvm-wrapper");

    let (llvm_kind, llvm_link_arg) = detect_llvm_link();

    // Link in all LLVM libraries, if we're using the "wrong" llvm-config then
    // we don't pick up system libs because unfortunately they're for the host
    // of llvm-config, not the target that we're attempting to link.
    let mut cmd = Command::new(&llvm_config);
    cmd.arg(llvm_link_arg).arg("--libs");

    if target.contains("windows-gnu") {
        println!("cargo:rustc-link-lib=shell32");
        println!("cargo:rustc-link-lib=uuid");
    } else if target.contains("netbsd") || target.contains("haiku") {
        println!("cargo:rustc-link-lib=z");
    }
    cmd.args(&components);

    for lib in output(&mut cmd).split_whitespace() {
        let name = if let Some(stripped) = lib.strip_prefix("-l") {
            stripped
        } else if let Some(stripped) = lib.strip_prefix('-') {
            stripped
        } else if Path::new(lib).exists() {
            // On MSVC llvm-config will print the full name to libraries, but
            // we're only interested in the name part
            let name = Path::new(lib).file_name().unwrap().to_str().unwrap();
            name.trim_end_matches(".lib")
        } else if lib.ends_with(".lib") {
            // Some MSVC libraries just come up with `.lib` tacked on, so chop
            // that off
            lib.trim_end_matches(".lib")
        } else {
            continue;
        };

        // Don't need or want this library, but LLVM's CMake build system
        // doesn't provide a way to disable it, so filter it here even though we
        // may or may not have built it. We don't reference anything from this
        // library and it otherwise may just pull in extra dependencies on
        // libedit which we don't want
        if name == "LLVMLineEditor" {
            continue;
        }

        let kind = if name.starts_with("LLVM") {
            llvm_kind
        } else {
            "dylib"
        };
        println!("cargo:rustc-link-lib={kind}={name}");
    }

    // Link in the system libraries that LLVM depends on
    #[cfg(not(target_os = "windows"))]
    link_llvm_system_libs(&llvm_config, required_components);

    // LLVM ldflags
    //
    // If we're a cross-compile of LLVM then unfortunately we can't trust these
    // ldflags (largely where all the LLVM libs are located). Currently just
    // hack around this by replacing the host triple with the target and pray
    // that those -L directories are the same!
    let mut cmd = Command::new(&llvm_config);
    cmd.arg(llvm_link_arg).arg("--ldflags");
    for lib in output(&mut cmd).split_whitespace() {
        if let Some(stripped) = lib.strip_prefix("-LIBPATH:") {
            println!("cargo:rustc-link-search=native={stripped}");
        } else if let Some(stripped) = lib.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={stripped}");
        } else if let Some(stripped) = lib.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={stripped}");
        }
    }

    // Some LLVM linker flags (-L and -l) may be needed even when linking
    // rustc_llvm, for example when using static libc++, we may need to
    // manually specify the library search path and -ldl -lpthread as link
    // dependencies.
    let llvm_linker_flags = tracked_env_var_os("LLVM_LINKER_FLAGS");
    if let Some(s) = llvm_linker_flags {
        for lib in s.into_string().unwrap().split_whitespace() {
            if let Some(stripped) = lib.strip_prefix("-l") {
                println!("cargo:rustc-link-lib={stripped}");
            } else if let Some(stripped) = lib.strip_prefix("-L") {
                println!("cargo:rustc-link-search=native={stripped}");
            }
        }
    }

    let llvm_static_stdcpp = tracked_env_var_os("LLVM_STATIC_STDCPP");
    let llvm_use_libcxx = tracked_env_var_os("LLVM_USE_LIBCXX");

    let stdcppname = if target.contains("openbsd") {
        if target.contains("sparc64") {
            "estdc++"
        } else {
            "c++"
        }
    } else if target.contains("freebsd") || target.contains("darwin") {
        "c++"
    } else if target.contains("netbsd") && llvm_static_stdcpp.is_some() {
        // NetBSD uses a separate library when relocation is required
        "stdc++_pic"
    } else if llvm_use_libcxx.is_some() {
        "c++"
    } else {
        "stdc++"
    };

    // RISC-V requires libatomic for sub-word atomic operations
    if target.starts_with("riscv") {
        println!("cargo:rustc-link-lib=atomic");
    }

    // C++ runtime library
    if !target.contains("msvc") {
        if let Some(s) = llvm_static_stdcpp {
            assert!(!cxxflags.contains("stdlib=libc++"));
            let path = PathBuf::from(s);
            println!(
                "cargo:rustc-link-search=native={}",
                path.parent().unwrap().display()
            );
            if target.contains("windows") {
                println!("cargo:rustc-link-lib=static-nobundle={stdcppname}");
            } else {
                println!("cargo:rustc-link-lib=static={stdcppname}");
            }
        } else if cxxflags.contains("stdlib=libc++") {
            println!("cargo:rustc-link-lib=c++");
        } else {
            println!("cargo:rustc-link-lib={stdcppname}");
        }
    }

    // Libstdc++ depends on pthread which Rust doesn't link on MinGW
    // since nothing else requires it.
    if target.contains("windows-gnu") {
        println!("cargo:rustc-link-lib=static-nobundle=pthread");
    }
}

#[cfg(not(target_os = "windows"))]
fn link_llvm_system_libs(llvm_config: &Path, components: &[&str]) {
    let (_, llvm_link_arg) = detect_llvm_link();
    let mut cmd: Command = Command::new(llvm_config);
    cmd.arg(llvm_link_arg).arg("--system-libs");

    for comp in components {
        cmd.arg(comp);
    }

    for lib in output(&mut cmd).split_whitespace() {
        let name = if let Some(stripped) = lib.strip_prefix("-l") {
            stripped
        } else {
            continue;
        };

        println!("cargo:rustc-link-lib=dylib={name}");
    }
}
