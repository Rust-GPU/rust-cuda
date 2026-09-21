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

struct LlvmFlavor {
    major: u8,
    config_env: &'static str,
    default_binary: &'static str,
    probe_cuda_home: bool,
    prebuilt_url: &'static str,
}

const LLVM7: LlvmFlavor = LlvmFlavor {
    major: 7,
    config_env: "LLVM_CONFIG",
    default_binary: "llvm-config",
    probe_cuda_home: false,
    prebuilt_url: PREBUILT_LLVM_URL_LLVM7,
};

const LLVM19: LlvmFlavor = LlvmFlavor {
    major: 19,
    config_env: "LLVM_CONFIG_19",
    default_binary: "llvm-config-19",
    probe_cuda_home: true,
    prebuilt_url: PREBUILT_LLVM_URL_LLVM19,
};

static PREBUILT_LLVM_URL_LLVM7: &str =
    "https://github.com/rust-gpu/rustc_codegen_nvvm-llvm/releases/download/llvm-7.1.0/";
static PREBUILT_LLVM_URL_LLVM19: &str =
    "https://github.com/rust-gpu/rustc_codegen_nvvm-llvm/releases/download/llvm-19.1.7/";
static CUDA_ROOT_ENVS: &[&str] = &[
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_ROOT",
    "CUDA_TOOLKIT_ROOT_DIR",
];

fn main() {
    let flavor = if llvm19_enabled() { &LLVM19 } else { &LLVM7 };
    rustc_llvm_build(flavor);
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
    // Ask llvm-config where its install tree lives rather than deriving lexically.
    // Lexical derivation breaks when llvm-config is exposed via a single symlink
    // into /usr/bin/ but the rest of the toolchain stays in the install prefix
    // (e.g. /usr/bin/llvm-config -> /opt/llvm-7/bin/llvm-config, with /opt/llvm-7/bin
    // off PATH). It also handles source-built toolchains where tool names are
    // unsuffixed (`llvm-as`) versus apt-packaged ones (`llvm-as-19`).
    let output = Command::new(llvm_config).arg("--bindir").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let bindir = String::from_utf8(output.stdout).ok()?.trim().to_string();
    Some(PathBuf::from(bindir).join(tool_prefix))
}

fn target_to_llvm_prebuilt(target: &str) -> String {
    let base = match target {
        "x86_64-pc-windows-msvc" => "windows-x86_64",
        "x86_64-unknown-linux-gnu" => "linux-x86_64",
        "aarch64-unknown-linux-gnu" => "linux-aarch64",
        _ => panic!(
            "Unsupported target with no matching prebuilt LLVM: `{target}`, install LLVM and set LLVM_CONFIG (or LLVM_CONFIG_19 when the `llvm19` feature is enabled)"
        ),
    };
    format!("{base}.tar.xz")
}

fn download_prebuilt_llvm(target: &str, base_url: &str) -> PathBuf {
    let prebuilt_name = target_to_llvm_prebuilt(target);
    let url = format!("{base_url}{prebuilt_name}");

    println!("cargo:warning=Downloading prebuilt LLVM from {url}");

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

    let response_code = easy.response_code().unwrap();
    if response_code != 200 {
        fail(&format!(
            "Failed to download prebuilt LLVM from {url}. HTTP response code: {response_code}"
        ));
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

fn find_llvm_config(target: &str, flavor: &LlvmFlavor) -> PathBuf {
    // USE_PREBUILT_LLVM=1 skips local probing and goes straight to download.
    if tracked_env_var_os("USE_PREBUILT_LLVM") != Some("1".into()) {
        let mut candidates = Vec::new();

        if let Some(path) = tracked_env_var_os(flavor.config_env) {
            candidates.push(PathBuf::from(path));
        }

        candidates.push(PathBuf::from(flavor.default_binary));

        if flavor.probe_cuda_home {
            for cuda_root in CUDA_ROOT_ENVS
                .iter()
                .filter_map(tracked_env_var_os)
                .map(PathBuf::from)
            {
                candidates.push(cuda_root.join("nvvm").join("bin").join("llvm-config"));
                candidates.push(cuda_root.join("bin").join("llvm-config"));
            }
        }

        for candidate in &candidates {
            if llvm_version_matches(candidate, flavor.major) {
                return candidate.clone();
            }
        }

        let tried = candidates
            .iter()
            .map(|candidate| format!("  - {}", candidate.display()))
            .collect::<Vec<_>>()
            .join("\n");

        println!(
            "cargo:warning=No matching LLVM {} toolchain found, falling back to prebuilt LLVM. Tried:\n{}",
            flavor.major, tried
        );
    }

    let url = tracked_env_var_os("PREBUILT_LLVM_URL")
        .map(|x| x.to_string_lossy().to_string())
        .unwrap_or_else(|| flavor.prebuilt_url.to_string());
    download_prebuilt_llvm(target, &url)
}

fn find_llvm_as(llvm_config: &Path, flavor: &LlvmFlavor) -> PathBuf {
    let mut candidates = Vec::new();

    if let Some(path) = sibling_llvm_tool(llvm_config, "llvm-as") {
        candidates.push(path);
    }

    candidates.push(PathBuf::from(format!("llvm-as-{}", flavor.major)));
    candidates.push(PathBuf::from("llvm-as"));

    for candidate in &candidates {
        if llvm_version_matches(candidate, flavor.major) {
            return candidate.clone();
        }
    }

    let tried = candidates
        .iter()
        .map(|candidate| format!("  - {}", candidate.display()))
        .collect::<Vec<_>>()
        .join("\n");

    fail(&format!(
        "LLVM {} support is enabled, but llvm-as {} was not found.\n\
         Tried:\n{tried}",
        flavor.major, flavor.major
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

fn configure_libintrinsics(llvm_config: &Path, flavor: &LlvmFlavor) {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR was not set"));

    build_helper::rerun_if_changed(Path::new("libintrinsics.ll"));

    let input = manifest_dir.join("libintrinsics.ll");
    let output = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR was not set"))
        .join(format!("libintrinsics_v{}.bc", flavor.major));
    let llvm_as = find_llvm_as(llvm_config, flavor);

    let status = Command::new(&llvm_as)
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .stderr(Stdio::inherit())
        .stdout(Stdio::inherit())
        .status()
        .unwrap_or_else(|err| {
            fail(&format!(
                "failed to execute llvm-as for LLVM {}: {llvm_as:?}\nerror: {err}",
                flavor.major
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
}

fn rustc_llvm_build(flavor: &LlvmFlavor) {
    let target = env::var("TARGET").expect("TARGET was not set");
    let llvm_config = find_llvm_config(&target, flavor);

    configure_libintrinsics(&llvm_config, flavor);

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

        // if we are on msvc, ignore all -W flags as msvc uses /W and -W is invalid.
        if target.contains("msvc") && flag.starts_with("-W") {
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

    let llvm_version_major = flavor.major.to_string();
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
