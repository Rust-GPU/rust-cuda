//! Compiler Explorer wrapper for rust-cuda.
//!
//! Accepts a `.rs` file containing a `#[kernel]` function, drops it into a
//! generated temporary Cargo project that depends on `cuda_std`, invokes
//! `cargo build` with `rustc_codegen_nvvm` as the codegen backend, and
//! writes the resulting PTX (or LLVM IR) to stdout. Compiler diagnostics
//! are relayed to stderr and the process exits with a non-zero status if
//! no artifact was produced.
//!
//! Expects two environment variables:
//!
//! * `RUST_CUDA_ROOT`: install prefix containing `lib/librustc_codegen_nvvm.so`
//!   and `crates/cuda_std/` (required).
//! * `CUDA_PATH`: CUDA toolkit root, used to locate `libnvvm` at runtime
//!   (defaults to `/usr/local/cuda` if unset).

use anyhow::{Context, Result};
use clap::Parser;
use clap::ValueEnum;
use std::path::PathBuf;

/// Rustflags applied on every build. These mirror the flags `cuda_builder`
/// passes when invoking rustc directly, so PTX produced through the wrapper
/// matches what a normal rust-cuda build would emit.
const STATIC_RUSTFLAGS: [&str; 6] = [
    "-Zunstable-options",
    "-Zcrate-attr=feature(register_tool)",
    "-Zcrate-attr=register_tool(nvvm_internal)",
    "-Zcrate-attr=no_std",
    "-Zsaturating_float_casts=false",
    "-Cpanic=immediate-abort",
];

#[derive(ValueEnum, Clone, Debug)]
enum Emit {
    Ptx,
    LlvmIr,
}

#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    #[arg(long, default_value_t = 3)]
    opt_level: u8,
    #[arg(long, default_value = "compute_75")]
    gpu_arch: String,
    #[arg(long, value_enum, default_value_t = Emit::Ptx)]
    emit: Emit,
    input: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let tmp = tempfile::TempDir::new().context("failed to create temp dir")?;
    let src_dir = tmp.path().join("gpu_kernel").join("src");
    std::fs::create_dir_all(&src_dir).context("failed to create src dir")?;

    let root = std::env::var("RUST_CUDA_ROOT").context("RUST_CUDA_ROOT not set")?;
    let template = include_str!("cargo_toml.template");
    let cargo_toml = template.replace("__CUDA_STD_PATH__", &format!("{}/crates/cuda_std", root));
    std::fs::write(
        tmp.path().join("gpu_kernel").join("Cargo.toml"),
        &cargo_toml,
    )
    .context("failed to write cargo toml")?;
    std::fs::copy(&args.input, src_dir.join("lib.rs")).context("failed to copy input file")?;

    let mut rustflags: Vec<String> = vec![format!(
        "-Zcodegen-backend={}/lib/librustc_codegen_nvvm.so",
        root
    )];
    for flag in STATIC_RUSTFLAGS {
        rustflags.push(flag.to_string());
    }
    let mut llvm_args = format!("-arch={} --override-libm", args.gpu_arch);
    if args.opt_level == 0 {
        llvm_args.push_str(" -opt=0");
    }

    if matches!(args.emit, Emit::LlvmIr) {
        rustflags.push("--emit=llvm-ir".to_string());
    }

    rustflags.push(format!("-Cllvm-args={}", llvm_args));

    // Cargo reads flags from CARGO_ENCODED_RUSTFLAGS as a list joined by the
    // ASCII unit-separator (0x1F), which avoids the quoting ambiguity that
    // RUSTFLAGS has with flags containing spaces (e.g. `-Cllvm-args=...`).
    let encoded = rustflags.join("\x1f");

    // The codegen backend dlopen()s libnvvm and friends at load time, so
    // their directories must be on LD_LIBRARY_PATH for rustc to start at all.
    let cuda_path = std::env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let codegen_dir = format!("{}/lib", root);
    let existing_ld = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    let ld_library_path = format!(
        "{}:{}/nvvm/lib64:{}/lib64:{}",
        codegen_dir, cuda_path, cuda_path, existing_ld
    );

    let crate_dir = tmp.path().join("gpu_kernel");
    let mut cmd = std::process::Command::new("cargo");
    cmd.current_dir(&crate_dir)
        .env("CARGO_ENCODED_RUSTFLAGS", &encoded)
        // `cuda_std` gates `f16`/`f128` support behind this feature flag; the
        // nvptx64 target does not support those types natively.
        .env("CARGO_FEATURE_NO_F16_F128", "1")
        .env("LD_LIBRARY_PATH", &ld_library_path)
        .arg("build")
        .arg("--lib")
        .arg("--message-format=json-render-diagnostics")
        .arg("-Zbuild-std=core,alloc")
        .arg("--target=nvptx64-nvidia-cuda");

    // Any non-zero opt-level maps to a cargo release build; libnvvm performs
    // its own optimisation level selection via `-Cllvm-args=-opt=N` above.
    if args.opt_level != 0 {
        cmd.arg("--release");
    }

    let output = cmd.output().context("failed to spawn cargo")?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut ptx_path: Option<String> = None;

    for line in stdout.lines() {
        let json: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        if json.get("reason").and_then(|v| v.as_str()) != Some("compiler-artifact") {
            continue;
        }

        if let Some(filenames) = json.get("filenames").and_then(|v| v.as_array()) {
            for f in filenames {
                if let Some(s) = f.as_str()
                    && s.ends_with(".ptx")
                {
                    ptx_path = Some(s.to_string());
                }
            }
        }
    }

    match ptx_path {
        Some(path) => {
            let contents = std::fs::read_to_string(&path).context("failed to read PTX file")?;
            print!("{}", contents);
        }
        None => {
            eprintln!("{}", String::from_utf8_lossy(&output.stderr));
            eprintln!("error: compilation failed, no PTX output produced");
            std::process::exit(1);
        }
    }

    Ok(())
}
