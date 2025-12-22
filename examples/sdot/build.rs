use std::env;
use std::path;
use std::process::{Command, Stdio};

use cuda_builder::CudaBuilder;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=kernels");

    let out_path = path::PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernels_path = path::PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("kernels");

    CudaBuilder::new(kernels_path.as_path())
        .copy_to(out_path.join("kernels.ptx"))
        .build()
        .unwrap();

    // Generate PTX from native CUDA kernels
    let cuda_kernel_path = kernels_path.join("cuda/sdot.cu");

    println!("cargo::rerun-if-changed={}", cuda_kernel_path.display());

    let cuda_ptx = out_path.join("kernels_cuda_mangles.ptx");
    let mut nvcc = Command::new("nvcc");
    nvcc.arg("--ptx")
        .args(["--Werror", "all-warnings"])
        .args(["--output-directory", out_path.as_os_str().to_str().unwrap()])
        .args(["-o", cuda_ptx.as_os_str().to_str().unwrap()])
        .arg(cuda_kernel_path.as_path());

    let build = nvcc
        .stderr(Stdio::inherit())
        .output()
        .expect("failed to execute nvcc kernel build");

    assert!(build.status.success());

    // Decodes (demangles) low-level identifiers
    let cat_out = Command::new("cat")
        .arg(cuda_ptx)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("Failed to start cat process")
        .stdout
        .expect("Failed to open cat stdout");

    let outputs = std::fs::File::create(out_path.join("kernels_cuda.ptx"))
        .expect("Can not open output ptc kernel file");

    let filt_out = Command::new("cu++filt")
        .arg("-p")
        .stdin(Stdio::from(cat_out))
        .stdout(Stdio::from(outputs))
        .stderr(Stdio::inherit())
        .spawn()
        .expect("Failed to start cu++filt process")
        .wait_with_output()
        .expect("Failed to wait on cu++filt");

    assert!(filt_out.status.success());
}
