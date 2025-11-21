use std::env;
use std::iter;

use cuda_builder::CudaBuilder;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let optix_include_paths = env::var_os("DEP_OPTIX_INCLUDE_DIR")
        .map(|s| env::split_paths(s.as_os_str()).collect::<Vec<_>>())
        .expect("Cannot find transitive metadata 'optix_include' from optix-sys package.");

    let args = optix_include_paths
        .iter()
        .map(|p| format!("-I{}", p.display()))
        .chain(iter::once(format!("-I{}/../common/gdt", manifest_dir)))
        .collect::<Vec<_>>();
    compile_to_ptx("src/ex02_pipeline.cu", &args);

    let ptx_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("kernels.ptx");

    CudaBuilder::new("kernels")
        .copy_to(ptx_path)
        .arch(cuda_builder::NvvmArch::Compute75)
        .optix(true)
        .build()
        .unwrap();
}

fn compile_to_ptx(cu_path: &str, args: &[String]) {
    println!("cargo::rerun-if-changed={}", cu_path);

    let full_path =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join(cu_path);

    let mut ptx_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join(cu_path);
    ptx_path.set_extension("ptx");
    std::fs::create_dir_all(ptx_path.parent().unwrap()).unwrap();

    let output = std::process::Command::new("nvcc")
        .arg("-ptx")
        .arg(&full_path)
        .arg("-o")
        .arg(&ptx_path)
        .args(args)
        .output()
        .expect("failed to fun nvcc");

    if !output.status.success() {
        panic!("{}", unsafe { String::from_utf8_unchecked(output.stderr) });
    }
}
