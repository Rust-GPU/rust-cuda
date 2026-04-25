use std::env;
use std::path;

use cuda_builder::CudaBuilder;

fn main() {
    // On Windows, nanorand's entropy uses SystemFunction036 (RtlGenRandom) from advapi32.
    // Explicitly link it so the MSVC linker resolves the symbol (avoids LNK2019 when
    // mixing CRTs or with certain link orders).
    #[cfg(target_os = "windows")]
    println!("cargo:rustc-link-lib=advapi32");

    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=kernels");
    println!("cargo::rerun-if-env-changed=RUST_CUDA_DUMP_FINAL_MODULE");
    println!("cargo::rerun-if-env-changed=RUST_CUDA_EMIT_LLVM_IR");

    let out_path = path::PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = path::PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let dump_final_module = env::var_os("RUST_CUDA_DUMP_FINAL_MODULE").is_some();
    let emit_llvm_ir = env::var_os("RUST_CUDA_EMIT_LLVM_IR").is_some();

    let mut builder = CudaBuilder::new(manifest_dir.join("kernels"));
    builder = builder.copy_to(out_path.join("kernels.ptx"));

    if dump_final_module {
        builder = builder.final_module_path(out_path.join("final-module.ll"));
    }

    if emit_llvm_ir {
        builder = builder.emit_llvm_ir(true);
    }

    builder.build().unwrap();
}
