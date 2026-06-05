use std::env;
use std::path;

fn main() {
    let cuda_include_paths = env::var_os("DEP_CUDA_INCLUDES")
        .map(|s| env::split_paths(s.as_os_str()).collect::<Vec<_>>())
        .expect("DEP_CUDA_INCLUDES not set; ensure cust_raw is a dependency");

    let cuda_root = env::var("DEP_CUDA_ROOT")
        .map(path::PathBuf::from)
        .expect("DEP_CUDA_ROOT not set; ensure cust_raw is a dependency");

    println!("cargo::rerun-if-changed=build");

    for dir in [
        cuda_root.join("lib64"),
        cuda_root.join("lib"),
        cuda_root.join("targets").join("x86_64-linux").join("lib"),
    ] {
        if dir.is_dir() {
            println!("cargo::rustc-link-search=native={}", dir.display());
        }
    }

    println!("cargo::rustc-link-lib=dylib=cufft");

    create_cufft_bindings(&cuda_include_paths);
}

fn create_cufft_bindings(cuda_include_paths: &[path::PathBuf]) {
    println!("cargo::rerun-if-changed=build/wrapper.h");

    let outdir = path::PathBuf::from(env::var("OUT_DIR").unwrap());
    let bindgen_path = outdir.join("cufft_raw.rs");

    let bindings = bindgen::Builder::default()
        .header("build/wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .clang_args(
            cuda_include_paths
                .iter()
                .map(|p| format!("-I{}", p.display())),
        )
        .allowlist_function("^cufft.*")
        .allowlist_type("^cufft.*")
        .allowlist_var("^CUFFT.*")
        // cuComplex/cuDoubleComplex are typedef'd as cufftComplex/cufftDoubleComplex
        .allowlist_type("^cu.*Complex.*")
        .allowlist_type("^float2$")
        .allowlist_type("^double2$")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .derive_ord(true)
        .size_t_is_usize(true)
        .layout_tests(true)
        .must_use_type("cufftResult")
        .wrap_unsafe_ops(true)
        .generate_comments(false)
        .generate()
        .expect("Unable to generate cuFFT bindings.");

    bindings
        .write_to_file(&bindgen_path)
        .expect("Cannot write cuFFT bindgen output to file.");
}
