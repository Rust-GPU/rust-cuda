use std::env;
use std::path;

fn main() {
    let cuda_include_paths = env::var_os("DEP_CUDA_INCLUDES")
        .map(|s| env::split_paths(s.as_os_str()).collect::<Vec<_>>())
        .expect("Cannot find transitive metadata 'cuda_include' from cust_raw package.");

    println!("cargo::rerun-if-changed=build");

    create_cufft_bindings(&cuda_include_paths);
    println!("cargo::rustc-link-lib=dylib=cufft");
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
