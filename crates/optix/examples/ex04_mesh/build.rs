use std::env;
use std::path;

use cuda_builder::CudaBuilder;

fn main() {
    let ptx_path = path::PathBuf::from(env::var("OUT_DIR").unwrap()).join("kernels.ptx");
    CudaBuilder::new("kernels")
        .copy_to(ptx_path)
        .build()
        .unwrap();
}
