<div align="center">
  <h1>The Rust CUDA Project</h1>

  <p>
    <strong>An ecosystem of libraries and tools for writing and executing extremely fast GPU code fully in
    <a href="https://www.rust-lang.org/">Rust</a></strong>
  </p>

  <h3>
    <a href="https://rust-gpu.github.io/rust-cuda/index.html">Guide</a>
    <span> | </span>
    <a href="https://rust-gpu.github.io/rust-cuda/guide/getting_started.html">Getting Started</a>
    <span> | </span>
    <a href="https://rust-gpu.github.io/rust-cuda/features.html">Features</a>
  </h3>
<strong>⚠️ The project is still in early development, expect bugs, safety issues, and things that don't work ⚠️</strong>
</div>

<br/>

> [!IMPORTANT]
> This project is no longer dormant and is [being
> rebooted](https://rust-gpu.github.io/blog/2025/01/27/rust-cuda-reboot). Read the [latest status update](https://rust-gpu.github.io/blog/2025/08/11/rust-cuda-update).
> Please contribute!

## Goal

The Rust CUDA Project is a project aimed at making Rust a tier-1 language for extremely fast GPU computing
using the CUDA Toolkit. It provides tools for compiling Rust to extremely fast PTX code as well as libraries
for using existing CUDA libraries with it.

## Background

Historically, general purpose high performance GPU computing has been done using the CUDA toolkit. The CUDA toolkit primarily
provides a way to use Fortran/C/C++ code for GPU computing in tandem with CPU code with a single source. It also provides
many libraries, tools, forums, and documentation to supplement the single-source CPU/GPU code.

CUDA is exclusively an NVIDIA-only toolkit. Many tools have been proposed for cross-platform GPU computing such as
OpenCL, Vulkan Computing, and HIP. However, CUDA remains the most used toolkit for such tasks by far. This is why it is
imperative to make Rust a viable option for use with the CUDA toolkit.

However, CUDA with Rust has been a historically very rocky road. The only viable option until now has been to use the LLVM PTX
backend, however, the LLVM PTX backend does not always work and would generate invalid PTX for many common Rust operations, and
in recent years it has been shown time and time again that a specialized solution is needed for Rust on the GPU with the advent
of projects such as rust-gpu (for Rust -> SPIR-V).

Our hope is that with this project we can push the Rust GPU computing industry forward and make Rust an excellent language
for such tasks. Rust offers plenty of benefits such as `__restrict__` performance benefits for every kernel, An excellent module/crate system,
delimiting of unsafe areas of CPU/GPU code with `unsafe`, high level wrappers to low level CUDA libraries, etc.

## Structure

The scope of the Rust CUDA Project is quite broad, it spans the entirety of the CUDA ecosystem, with libraries and tools to make it
usable using Rust. Therefore, the project contains many crates for all corners of the CUDA ecosystem.

The current line-up of libraries is the following:

- `rustc_codegen_nvvm` Which is a rustc backend that targets NVVM IR (a subset of LLVM IR) for the [libnvvm](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html) library.
  - Generates highly optimized PTX code which can be loaded by the CUDA Driver API to execute on the GPU.
  - For the near future it will be CUDA-only, but it may be used to target amdgpu in the future.
- `cuda_std` for GPU-side functions and utilities, such as thread index queries, memory allocation, warp intrinsics, etc.
  - _Not_ a low level library, provides many utility functions to make it easier to write cleaner and more reliable GPU kernels.
  - Closely tied to `rustc_codegen_nvvm` which exposes GPU features through it internally.
- [`cudnn`](https://github.com/Rust-GPU/Rust-CUDA/tree/master/crates/cudnn) for a collection of GPU-accelerated primitives for deep neural networks.
- `cust` for CPU-side CUDA features such as launching GPU kernels, GPU memory allocation, device queries, etc.
  - High level with features such as RAII and Rust Results that make it easier and cleaner to manage the interface to the GPU.
  - A high level wrapper for the CUDA Driver API, the lower level version of the more common CUDA Runtime API used from C++.
  - Provides much more fine grained control over things like kernel concurrency and module loading than the C++ Runtime API.
- `gpu_rand` for GPU-friendly random number generation, currently only implements xoroshiro RNGs from `rand_xoshiro`.
- `optix` for CPU-side hardware raytracing and denoising using the CUDA OptiX library.

In addition to many "glue" crates for things such as high level wrappers for certain smaller CUDA libraries.

## Related Projects

Other projects related to using Rust on the GPU:

- 2016: [glassful](https://github.com/kmcallister/glassful) Subset of Rust that compiles to GLSL.
- 2017: [inspirv-rust](https://github.com/msiglreith/inspirv-rust) Experimental Rust MIR -> SPIR-V Compiler.
- 2018: [nvptx](https://github.com/japaric-archived/nvptx) Rust to PTX compiler using the `nvptx` target for rustc (using the LLVM PTX backend).
- 2020: [accel](https://github.com/termoshtt/accel) Higher-level library that relied on the same mechanism that `nvptx` does.
- 2020: [rlsl](https://github.com/MaikKlein/rlsl) Experimental Rust -> SPIR-V compiler (predecessor to rust-gpu)
- 2020: [rust-gpu](https://github.com/Rust-GPU/rust-gpu) `rustc` compiler backend to compile Rust to SPIR-V for use in shaders, similar mechanism as our project.

## Usage
```bash
## setup your environment like:
### export OPTIX_ROOT=/opt/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64
### export OPTIX_ROOT_DIR=/opt/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64

## build proj
cargo build
```

## Use Rust-CUDA in Container Environments

The distribution related Dockerfile are located in `container` folder.
Taking ubuntu 24.04 as an example, run the following command in repository root:
```bash
docker build -f ./container/ubuntu24/Dockerfile -t rust-cuda-ubuntu24 .
docker run --rm --runtime=nvidia --gpus all -it rust-cuda-ubuntu24
```

A sample `.devcontainer.json` file is also included, configured for Ubuntu 24.02. Copy this to `.devcontainer/devcontainer.json` to make additonal customizations.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your discretion.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
