# Introduction

Welcome to the Rust CUDA Guide!

## Goal

The Rust CUDA Project is a project aimed at making Rust a tier-1 language for GPU computing using
the CUDA Toolkit. It provides tools for compiling Rust to fast PTX code as well as libraries for
using existing CUDA libraries with it.

## Background

Historically, general-purpose high-performance GPU computing has been done using the CUDA toolkit.
The CUDA toolkit primarily provides a way to use Fortran/C/C++ code for GPU computing in tandem
with CPU code with a single source. It also provides many libraries, tools, forums, and
documentation to supplement the single-source CPU/GPU code.

CUDA is exclusively an NVIDIA-only toolkit. Many tools have been proposed for cross-platform GPU
computing such as OpenCL, Vulkan Computing, and HIP. However, CUDA remains the most used toolkit
for such tasks by far. This is why it is imperative to make Rust a viable option for use with the
CUDA toolkit.

However, CUDA with Rust has been a historically very rocky road. The only viable option until now
has been to use the LLVM PTX backend. However, the LLVM PTX backend does not always work and would
generate invalid PTX for many common Rust operations. In recent years it has been shown time and
time again that a specialized solution is needed for Rust on the GPU with the advent of projects
such as rust-gpu (for translating Rust to SPIR-V).

Our hope is that with this project we can push the Rust on GPUs forward and make Rust an excellent
language for such tasks. Rust offers plenty of benefits such as `__restrict__` performance benefits
for every kernel, an excellent module/crate system, delimiting of unsafe areas of CPU/GPU code with
`unsafe`, high-level wrappers to low-level CUDA libraries, etc.

## Structure

The scope of the Rust CUDA Project is broad, spanning the entirety of the CUDA ecosystem, with
libraries and tools to make it usable using Rust. Therefore, the project contains many crates for
all corners of the CUDA ecosystem.

- `rustc_codegen_nvvm` is a rustc backend that targets NVVM IR (a subset of LLVM IR) for the
  [libnvvm](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html) library.
  - Generates highly optimized PTX code which can be loaded by the CUDA Driver API to execute on
    the GPU.
  - For now it is CUDA-only, but it may be used to target AMD GPUs in the future.
- `cuda_std` contains GPU-side functions and utilities, such as thread index queries, memory
  allocation, warp intrinsics, etc.
  - It is _not_ a low level library. It provides many utility functions to make it easier to write
    cleaner and more reliable GPU kernels.
  - It is Closely tied to `rustc_codegen_nvvm` which exposes GPU features through it internally.
- `cust` contains CPU-side CUDA features such as launching GPU kernels, GPU memory allocation,
  device queries, etc.
  - It is a high-level wrapper for the CUDA Driver API, the lower level alternative to the more
    common CUDA Runtime API used from C++. It provides more fine-grained control over things like
    kernel concurrency and module loading than the Runtime API.
  - High-level Rust features such as RAII and `Result` make it easier and cleaner to manage
    the interface to the GPU.
- `cudnn` is a collection of GPU-accelerated primitives for deep neural networks.
- `gpu_rand` does GPU-friendly random number generation. It currently only implements xoroshiro
  RNGs from `rand_xoshiro`.
- `optix` provides CPU-side hardware raytracing and denoising using the CUDA OptiX library.
  (This library is currently commented out because the OptiX SDK is difficult to install.)

There are also several "glue" crates for things such as high level wrappers for certain smaller
CUDA libraries.

## Related Projects

Other projects related to using Rust on the GPU:

- 2016: [glassful](https://github.com/kmcallister/glassful) translates a subset of Rust to GLSL.
- 2017: [inspirv-rust](https://github.com/msiglreith/inspirv-rust) is an experimental
  Rust-MIR-to-SPIR-V compiler.
- 2018: [nvptx](https://github.com/japaric-archived/nvptx) is a Rust-to-PTX compiler using the
  `nvptx` target for rustc (using the LLVM PTX backend).
- 2020: [accel](https://github.com/termoshtt/accel) is a higher-level library that relied on the
  same mechanism that `nvptx` does.
- 2020: [rlsl](https://github.com/MaikKlein/rlsl) is an experimental Rust-to-SPIR-V compiler
  (and a predecessor to rust-gpu).
- 2020: [rust-gpu](https://github.com/Rust-GPU/rust-gpu) is a `rustc` compiler backend to compile
  Rust to SPIR-V for use in shaders. Like Rust CUDA, it is part of the broader [Rust
  GPU](https://rust-gpu.github.io/) project.
