# Getting Started

This section covers how to get started writing GPU crates with `cuda_std` and `cuda_builder`.

## Required Libraries

Before you can use the project to write GPU crates, you will need a couple of prerequisites:

- [The CUDA SDK](https://developer.nvidia.com/cuda-downloads), version 12.0 or later (and the
  appropriate driver - [see CUDA release
  notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)).

  This is only for building GPU crates, to execute built PTX you only need CUDA `9+`.

- LLVM 7.x (7.0 to 7.4), The codegen searches multiple places for LLVM:

  - If `LLVM_CONFIG` is present, it will use that path as `llvm-config`.
  - Or, if `llvm-config` is present as a binary, it will use that, assuming that `llvm-config --version` returns `7.x.x`.
  - Finally, if neither are present or unusable, it will attempt to download and use prebuilt LLVM. This currently only
    works on Windows however.

- The OptiX SDK if using the optix library (the pathtracer example uses it for denoising).

- You may also need to add `libnvvm` to PATH, the builder should do it for you but in case it does not work, add libnvvm to PATH, it should be somewhere like `CUDA_ROOT/nvvm/bin`,

- You may wish to use or consult the bundled [Dockerfile](#docker) to assist in your local config

## rust-toolchain.toml

NVVM codegen currently requires a specific version of Rust nightly, because it uses rustc internals
that are subject to change. Therefore, you must copy the `rust-toolchain.toml` file in the project
repository so that your own project uses the correct nightly version.

Note: `cust` and other CPU-side libraries work with stable Rust, but they will end up being
compiled with the version of nightly specified in `rust-toolchain.toml`.

## Cargo.toml

Now we can actually get started creating our GPU crate 🎉

Start by making a normal crate as you normally would, manually or with `cargo init`: `cargo init name --lib`.

After this, we just need to add a couple of things to our Cargo.toml:

```diff
[package]
name = "name"
version = "0.1.0"
edition = "2021"

+[lib]
+crate-type = ["cdylib", "rlib"]

[dependencies]
+cuda_std = "XX"
```

Where `XX` is the latest version of `cuda_std`.

We changed our crate's crate types to `cdylib` and `rlib`. We specified `cdylib` because the nvptx targets do not support binary crate types.
`rlib` is so that we will be able to use the crate as a dependency, such as if we would like to use it on the CPU.

## lib.rs

Before we can write any GPU kernels, we must add a few directives to our `lib.rs` which are required by the codegen:

```rs
#![cfg_attr(
    target_os = "cuda",
    no_std,
    register_attr(nvvm_internal)
)]

use cuda_std::*;
```

This does a couple of things:

- It only applies the attributes if we are compiling the crate for the GPU (target_os = "cuda").
- It declares the crate to be `no_std` on CUDA targets.
- It registers a special attribute required by the codegen for things like figuring out
  what functions are GPU kernels.
- It explicitly includes `kernel` macro and `thread`

If you would like to use `alloc` or things like printing from GPU kernels (which requires alloc) then you need to declare `alloc` too:

```rs
extern crate alloc;
```

Finally, if you would like to use types such as slices or arrays inside of GPU kernels you must allow `improper_cytypes_definitions` either on the whole crate or the individual GPU kernels. This is because on the CPU, such types are not guaranteed to be passed a certain way, so they should not be used in `extern "C"` functions (which is what kernels are implicitly declared as). However, `rustc_codegen_nvvm` guarantees the way in which things like structs, slices, and arrays are passed. See [Kernel ABI](./kernel_abi.md).

```rs
#![allow(improper_ctypes_definitions)]
```

## Writing our first GPU kernel

Now we can finally start writing an actual GPU kernel.

<details>
  <summary>Expand this section if you are not familiar with how GPU-side CUDA works</summary>

Firstly, we must explain a couple of things about GPU kernels, specifically, how they are executed. GPU Kernels (functions) are the entry point for executing anything on the GPU, they are the functions which will be executed from the CPU. GPU kernels do not return anything, they write their data to buffers passed into them.

CUDA's execution model is very very complex and it is unrealistic to explain all of it in
this section, but the TLDR of it is that CUDA will execute the GPU kernel once on every
thread, with the number of threads being decided by the caller (the CPU).

We call these parameters the launch dimensions of the kernel. Launch dimensions are split
up into two basic concepts:

- **Threads:** A single thread executes the GPU kernel **once**, and it makes the index
  of itself available to the kernel through special registers (functions in our case).
- **Blocks:** A single block houses multiple threads that it execute on its own. Thread indices
  are only unique across the thread's block, therefore CUDA also exposes the index
  of the current block.

One important thing to note is that block and thread dimensions may be 1d, 2d, or 3d.
That is to say, i can launch `1` block of `6x6x6`, `6x6`, or `6` threads. I could
also launch `5x5x5` blocks. This is very useful for 2d/3d applications because it makes
the 2d/3d index calculations much simpler. CUDA exposes thread and block indices
for each dimension through special registers. We expose thread index queries through
`cuda_std::thread`.

</details>

Now that we know how GPU functions work, let's write a simple kernel. We will write
a kernel which does `[1, 2, 3, 4] + [1, 2, 3, 4] = [2, 4, 6, 8]`. We will use
a 1-dimensional index and use the `cuda_std::thread::index_1d` utility method to
calculate a globally-unique thread index for us (this index is only unique if the kernel was launched with a 1d launch config!).

```rs
#[kernel]
pub unsafe fn add(a: &[f32], b: &[f32], c: *mut f32) {
    let idx = thread::index_1d() as usize;
    if idx < a.len() {
        let elem = &mut *c.add(idx);
        *elem = a[idx] + b[idx];
    }
}
```

If you have used CUDA C++ before, this should seem fairly familiar, with a few oddities:

- Kernel functions must be unsafe currently, this is because the semantics of Rust safety
  on the GPU are still very much undecided. This restriction will probably be removed in the future.
- We use `*mut f32` and not `&mut [f32]`. This is because using `&mut` in function arguments
  is unsound. The reason being that rustc assumes `&mut` does not alias. However, because every thread gets a copy of the arguments, this would cause it to alias, thereby violating
  this invariant and yielding technically unsound code. Pointers do not have such an invariant on the other hand. Therefore, we use a pointer and only make a mutable reference once we
  are sure the elements are disjoint: `let elem = &mut *c.add(idx);`.
- We check that the index is not out of bounds before doing anything, because it is common to
  launch kernels with thread counts that are not exactly divisible by the length for optimization.

Internally what this does is it first checks that a couple of things are right in the kernel:

- All parameters are `Copy`.
- The function is `unsafe`.
- The function does not return anything.

Then it declares this kernel to the codegen so that the codegen can tell CUDA this is a GPU kernel.
It also applies `#[no_mangle]` so the name of the kernel is the same as it is declared in the code.

## Building the GPU crate

Now that you have some kernels defined in a crate, you can build them easily using `cuda_builder`.
which builds GPU crates while passing everything needed by rustc.

To use it you can simply add it as a build dependency in your CPU crate (the crate running the GPU kernels):

```diff
+[build-dependencies]
+cuda_builder = "XX"
```

Where `XX` is the current version of cuda_builder.

Then, you can simply invoke it in the build.rs of your CPU crate:

```rs
use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("path/to/gpu/crate/root")
        .copy_to("some/path.ptx")
        .build()
        .unwrap();
}
```

The first argument is the path to the root of the GPU crate you are trying to build, which would probably be `../name` in our case.
The second function `.copy_to(path)` tells the builder to copy the built PTX file somewhere. By default the builder puts the PTX file
inside of `target/cuda-builder/nvptx64-nvidia-cuda/release/crate_name.ptx`, but it is usually helpful to copy it to another path, which is
what such method does. Finally, `build()` actually runs rustc to compile the crate. This may take a while since it needs to build things like core
from scratch, but after the first compile, incremental will make it much faster.

You can also specify a different compute capability with `.arch()`:

```rs
CudaBuilder::new("path/to/gpu/crate")
    .arch(cuda_builder::NvvmArch::Compute75)  // Target compute 7.5 GPUs
    .copy_to("kernel.ptx")
    .build()
    .unwrap();
```

The architecture you choose affects which GPU features are available. See the [Compute Capability Gating](./compute_capabilities.md) guide for details on writing code that adapts to different GPU capabilities.

Finally, you can include the PTX as a static string in your program:

```rs
static PTX: &str = include_str!("some/path.ptx");
```

Then execute it using cust.

Don't forget to include the current `rust-toolchain.toml` at the top of your project.

## Docker

There are also some [Dockerfiles](https://github.com/Rust-GPU/rust-cuda/tree/main/container) prepared as a quickstart with all the necessary libraries for base CUDA development.

You can use it as follows (assuming your clone of Rust CUDA is at the absolute path `RUST_CUDA`):

- Ensure you have Docker setup to [use gpus](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- Build `docker build -t rust-cuda $RUST_CUDA`
- Run `docker run -it --gpus all -v $RUST_CUDA:/root/rust-cuda --entrypoint /bin/bash rust-cuda`
  - Running will drop you into the container's shell and you will find the project at `~/rust-cuda`
- If all is well, you'll be able to `cargo run` in `~/rust-cuda/examples/cuda/cpu/add`

**Notes:**

1. refer to [rust-toolchain.toml](#rust-toolchain.toml) to ensure you are using the correct toolchain in your project.
2. despite using Docker, your machine will still need to be running a compatible driver, in this case for Cuda 11.4.1 it is >=470.57.02
3. if you have issues within the container, it can help to start ensuring your gpu is recognized
   - ensure `nvidia-smi` provides meaningful output in the container
   - NVidia provides a number of samples https://github.com/NVIDIA/cuda-samples. In particular, you may want to try `make`ing and running the [`deviceQuery`](https://github.com/NVIDIA/cuda-samples/tree/ba04faaf7328dbcc87bfc9acaf17f951ee5ddcf3/Samples/deviceQuery) sample. If all is well you should see many details about your gpu
