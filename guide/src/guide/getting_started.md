# Getting started

## Required libraries

Rust CUDA has several prerequisites.

- A machine with an NVIDIA GPU with a Compute Capability of 5.0 (Maxwell) or later.
- [CUDA](https://developer.nvidia.com/cuda-downloads) version 12.0 or later.
- An appropriate NVIDIA driver.
  - For CUDA 12, the driver version `N` should be in the range `525 <= N < 580`.
  - For CUDA 13, the driver version `N` should be in the range `580 <= N`.
- LLVM 7.x (7.0 to 7.4). This is (unfortunately) a very old version of LLVM. The codegen backend
  searches multiple places for LLVM.
  - If `LLVM_CONFIG` is present, the backend will use that path as `llvm-config`.
  - Or, if `llvm-config` is present as a binary, the backend will use that, assuming that
    `llvm-config --version` returns `7.x.x`.
  - Failing that, the backend will attempt to download and use a prebuilt LLVM. This currently only
    works on Windows, however.

Because the required libraries can be difficult to install, we provide Docker images containing
CUDA and LLVM 7. There are instructions on using these Docker images [below](#docker).
Alternatively, if you do want to install these libraries yourself, the steps within the [Docker
files] are a good starting point.

[Docker files]: https://github.com/Rust-GPU/rust-cuda/tree/main/container

## CUDA basics

GPU kernels are functions launched from the CPU that run on the GPU. They do not have a return
value, instead writing data into mutable buffers passed to them. CUDA executes multiple (possibly
hundreds) of invocations of a GPU kernel at once, each one on a different thread, and each thread
typically works on only part of the input and output buffers, sometimes just a single element
thereof.

The caller decides the *launch dimensions*.
- **Threads:** A single thread executes the GPU kernel **once**. CUDA makes the thread's index
  available to the kernel.
- **Blocks:** A single block houses multiple threads that it execute on its own. CUDA also makes
  the blocks index avaiable to the kernel.

Block and thread dimensions may be 1D, 2D, or 3D. For example, you can launch 1 block of 6 threads,
or `6x6` threads, or `6x6x6` threads. Likewise, you can launch 5 or 5x5 or 5x5x5 blocks. This can
make index calculations for programs with 2D or 3D data simpler.

## A first example: the code

This section will walk through a simple Rust CUDA program that adds two small 1D vectors on the
GPU. It consists of two tiny crates and some connecting pieces.

The file structure looks like this:
```
.
├── rust-toolchain.toml  # Specifies which nightly version to use
├── build.rs             # Build script that compiles the code that runs on the GPU
├── kernels
│   ├── Cargo.toml       # Cargo manifest for code that runs on the GPU
│   └── src
│       └── lib.rs       # Code that runs on the GPU
├── Cargo.toml           # Cargo manifest for code that runs on the CPU
└── src
    └── main.rs          # Code that runs on the CPU
```

### `rust-toolchain.toml`

`rustc_codegen_nvvm` currently requires a specific version of Rust nightly because it uses rustc
internals that are subject to change. You must copy the appropriate revision of
[`rust-toolchain.toml` from the rust-cuda repository][repo] so that your own project uses the
correct nightly version.

[repo]: https://github.com/Rust-GPU/rust-cuda/blob/7fa76f3d717038a92c90bf4a482b0b8dd3259344/rust-toolchain.toml

### `Cargo.toml` and `kernels/Cargo.toml`

The top-level `Cargo.toml` looks like this:
```toml
[package]
name = "rust-cuda-basic"
version = "0.1.0"
edition = "2024"

[dependencies]
cust = { git = "https://github.com/rust-gpu/rust-cuda", rev = "7fa76f3d717038a92c90bf4a482b0b8dd3259344" }
kernels = { path = "kernels" }

[build-dependencies]
cuda_builder = { git = "https://github.com/rust-gpu/rust-cuda", rev = "7fa76f3d717038a92c90bf4a482b0b8dd3259344", features = ["rustc_codegen_nvvm"] }
```

`kernels/Cargo.toml` looks like this:
```toml
[package]
name = "kernels"
version = "0.1.0"
edition = "2024"

[dependencies]
cuda_std = { git = "https://github.com/rust-gpu/rust-cuda", rev = "7fa76f3d717038a92c90bf4a482b0b8dd3259344" }

[lib]
# - cdylib: because the nvptx targets do not support binary crate types.
# - rlib: so the `kernels` crate can be used as a dependency by `rust-cuda-basic`.
crate-type = ["cdylib", "rlib"]
```

At the time of writing there are no recent releases of any Rust CUDA crates so it is best
to use code directly from the GitHub repository via `git` and `rev`. The above revision works but
later revisions should also work.

### `kernels/src/lib.rs`

This file defines the code that will run on the GPU.
```rust
use cuda_std::prelude::*;

// Input/output type shared with the `rustc-cuda-basic` crate.
pub type T = f32;

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn add(a: &[T], b: &[T], c: *mut T) {
    let i = thread::index_1d() as usize;
    if i < a.len() {
        let elem = unsafe { &mut *c.add(i) };
        *elem = a[i] + b[i];
    }
}
```

It defines the addition of a single pair of elements in `a` and `b`. Some parts of this file look
like normal Rust code, but some parts are unusual.
- The type `T` will be shared with the CPU code in a way that minimizes the chances of certain
  kinds of errors. More on this below.
- The `#[kernel]` attribute indicates this is code that runs on the GPU. It is similar to
  `__global__` in CUDA C++. Multiple invocations of this kernel will run in parallel and share
  `a`, `b`, and `c`.
- The proc macro that processes the `#[kernel]` attribute marks the kernel as `no_mangle` so that
  the name is obvious in both GPU code and CPU code. The proc macro also checks that the kernel is
  marked `unsafe`, all parameters are `Copy`, and there is no return value.
- All GPU functions are unsafe because the parallel execution and sharing of data typical for GPU
  kernels is incompatible with safe Rust.
- The inputs (`a` and `b`) are normal slices but the output (`c`) is a raw pointer. Again, this
  is because `c` is mutable state shared by multiple kernels executing in parallel. Using `&mut
  [T]` would incorrectly indicate that it is non-shared mutable state, and therefore Rust CUDA does
  not allow mutable references as argument to kernels. Raw pointers do not have this restriction.
  Therefore, we use a pointer and only make a mutable reference once we have an element
  (`c.add(i)`) that we know won't be touched by other kernel invocations.
- The `#[allow(improper_ctypes_definitions)]` follows on from this. The kernel boundary is like an
  FFI boundary, and slices are not normally allowed there because they are not guaranteed to be
  passed in a particular way. However, `rustc_codegen_nvvm` *does* guarantee the way in which
  things like structs, slices, and arrays are passed (see [Kernel ABI](./kernel_abi.md)). Therefore
  this lint can be disabled.
- `thread::index_1d()` gives the globally-unique thread index. The check `i < a.len()` bounds check
  is necessary because threads run in blocks, and sometimes indices that exceed an inputs bounds
  occur.
- The entire crate is compiled as `no_std`. If you want to use `alloc`, just add `extern crate
  alloc;` to the file.
- The crate is actually compiled twice. Once by `cuda_builder` to produce PTX code for the kernels,
  and once normally by Cargo to produce the rlib for definitions (such as `T`) shared with the
  top-level crate.

Although this example only includes one kernel, larger examples contain multiple kernels, which is
why the name `kernels` is used.

### `build.rs`

The build script uses `cuda_builder` to compile the kernel to PTX code. Under the covers,
`cuda_builder` uses rustc with `rustc_codegen_nvvm`. `kernels.ptx` will be embedded in the main
executable.
```rust
use std::env;
use std::path;

use cuda_builder::CudaBuilder;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=kernels");

    let out_dir = path::PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = path::PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // Compile the `kernels` crate to `$OUT_DIR/kernels.ptx`.
    CudaBuilder::new(manifest_dir.join("kernels"))
        .copy_to(out_dir.join("kernels.ptx"))
        .build()
        .unwrap();
}
```

You can specify a different compilation target by inserting an `arch` call in the method chain,
e.g.:

```rust
        .arch(cuda_builder::NvvmArch::Compute90)  // Target compute capability 9.0
```
The compile target determines which GPU features are available. See the [Compute Capability
Gating](./compute_capabilities.md) guide for details on writing code that adapts to different GPU
capabilities.

### `src/main.rs`

The final file contains `main`, which ties everything together.
```rust
use cust::prelude::*;
use kernels::T;
use std::error::Error;

// Embed the PTX code as a static string.
static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the CUDA Driver API. `_ctx` must be kept alive until the end.
    let _ctx = cust::quick_init()?;

    // Create a module from the PTX code compiled by `cuda_builder`.
    let module = Module::from_ptx(PTX, &[])?;

    // Create a stream, which is like a thread for dispatching GPU calls.
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Initialize input and output buffers in CPU memory.
    let a: [T; _] = [1.0, 2.0, 3.0, 4.0];
    let b: [T; _] = [2.0, 3.0, 4.0, 5.0];
    let mut c: Vec<T> = vec![0.0 as T; a.len()];

    // Allocate memory on the GPU and copy the contents from the CPU memory.
    let a_gpu = a.as_dbuf()?;
    let b_gpu = b.as_dbuf()?;
    let c_gpu = c.as_slice().as_dbuf()?;

    // Launch the kernel on the GPU.
    // - The first two parameters between the triple angle brackets specify 1
    //   block of 4 threads.
    // - The third parameter is the number of bytes of dynamic shared memory.
    //   This is usually zero.
    // - These threads run in parallel, so each kernel invocation must modify
    //   separate parts of `c_gpu`. It is the kernel author's responsibility to
    //   ensure this.
    // - Immutable slices are passed via pointer/length pairs. This is unsafe
    //   because the kernel function is unsafe, but also because, like an FFI
    //   call, any mismatch between this call and the called kernel could
    //   result in incorrect behaviour or even uncontrolled crashes.
    let add_kernel = module.get_function("add")?;
    unsafe {
        launch!(
            add_kernel<<<1, 4, 0, stream>>>(
                a_gpu.as_device_ptr(),
                a_gpu.len(),
                b_gpu.as_device_ptr(),
                b_gpu.len(),
                c_gpu.as_device_ptr(),
            )
        )?;
    }

    // Synchronize all threads, i.e. ensure they have all completed before continuing.
    stream.synchronize()?;

    // Copy the GPU memory back to the CPU.
    c_gpu.copy_to(&mut c)?;

    println!("c = {:?}", c);

    Ok(())
}
```

Because `T` is shared between the crates, the type used in the buffers could be changed from `f32`
to `f64` by modifying just the definition of `T`. Without that, such a change would require
modifying lines in both crates, and any inconsistencies could cause correctness problems.

## A first example: building and running

There are two ways to build and run this example: natively, and with docker.

### Native

If you have all the required libraries installed, try building with `cargo build`.

If you get an error "libnvvm.so.4: cannot open shared object file", you will need to adjust
`LD_LIBRARY_PATH`, something like this:
```
export LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64:${LD_LIBRARY_PATH}"
```

If you get an error "error: couldn't load codegen backend" on Windows, you will need to adjust
`PATH`, something like this with CUDA 12:
```
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\nvvm\bin"
```
or this with CUDA 13:
```
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\nvvm\bin\x64"
```

You should then be able to `cargo run`, and see the expected output:
```
c = [3.0, 5.0, 7.0, 9.0]
```

### Docker

Docker is complicated. If you already know how it works, feel free to use the provided images
however you like. The rest of this section aims to provide basic instructions for those who are
less confident.

First, ensure you have Docker setup to [use GPUs]. Even with Docker, your machine will need an
appropriate driver.

[use GPUs]: https://docs.docker.com/config/containers/resource_constraints/#gpu

You can build your own docker image but it is easier to use a prebuilt one. The [`dcr`] script
uses `docker create` with a prebuilt image to create a container that contains the required
libraries. It then uses `docker start` to start the container in such a way that it will run
indefinitely unless explicitly stopped. Even if the host machine is rebooted the container will
automatically restart.

Once the container is started, the [`dex`] script uses `docker exec` to run arbitrary commands
within the container. For example, `dex cargo build` will execute `cargo build` within the
container.

[`dcr`]: https://github.com/Rust-GPU/rust-cuda/blob/main/container/scripts/dcr
[`dex`]: https://github.com/Rust-GPU/rust-cuda/blob/main/container/scripts/dex

Some useful docker commands:
- `docker exec -it rust-cuda bash`: run a bash shell within the container. This lets you operate
  inside the container indefinitely. But facilities within the container are limited, so using
  `dex` to run commands one at a time is generally easier.
- `docker images`: show the status of all local images.
- `docker ps`: show the status of running containers.
- `docker ps --all`: show the status of all containers.
- `docker stop rust-cuda`: stop the `rust-cuda` container.
- `docker rm rust-cuda`: remove the `rust-cuda` container, which must have been stopped.

If you have problems with the container, the following steps may help with checking that your GPU
is recognized.
- Check if `dex nvidia-smi` provides meaningful output.
- NVIDIA provides a number of [samples](https://github.com/NVIDIA/cuda-samples). You could try
  `make`ing and running the [`deviceQuery`] sample. If all is well it will print various details
  about your GPU.

A sample `.devcontainer.json` file is also included, configured for Ubuntu 24.04. Copy this to
`.devcontainer/devcontainer.json` to make additional customizations.

[`deviceQuery`]: https://github.com/NVIDIA/cuda-samples/tree/ba04faaf7328dbcc87bfc9acaf17f951ee5ddcf3/Samples/deviceQuery

## More examples

The [`examples`] directory has more complex examples. They all follow the same basic structure as
this first example.

[`examples`]: https://github.com/Rust-GPU/rust-cuda/tree/main/examples
