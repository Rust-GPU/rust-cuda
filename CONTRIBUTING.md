# Contributing to Rust CUDA

Welcome! We're glad you're interested in contributing to the Rust CUDA project. We welcome
contributions from people of all backgrounds who are interested in making great software with us.

## Getting Help

For questions, clarifications, and general help:

1. Search existing [GitHub issues](https://github.com/Rust-GPU/rust-cuda/issues)
2. If you can't find the answer, open a new issue or start a discussion

## Prerequisites

### Required

- **CUDA Toolkit** (12.x or 13.x recommended). Install from
  [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).
- **Rust nightly toolchain** -- the project pins a specific nightly via
  [`rust-toolchain.toml`](rust-toolchain.toml). Running any `cargo` command in the repo
  will automatically install the correct version if you have `rustup`.
- **LLVM tools** -- installed automatically by `rustup` as part of the pinned toolchain
  components.
- A **CUDA-capable GPU** with compute capability >= 3.0.

### Optional

- **cuDNN** -- required only if you're building the `cudnn` / `cudnn-sys` crates. Install
  from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn).
- **mdBook** -- required to build the guide locally. Install with
  `cargo install mdbook`.

### Windows-Specific Notes

- Ensure the CUDA Toolkit `bin` directory is on your `PATH` (e.g.
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin`).
- The MSVC build tools are required. Install via
  [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) with the
  "Desktop development with C++" workload.
- If using cuDNN, place the cuDNN files in your CUDA Toolkit directory or set
  `CUDNN_PATH` to point to the cuDNN installation.
- Some crates require `advapi32` for linking (handled automatically by build scripts).

### Linux-Specific Notes

- Ensure `nvcc` is on your `PATH` and `LD_LIBRARY_PATH` includes the CUDA lib directory.
- The project provides container images for CI; see
  `.github/workflows/ci_linux.yml` for reference.

## Building

Build the entire workspace:

```sh
cargo build
```

Build a specific crate:

```sh
cargo build -p cust
cargo build -p cudnn
```

Run clippy:

```sh
cargo clippy --workspace
```

Run tests (requires a CUDA-capable GPU):

```sh
cargo test --workspace
```

### Building the Guide

The user-facing documentation is an [mdBook](https://rust-lang.github.io/mdBook/) located
in the `guide/` directory.

```sh
# Install mdBook (one-time)
cargo install mdbook

# Build and serve locally
mdbook serve guide --open
```

## Running Examples

Examples live in the `examples/` and `samples/` directories:

```sh
# Vector addition
cargo run -p vecadd

# Matrix multiplication (GEMM)
cargo run -p gemm
```

See [`examples/README.md`](examples/README.md) for the full list.

## Issues

### Feature Requests

If you have ideas for improvements, suggest features by opening a GitHub issue. Include
details about the feature and describe any use cases it would enable.

### Bug Reports

When reporting a bug, make sure your issue describes:

- Steps to reproduce the behavior
- Your platform (OS, GPU, CUDA version, Rust toolchain version)
- Any error messages or logs

### Wontfix

Issues may be closed as `wontfix` if they are misaligned with the project vision or out of
scope. We will comment on the issue with detailed reasoning.

## Contribution Workflow

### Finding Work

Start by looking at open issues tagged as
[`help wanted`](https://github.com/Rust-GPU/rust-cuda/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22)
or
[`good first issue`](https://github.com/Rust-GPU/rust-cuda/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).

Comment on the issue to let others know you're working on it.

### Pull Request Process

1. **Fork** the repository.
2. **Create a new feature branch** from `main`.
3. **Make your changes.** Ensure there are no build errors by running `cargo build` and
   `cargo clippy --workspace` locally.
4. **Open a pull request** with a clear title and description of what you did.
5. A maintainer will review your pull request and may ask you to make changes.

### Commit Messages

This project follows the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
specification. Each commit message should have the format:

```
<type>(<scope>): <description>
```

**Types:** `feat`, `fix`, `docs`, `chore`, `ci`, `test`, `refactor`, `perf`, `style`

**Scopes** (common examples): `cust`, `cudnn`, `cudnn-sys`, `cust_raw`, `cuda_std`,
`nvvm`, `vecadd`, `guide`, `windows`

**Examples:**

```
feat(cudnn): add batch normalization forward/backward
fix(cust_raw): correct Windows CUDA path discovery
docs(guide): add Windows getting-started section
ci(windows): include vecadd in workspace build
```

## Project Structure

| Directory | Description |
| --- | --- |
| `crates/cust` | High-level safe wrapper around the CUDA Driver API |
| `crates/cust_core` | Core `DeviceCopy` trait shared between host and device |
| `crates/cust_raw` | Low-level `bindgen` bindings to CUDA SDK |
| `crates/cudnn` | Type-safe cuDNN wrapper |
| `crates/cudnn-sys` | Low-level `bindgen` bindings to cuDNN |
| `crates/cuda_std` | GPU-side standard library |
| `crates/cuda_std_macros` | Proc macros (`#[kernel]`, `#[gpu_only]`, etc.) |
| `crates/cuda_builder` | Build-time helper for compiling GPU kernels |
| `crates/rustc_codegen_nvvm` | Custom rustc backend targeting NVVM/PTX |
| `crates/nvvm` | Wrapper around NVIDIA's libNVVM |
| `crates/blastoff` | cuBLAS bindings |
| `examples/` | Example programs |
| `samples/` | Ports of NVIDIA CUDA samples |
| `guide/` | mdBook source for the Rust CUDA Guide |

## Licensing

This project is dual-licensed under Apache-2.0 or MIT, at your discretion. Unless you
explicitly state otherwise, any contribution intentionally submitted for inclusion in the
work shall be dual-licensed as above, without any additional terms or conditions.
