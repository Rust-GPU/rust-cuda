# Compute Capability Gating

This section covers how to write code that adapts to different CUDA compute capabilities
using conditional compilation.

## What are Compute Capabilities?

CUDA GPUs have different "compute capabilities" that determine which features they
support. Each capability is identified by a version number like `3.5`, `5.0`, `6.1`,
`7.5`, etc. Higher numbers generally mean more features are available.

For example:

- Compute capability 5.0+ supports 64-bit integer min/max and bitwise atomic operations
- Compute capability 6.0+ supports double-precision (f64) atomic operations
- Compute capability 7.0+ supports tensor core operations

For comprehensive details, see [NVIDIA's CUDA documentation on GPU architectures](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-compilation).

## Virtual vs Real Architectures

In CUDA terminology:

- **Virtual architectures** (`compute_XX`) define the PTX instruction set and available
  features
- **Real architectures** (`sm_XX`) represent actual GPU hardware

Rust CUDA works exclusively with virtual architectures since it only generates PTX. The
`NvvmArch::ComputeXX` enum values correspond to CUDA's virtual architectures.

## Using Target Features

When building your kernel, the `NvvmArch::ComputeXX` variant you choose enables specific
`target_feature` flags. These can be used with `#[cfg(...)]` to conditionally compile
code based on the capabilities of the target GPU.

For example, this checks whether the target architecture supports running compute 6.0
code or newer:

```rust
#[cfg(target_feature = "compute_60")]
```

Think of it as asking: “Is the GPU I’m building for at least compute 6.0?” Depending on
which `NvvmArch::ComputeXX` is used to build the kernel, there is a different answer:

- Building for `Compute60` → ✓ Yes (exact match)
- Building for `Compute70` → ✓ Yes (7.0 GPUs support 6.0 code)
- Building for `Compute50` → ✗ No (5.0 GPUs can't run 6.0 code)

These features let you write optimized code paths for specific GPU generations while
still supporting older ones.

## Specifying Compute Capabilites

Starting with CUDA 12.9, NVIDIA introduced architecture suffixes that affect
compatibility.

### Base Architecture (No Suffix)

Example: `NvvmArch::Compute70`

This is everything mentioned above, and was the only option in CUDA 12.8 and lower.

**When to use**: Default choice for maximum compatibility.

Example usage:

```rust
// In build.rs
CudaBuilder::new("kernels")
    .arch(NvvmArch::Compute70)
    .build()
    .unwrap();

// In your kernel code:
#[cfg(target_feature = "compute_60")]  // ✓ Pass (lower base variant)
#[cfg(target_feature = "compute_70")]  // ✓ Pass (this base variant))
#[cfg(target_feature = "compute_80")]  // ✗ Fail (higher base variant)
```

### Family Suffix ('f')

Example: `NvvmArch::Compute101f`

Specifies code compatible with the same major compute capability version and with an
equal or higher minor compute capability version.

**When to use**: When you need features from a specific minor version but want forward
compatibility within the family.

Example usage:

```rust
// In build.rs
CudaBuilder::new("kernels")
    .arch(NvvmArch::Compute101f)
    .build()
    .unwrap();

// In your kernel code:
#[cfg(target_feature = "compute_90")]    // ✓ Pass (lower base variant)
#[cfg(target_feature = "compute_100")]   // ✓ Pass (lower base variant)
#[cfg(target_feature = "compute_100f")]  // ✓ Pass (lower 'f' variant)
#[cfg(target_feature = "compute_101")]   // ✓ Pass (this base variant)
#[cfg(target_feature = "compute_101f")]  // ✓ Pass (the 'f' variant itself)
#[cfg(target_feature = "compute_103")]   // ✗ Fail (higher base variant)
#[cfg(target_feature = "compute_110")]   // ✗ Fail (higher base variant)
```

### Architecture Suffix ('a')

Example: `NvvmArch::Compute100a`

Specifies code that only runs on GPUs of that specific compute capability and no others.
However, during compilation, it enables all available instructions for the architecture,
including all base variants up to the same version and all family variants with the same
major version and equal or lower minor version.

**When to use**: When you need to use architecture-specific features (like certain
Tensor Core operations) that are only available on that exact GPU model.

Example usage:

```rust
// In build.rs
CudaBuilder::new("kernels")
    .arch(NvvmArch::Compute100a)
    .build()
    .unwrap();

// In your kernel code:
#[cfg(target_feature = "compute_90")]    // ✓ Pass (lower base variant)
#[cfg(target_feature = "compute_100")]   // ✓ Pass (base variant)
#[cfg(target_feature = "compute_100f")]  // ✓ Pass (family variant with same major/minor)
#[cfg(target_feature = "compute_100a")]  // ✓ Pass (the 'a' variant itself)
#[cfg(target_feature = "compute_101f")]  // ✗ Fail (higher family variant)
#[cfg(target_feature = "compute_110")]   // ✗ Fail (higher base variant)
```

Note: While the 'a' variant enables all these features during compilation (allowing you to use all available instructions), the generated PTX code will still only run on the exact GPU architecture specified.

For more details on suffixes, see [NVIDIA's blog post on family-specific architecture features](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/).

### Manual Compilation (Without `cuda_builder`)

If you're invoking `rustc` directly instead of using `cuda_builder`, you only need to specify the architecture through LLVM args:

```bash
rustc --target nvptx64-nvidia-cuda \
    -C llvm-args=-arch=compute_61 \
    -Z codegen-backend=/path/to/librustc_codegen_nvvm.so \
    ...
```

Or with cargo:

```bash
export RUSTFLAGS="-C llvm-args=-arch=compute_61 -Z codegen-backend=/path/to/librustc_codegen_nvvm.so"
cargo build --target nvptx64-nvidia-cuda
```

The codegen backend automatically synthesizes target features based on the architecture type as described above.

### Common Patterns for Base Architectures

These patterns work when using base architectures (no suffix), which enable all lower capabilities:

#### At Least a Capability (Default)

```rust,no_run
// Code that requires compute 6.0 or higher
#[cfg(target_feature = "compute_60")]
{
    cuda_std::atomic::atomic_add(data, 1.0); // f64 atomics need 6.0+
}
```

#### Exactly One Capability

```rust,no_run
// Code that targets exactly compute 6.1 (not 6.2+)
#[cfg(all(target_feature = "compute_61", not(target_feature = "compute_62")))]
{
    // Features specific to compute 6.1
}
```

#### Up To a Maximum Capability

```rust,no_run
// Code that works up to compute 6.0 (not 6.1+)
#[cfg(all(target_feature = "compute_35", not(target_feature = "compute_61")))]
{
    // Maximum compatibility implementation
}
```

#### Targeting Specific Architecture Ranges

```rust,no_run
// This block compiles when building for architectures >= 6.0 but < 8.0
#[cfg(all(target_feature = "compute_60", not(target_feature = "compute_80")))]
{
    // Code here can use features from 6.0+ but must not use 8.0+ features
}
```

## Debugging Capability Issues

If you encounter errors about missing functions or features:

1. Check the compute capability you're targeting in `cuda_builder`
2. Verify your GPU supports the features you're using
3. Use `nvidia-smi` to check your GPU's compute capability
4. Add appropriate `#[cfg]` guards or increase the target architecture

## Runtime Behavior

Again, Rust CUDA **only generates PTX**, not pre-compiled GPU binaries
("[fatbinaries](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#fatbinaries)").
This PTX is then JIT-compiled by the CUDA driver at _runtime_.

For more details, see [NVIDIA's documentation on GPU
compilation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-compilation)
and [JIT
compilation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#just-in-time-compilation).
