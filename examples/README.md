# CUDA + Rust examples

The examples in here showcase both the GPU side and the CPU side of writing a tool which uses the
GPU.

## Available Examples

### [vecadd](vecadd)

A simple vector addition example demonstrating basic CUDA kernel usage.

### [gemm](gemm)

General Matrix Multiplication (GEMM) implementation showing more complex CUDA operations.

### [i128_demo](i128_demo)

A simple demo that uses 128-bit integer operations.

### [sha2_crates_io](sha2_crates_io)

Demonstrates using an existing Rust crate ([`sha2`](https://crates.io/crates/sha2) from crates.io)
on both CPU and GPU without modification. Shows that the same cryptographic hashing code can run on
CUDA, producing identical results to the CPU implementation.

### [Interactive Path Tracer](../crates/optix/examples/path_tracer/)

A very simple interactive Path Tracer inspired by [Ray Tracing In One
Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) which runs on CPU or GPU,
with the additional option of running OptiX denoising.

![Path Tracer](assets/path_tracer.png)

The Path Tracer uses `cuda_builder` to compile the core path tracer for the GPU and GPU (hardware
raytracing), and uses the core path tracer as a normal crate for CPU rendering and sharing
structures.
