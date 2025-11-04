# Chapter 0: Introduction

## [asyncAPI](https://github.com/Rust-GPU/rust-cuda/samples/introduction/async_api)
This example demonstrates two key capabilities of CUDA events: measuring GPU execution time and enabling concurrent CPU-GPU operations.

1. Events are recorded at specific points within a CUDA stream to mark the beginning and end of GPU operations.
2. Because CUDA stream operations execute asynchronously, the CPU remains free to perform other work while the GPU processes tasks (including memory transfers between host and device)
3. The CPU can query these events to check whether the GPU has finished its work, allowing for coordination between the two processors without blocking the CPU.
