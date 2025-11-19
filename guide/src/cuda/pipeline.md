# The CUDA Pipeline

CUDA is traditionally used via CUDA C/C++ files which have a `.cu` extension. These files can be
compiled using NVCC (NVIDIA CUDA Compiler) into an executable.

CUDA files consist of **device** and **host** functions. **Device** functions run on the GPU, and
are also called kernels. **Host** functions run on the CPU and usually include logic on how to
allocate GPU memory and call device functions.

Behind the scenes, NVCC has several stages of compilation.

First, NVCC separates device and host functions and compiles them separately. Device functions are
compiled to [NVVM IR](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html), a subset of LLVM IR
with additional restrictions including the following.
- Many intrinsics are unsupported.
- "Irregular" integer types such as `i4` or `i111` are unsupported and will segfault (however in
  theory they should be supported).
- Global names cannot include `.`.
- Some linkage types are not supported.
- Function ABIs are ignored; everything uses the PTX calling convention.

libNVVM is a closed source library which takes NVVM IR, optimizes it further, then converts it to
PTX. PTX is a low level, assembly-like format with an open specification which can be targeted by
any language. For an assembly format, PTX is fairly user-friendly.
- It is well formatted.
- It is mostly fully specified (other than the iffy grammar specification).
- It uses named registers/parameters.
- It uses virtual registers. (Because GPUs have thousands of registers, listing all of them out
  would be unrealistic.)
- It uses ASCII as a file encoding.

PTX can be run on NVIDIA GPUs using the driver API or runtime API. Those APIs will convert the PTX
into a final format called SASS which is register allocated and executed on the GPU.
