# Frequently Asked Questions 

This page will cover a lot of the questions people often have when they encounter this project,
so they are addressed all at once.

## Why not use rustc with the LLVM PTX backend?

Good question, a good amount of reasons:
- The LLVM PTX backend is still very much WIP and often doesn't have things and/or breaks.
- Due to odd dylib issues, the LLVM PTX backend does not work on windows, it will fail to link in intrinsics. 
This can be circumvented by building LLVM in a special way, but this is far beyond what most users will do to get a backend to work.
- NVVM is used in NVCC itself, therefore NVIDIA is much more receptive to bugs inside of it. 
- NVVM contains proprietary optimizations (which is why it's closed source) that are simply not present in the LLVM PTX backend
which yield considerable performance differences (especially on more complex kernels with more information in the IR).
- For some reason (either rustc giving weird LLVM IR or the LLVM PTX backend being broken) the LLVM PTX backend often
generates completely invalid PTX for trivial programs, so it is not an acceptable workflow for a production pipeline.
- GPU and CPU codegen is fundamentally different, creating a codegen that is only for the GPU allows us to 
seamlessly implement features which would have been impossible or very difficult to implement in the existing codegen, such as:
  - Shared memory, this requires some special generation of globals with custom addrspaces, its just not possible to do without backend explicit handling.
  - Custom linking logic to do dead code elimination so as to not end up with large PTX files full of dead functions/globals.
  - Stripping away everything we do not need, no complex ABI handling, no shared lib handling, control over how function calls are generated, etc.

So overall, the LLVM PTX backend is fit for smaller kernels/projects/proofs of concept.
It is however not fit for compiling an entire language (core is __very__ big) with dependencies and more. The end goal is for Rust to be able to be used 
over CUDA C/C++ with the same (or better!) performance and features, therefore, we must take advantage of all optimizations NVCC has over us.

## If NVVM IR is a subset of LLVM IR, can we not give rustc-generated LLVM IR to NVVM?

Short answer, no.

Long answer, there are a couple of things that make this impossible:
- At the time of writing, libNVVM expects LLVM 7 bitcode, which is a very old format. Giving it bitcode from later LLVM version (which is what rustc uses) does not work.
- NVVM IR is a __subset__ of LLVM IR, there are tons of things that NVVM will not accept. Such as a lot of function attrs not being allowed. 
This is well documented and you can find the spec [here](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html). Not to mention
many bugs in libNVVM that I have found along the way, the most infuriating of which is nvvm not accepting integer types that arent `i1, i8, i16, i32, or i64`.
This required special handling in the codegen to convert these "irregular" types into vector types.

## What is the point of using Rust if a lot of things in kernels are unsafe?

This is probably the most asked question by far, so let's break it down in detail. 

TL;DR There are things we fundamentally can't check, but just because that is the case does not mean
we cannot still prevent a lot of problems we *can* check.

Yes it is true that GPU kernels have much more unsafe than CPU code usually, but why is that?

The reason is that CUDA's entire model is not based on safety in any way, there are almost zero
safety nets in CUDA. Rust is the polar opposite of this model, everything is safe unless there 
are some invariants that cannot be checked by the compiler. Let's take a look at some of the
invariants we face here.

Take this program as an example, written in CUDA C++:

```cpp
__global__ void kernel(int* buf, int* other)
{
  int idx = threadIdx.x;
  buf[idx] = other[idx];
}

int main(void)
{
  int N = 50;
  int* a, b, d_a, d_b;
  a = (int*)malloc(N*sizeof(int));
  b = (int*)malloc(N*sizeof(int));

  cudaMalloc(&d_a, N*sizeof(int));
  cudaMalloc(&d_b, N*sizeof(int));

  for (int i = 0; i < N; i++) {
    a[i] = 0.0f;
    b[i] = 2.0f;
  }

  cudaMemcpy(d_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, N*sizeof(float), cudaMemcpyHostToDevice);

  kernel<<<1, N>>>(d_a, d_b);

  cudaMemcpy(d_a, a, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(d_b, b, N*sizeof(float), cudaMemcpyDeviceToHost);

  /* do something with the data */

  cudaFree(d_a);
  cudaFree(d_b);
  free(a);
  free(b);
}
```

You may think this looks innocent enough, it's a very easy and understandable program. But
if you really think about it, this is a minefield of things that could go wrong. Let's list most of them:
- `buf` could be too small, that is undefined behavior (reading beyond allocated memory)
- similarly, `other` could also be too small.
- The kernel could have been called with too many or not enough parameters.
- The kernel could have been called with a different grid/block dimension than expected, which would cause a data race.
- Any of the `cudaMalloc`, `cudaMemcpy`, kernel launches, or `cudaFree` calls could have errored, which we dont handle and simply ignore.
- We could have forgotten to initialize the buffers.
- We could have forgotten to free the buffers.

This goes to show that CUDA C/C++ and CUDA overall rely on shifting the burden of correctness from the API to the developer.
However, Rust uses a completely opposite design model, the compiler verifies as much as it can, and burden is only shifted to the 
developer if its absolutely essential, behind `unsafe`.

This creates a big problem for us, it is very difficult (and sometimes impossible) to prove correctness statically when wrapping 
how CUDA works. We can solve a lot of the points using things like RAII and providing a high level wrapper, but we fundamentally
cannot prove a lot of things, the most common place where this is shown is the CPU-GPU boundary, e.g. launching kernels.

Firstly, we cannot verify that the PTX we are calling is sound, that it has no data races, writes into the right buffers, doesnt rely
on undocumented invariants, and does not write invalid data to buffers. This already makes launching kernels perma-unsafe.

Second, CUDA does zero validation in terms of kernel parameter mismatch, it will simply segfault on you, or even keep going 
but produce invalid data (or cause the kernel to cause undefined behavior). This is a design flaw in CUDA itself, we have 
no control over it and no 100% reliable way to fix it, therefore we must shift this burden of correctness to the developer. 

Moreover, the CUDA GPU kernel model is entirely based on trust, trusting each thread to index into the correct place in buffers,
trusting the caller of the kernel to uphold some dimension invariants, etc. This is once again, completely incompatible with how 
Rust does things. We can provide wrappers to calculate an index that always works, and macros to index a buffer automatically, but 
indexing in complex ways is a core operation in CUDA and it is impossible for us to prove that whatever the developer is doing is correct.

Finally, We would love to be able to use mut refs in kernel parameters, but this is would be unsound. Because
each kernel function is *technically* called multiple times in parallel with the same parameters, we would be
aliasing the mutable ref, which rustc declares as unsound (aliasing mechanics). So raw pointers or slightly-less-unsafe
need to be used. However, they are usually only used for the initial buffer indexing, after which you can turn them into a
mutable reference just fine (because you indexed in a way where no other thread will index that element). Also note
that shared refs can be used as parameters just fine.

Now that we outlined why this is a thing, why is using Rust a benefit if we still need to use unsafe?

Well it's simple, eliminating most of the things that a developer needs to think about to have a safe program
is still exponentially safer than leaving __everything__ to the developer to think about. 

By using Rust, we eliminate:
- The forgotten/unhandled CUDA errors problem (yay results!).
- The uninitialized memory problem.
- The forgetting to dealloc memory problem.
- All of the inherent C++ problems in the kernel beyond the initial buffer indexing.
- The mismatched grid/block dimension problem (by providing `thread::index`).
- The forgetting to memcpy data back problem.

And countless other problems with things like graphs, streams, devices, etc.

So, just because we cannot solve *every* problem with CUDA safety, does not mean we cannot solve 
a lot of them, and ease the burden of correctness from the developer. 

Besides, using Rust only adds to safety, it does not make CUDA *more* unsafe. This means there are only
things to gain in terms of safety using Rust.

## Why not use Rust GPU with compute shaders?

The reasoning for this is the same reasoning as to why you would use CUDA over opengl/vulkan compute shaders:
- CUDA usually outperforms shaders if kernels are written well and launch configurations are optimal.
- CUDA has many useful features such as shared memory, unified memory, graphs, fine grained thread control, streams, the PTX ISA, etc.
- Rust GPU does not perform many optimizations, and with `rustc_codegen_ssa`'s less than ideal codegen, the optimizations by LLVM and libNVVM are needed.
- SPIR-V is arguably still not suitable for serious GPU kernel codegen, it is underspecced, complex, and does not mention many things which are needed.
While libNVVM (which uses a well documented subset of LLVM IR) and the PTX ISA are very thoroughly documented/specified.
- Rust GPU is primarily focused on graphical shaders, compute shaders are secondary, which the Rust ecosystem needs, but it also 
needs a project 100% focused on computing, and computing only.
- SPIR-V cannot access many useful CUDA libraries such as OptiX, cuDNN, cuBLAS, etc.
- SPIR-V debug info is still very young and Rust GPU cannot generate it. While `rustc_codegen_nvvm` does, which can be used
for profiling kernels in something like nsight compute.

Moreover, CUDA is the primary tool used in big computing industries such as VFX and scientific computing. Therefore 
it is much easier for CUDA C++ users to use Rust for GPU computing if most of the concepts are still the same. Plus,
we can interface with existing CUDA code by compiling it to PTX then linking it with our Rust code using the CUDA linker
API (which is exposed in a high level wrapper in cust).

## Why use the CUDA Driver API over the Runtime API?

Simply put, the driver API provides better control over concurrency, context, and module management, and overall has better performance 
control than the runtime API.

Let's break it down into the main new concepts introduced in the Driver API.

### Contexts

The first big difference in the driver API is that CUDA context management is explicit and not implicit.

Contexts are similar to CPU processes, they manage all of the resources, streams, allocations, etc associated with
operations done inside them. 

The driver API provides control over these contexts. You can create new contexts and drop them at any time. 
As opposed to the runtime API which works off of an implicit context destroyed on device reset. This
causes a problem for larger applications because a new integration of CUDA could call device reset
when it is finished, which causes further uses of CUDA to fail.

### Modules

Modules are the second big difference in the driver API. Modules are similar to shared libraries, they
contain all of the globals and functions (kernels) inside of a PTX/cubin file. The driver API
is language-agnostic, it purely works off PTX/cubin files. To answer why this is important we
need to cover what cubins and PTX files are briefly.

PTX is a low level assembly-like language which is the penultimate step before what the GPU actually
executes. It is human-readable and you can dump it from a CUDA C++ program with `nvcc ./file.cu --ptx`.
This PTX is then optimized and lowered into a final format called SASS (Source and Assembly) and 
turned into a cubin (CUDA binary) file. 

Driver API modules can be loaded as either PTX, cubin, or fatbin files. If they are loaded as 
PTX then the driver API will JIT compile the PTX to cubin then cache it. You can also
compile PTX to cubin yourself using ptx-compiler and cache it.

This pipeline provides much better control over what functions you actually need to load and cache.
You can separate different functions into different modules you can load dynamically (and even dynamically reload).
This can yield considerable performance benefits when dealing with a lot of functions.

### Streams

Streams are (one of) CUDA's way of dispatching multiple kernels in parallel. You can kind of think of them 
as OS threads essentially. Kernels dispatched one after the other inside of a particular stream
will execute one after the other on the GPU, which is helpful for kernels that rely on a previous kernel's result.

The CUDA runtime API operates off of a single global stream. This causes a lot of issues for users of large programs or libraries that
need to manage many kernels being dispatched at the same time as efficiently as possible.

## Why target NVIDIA GPUs only instead of using something that can work on AMD?

This is a complex issue with many arguments for both sides, so I will give you
both sides as well as my opinion. 

Pros for using OpenCL over CUDA:
- OpenCL (mostly) works on everything because it is a specification, not an actual centralized tool.
- OpenCL will be decently fast on most systems.

Cons for using OpenCL over CUDA:
- Just like all open specifications, not every implementation is as good or supports the same things.
Just because the absolute basics work, does not mean more exotic features work on everything because
some vendors may lag behind others.
- OpenCL is slow to add new features, this is a natural consequence of being an open specification many
vendors need to implement. For example, OpenCL 3.0 (which was announced in around April 2020) is supported
by basically nobody. NVIDIA cards support OpenCL 2.0 while AMD cards support OpenCL 2.1. This means
new features cannot be reliably relied upon because they are unlikely to work on a lot of cards for a LONG time.
- OpenCL can only be written in OpenCL C (based on C99), OpenCL C++ is a thing, but again, not everything
supports it. This makes complex programs more difficult to create.
- OpenCL has less tools and libraries.
- OpenCL is nowhere near as language-agnostic as CUDA. CUDA works almost fully off of an assembly format (PTX)
and debug info. Essentially how CPU code works. This makes writing language-agnostic things in OpenCL near impossible and
locks you into using OpenCL C.
- OpenCL is plagued with serious driver bugs which have not been fixed, or that occur only on certain vendors.

Pros for using CUDA over OpenCL:
- CUDA is for the most part the industry-standard tool for "higher level" computing such as scientific or
VFX computing.
- CUDA is a proprietary tool, meaning that NVIDIA is able to push out bug fixes and features much faster
than releasing a new spec and waiting for vendors to implement it. This allows for more features being added, 
such as cooperative kernels, CUDA graphs, unified memory, new profilers, etc.
- CUDA is a single entity, meaning that if something does or does not work on one system it is unlikely 
that that will be different on another system. Assuming you are not using different architectures, where
one GPU may be lacking a feature.
- CUDA is usually 10-30% faster than OpenCL overall, this is likely due to subpar OpenCL drivers by NVIDIA,
but it is unlikely this performance gap will change in the near future.
- CUDA has a much richer set of libraries and tools than OpenCL, such as cuFFT, cuBLAS, cuRand, cuDNN, OptiX, NSight Compute, cuFile, etc.
- You can seamlessly use existing CUDA C/C++ code with `cust` or `rustc_codegen_nvvm`-generated PTX by
using the CUDA linker APIs which are exposed in `cust`. Allowing for incremental switching to Rust.
- There is a generally larger set of code samples in CUDA C/C++ over OpenCL.
- Documentation is __far__ better, there are (mostly) complete API docs for every single CUDA library and function out there.
- CUDA generally offers more control over the internals of how CUDA executes your GPU code. For example, you can choose
to keep PTX which uses a virtual architecture, or you can compile that to cubin (SASS) and cache that for faster load times.

Cons for using CUDA over OpenCL:
- CUDA only works on NVIDIA GPUs.

# What makes cust and RustaCUDA different?

Cust is a fork of RustaCUDA which changes a lot of things inside of it, as well as adds new features that
are not inside of RustaCUDA. 

The most significant changes (This list is not complete!!) are:
- Drop code no longer panics on failure to drop raw CUDA handles, this is so that InvalidAddress errors, which cause 
CUDA to nuke the driver and nuke any memory allocations no longer cause piles of panics from device boxes trying to be 
dropped when returning from the function with `?`.
- cuda-sys is no longer used, instead, we have our own bindings `cust_raw` so we can ensure updates to the latest CUDA features.
- CUDA occupancy functions have been added.
- PTX linking functions have been added.
- Native support for `vek` linear algebra types for grid/block dimensions and DeviceCopy has been added under the `vek` feature.
- Util traits have been added.
- Basic graph support has been added.
- Some functions have been renamed.
- Some functions have been added.

Changes that are currently in progress but not done/experimental:
- Surfaces
- Textures
- Graphs
- PTX validation

Just like RustaCUDA, cust makes no assumptions of what language was used to generate the PTX/cubin. It could be 
C, C++, futhark, or best of all, Rust!

Cust's name is literally just Rust + CUDA mashed together in a horrible way.
Or you can pretend it stands for custard if you really like custard.
