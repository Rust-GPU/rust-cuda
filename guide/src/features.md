# Supported Features 

This page is used for tracking Cargo/Rust and CUDA features that are currently supported 
or planned to be supported in the future. As well as tracking some information about how they could 
be supported.

Note that `Not supported` does __not__ mean it won't ever be supported, it just means we haven't gotten
around to adding it yet.

| Indicator | Meaning |
| --------- | ------- |
| ➖ | Not Applicable |
| ❌ | Not Supported |
| ✔️ | Fully Supported |
| 🟨 | Partially Supported |

# Rust Features

| Feature Name | Support Level | Notes |
| ------------ | ------------- | ----- |
| Opt-Levels | ✔️ | behaves mostly the same (because LLVM is still used for optimizations). Except that libNVVM opts are run on anything except no-opts because NVVM only has -O0 and -O3 |
| codegen-units | ✔️ |
| LTO | ➖ | we load bitcode modules lazily using dependency graphs, which then forms a single module optimized by libNVVM, so all the benefits of LTO are on without pre-libNVVM LTO being needed. |
| Closures | ✔️ |
| Enums | ✔️ |
| Loops | ✔️ |
| If | ✔️ |
| Match | ✔️ |
| Proc Macros | ✔️ |
| Try (`?`) | ✔️ |
| 128 bit integers | 🟨 | Basic ops should work (and are emulated), advanced intrinsics like `ctpop`, `rotate`, etc are unsupported. |
| Unions | ✔️ |
| Iterators | ✔️ |
| Dynamic Dispatch | ✔️ |
| Pointer Casts | ✔️ |
| Unsized Slices | ✔️ |
| Alloc | ✔️ |
| Printing | ✔️ |
| Panicking | ✔️ | Currently just traps (aborts) because of weird printing failures in the panic handler |
| Float Ops | ✔️ | Maps to libdevice intrinsics, calls to libm are not intercepted though, which we may want to do in the future |
| Atomics | ❌ | 

# CUDA Libraries

| Library Name | Support Level | Notes |
| ------------ | ------------- | ----- |
| CUDA Runtime API | ➖ | The CUDA Runtime API is for CUDA C++, we use the driver API | 
| CUDA Driver API | 🟨 | Most functions are implemented, but there is still a lot left to wrap because it is gigantic | 
| cuBLAS | ❌ | In-progress |
| cuFFT | ❌ |
| cuSOLVER | ❌ |
| cuRAND | ➖ | cuRAND only works with the runtime API, we have our own general purpose GPU rand library called `gpu_rand` |
| cuDNN | ❌ | In-progress |
| cuSPARSE | ❌ |
| AmgX | ❌ |
| cuTENSOR | ❌ |
| OptiX | 🟨 | CPU OptiX is mostly complete, GPU OptiX is still heavily in-progress because it needs support from the codegen | 

# GPU-side Features

Note: Most of these categories are used __very__ rarely in CUDA code, therefore
do not be alarmed that it seems like many things are not supported. We just focus
on things used by the wide majority of users.

| Feature Name | Support Level | Notes |
| ------------ | ------------- | ----- |
| Function Execution Space Specifiers | ➖ |
| Variable Memory Space Specifiers | ✔️ | Handled Implicitly but can be explicitly stated for statics with `#[address_space(...)]` |
| Built-in Vector Types | ➖ | Use linear algebra libraries like vek or glam |
| Built-in Variables | ✔️ |
| Memory Fence Instructions | ✔️ |
| Synchronization Functions | ✔️ |
| Mathematical Functions | 🟨 | Less common functions like native f16 math are not supported |
| Texture Functions | ❌ |
| Surface Functions | ❌ |
| Read-Only Data Cache Load Function | ❌ | No real need, immutable references hint this automatically |
| Load Functions Using Cache Hints | ❌ |
| Store Functions Using Cache Hints | ❌ |
| Time Function | ✔️ | 
| Atomic Functions | ❌ |
| Address Space Predicate Functions | ✔️ | Address Spaces are implicitly handled, but they may be added for exotic interop with CUDA C/C++ |
| Address Space Conversion Functions | ✔️ |
| Alloca Function | ➖ |
| Compiler Optimization Hint Functions | ➖ | Existing `core` hints work |
| Warp Vote Functions | ❌ |
| Warp Match Functions | ❌ |
| Warp Reduce Functions | ❌ |
| Warp Shuffle Functions | ❌ |
| Nanosleep | ✔️ |
| Warp Matrix Functions (Tensor Cores) | ❌ |
| Asynchronous Barrier | ❌ |
| Asynchronous Data Copies | ❌ |
| Profiler Counter Function | ✔️ |
| Assertion | ✔️ |
| Trap Function | ✔️ |
| Breakpoint | ✔️ |
| Formatted Output | ✔️ |
| Dynamic Global Memory Allocation | ✔️ |
| Execution Configuration | ✔️ |
| Launch Bounds | ❌ |
| Pragma Unroll | ❌ |
| SIMD Video Instructions | ❌ |
| Cooperative Groups | ❌ |
| Dynamic Parallelism | ❌ |
| Stream Ordered Memory | ✔️ |
| Graph Memory Nodes | ❌ |
| Unified Memory | ✔️ |
| `__restrict__` | ➖ | Not needed, you get that performance boost automatically through rust's noalias :) |
