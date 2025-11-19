# GPU Computing

You probably already know what GPU computing is, but if you don't, it is utilizing the extremely parallel
nature of GPUs for purposes other than rendering. It is widely used in many scientific and consumer fields.
Some of the most common uses being fluid/smoke simulation, protein folding, physically based rendering, 
cryptocurrency mining, AI model training, etc. 

GPUs excel at tasks that do mostly the same thing every time, and need to do it millions of times. 
They do not excel at so-called "divergent" tasks, tasks where each run of the task may take different amounts
of time and/or take different code paths.

# Why CUDA?

CUDA is currently one of the best choices for fast GPU computing for multiple reasons:
- It offers deep control over how kernels are dispatched and how memory is managed.
- It has a rich ecosystem of tutorials, guides, and libraries such as cuRAND, cuBLAS, libNVVM, OptiX, the PTX ISA, etc.
- It is mostly unmatched in performance because it is solely meant for computing and offers rich control.
And more...

However, CUDA can only run on NVIDIA GPUs, which precludes AMD GPUs from tools that use it. However, this is a drawback that 
is acceptable by many because of the significant developer cost of supporting both NVIDIA GPUs with CUDA and 
AMD GPUs with OpenCL, since OpenCL is generally slower, clunkier, and lacks libraries and docs on par with CUDA.

# Why Rust?

Rust is a great choice for GPU programming, however, it has needed a kickstart, which is what `rustc_codegen_nvvm` tries to 
accomplish; The initial hurdle of getting Rust to compile to something CUDA can run is over, now comes the design and 
polish part. 

On top of its rich language features (macros, enums, traits, proc macros, great errors, etc), Rust's safety guarantees
can be applied in GPU programming too; A field that has historically been full of implied invariants and unsafety, such
as (but not limited to):
- Expecting some amount of dynamic shared memory from the caller.
- Expecting a certain layout for thread blocks/threads.
- Manually handling the indexing of data, leaving code prone to data races if not managed correctly.
- Forgetting to free memory, using uninitialized memory, etc.

Not to mention the standardized tooling that makes the building, documentation, sharing, and linting of GPU kernel libraries easily possible.
Most of the reasons for using rust on the CPU apply to using Rust for the GPU, these reasons have been stated countless times so
I will not repeat them here. 

A couple of particular rust features make writing CUDA code much easier: RAII and Results.
In `cust` everything uses RAII (through `Drop` impls) to manage freeing memory and returning handles, which 
frees users from having to think about that, which yields safer, more reliable code.

Results are particularly helpful, almost every single call in every CUDA library returns a status code in the form of a CUDA result.
Ignoring these statuses is very dangerous and can often lead to random segfaults and overall unreliable code. For this purpose,
both the CUDA SDK, and other libraries provide macros to handle such statuses. This handling is not very reliable and causes
dependency issues down the line. 

Instead of an unreliable system of macros, we can leverage rust results for this. In cust we return special `CudaResult<T>`
results that can be bubbled up using rust's `?` operator, or, similar to `CUDA_SAFE_CALL` can be unwrapped or expected if 
proper error handling is not needed. 
