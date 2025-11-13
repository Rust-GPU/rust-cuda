# PTX Generation 

This is the final and most fun part of codegen, taking our LLVM bitcode and giving it to libnvvm. 
It is in theory as simple as just giving nvvm every single bitcode module, but in practice, we do a couple 
of things before and after to reduce ptx size and speed things up.

# The NVVM API

libnvvm is a dynamically linked library which is distributed in every download of the CUDA SDK. 
If you are on windows, it should be somewhere around `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/nvvm/bin`
where `v11.3` is the version of cuda you have downloaded. On Windows it's usually called `nvvm64_40_0.dll`. If you are 
on linux it should be somewhere around `/opt/cuda/nvvm-prev/lib64/libnvvm.so`. You can see its API either in the
[API docs](https://docs.nvidia.com/cuda/libnvvm-api/group__compilation.html) or in its header file in the `include` folder.

We have our own high level bindings to it published as a crate called `nvvm`.

The libnvvm API could not be simpler, it is just a couple of functions:
- Make new program
- Add bitcode module
- Lazy add bitcode module
- Verify program
- Compile program

The first step is always making a new program, a program is just a container for 
modules that then gets compiled. 

# Module loading

This is the most important part, we need to add our LLVM bitcode to the program, that 
should be a very simple thing that would involve no calls to random functions in the rustc
haystack, ...right? Why of course not, you didn't seriously think we would make this
straight-forward, right?

So, in theory it is very simple, just load the bitcode from the rlib and tell nvvm to load it. 
While this is easy and it works, it has its own very visible issues.

Traditionally, if you never use a function, either the compiler destroys it when using LTO, or 
the linker destroys it in its own dead code pass. The issue is that LTO is not always run,
and we do not have a linker, nvvm *is* our linker. However, nvvm does not eliminate dead functions.
I think you can guess why that is a problem, so unless we want `11mb` ptx files (yes this is actually
how big it was) we need to do something about it.

# Module Merging and DCE

To solve our dead code issue, we take a pretty simple approach. We merge every module (one crate maybe be multiple modules
because of codegen units) into a single module to start. Then, we do the following:
- (Internalize) Iterate over every global and function then:
  - If the global/function is not a declaration (i.e. an extern decl) and not a kernel, then mark its linkage as `internal` and give it default visibility.
- (Global DCE) Run the `globalDCE` LLVM Pass over the module. This will delete any globals/functions we do not use.

Internal linkage tells LLVM that the symbol is not externally-needed, meaning that it can delete the symbol if
it is not used by other non-internal functions. In this case, our non-internal functions are kernel functions.

In the future we could probably make this even better by combining our previous lazy-loading approach, by only loading functions/modules
into the module if they are used, doing so using dependency graphs.

# libdevice

There are a couple of special modules we need to load before we are done, `libdevice` and `libintrinsics`. The first and most
important one is libdevice, libdevice is essentially a bitcode module containing hyper-optimized math intrinsics
that nvidia provides for us. You can find it as a `.bc` file in the libdevice folder inside your nvvm install location.
Every function inside of it is prefixed with `__nv_`, you can find docs for it [here](https://docs.nvidia.com/cuda/libdevice-users-guide/index.html).

We declare these intrinsics inside of `ctx_intrinsics.rs` and link to them inside cuda_std. We also use them to codegen
a lot of intrinsics inside `intrinsic.rs`, such as `sqrtf32`.

libdevice is also lazy loaded so we do not import useless intrinsics.

# libintrinsics

This is the last special module we load, it is simple, it is just a dumping ground for random wrapper functions 
we need to define that `cuda_std` or the codegen needs. You can find the llvm ir definition for it in the codegen directory
called `libintrinsics.ll`. All of its functions should be declared with the `__nvvm_` prefix.

# Compilation

Finally, we have everything loaded and we can compile our program. We do one last thing however.

Nvvm has a function for verifying our program to make sure we did not add anything nvvm does not like. We run this 
before compilation just to be safe. Although annoyingly this does not catch all errors, nvvm just segfaults sometimes which is unfortunate.

Compiling is simple, we just call nvvm's program compile function and panic if it fails, if it doesn't, we get a final PTX string. We 
can then just write that to the file that rustc wants us to put the final item in.
