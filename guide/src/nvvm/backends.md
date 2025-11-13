# Custom Rustc Backends

Before we get into the details of rustc_codegen_nvvm, we obviously need to explain what a codegen is!

Custom codegens are rustc's answer to "well what if i want rust to compile to X?". This is a problem
that comes up in many situations, especially conversations of "well LLVM cannot target this, so we are screwed".
To solve this problem, rustc decided to incrementally decouple itself from being attached/reliant on LLVM exclusively.

Previously, rustc only had a single codegen, the LLVM codegen. The LLVM codegen translated MIR directly to LLVM IR.
This is great if you just want to support LLVM, but LLVM is not perfect, and inevitably you will hit limits to what LLVM
is able to do. Or, you may just want to stop using LLVM, LLVM is not without problems (it is often slow, clunky to deal with, 
and does not support a lot of targets). 

Nowadays, Rustc is almost fully decoupled from LLVM and it is instead generic over the "codegen" backend used.
Rustc instead uses a system of codegen backends that implement traits and then get loaded as dynamically linked libraries.
This allows rust to compile to virtually anything with a surprisingly small amount of work. At the time of writing, there are
five publicly known codegens that exist:
- rustc_codegen_clif, cranelift
- rustc_codegen_llvm
- rustc_codegen_gcc
- rustc_codegen_spirv
- rustc_codegen_nvvm, obviously the best codegen ;)

`rustc_codegen_clif` targets the cranelift backend, which is a codegen backend written in rust that is faster than LLVM but does not have many optimizations
compared to LLVM. `rustc_codegen_llvm` is obvious, it is the backend almost everybody uses which targets LLVM. `rustc_codegen_gcc` targets GCC (GNU Compiler Collection)
which is able to target more exotic targets than LLVM, especially for embedded. `rustc_codegen_spirv` targets the SPIR-V (Standard Portable Intermediate Representation 5)
format, which is a format mostly used for compiling shader languages such as GLSL or WGSL to a standard representation that Vulkan/OpenGL can use, the reasons
why SPIR-V is not an alternative to CUDA/rustc_codegen_nvvm have been covered in the [FAQ](../../faq.md).

Finally, we come to the star of the show, `rustc_codegen_nvvm`. This backend targets NVVM IR for compiling rust to gpu kernels that can be run by CUDA. 
What NVVM IR/libnvvm are has been covered in the [CUDA section](../../cuda/pipeline.md).

# rustc_codegen_ssa

Despite its name, `rustc_codegen_ssa` does not actually codegen to anything, it is however the central crate behind every single codegen.
The SSA codegen does most of the hard work in codegen, which is actually codegenning MIR and taking care of managing codegen altogether.

The SSA codegen abstracts away the MIR lowering logic so that custom codegens do not have to implement the time consuming logic of lowering MIR,
they can just implement a bunch of traits and the SSA codegen does everything else.

The SSA codegen is literally just a bunch of traits, for example:
- A trait for getting a type like an integer type
- A trait for optimizing a module
- A trait for linking everything
- A trait for declaring a function
...etc
You will find an SSA codegen trait in almost every single file.
