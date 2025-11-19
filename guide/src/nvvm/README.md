# `rustc_codegen_nvvm`

This section will cover the more technical details of how `rustc_codegen_nvvm` works 
as well as the issues that came with it.

It will also explain some technical details about CUDA/PTX/etc, it is not necessarily 
limited to `rustc_codegen_nvvm`.

Basic knowledge of how rustc and LLVM work and what they do is assumed. You can find
info about rustc in the [rustc dev guide](https://rustc-dev-guide.rust-lang.org/).
