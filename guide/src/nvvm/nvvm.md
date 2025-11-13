# rustc_codegen_nvvm

At the highest level, our codegen workflow goes like this:

```
Source code -> Typechecking -> MIR -> SSA Codegen -> LLVM IR (NVVM IR) -> PTX -> PTX opts/function DCE -> Final PTX
               |                                     |                  |      |                                  ^
               |                                     |          libnvvm +------+                                  |
               |                                     |                                                            |
               |                  rustc_codegen_nvvm +------------------------------------------------------------|
         Rustc +---------------------------------------------------------------------------------------------------
```

Before we do anything, rustc does its normal job, it typechecks, converts everything to MIR, etc. Then, 
rustc loads our codegen shared lib and invokes it to codegen the MIR. It creates an instance of
`NvvmCodegenBackend` and it invokes `codegen_crate`. You could do anything inside `codegen_crate` but 
we just defer back to rustc_codegen_ssa and tell it to do the job for us:

```rs
fn codegen_crate<'tcx>(
    &self,
    tcx: TyCtxt<'tcx>,
    metadata: EncodedMetadata,
    need_metadata_module: bool,
) -> Box<dyn std::any::Any> {
    Box::new(rustc_codegen_ssa::base::codegen_crate(
        NvvmCodegenBackend,
        tcx,
        String::new(),
        metadata,
        need_metadata_module,
    ))
}
```

After that, the codegen logic is kind of abstracted away from us, which is a good thing!
We just need to provide the SSA codegen whatever it needs to do its thing. This is 
done in the form of traits, lots and lots and lots of traits, more traits than you've ever seen, traits
your subconscious has warned you of in nightmares, anyways. Because talking about how the SSA codegen
works is kind of useless, we will instead talk first about general concepts and terminology, then 
dive into each trait. 

But first, let's talk about the end of the codegen, it is pretty simple, we do a couple of things:
*after codegen is done and LLVM has been run to optimize each module*
- 1: We gather every llvm bitcode module we created.
- 2: We create a new libnvvm program.
- 3: We add every bitcode module to the libnvvm program.
- 4: We try to find libdevice and add it to the program (see [nvidia docs](https://docs.nvidia.com/cuda/libdevice-users-guide/introduction.html#what-is-libdevice) on what libdevice is).
- 5: We run the verifier on the nvvm program just to check that we did not create any invalid nvvm ir.
- 6: We run the compiler which gives us a final PTX string, hooray!
- 7: Finally, the PTX goes through a small stage where its parsed and function DCE is run to eliminate
     Most of the bloat in the file, traditionally this is done by the linker but theres no linker to be found for miles here.
- 8: We write this ptx file to wherever rustc tells us to write the final file.

We will cover the libnvvm steps in more detail later on.

# Codegen Units (CGUs)

Ah codegen units, the thing everyone just tells you to set to `1` in Cargo.toml, but what are they?
Well, to put it simply, codegen units are rustc splitting up a crate into different modules to then 
run LLVM in parallel over. For example, rustc can run LLVM over two different modules in parallel and 
save time.

This gets a little bit more complex with generics, because MIR is not monomorphized and monomorphized MIR is not a thing,
the codegen monomorphizes instances on the fly. Therefore rustc needs to put any generic functions that one CGU relies on
inside of the same CGU because it needs to monomorphize them.

# Rlibs

rlibs are mysterious files, their origins are mysterious and their contents are the deepest layer of the iceberg. Just kidding,
but rlibs often confuse people (including me at first). Rlibs are rustc's way of encoding basically everything it needs to know 
about a crate into a file. Rlibs usually contain the following:
- Object files for each CGU.
- LLVM Bitcode.
- a Symbol table.
- metadata:
  - rustc version (because things can go kaboom if version mismatches, ABIs are fun amirite)
  - A crate hash
  - a crate id
  - info about the source files
  - the exported API, things like macros, traits, etc.
  - MIR, for things such as generic functions and `#[inline]`d functions (please don't put `#[inline]` on everything, rustc will cry)
