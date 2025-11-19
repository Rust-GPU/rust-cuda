# Kernel ABI

This section details how parameters are passed to GPU kernels by the Codegen at the current time. 
In other words, how the codegen expects you to pass different types to GPU kernels from the CPU.

⚠️ If you find any bugs in the ABI please report them. ⚠️

## Preface

Please note that the following __only__ applies to non-rust call conventions, we make zero guarantees 
about the rust call convention, just like rustc. 

While we currently override every ABI except rust, you should generally only use `"C"`, any 
other ABI we override purely to avoid footguns.

Functions marked as `#[kernel]` are enforced to be `extern "C"` by the kernel macro, and it is expected
that __all__ GPU kernels be `extern "C"`, not that you should be declaring any kernels without the `#[kernel]` macro,
because the codegen/`cuda_std` is allowed to rely on the behavior of `#[kernel]` for correctness.

## Structs 

Structs are always passed directly using byte arrays if they are passed by value in the function. This
corresponds to what is expected by CUDA/the PTX ABI.

For example:

```rs
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Foo {
    pub a: u16,
    pub b: u64,
    pub c: u128,
}

#[kernel]
pub unsafe fn kernel(a: Foo) {
    /* ... */
}
```

will map to the following PTX:

```
.visible .entry kernel(
	.param .align 16 .b8 kernel_param_0[32]
)
```

Consequently, it is expected that you will pass the struct by value when launching the kernel, and not
by reference (by allocating a device box):

```rs
let foo = Foo { 
  a: 5,
  b: 6,
  c: 7
};

unsafe {
  launch!(
    module.kernel<<<1, 1, 0, stream>>>(foo)
  )?;
}
```

And not

```rs
let foo = DeviceBox::new(Foo { 
  a: 5,
  b: 6,
  c: 7
});

unsafe {
  launch!(
    module.kernel<<<1, 1, 0, stream>>>(foo.as_device_ptr())
  )?;
}
```

## Arrays 

Like structs, arrays are always passed by value as byte arrays.

## Slices 

Slices are passed as **two word-sized parameters**: a pointer to the beginning of the data, and an
integer giving the length of the slice.

For example:

```rs
#[kernel]
pub unsafe fn kernel(a: &[u8]) {
  /* ... */
}
```

Will map to the following PTX (on nvptx64):

```
.visible .entry kernel(
	.param .u64 kernel_param_0,
	.param .u64 kernel_param_1
)
```

Consequently, it is expected that you will pass the pointer and the length as multiple parameters when calling the kernel:

```rs
let mut buf = [5u8; 10].as_dbuf()?;

unsafe {
  launch!(
    module.kernel<<<1, 1, 0, stream>>>(buf.as_device_ptr(), buf.len())
  )?;
}
```

You may get warnings about slices being an improper C-type, but the warnings are safe to ignore, the codegen guarantees 
that slices are passed as pairs of params.

You cannot however pass mutable slices, this is because it would violate aliasing rules, each thread receiving a copy of the mutable
slice would violate aliasing rules. You may use a `&[UnsafeCell<T>]` then convert an element to a mutable ref (once you know the element accesses
are disjoint), or more commonly, use a raw pointer.

## ZSTs

ZSTs (zero-sized types) are ignored and become nothing in the final PTX.

## Primitives

Primitive types are passed directly by value, same as structs. They map to the special PTX types `.s8`, `.s16`, `.s32`, `.s64`, `.u8`, `.u16`, `.u32`, `.u64`, `.f32`, and `.f64`.
With the exception that `u128` and `i128` are passed as byte arrays (but this has no impact on how they are passed from the CPU).

## References And Pointers

References and Pointers are both passed as expected, as pointers. It is therefore expected that you pass such parameters using device memory:

```rs
#[kernel]
pub unsafe fn kernel(a: &u8) {
  /* ... */
}
```

```rs
let mut val = DeviceBox::new(&5)?;

unsafe {
  launch!(
    module.kernel<<<1, 1, 0, stream>>>(val.as_device_ptr())
  )?;
}
```

## repr(Rust) Types

using repr(Rust) types inside of kernels is not disallowed but it is highly discouraged. This is because rustc is allowed to switch up
how the types are represented across compiler invocations which leads to hard to track errors.

Therefore, you should generally only use repr(C) inside of kernel parameters. With the exception of slices that have a guaranteed parameter layout.
