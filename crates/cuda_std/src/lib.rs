//! # CUDA Standard Library
//!
//! The CUDA Standard Library provides a curated set of abstractions for writing performant, reliable, and
//! understandable GPU kernels using the Rustc NVVM backend.
//!
//! This library will build on non-nvptx targets or targets not using the nvvm backend. However, it will not
//! be usable, and it will throw linker errors if you attempt to use most of the functions in the library.
//! However, [`kernel`] automatically cfg-gates the function annotated for `nvptx64`, therefore,
//! no "actual" functions from this crate should be used when compiling for a non-nvptx target.
//!
//! This crate cannot be used with the llvm ptx backend either, it heavily relies on external functions implicitly
//! defined by the nvvm backend, as well as internal attributes.
//!
//! # Structure
//!
//! This library tries to follow the structure of the Rust standard library to some degree, where
//! different concepts are separated into their own modules.
//!
//! # The Prelude
//!
//! In order to simplify imports, we provide a prelude module which contains GPU analogues to standard library
//! structures as well as common imports such as [`thread`].

#![allow(internal_features)]
#![cfg_attr(
    target_os = "cuda",
    feature(
        alloc_error_handler,
        asm_experimental_arch,
        link_llvm_intrinsics,
        stdarch_nvptx
    )
)]

extern crate alloc;

pub mod float;
#[allow(warnings)]
pub mod intrinsics;
pub mod io;
pub mod mem;
pub mod misc;
// WIP
// pub mod rt;
pub mod atomic;
pub mod ptr;
pub mod shared;
pub mod thread;
pub mod warp;

mod float_ext;

pub use cuda_std_macros::*;
pub use float::GpuFloat;
pub use float_ext::*;
pub use glam;
pub use half;
#[deprecated(note = "The `vek` module is deprecated, use `glam` instead.")]
pub use vek;

pub use half::{bf16, f16};

pub mod prelude {
    pub use crate::f16;
    pub use crate::kernel;
    pub use crate::thread;
    pub use crate::{assert_eq, assert_ne, print, println};
    pub use alloc::{
        borrow::ToOwned,
        boxed::Box,
        format,
        rc::Rc,
        string::{String, ToString},
        vec::Vec,
    };
}

#[cfg(target_arch = "nvptx64")]
#[alloc_error_handler]
fn alloc_handler(layout: core::alloc::Layout) -> ! {
    core::panic!("Memory allocation of {} bytes failed", layout.size());
}

// FIXME(RDambrosio016): For some very odd reason, this function causes an InvalidAddress error when called,
// despite it having no reason for doing that. It needs more debugging to see what is causing it exactly. For now we just trap.
#[cfg(target_arch = "nvptx64")]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // use crate::prelude::*;
    // let block = thread::block_idx();
    // let thread = thread::thread_idx();

    // let thread_str = if thread.z == 0 && thread.y == 0 {
    //     format!("{}", thread.x)
    // } else if thread.z == 0 {
    //     format!("({}, {})", thread.x, thread.y)
    // } else {
    //     format!("({}, {}, {})", thread.x, thread.y, thread.z)
    // };

    // let block_str = if block.z == 0 && block.y == 0 {
    //     format!("{}", block.x)
    // } else if block.z == 0 {
    //     format!("({}, {})", block.x, block.y)
    // } else {
    //     format!("({}, {}, {})", block.x, block.y, block.z)
    // };

    // let locstr = if let Some(loc) = info.location() {
    //     let file = loc.file().to_string();
    //     let line = loc.line().to_string();
    //     format!("(in file `{}` at line `{}`\n) ", file, line)
    // } else {
    //     String::new()
    // };

    // // let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
    // //     format!(
    // //         "thread {} in block {} {}panicked: {}",
    // //         thread, block, locstr, s
    // //     )
    // // } else {
    // //     format!("thread {} in block {} {}panicked", thread, block, locstr)
    // // };

    // // crate::println!("{}", msg);

    unsafe extern "C" {
        fn __nvvm_trap() -> !;
    }

    unsafe { __nvvm_trap() };
}
