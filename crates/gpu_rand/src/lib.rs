//! gpu_rand is the Rust CUDA Project's equivalent of cuRAND. cuRAND unfortunately does not work
//! with the CUDA Driver API, therefore, we reimplement (and extend) some of its algorithms and
//! provide them in this crate.
//!
//! This crate is meant to be GPU-centric, which means it may special-case certain things to run
//! faster on the GPU by using PTX assembly. However, it is supposed to also work on the CPU,
//! allowing you to reuse the same random states across CPU and GPU.
//!
//! A lot of the initial code is taken from the [rust-random
//! project](https://github.com/rust-random) and modified to make it able to pass to the GPU, as
//! well as cleaning up certain things and updating it to edition 2024.
//!
//! The following generators are implemented:
//!
//! The random generators currently implemented are:
//!
//! 32-bit:
//! - Xoroshiro64**
//! - Xoroshiro64*
//! - Xoroshiro128+
//! - Xoroshiro128++
//! - Xoroshiro128**
//!
//! 64-bit:
//! - Xoroshiro128+
//! - Xoroshiro128++
//! - Xoroshiro128**
//! - Xoroshiro256+
//! - Xoroshiro256++
//! - Xoroshiro256**
//! - Xoroshiro512+
//! - Xoroshiro512++
//! - Xoroshiro512**
//! - SplitMix64
//!
//! We also provide a default 64-bit generator which should be more than enough for most
//! applications. The default currently uses Xoroshiro128** but that is subject to change in the
//! future.

#![deny(missing_docs)]
#![deny(missing_debug_implementations)]
#![allow(clippy::unreadable_literal)]
#![feature(doc_cfg)]

pub mod xoroshiro;

mod default;
mod gpurng;

pub use default::*;
pub use gpurng::*;
