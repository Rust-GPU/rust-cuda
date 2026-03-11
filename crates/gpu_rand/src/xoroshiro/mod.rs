//! Xoroshiro pseudorandom generators, Fast, fairly random, and small-state.
//!
//! # 64-bit generators
//! - [`Xoshiro256StarStar`]: Recommended for all purposes. Excellent speed and
//!   a state space (256 bits) large enough for any parallel application.
//! - [`Xoshiro256PlusPlus`]: Recommended for all purposes. Excellent speed and
//!   a state space (256 bits) large enough for any parallel application.
//! - [`Xoshiro256Plus`]: Recommended for generating 64-bit floating-point
//!   numbers. About 15% faster than `Xoshiro256StarStar`, but has a [low linear
//!   complexity] in the lowest bits (which are discarded when generating
//!   floats), making it fail linearity tests. This is unlikely to have any
//!   impact in practice.
//! - [`Xoroshiro128StarStar`]: An alternative to `Xoshiro256StarStar`, having
//!   the same speed but using half the state. Only suited for low-scale parallel
//!   applications.
//! - [`Xoroshiro128PlusPlus`]: An alternative to `Xoshiro256PlusPlus`, having
//!   the same speed but using half the state. Only suited for low-scale parallel
//!   applications.
//! - [`Xoroshiro128Plus`]: An alternative to `Xoshiro256Plus`, having the same
//!   speed but using half the state. Only suited for low-scale parallel
//!   applications. Has a [low linear complexity] in the lowest bits (which are
//!   discarded when generating floats), making it fail linearity tests. This is
//!   unlikely to have any impact in practice.
//! - [`Xoshiro512StarStar`]: An alternative to `Xoshiro256StarStar` with more
//!   state and the same speed.
//! - [`Xoshiro512PlusPlus`]: An alternative to `Xoshiro256PlusPlus` with more
//!   state and the same speed.
//! - [`Xoshiro512Plus`]: An alternative to `Xoshiro512Plus` with more
//!   state and the same speed. Has a [low linear complexity] in the lowest bits
//!   (which are discarded when generating floats), making it fail linearity
//!   tests. This is unlikely to have any impact in practice.
//! - [`SplitMix64`]: Recommended for initializing generators of the xoshiro
//!   familiy from a 64-bit seed. Used for implementing `seed_from_u64`.
//!
//! # 32-bit generators
//! - [`Xoshiro128StarStar`]: Recommended for all purposes. Excellent speed.
//! - [`Xoshiro128PlusPlus`]: Recommended for all purposes. Excellent speed.
//! - [`Xoshiro128Plus`]: Recommended for generating 32-bit floating-point
//!   numbers. Faster than `Xoshiro128StarStar`, but has a [low linear
//!   complexity] in the lowest bits (which are discarded when generating
//!   floats), making it fail linearity tests. This is unlikely to have any
//!   impact in practice.
//! - [`Xoroshiro64StarStar`]: An alternative to `Xoshiro128StarStar`, having
//!   the same speed but using half the state.
//! - [`Xoroshiro64Star`]: An alternative to `Xoshiro128Plus`, having the
//!   same speed but using half the state. Has a [low linear complexity] in the
//!   lowest bits (which are discarded when generating floats), making it fail
//!   linearity tests. This is unlikely to have any impact in practice.
//!
//! The `*PlusPlus` generators perform similarily to the `*StarStar` generators.
//! See the [xoshiro paper], where the differences are discussed in detail.
//!
//! [xoshiro]: http://xoshiro.di.unimi.it/
//! [xoshiro paper]: http://vigna.di.unimi.it/ftp/papers/ScrambledLinear.pdf
//! [low linear complexity]: http://xoshiro.di.unimi.it/lowcomp.php

#[macro_use]
mod common;
mod splitmix64;
mod xoroshiro128plus;
mod xoroshiro128plusplus;
mod xoroshiro128starstar;
mod xoroshiro64star;
mod xoroshiro64starstar;
mod xoshiro128plus;
mod xoshiro128plusplus;
mod xoshiro128starstar;
mod xoshiro256plus;
mod xoshiro256plusplus;
mod xoshiro256starstar;
mod xoshiro512plus;
mod xoshiro512plusplus;
mod xoshiro512starstar;

pub use common::Seed512;
pub use rand_core;
pub use splitmix64::SplitMix64;
pub use xoroshiro64star::Xoroshiro64Star;
pub use xoroshiro64starstar::Xoroshiro64StarStar;
pub use xoroshiro128plus::Xoroshiro128Plus;
pub use xoroshiro128plusplus::Xoroshiro128PlusPlus;
pub use xoroshiro128starstar::Xoroshiro128StarStar;
pub use xoshiro128plus::Xoshiro128Plus;
pub use xoshiro128plusplus::Xoshiro128PlusPlus;
pub use xoshiro128starstar::Xoshiro128StarStar;
pub use xoshiro256plus::Xoshiro256Plus;
pub use xoshiro256plusplus::Xoshiro256PlusPlus;
pub use xoshiro256starstar::Xoshiro256StarStar;
pub use xoshiro512plus::Xoshiro512Plus;
pub use xoshiro512plusplus::Xoshiro512PlusPlus;
pub use xoshiro512starstar::Xoshiro512StarStar;
