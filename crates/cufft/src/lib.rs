//! Rust wrapper for the [cuFFT library](https://docs.nvidia.com/cuda/cufft/).
//!
//! Create a `FftPlan<T>` with one of the plan constructors,
//! where `T` implements the `FftType` trait (`C2C`, `R2C`, `C2R`, `Z2Z`, `D2Z`, `Z2D`).
//! Optionally attach a stream with `FftPlan::set_stream`.
//! Execute the plan with `FftPlan::exec`.
//! Plans are destroyed when dropped.
//!
//! Raw bindgen bindings are available in `cufft_raw`.

mod error;
mod plan;

pub use error::{CufftError, IntoResult};
pub use plan::{C2C, C2R, D2Z, Direction, FftPlan, FftType, R2C, Z2D, Z2Z};
