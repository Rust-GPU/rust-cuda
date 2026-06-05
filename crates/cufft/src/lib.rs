//! Rust wrapper for the [cuFFT library](https://docs.nvidia.com/cuda/cufft/).
//!
//! Create a `FftPlan` with one of the plan constructors.
//! Optionally attach a stream with `FftPlan::set_stream`
//! Execute the plan with one of the `exec_*` methods.
//! Plans are destroyed when they dropped.
//!
//! Raw bindgen bindings are available in `cufft_raw`.

mod error;
mod plan;

pub use error::{CufftError, IntoResult};
pub use plan::{Direction, FftPlan, FftType};
