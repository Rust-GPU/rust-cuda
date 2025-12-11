//! # cudnn
//! Type safe cuDNN wrapper for the Rust programming language.
//!
//! ## Project status
//!
//! The current version of cuDNN targeted by this wrapper is 8.3.2. You can refer to the official
//! [release notes] and to the [support matrix] by NVIDIA.
//!
//! [release notes]: https://docs.nvidia.com/deeplearning/cudnn/release-notes/index.html
//! [support matrix]: https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html
//!
//! The legacy API is somewhat complete and usable but the backend API is still a work in progress
//! and its usage is discouraged. Both APIs are still being developed so expect bugs and reasonable
//! breaking changes whilst using this crate.
//!
//! The project is part of the Rust CUDA ecosystem and is actively maintained by
//! [frjnn](https://github.com/frjnn).
//!
//! ## Primer
//!
//! Here follows a list of useful concepts that should be taken as a handbook for the users of the
//! crate. This is not intended to be the full documentation, as each wrapped struct, enum and
//! function has its own docs, but rather a quick sum up of the key points of the API. For a deeper
//! view, you should refer both to the docs of each item and to the [official ones] by NVIDIA.
//! Furthermore, if you are new to cuDNN we strongly suggest reading the [official developer
//! guide].
//!
//! [official ones]: https://docs.nvidia.com/deeplearning/cudnn/api/index.html
//! [official developer guide]: https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#overview
//!
//! ### Device buffers
//!
//! This crate is built around [`cust`](https://docs.rs/cust/latest/cust/index.html) which is the
//! core wrapper for interfacing with the CUDA driver API of our choice.
//!
//! ### cuDNN statuses and Result
//!
//! All cuDNN library functions return their status. This crate uses
//! [`Result`](https://doc.rust-lang.org/std/result/enum.Result.html) to achieve a leaner,
//! idiomatic and easier to manage API.
//!
//! ### cuDNN handles and RAII
//!
//! The main entry point of the cuDNN library is the `CudnnContext` struct. This handle is tied to
//! a device and it is explicitly passed to every subsequent library function that operates on GPU
//! data. It manages resources allocations both on the host and the device and takes care of the
//! synchronization of all the the cuDNN primitives.
//!
//! The handles, and the other cuDNN structs wrapped by this crate, are implementors of the
//! [`Drop`](https://doc.rust-lang.org/std/ops/trait.Drop.html) trait which implicitly calls their
//! destructors on the cuDNN side when they go out of scope.
//!
//! cuDNN contexts can be created as shown in the following snippet:
//!
//! ```rust
//! use cudnn::CudnnContext;
//!
//! let ctx = CudnnContext::new().unwrap();
//! ```
//!
//! ### cuDNN data types
//!
//! In order to enforce type safety as much as possible at compile time, we shifted away from the
//! original cuDNN enumerated data types and instead opted to leverage Rust's generics. In
//! practice, this means that specifying the data type of a cuDNN tensor descriptor is done as
//! follows:
//!
//! ```rust
//! use cudnn::{CudnnContext, TensorDescriptor};
//!
//! let ctx = CudnnContext::new().unwrap();
//!
//! let shape = &[5, 5, 10, 25];
//! let strides = &[1250, 250, 25, 1];
//!
//! // f32 tensor
//! let desc = TensorDescriptor::<f32>::new_strides(shape, strides).unwrap();
//! ```
//!
//! This API also allows for using Rust own types as cuDNN data types, which we see as a desirable
//! property.
//!
//! Safely manipulating cuDNN data types that do not have any such direct match, such as vectorized
//! ones, whilst still performing compile time compatibility checks can be done as follows:
//!
//! ```rust
//! use cudnn::{CudnnContext, TensorDescriptor, Vec4};
//!
//! let ctx = CudnnContext::new().unwrap();
//!
//! let shape = &[4, 32, 32, 32];
//!
//! // in cuDNN this is equal to the INT8x4 data type and CUDNN_TENSOR_NCHW_VECT_C format
//! let desc = TensorDescriptor::<i8>::new_vectorized::<Vec4>(shape).unwrap();
//! ```
//!
//! The previous tensor descriptor can be used together with a `i8` device buffer and cuDNN will
//! see it as being a tensor of `CUDNN_TENSOR_NCHW_VECT_C` format and `CUDNN_DATA_INT8x4` data
//! type.
//!
//! Currently this crate does not support `f16` and `bf16` data types.
//!
//! ### cuDNN tensor formats
//!
//! We decided not to check tensor format configurations at compile time, because it is too strong
//! a requirement. As a consequence, should you mess up, the program will fail at run-time. A
//! proper understanding of the cuDNN API mechanics is thus fundamental to properly use this crate.
//!
//! You can refer to this [extract] from the cuDNN developer guide to learn more about tensor
//! formats.
//!
//! [extract]: https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#data-layout-formats
//!
//! We split the original cuDNN tensor format enum, which counts 3 variants, in 2 parts: the
//! `ScalarC` enum and the `TensorFormat::NchwVectC` enum variant. The former stands for "scalar
//! channel" and it encapsulates the `Nchw` and `Nhwc` formats. Scalar channel formats can be both
//! converted to the `TensorFormat` enum with
//! [`.into()`](https://doc.rust-lang.org/std/convert/trait.Into.html).
//!
//! ```rust
//! use cudnn::{ScalarC, TensorFormat};
//!
//! let sc_fmt = ScalarC::Nchw;
//!
//! let vc_fmt = TensorFormat::NchwVectC;
//!
//! let sc_to_tf: TensorFormat = sc_fmt.into();
//! ```

#![deny(rustdoc::broken_intra_doc_links)]

mod activation;
mod attention;
mod backend;
mod context;
mod convolution;
mod data_type;
mod determinism;
mod dropout;
mod error;
mod math_type;
mod nan_propagation;
mod op;
mod pooling;
mod reduction;
mod rnn;
mod softmax;
mod tensor;
mod w_grad_mode;

pub use activation::*;
pub use attention::*;
pub use context::*;
pub use convolution::*;
pub use data_type::*;
pub use determinism::*;
pub use dropout::*;
pub use error::*;
pub use math_type::*;
pub use nan_propagation::*;
pub use op::*;
pub use pooling::*;
pub use reduction::*;
pub use rnn::*;
pub use softmax::*;
pub use tensor::*;
pub use w_grad_mode::*;

pub(crate) mod private {
    pub trait Sealed {}
}
