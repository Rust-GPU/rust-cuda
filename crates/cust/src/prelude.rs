//! This module re-exports a number of commonly-used types for working with cust.
//!
//! This allows the user to `use cust::prelude::*;` and have the most commonly-used types
//! available quickly.

pub use crate::CudaFlags;
pub use crate::context::{Context, ContextFlags};
pub use crate::device::Device;
pub use crate::event::{Event, EventFlags, EventStatus};
pub use crate::external::*;
pub use crate::function::Function;
pub use crate::launch;
pub use crate::memory::{
    CopyDestination, DeviceBuffer, DevicePointer, DeviceSlice, DeviceVariable, UnifiedBuffer,
};
pub use crate::module::Module;
pub use crate::stream::{Stream, StreamFlags};
pub use crate::util::*;
