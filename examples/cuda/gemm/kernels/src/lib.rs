mod gemm_naive;
mod gemm_tiled;

pub use crate::gemm_naive::gemm_naive;
pub use crate::gemm_tiled::{TILE_SIZE, gemm_tiled};
