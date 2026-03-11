use crate::{
    CudnnContext, CudnnError, IntoResult,
    backend::{Descriptor, EngineCfgBuilder, Graph},
};

pub enum HeuristicMode {
    A,
    B,
}

impl From<HeuristicMode> for cudnn_sys::cudnnBackendHeurMode_t {
    fn from(mode: HeuristicMode) -> Self {
        match mode {
            HeuristicMode::A => cudnn_sys::cudnnBackendHeurMode_t::CUDNN_HEUR_MODE_A,
            HeuristicMode::B => cudnn_sys::cudnnBackendHeurMode_t::CUDNN_HEUR_MODE_B,
        }
    }
}
