use crate::{
    CudnnError, IntoResult,
    backend::{Descriptor, Operation, ReductionCfg, Tensor},
};

#[derive(Default, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReductionBuilder {
    cfg: Option<ReductionCfg>,
    x: Option<Tensor>,
    y: Option<Tensor>,
}

impl ReductionBuilder {
    pub fn set_cfg(mut self, cfg: ReductionCfg) -> Self {
        self.cfg = Some(cfg);
        self
    }

    pub fn set_x(mut self, x: Tensor) -> Self {
        self.x = Some(x);
        self
    }

    pub fn set_y(mut self, y: Tensor) -> Self {
        self.y = Some(y);
        self
    }

    pub fn build(self) -> Result<Operation, CudnnError> {
        let cfg = self.cfg.expect("reduce configuration is required.");
        let x = self.x.expect("x tensor is required.");
        let y = self.y.expect("y tensor is required");

        unsafe {
            let mut raw = Descriptor::new(
                cudnn_sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR,
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_REDUCTION_DESC,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &cfg.raw.inner(),
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_REDUCTION_XDESC,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &x.raw.inner(),
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_REDUCTION_YDESC,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &y.raw.inner(),
            )?;

            raw.finalize()?;

            Ok(Operation::Reduction { raw, cfg, x, y })
        }
    }
}
