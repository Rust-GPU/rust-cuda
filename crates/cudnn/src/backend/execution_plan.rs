use crate::{
    CudnnContext, CudnnError, IntoResult,
    backend::{Descriptor, EngineCfg},
};

#[derive(Default, PartialEq, Debug)]
pub struct ExecutionPlanBuilder {
    engine_cfg: Option<EngineCfg>,
}

impl ExecutionPlanBuilder {
    pub fn set_engine_cfg(mut self, engine_cfg: EngineCfg) -> Self {
        self.engine_cfg = Some(engine_cfg);
        self
    }

    pub fn build(self) -> Result<ExecutionPlan, CudnnError> {
        let engine_cfg = self.engine_cfg.expect("engine configuration is required.");

        unsafe {
            let mut descriptor = Descriptor::new(
                cudnn_sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
            )?;

            descriptor.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &engine_cfg.descriptor.inner(),
            )?;

            descriptor.finalize()?;

            Ok(ExecutionPlan {
                descriptor,
                engine_cfg,
            })
        }
    }
}

pub struct ExecutionPlan {
    pub(crate) descriptor: Descriptor,
    engine_cfg: EngineCfg,
}
