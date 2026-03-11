use crate::{
    CudnnError, DataType, IntoResult,
    backend::{Descriptor, ReductionMode},
};

#[derive(Clone, Default, PartialEq, Eq, Hash, Debug)]
pub struct ReductionCfgBuilder {
    math_precision: Option<cudnn_sys::cudnnDataType_t>,
    mode: Option<ReductionMode>,
}

impl ReductionCfgBuilder {
    pub fn set_math_precision<T>(mut self) -> Self
    where
        T: DataType,
    {
        self.math_precision = Some(T::into_raw());
        self
    }

    pub fn set_mode(mut self, mode: ReductionMode) -> Self {
        self.mode = Some(mode);
        self
    }

    pub fn build(self) -> Result<ReductionCfg, CudnnError> {
        let math_precision = self
            .math_precision
            .unwrap_or(cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT);

        let mode: cudnn_sys::cudnnReduceTensorOp_t =
            self.mode.expect("reduction mode is required.").into();

        unsafe {
            let mut raw = Descriptor::new(
                cudnn_sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_REDUCTION_DESCRIPTOR,
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_REDUCTION_COMP_TYPE,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DATA_TYPE,
                1,
                &math_precision,
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_REDUCTION_OPERATOR,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_REDUCTION_OPERATOR_TYPE,
                1,
                &mode,
            )?;

            raw.finalize()?;

            Ok(ReductionCfg { raw })
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReductionCfg {
    pub(crate) raw: Descriptor,
}
