use crate::{
    CudnnError, DataType, IntoResult,
    backend::{ConvCfg, Descriptor, FloatDataType, Operation, Real, Tensor},
};

pub struct ConvBwdFilterBuilder {
    cfg: Option<ConvCfg>,
    alpha: Option<Real>,
    beta: Option<Real>,
    dw: Option<Tensor>,
    x: Option<Tensor>,
    dy: Option<Tensor>,
}

impl ConvBwdFilterBuilder {
    pub fn set_cfg(mut self, cfg: ConvCfg) -> Self {
        self.cfg = Some(cfg);
        self
    }

    pub fn set_alpha<T>(mut self, alpha: T) -> Self
    where
        T: FloatDataType,
    {
        self.alpha = Some(alpha.wrap());
        self
    }

    pub fn set_beta<T>(mut self, beta: T) -> Self
    where
        T: FloatDataType,
    {
        self.beta = Some(beta.wrap());
        self
    }

    pub fn set_dw(mut self, dw: Tensor) -> Self {
        self.dw = Some(dw);
        self
    }

    pub fn set_dx(mut self, x: Tensor) -> Self {
        self.x = Some(x);
        self
    }

    pub fn set_dy(mut self, dy: Tensor) -> Self {
        self.dy = Some(dy);
        self
    }

    pub fn build(self) -> Result<Operation, CudnnError> {
        let cfg = self.cfg.expect("convolution configuration is required.");
        let dw = self.dw.expect("dw tensor is required");
        let x = self.x.expect("x tensor is required.");
        let dy = self.dy.expect("dy tensor is required.");

        let alpha = self.alpha.unwrap_or(Real::Float(1.0));
        let beta = self.beta.unwrap_or(Real::Float(0.0));

        unsafe {
            let mut raw = Descriptor::new(
                cudnn_sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &cfg.raw.inner(),
            )?;

            match self.alpha {
                Some(Real::Float(ref alpha)) => raw.set_attribute(
                    cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
                    cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_FLOAT,
                    1,
                    alpha,
                )?,
                Some(Real::Double(ref alpha)) => raw.set_attribute(
                    cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
                    cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DOUBLE,
                    1,
                    alpha,
                )?,
                None => (),
            }

            match self.beta {
                Some(Real::Float(ref beta)) => raw.set_attribute(
                    cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
                    cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_FLOAT,
                    1,
                    beta,
                )?,
                Some(Real::Double(ref beta)) => raw.set_attribute(
                    cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
                    cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DOUBLE,
                    1,
                    beta,
                )?,
                None => (),
            }

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &dw.raw.inner(),
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &x.raw.inner(),
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &dy.raw.inner(),
            )?;

            raw.finalize()?;

            Ok(Operation::ConvBwdFilter {
                raw,
                cfg,
                alpha,
                beta,
                dw,
                x,
                dy,
            })
        }
    }
}
