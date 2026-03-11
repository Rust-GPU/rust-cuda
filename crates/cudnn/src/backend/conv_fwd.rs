use crate::{
    CudnnError, DataType, IntoResult,
    backend::{ConvCfg, Descriptor, FloatDataType, Operation, Real, Tensor},
};

pub struct ConvFwdBuilder {
    cfg: Option<ConvCfg>,
    alpha: Option<Real>,
    beta: Option<Real>,
    w: Option<Tensor>,
    x: Option<Tensor>,
    y: Option<Tensor>,
}

impl ConvFwdBuilder {
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

    pub fn set_w(mut self, w: Tensor) -> Self {
        self.w = Some(w);
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
        let cfg = self.cfg.expect("convolution configuration is required.");

        let w = self.w.expect("w tensor is required");
        let x = self.x.expect("x tensor is required.");
        let y = self.y.expect("y tensor is required.");

        let alpha = self.alpha.unwrap_or(Real::Float(1.0));
        let beta = self.beta.unwrap_or(Real::Float(0.0));

        unsafe {
            let mut raw = Descriptor::new(
                cudnn_sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &cfg.raw.inner(),
            )?;

            match self.alpha {
                Some(Real::Float(ref alpha)) => raw.set_attribute(
                    cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                    cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_FLOAT,
                    1,
                    alpha,
                )?,
                Some(Real::Double(ref alpha)) => raw.set_attribute(
                    cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                    cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DOUBLE,
                    1,
                    alpha,
                )?,
                None => (),
            }

            match self.beta {
                Some(Real::Float(ref beta)) => raw.set_attribute(
                    cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                    cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_FLOAT,
                    1,
                    beta,
                )?,
                Some(Real::Double(ref beta)) => raw.set_attribute(
                    cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                    cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DOUBLE,
                    1,
                    beta,
                )?,
                None => (),
            }

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &w.raw.inner(),
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &x.raw.inner(),
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &y.raw.inner(),
            )?;

            raw.finalize()?;

            Ok(Operation::ConvFwd {
                raw,
                cfg,
                alpha,
                beta,
                w,
                x,
                y,
            })
        }
    }
}
