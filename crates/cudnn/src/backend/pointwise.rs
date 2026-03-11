use crate::{
    CudnnError, DataType, IntoResult, NanPropagation,
    backend::{Descriptor, FloatDataType, Operation, PointwiseCfg, PointwiseMode, Real, Tensor},
};

#[derive(Clone, Default, Debug, PartialEq)]
pub struct PointwiseBuilder {
    cfg: Option<PointwiseCfg>,
    x: Option<Tensor>,
    b: Option<Tensor>,
    y: Option<Tensor>,
    alpha: Option<Real>,
    beta: Option<Real>,
}

impl PointwiseBuilder {
    pub fn set_cfg(mut self, cfg: PointwiseCfg) -> Self {
        self.cfg = Some(cfg);
        self
    }

    pub fn set_x(mut self, x: Tensor) -> Self {
        self.x = Some(x);
        self
    }

    pub fn set_b(mut self, b: Tensor) -> Self {
        self.b = Some(b);
        self
    }

    pub fn set_y(mut self, y: Tensor) -> Self {
        self.y = Some(y);
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

    pub fn build(mut self) -> Result<Operation, CudnnError> {
        let cfg = self.cfg.expect("pointwise configuration is required.");
        let x = self.x.expect("primary input is required.");
        let y = self.y.expect("output is required.");

        unsafe {
            let mut raw = Descriptor::new(
                cudnn_sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &cfg.raw.inner(),
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_XDESC,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &x.raw.inner(),
            )?;

            if let Some(ref b) = self.b {
                raw.set_attribute(
                    cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_BDESC,
                    cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                    1,
                    &b.raw.inner(),
                )?;
            }

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_YDESC,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &y.raw.inner(),
            )?;

            match self.alpha {
                Some(Real::Float(ref alpha)) => raw.set_attribute(
                    cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1,
                    cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_FLOAT,
                    1,
                    alpha,
                )?,
                Some(Real::Double(ref alpha)) => raw.set_attribute(
                    cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1,
                    cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DOUBLE,
                    1,
                    alpha,
                )?,
                None => (),
            }

            match self.beta {
                Some(Real::Float(ref beta)) => raw.set_attribute(
                    cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2,
                    cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_FLOAT,
                    1,
                    beta,
                )?,
                Some(Real::Double(ref beta)) => raw.set_attribute(
                    cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2,
                    cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DOUBLE,
                    1,
                    beta,
                )?,
                None => (),
            }

            raw.finalize()?;

            Ok(Operation::Pointwise {
                raw,
                cfg,
                x,
                y,
                b: self.b,
                alpha: self.alpha,
                beta: self.beta,
            })
        }
    }
}
