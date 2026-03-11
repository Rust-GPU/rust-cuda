use crate::{
    CudnnError, DataType, IntoResult, NanPropagation,
    backend::{Descriptor, PointwiseMode},
};

#[derive(Clone, Default, PartialEq, Debug)]
pub struct PointwiseCfgBuilder {
    math_precision: Option<cudnn_sys::cudnnDataType_t>,
    mode: Option<PointwiseMode>,
    nan_propagation: Option<NanPropagation>,
    relu_lower_clip: Option<f64>,
    relu_upper_clip: Option<f64>,
    relu_lower_clip_slope: Option<f64>,
    elu_alpha: Option<f64>,
    softplus_beta: Option<f64>,
    swish_beta: Option<f64>,
}

impl PointwiseCfgBuilder {
    pub fn set_math_precision<T>(&mut self) -> &mut Self
    where
        T: DataType,
    {
        self.math_precision = Some(T::into_raw());
        self
    }

    pub fn set_mode(&mut self, mode: PointwiseMode) -> &mut Self {
        self.mode = Some(mode);
        self
    }

    pub fn set_nan_propagation(&mut self, nan_propagation: NanPropagation) -> &mut Self {
        self.nan_propagation = Some(nan_propagation);
        self
    }

    pub fn set_relu_lower_clip(&mut self, lower_clip: f64) -> &mut Self {
        self.relu_lower_clip = Some(lower_clip);
        self
    }

    pub fn set_relu_upper_clip(&mut self, upper_clip: f64) -> &mut Self {
        self.relu_upper_clip = Some(upper_clip);
        self
    }

    pub fn set_relu_lower_clip_slope(&mut self, lower_clip_slope: f64) -> &mut Self {
        self.relu_lower_clip_slope = Some(lower_clip_slope);
        self
    }

    pub fn set_elu_alpha(&mut self, alpha: f64) -> &mut Self {
        self.elu_alpha = Some(alpha);
        self
    }

    pub fn set_softplus_beta(&mut self, beta: f64) -> &mut Self {
        self.softplus_beta = Some(beta);
        self
    }

    pub fn set_swish_beta(&mut self, beta: f64) -> &mut Self {
        self.swish_beta = Some(beta);
        self
    }

    pub fn build(&mut self) -> Result<PointwiseCfg, CudnnError> {
        let mode: cudnn_sys::cudnnPointwiseMode_t =
            self.mode.expect("pointwise mode is required.").into();

        let math_precision = self
            .math_precision
            .unwrap_or(cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT);

        unsafe {
            let mut raw = Descriptor::new(
                cudnn_sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_POINTWISE_DESCRIPTOR,
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_MATH_PREC,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DATA_TYPE,
                1,
                &math_precision,
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_MODE,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_POINTWISE_MODE,
                1,
                &mode,
            )?;

            if let Some(ref nan_propagation) = self.nan_propagation {
                raw.set_attribute(
                    cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_NAN_PROPAGATION,
                    cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_NAN_PROPOGATION,
                    1,
                    nan_propagation,
                )?;
            }

            if let Some(ref relu_lower_clip) = self.relu_lower_clip {
                raw.set_attribute(
                    cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP,
                    cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DOUBLE,
                    1,
                    relu_lower_clip,
                )?;
            }

            if let Some(ref relu_upper_clip) = self.relu_upper_clip {
                raw.set_attribute(
                    cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP,
                    cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DOUBLE,
                    1,
                    relu_upper_clip,
                )?;
            }

            if let Some(ref relu_lower_clip_slope) = self.relu_lower_clip_slope {
                raw.set_attribute(
                    cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE,
                    cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DOUBLE,
                    1,
                    relu_lower_clip_slope,
                )?;
            }

            if let Some(ref elu_alpha) = self.elu_alpha {
                raw.set_attribute(
                    cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_ELU_ALPHA,
                    cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DOUBLE,
                    1,
                    elu_alpha,
                )?;
            }

            if let Some(ref softplus_beta) = self.softplus_beta {
                raw.set_attribute(
                    cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA,
                    cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DOUBLE,
                    1,
                    softplus_beta,
                )?;
            }

            if let Some(ref swish_beta) = self.swish_beta {
                raw.set_attribute(
                    cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_SWISH_BETA,
                    cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DOUBLE,
                    1,
                    swish_beta,
                )?;
            }

            raw.finalize()?;

            Ok(PointwiseCfg { raw })
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PointwiseCfg {
    pub(crate) raw: Descriptor,
}
