use crate::{ConvMode, CudnnError, DataType, IntoResult, backend::Descriptor};

#[derive(Default, Clone, PartialEq, Eq, Hash, Debug)]
pub struct ConvCfgBuilder<'a> {
    comp_type: Option<cudnn_sys::cudnnDataType_t>,
    mode: Option<ConvMode>,
    dilations: Option<&'a [i64]>,
    strides: Option<&'a [i64]>,
    pre_paddings: Option<&'a [i64]>,
    post_paddings: Option<&'a [i64]>,
}

impl<'a> ConvCfgBuilder<'a> {
    pub fn set_comp_type<T>(mut self) -> Self
    where
        T: DataType,
    {
        self.comp_type = Some(T::into_raw());
        self
    }

    pub fn set_convolution_mode(mut self, mode: ConvMode) -> Self {
        self.mode = Some(mode);
        self
    }

    pub fn set_dilations(mut self, dilations: &'a [i64]) -> Self {
        self.dilations = Some(dilations);
        self
    }

    pub fn set_strides(mut self, strides: &'a [i64]) -> Self {
        self.strides = Some(strides);
        self
    }

    pub fn set_pre_paddings(mut self, pre_paddings: &'a [i64]) -> Self {
        self.pre_paddings = Some(pre_paddings);
        self
    }

    pub fn set_post_paddings(mut self, post_paddings: &'a [i64]) -> Self {
        self.post_paddings = Some(post_paddings);
        self
    }

    pub fn build(self) -> Result<ConvCfg, CudnnError> {
        let comp_type = self.comp_type.expect("computation type is required");
        let mode = self.mode.expect("convolution mode is required");
        let dilations = self.dilations.expect("dilations are required");
        let strides = self.strides.expect("strides are required");
        let pre_paddings = self.pre_paddings.expect("pre-paddings are required.");
        let post_paddings = self.post_paddings.expect("post-paddings are required.");

        unsafe {
            let mut raw = Descriptor::new(
                cudnn_sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR,
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_COMP_TYPE,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DATA_TYPE,
                1,
                &comp_type,
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_CONV_MODE,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_CONVOLUTION_MODE,
                1,
                &mode,
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_DILATIONS,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
                dilations.len() as i64,
                dilations,
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_DILATIONS,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
                dilations.len() as i64,
                dilations,
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
                strides.len() as i64,
                strides,
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
                pre_paddings.len() as i64,
                pre_paddings,
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
                post_paddings.len() as i64,
                post_paddings,
            )?;

            raw.finalize()?;

            Ok(ConvCfg { raw })
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ConvCfg {
    pub(crate) raw: Descriptor,
}
