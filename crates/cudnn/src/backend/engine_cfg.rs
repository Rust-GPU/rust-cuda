use crate::{
    CudnnError, IntoResult,
    backend::{Descriptor, Engine},
};

#[derive(Default, PartialEq, Debug)]
pub struct EngineCfgBuilder {
    descriptor: Option<Descriptor>,
    engine: Option<Engine>,
}

impl EngineCfgBuilder {
    pub(crate) fn set_descriptor(mut self, descriptor: Descriptor) -> Self {
        self.descriptor = Some(descriptor);
        self
    }

    pub fn set_engine(mut self, engine: Engine) -> Self {
        self.engine = Some(engine);
        self
    }

    pub fn build(self) -> Result<EngineCfg, CudnnError> {
        let engine = self.engine.expect("engine is required.");

        unsafe {
            let mut descriptor = match self.descriptor {
                None => Descriptor::new(
                    cudnn_sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINECFG_DESCRIPTOR,
                )?,
                Some(descriptor) => descriptor,
            };

            descriptor.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINECFG_ENGINE,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &engine.descriptor.inner(),
            )?;

            descriptor.finalize()?;

            Ok(EngineCfg { descriptor, engine })
        }
    }
}

#[derive(PartialEq, Debug)]
pub struct EngineCfg {
    pub(crate) descriptor: Descriptor,
    engine: Engine,
}

impl EngineCfg {
    pub fn get_engine(&self) -> &Engine {
        &self.engine
    }
}
