use crate::{
    CudnnContext, CudnnError,
    backend::{Descriptor, Operation},
};

#[derive(Default, PartialEq, Debug)]
pub struct GraphBuilder {
    context: Option<CudnnContext>,
    operations: Option<Vec<Operation>>,
}

impl GraphBuilder {
    pub fn set_context(mut self, context: CudnnContext) -> Self {
        self.context = Some(context);
        self
    }

    pub fn set_operations(mut self, operations: Vec<Operation>) -> Self {
        self.operations = Some(operations);
        self
    }

    pub fn build(self) -> Result<Graph, CudnnError> {
        let context = self.context.expect("cudnn context is required.");
        let operations = self.operations.expect("operations are required");

        unsafe {
            let mut descriptor = Descriptor::new(
                cudnn_sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
            )?;

            descriptor.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_HANDLE,
                1,
                &context.raw,
            )?;

            let descriptors = operations
                .iter()
                .map(|op| match op {
                    Operation::ConvBwdData { raw, .. } => raw.inner(),
                    Operation::ConvBwdData { raw, .. } => raw.inner(),
                    Operation::ConvBwdFilter { raw, .. } => raw.inner(),
                    Operation::ConvFwd { raw, .. } => raw.inner(),
                    Operation::MatMul { raw, .. } => raw.inner(),
                    Operation::Pointwise { raw, .. } => raw.inner(),
                    Operation::Reduction { raw, .. } => raw.inner(),
                })
                .collect::<Vec<_>>();

            descriptor.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_OPS,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                descriptors.len() as i64,
                descriptors.as_slice(),
            )?;

            descriptor.finalize()?;

            Ok(Graph {
                descriptor,
                context,
                operations,
            })
        }
    }
}

#[derive(PartialEq, Debug)]
pub struct Graph {
    pub(crate) descriptor: Descriptor,
    context: CudnnContext,
    operations: Vec<Operation>,
}

impl Graph {
    pub fn get_context(&self) -> &CudnnContext {
        &self.context
    }

    pub fn get_operations(&self) -> &[Operation] {
        &self.operations
    }
}
