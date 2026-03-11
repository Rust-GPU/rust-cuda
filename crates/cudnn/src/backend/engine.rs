use crate::{
    CudnnError, IntoResult,
    backend::{Descriptor, Graph},
};

#[derive(Default, Debug, PartialEq)]
pub struct EngineBuilder {
    graph: Option<Graph>,
    global_index: Option<i64>,
}

impl EngineBuilder {
    pub fn set_graph(mut self, graph: Graph) -> Self {
        self.graph = Some(graph);
        self
    }

    pub fn set_global_index(mut self, global_index: i64) -> Self {
        self.global_index = Some(global_index);
        self
    }

    pub fn build(self) -> Result<Engine, CudnnError> {
        let graph = self.graph.expect("operation graph is required");
        let global_index = self.global_index.expect("global index is required.");

        unsafe {
            let mut descriptor = Descriptor::new(
                cudnn_sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINE_DESCRIPTOR,
            )?;

            descriptor.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &graph.descriptor.inner(),
            )?;

            descriptor.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
                1,
                &global_index,
            )?;

            descriptor.finalize()?;

            Ok(Engine {
                descriptor,
                graph,
                global_index,
            })
        }
    }
}

#[derive(PartialEq, Debug)]
pub struct Engine {
    pub(crate) descriptor: Descriptor,
    graph: Graph,
    global_index: i64,
}

impl Engine {
    pub fn get_graph(&self) -> &Graph {
        &self.graph
    }

    pub fn get_global_index(&self) -> i64 {
        self.global_index
    }
}
