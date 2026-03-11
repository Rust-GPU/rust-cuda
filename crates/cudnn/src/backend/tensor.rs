use crate::{CudnnError, DataType, IntoResult, backend::Descriptor};

#[derive(Clone, Default, Debug, PartialEq, Eq, Hash)]
pub struct TensorBuilder<'a> {
    id: Option<i64>,
    data_type: Option<cudnn_sys::cudnnDataType_t>,
    byte_alignment: Option<i64>,
    dimensions: Option<&'a [i64]>,
    strides: Option<&'a [i64]>,
    vector_count: Option<i64>,
    vectorized_dimension: Option<i64>,
    is_virtual: bool,
}

impl<'a> TensorBuilder<'a> {
    pub fn set_id(mut self, id: i64) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn set_data_type<T>(mut self) -> Self
    where
        T: DataType,
    {
        self.data_type = Some(T::into_raw());
        self.byte_alignment = Some(std::mem::size_of::<T>() as i64);
        self
    }

    pub fn set_dimensions(mut self, dimensions: &'a [i64]) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    pub fn set_strides(mut self, strides: &'a [i64]) -> Self {
        self.strides = Some(strides);
        self
    }

    pub fn set_vector_count(mut self, vector_count: i64) -> Self {
        self.vector_count = Some(vector_count);
        self
    }

    pub fn set_vectorized_dimension(mut self, vectorized_dimension: i64) -> Self {
        self.vectorized_dimension = Some(vectorized_dimension);
        self
    }

    pub fn is_virtual(mut self, is_virtual: bool) -> Self {
        self.is_virtual = is_virtual;
        self
    }

    pub fn build(mut self) -> Result<Tensor, CudnnError> {
        let id = self.id.expect("id is required.");
        let data_type = self.data_type.expect("data type is required.");
        let byte_alignment = self.byte_alignment.expect("byte alignment is required.");
        let dimensions = self.dimensions.expect("dimensions are required.");
        let strides = self.strides.expect("strides are required.");

        unsafe {
            let mut raw = Descriptor::new(
                cudnn_sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_TENSOR_DESCRIPTOR,
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_UNIQUE_ID,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
                1,
                &id,
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_DATA_TYPE,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DATA_TYPE,
                1,
                &data_type,
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
                1,
                &byte_alignment,
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_DIMENSIONS,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
                dimensions.len() as i64,
                dimensions,
            )?;

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_STRIDES,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
                strides.len() as i64,
                strides,
            )?;

            if let Some(vector_count) = self.vector_count {
                raw.set_attribute(
                    cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_VECTOR_COUNT,
                    cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
                    1,
                    &vector_count,
                )?;

                if vector_count != 1 {
                    let vectorized_dimension = self
                        .vectorized_dimension
                        .expect("vectorized_dimension is required when vector_count > 1");

                    raw.set_attribute(
                        cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION,
                        cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
                        1,
                        &vectorized_dimension,
                    )?;
                }
            }

            raw.set_attribute(
                cudnn_sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_IS_VIRTUAL,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BOOLEAN,
                1,
                &self.is_virtual,
            )?;

            raw.finalize()?;

            Ok(Tensor {
                raw,
                id,
                byte_alignment,
                dimensions: dimensions.to_vec(),
                strides: strides.to_vec(),
                vector_count: self.vector_count.unwrap_or(1),
                vectorized_dimension: self.vectorized_dimension,
                is_virtual: self.is_virtual,
            })
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Tensor {
    pub(crate) raw: Descriptor,
    id: i64,
    byte_alignment: i64,
    dimensions: Vec<i64>,
    strides: Vec<i64>,
    vector_count: i64,
    vectorized_dimension: Option<i64>,
    is_virtual: bool,
}

impl Tensor {
    pub fn get_id(&self) -> i64 {
        self.id
    }

    pub fn get_byte_alignment(&self) -> i64 {
        self.byte_alignment
    }

    pub fn get_dimensions(&self) -> &[i64] {
        &self.dimensions
    }

    pub fn get_strides(&self) -> &[i64] {
        &self.strides
    }
    pub fn get_vector_count(&self) -> i64 {
        self.vector_count
    }

    pub fn get_vectorized_dimension(&self) -> Option<i64> {
        self.vectorized_dimension
    }

    pub fn is_virtual(&self) -> bool {
        self.is_virtual
    }
}
