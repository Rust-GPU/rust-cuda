use cust::memory::GpuBuffer;

use crate::{
    CudnnContext, CudnnError, DataType, IntoResult, ScalingDataType, TensorDescriptor, private,
};

mod pooling_descriptor;
mod pooling_mode;

pub use pooling_descriptor::*;
pub use pooling_mode::*;

impl CudnnContext {
    /// This function computes the pooling of the input tensor and produces a smaller
    /// tensor in output.
    ///
    /// # Arguments
    ///
    /// * `pooling_desc` - descriptor of the pooling operation.
    /// * `alpha` - scaling factor for the result.
    /// * `x_desc` - descriptor for the input tensor.
    /// * `x` - data for the input tensor.
    /// * `beta` - scaling factor for the destination tensor.
    /// * `y_desc` - descriptor for the destination tensor.
    /// * `y` - data for the destination tensor.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnPoolingForward)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if the batch size or channels dimensions of the two tensor differ
    /// or an invalid combination of arguments is detected.
    #[allow(clippy::too_many_arguments)]
    pub fn pooling_forward<CompT, T>(
        &self,
        pooling_desc: &PoolingDescriptor,
        alpha: CompT,
        x_desc: &TensorDescriptor<T>,
        x: &impl GpuBuffer<T>,
        beta: CompT,
        y_desc: &TensorDescriptor<T>,
        y: &mut impl GpuBuffer<T>,
    ) -> Result<(), CudnnError>
    where
        CompT: ScalingDataType<T>,
        T: DataType,
    {
        let alpha_ptr = &alpha as *const CompT as *const _;
        let x_ptr = x.as_device_ptr().as_ptr() as *const _;

        let beta_ptr = &beta as *const CompT as *const _;
        let y_ptr = y.as_device_ptr().as_mut_ptr() as *mut _;

        unsafe {
            cudnn_sys::cudnnPoolingForward(
                self.raw,
                pooling_desc.raw,
                alpha_ptr,
                x_desc.raw,
                x_ptr,
                beta_ptr,
                y_desc.raw,
                y_ptr,
            )
            .into_result()
        }
    }

    /// Computes the gradient of a pooling operation.
    ///
    /// # Arguments
    ///
    /// * `pooling_desc` - descriptor of the pooling operation.
    /// * `alpha` - scaling factor for the result.
    /// * `y_desc` - tensor descriptor for the output map.
    /// * `y` - data for the output map.
    /// * `dy_desc` - tensor descriptor for the differential of the output map.
    /// * `dy` - data foe the differential of the output map.
    /// * `x_desc` - tensor descriptor for the pooling input.
    /// * `x` - data for the pooling input.
    /// * `beta` - scaling factor for the destination tensor.
    /// * `dx_desc` - tensor descriptor for the input differential.
    /// * `dx` - data for the input differential.
    ///
    /// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnPoolingBackward)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if the dimensions or the strides of `y` and `dy` tensors differ or if the
    /// dimensions or the strides of `x` and `dx` tensors differ or if an unsupported combination
    /// of arguments is detected.
    #[allow(clippy::too_many_arguments)]
    pub fn pooling_backward<CompT, T>(
        &self,
        pooling_desc: &PoolingDescriptor,
        alpha: CompT,
        y_desc: &TensorDescriptor<T>,
        y: &impl GpuBuffer<T>,
        dy_desc: &TensorDescriptor<T>,
        dy: &impl GpuBuffer<T>,
        x_desc: &TensorDescriptor<T>,
        x: &impl GpuBuffer<T>,
        beta: CompT,
        dx_desc: &TensorDescriptor<T>,
        dx: &mut impl GpuBuffer<T>,
    ) -> Result<(), CudnnError>
    where
        CompT: SupportedPoolBwd<T>,
        T: DataType,
    {
        let alpha_ptr = &alpha as *const CompT as *const _;

        let y_ptr = y.as_device_ptr().as_ptr() as *const _;
        let dy_ptr = dy.as_device_ptr().as_ptr() as *const _;
        let x_ptr = x.as_device_ptr().as_ptr() as *const _;

        let beta_ptr = &beta as *const CompT as *const _;

        let dx_ptr = dx.as_device_ptr().as_mut_ptr() as *mut _;

        unsafe {
            cudnn_sys::cudnnPoolingBackward(
                self.raw,
                pooling_desc.raw,
                alpha_ptr,
                y_desc.raw,
                y_ptr,
                dy_desc.raw,
                dy_ptr,
                x_desc.raw,
                x_ptr,
                beta_ptr,
                dx_desc.raw,
                dx_ptr,
            )
            .into_result()
        }
    }
}

/// Supported type configurations for the pooling backward operation as specified in the cuDNN
/// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnPoolingBackward).
pub trait SupportedPoolBwd<T>: DataType + private::Sealed
where
    T: DataType,
{
}

impl SupportedPoolBwd<f32> for f32 {}
impl SupportedPoolBwd<f64> for f64 {}
