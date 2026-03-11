use cust::memory::GpuBuffer;

use crate::{CudnnContext, CudnnError, DataType, IntoResult, TensorDescriptor, private};

mod softmax_algo;
mod softmax_mode;

pub use softmax_algo::*;
pub use softmax_mode::*;

impl CudnnContext {
    /// Computes the softmax function.
    ///
    /// # Arguments
    ///
    /// * `algo` - softmax algorithm to compute.
    ///
    /// * `mode` - specifies the softmax mode.
    ///
    /// * `alpha` - scaling factor for the result. Must be stored in host memory.
    ///
    /// * `x_desc` - tensor descriptor for the operand.
    ///
    /// * `x` - operand data in device memory.
    ///
    /// * `beta` - scaling factor for the destination tensor.
    ///
    /// * `y_desc` - tensor descriptor for the result.
    ///
    /// * `y` - output data in device memory.
    ///
    /// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSoftmaxForward)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if the configuration in input is not supported, the tensor shapes differ or
    /// the data types of the input and destination tensor are not the same.
    #[allow(clippy::too_many_arguments)]
    pub fn softmax_forward<T, CompT>(
        &self,
        algo: SoftmaxAlgo,
        mode: SoftmaxMode,
        alpha: CompT,
        x_desc: &TensorDescriptor<T>,
        x: &impl GpuBuffer<T>,
        beta: CompT,
        y_desc: &TensorDescriptor<T>,
        y: &mut impl GpuBuffer<T>,
    ) -> Result<(), CudnnError>
    where
        T: DataType,
        CompT: SupportedSoftmax<T>,
    {
        let alpha_ptr = &alpha as *const CompT as *const _;
        let x_ptr = x.as_device_ptr().as_ptr() as *const _;

        let beta_ptr = &beta as *const CompT as *const _;
        let y_ptr = y.as_device_ptr().as_mut_ptr() as *mut _;

        unsafe {
            cudnn_sys::cudnnSoftmaxForward(
                self.raw,
                algo.into(),
                mode.into(),
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

    /// Computes the gradient of the softmax function
    ///
    /// # Arguments
    ///
    /// * `algo` - softmax algorithm to compute the gradient of.
    /// * `mode` - specifies the softmax mode to compute the gradient of.
    /// * `alpha` - scaling factor for the result. Must be stored in host memory.
    /// * `y_desc` - tensor descriptor for the operand.
    /// * `y` - operand data in device memory.
    /// * `dy_desc` - tensor descriptor for the result.
    /// * `dy` - output data in device memory.
    /// * `beta` - scaling factor for the differential tensor.
    /// * `dx_desc` - differential tensor descriptor.
    /// * `dx` - differential data in device memory.
    ///
    /// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSoftmaxBackward)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if the configuration in input is not supported, the tensor shapes differ or
    /// the data types of the input and differential tensor are not the same.
    #[allow(clippy::too_many_arguments)]
    pub fn softmax_backward<T, CompT>(
        &self,
        algo: SoftmaxAlgo,
        mode: SoftmaxMode,
        alpha: CompT,
        y_desc: &TensorDescriptor<T>,
        y: &impl GpuBuffer<T>,
        dy_desc: &TensorDescriptor<T>,
        dy: &impl GpuBuffer<T>,
        beta: CompT,
        dx_desc: &TensorDescriptor<T>,
        dx: &mut impl GpuBuffer<T>,
    ) -> Result<(), CudnnError>
    where
        T: DataType,
        CompT: SupportedSoftmax<T>,
    {
        let alpha_ptr = &alpha as *const CompT as *const _;
        let y_ptr = y.as_device_ptr().as_ptr() as *const _;

        let beta_ptr = &beta as *const CompT as *const _;
        let dy_ptr = dy.as_device_ptr().as_ptr() as *const _;

        let dx_ptr = dx.as_device_ptr().as_mut_ptr() as *mut _;

        unsafe {
            cudnn_sys::cudnnSoftmaxBackward(
                self.raw,
                algo.into(),
                mode.into(),
                alpha_ptr,
                y_desc.raw,
                y_ptr,
                dy_desc.raw,
                dy_ptr,
                beta_ptr,
                dx_desc.raw,
                dx_ptr,
            )
            .into_result()
        }
    }
}

/// Supported data type configurations for softmax operations.
pub trait SupportedSoftmax<T>: DataType + private::Sealed {}

impl SupportedSoftmax<f32> for f32 {}
impl SupportedSoftmax<f64> for f64 {}
