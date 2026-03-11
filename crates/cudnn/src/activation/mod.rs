use cust::memory::GpuBuffer;

use crate::{
    CudnnContext, CudnnError, DataType, IntoResult, ScalingDataType, TensorDescriptor, private,
};

mod activation_descriptor;
mod activation_mode;

pub use activation_descriptor::*;
pub use activation_mode::*;

impl CudnnContext {
    /// Applies a specific neuron activation functions element wise of the provided
    /// tensor.
    ///
    /// # Arguments
    ///
    ///   * `activation_desc` - activation descriptor.
    ///   * `alpha` - scaling factor for the result.
    ///   * `x_desc` - tensor descriptor for the input.
    ///   * `x` - data for the input tensor.
    ///   * `beta` - scaling factor for the destination tensor.
    ///   * `y_desc` - tensor descriptor for the output.
    ///   * `y` - data for the output.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnActivationForward)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if the shapes of the `y` and `x` tensors do not match or an
    /// unsupported configuration of arguments is detected.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{ActivationDescriptor, ActivationMode, CudnnContext, NanPropagation, TensorDescriptor};
    /// use cust::memory::DeviceBuffer;
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let mode = ActivationMode::Swish;
    /// let nan_opt = NanPropagation::PropagateNaN;
    /// let coefficient = None;
    ///
    /// let desc = ActivationDescriptor::new(mode, nan_opt, coefficient)?;
    ///
    /// let alpha = 1.0;
    /// let x_desc = TensorDescriptor::<i8>::new_strides(&[1, 1, 1, 5], &[5, 5, 5, 1])?;
    /// let x = DeviceBuffer::<i8>::from_slice(&[10, 10, 10, 10, 10])?;
    ///
    /// let beta = 0.0;
    /// let y_desc = TensorDescriptor::<i8>::new_strides(&[1, 1, 1, 5], &[5, 5, 5, 1])?;
    /// let mut y = DeviceBuffer::<i8>::from_slice(&[0, 0, 0, 0, 0])?;
    ///
    /// ctx.activation_forward(&desc, alpha, &x_desc, &x, beta, &y_desc, &mut y)?;
    ///
    /// let y_host = y.as_host_vec()?;
    ///
    /// assert!(y_host.iter().all(|el| *el == 10));
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn activation_forward<CompT, T>(
        &self,
        activation_desc: &ActivationDescriptor,
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
            cudnn_sys::cudnnActivationForward(
                self.raw,
                activation_desc.raw,
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

    /// Computes the gradient of a neuron activation function.
    ///
    /// # Arguments
    ///
    ///   * `activation_descriptor` - descriptor of a neuron activation operation.
    ///   * `alpha` - scaling factor for the result.
    ///   * `y_desc` - tensor descriptor for the output map.
    ///   * `y` - data for the output map.
    ///   * `dy_desc` - tensor descriptor for the differential of the output map.
    ///   * `dy` - data foe the differential of the output map.
    ///   * `x_desc` - tensor descriptor for the activation input.
    ///   * `x` - data for the activation input.
    ///   * `beta` - scaling factor for the destination tensor.
    ///   * `dx_desc` - tensor descriptor for the input differential.
    ///   * `dx` - data for the input differential.
    ///
    /// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnActivationBackward)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if the shapes of the `dx` and `x` tensors do not match, the strides of the
    /// tensors and their differential do not match,  or an unsupported configuration of arguments
    /// is detected.
    #[allow(clippy::too_many_arguments)]
    pub fn activation_backward<CompT, T>(
        &self,
        activation_desc: &ActivationDescriptor,
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
        CompT: SupportedActBwd<T>,
        T: DataType,
    {
        let alpha_ptr = &alpha as *const CompT as *const _;

        let y_ptr = y.as_device_ptr().as_ptr() as *const _;
        let dy_ptr = dy.as_device_ptr().as_ptr() as *const _;
        let x_ptr = x.as_device_ptr().as_ptr() as *const _;

        let beta_ptr = &beta as *const CompT as *const _;

        let dx_ptr = dx.as_device_ptr().as_mut_ptr() as *mut _;

        unsafe {
            cudnn_sys::cudnnActivationBackward(
                self.raw,
                activation_desc.raw,
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

/// Supported type configurations for the activation backward operation.
pub trait SupportedActBwd<T>: DataType + private::Sealed
where
    T: DataType,
{
}

impl SupportedActBwd<f32> for f32 {}
impl SupportedActBwd<f64> for f64 {}
