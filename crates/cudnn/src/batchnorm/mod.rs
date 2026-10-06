use std::mem::MaybeUninit;

use cust::memory::GpuBuffer;

use crate::{CudnnContext, CudnnError, DataType, IntoResult, TensorDescriptor, private};

mod batchnorm_mode;

pub use batchnorm_mode::*;

impl CudnnContext {
    /// Derives the secondary tensor descriptor for the scale, bias, mean and variance tensors used
    /// by the batch normalization operations, given the descriptor of the operand `x` and the
    /// batch normalization mode.
    ///
    /// This wraps `cudnnDeriveBNTensorDescriptor`. The returned descriptor is used as the
    /// `scale_bias_mean_var_desc` argument of the batch normalization functions.
    ///
    /// # Arguments
    ///
    /// * `x_desc` - tensor descriptor of the operand `x`.
    /// * `mode` - batch normalization mode that will be used with the derived descriptor.
    ///
    /// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDeriveBNTensorDescriptor)
    /// may offer additional information about the API behavior.
    pub fn derive_batch_norm_descriptor<T>(
        &self,
        x_desc: &TensorDescriptor<T>,
        mode: BatchNormMode,
    ) -> Result<TensorDescriptor<T>, CudnnError>
    where
        T: DataType,
    {
        let mut raw = MaybeUninit::uninit();

        unsafe {
            cudnn_sys::cudnnCreateTensorDescriptor(raw.as_mut_ptr()).into_result()?;
            let raw = raw.assume_init();

            cudnn_sys::cudnnDeriveBNTensorDescriptor(raw, x_desc.raw, mode.into()).into_result()?;

            Ok(TensorDescriptor::from_raw(raw))
        }
    }

    /// Performs the forward batch normalization pass in inference mode, using pre-computed
    /// (population) statistics.
    ///
    /// The result is computed as `y = scale * (x - estimatedMean) / sqrt(estimatedVariance +
    /// epsilon) + bias` and blended into the destination as `alpha * result + beta * y_prior`.
    ///
    /// # Arguments
    ///
    /// * `mode` - batch normalization mode.
    /// * `alpha` - scaling factor for the result. Must be stored in host memory.
    /// * `beta` - scaling factor for the destination tensor. Must be stored in host memory.
    /// * `x_desc` - tensor descriptor for the operand.
    /// * `x` - operand data in device memory.
    /// * `y_desc` - tensor descriptor for the result.
    /// * `y` - output data in device memory.
    /// * `scale_bias_mean_var_desc` - descriptor derived with
    ///   [`derive_batch_norm_descriptor`](CudnnContext::derive_batch_norm_descriptor).
    /// * `scale` - scale (`gamma`) data in device memory.
    /// * `bias` - bias (`beta`) data in device memory.
    /// * `estimated_mean` - pre-computed mean data in device memory.
    /// * `estimated_variance` - pre-computed variance data in device memory.
    /// * `epsilon` - epsilon value used in the batch normalization formula. Must be the same value
    ///   in the forward and backward passes.
    ///
    /// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnBatchNormalizationForwardInference)
    /// may offer additional information about the API behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if the configuration in input is not supported, the tensor shapes differ or
    /// the data types of the input and destination tensor are not the same.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_norm_forward_inference<T, CompT>(
        &self,
        mode: BatchNormMode,
        alpha: CompT,
        beta: CompT,
        x_desc: &TensorDescriptor<T>,
        x: &impl GpuBuffer<T>,
        y_desc: &TensorDescriptor<T>,
        y: &mut impl GpuBuffer<T>,
        scale_bias_mean_var_desc: &TensorDescriptor<T>,
        scale: &impl GpuBuffer<T>,
        bias: &impl GpuBuffer<T>,
        estimated_mean: &impl GpuBuffer<T>,
        estimated_variance: &impl GpuBuffer<T>,
        epsilon: f64,
    ) -> Result<(), CudnnError>
    where
        T: DataType,
        CompT: SupportedBatchNorm<T>,
    {
        let alpha_ptr = &alpha as *const CompT as *const _;
        let beta_ptr = &beta as *const CompT as *const _;

        let x_ptr = x.as_device_ptr().as_ptr() as *const _;
        let y_ptr = y.as_device_ptr().as_mut_ptr() as *mut _;

        let scale_ptr = scale.as_device_ptr().as_ptr() as *const _;
        let bias_ptr = bias.as_device_ptr().as_ptr() as *const _;
        let mean_ptr = estimated_mean.as_device_ptr().as_ptr() as *const _;
        let variance_ptr = estimated_variance.as_device_ptr().as_ptr() as *const _;

        unsafe {
            cudnn_sys::cudnnBatchNormalizationForwardInference(
                self.raw,
                mode.into(),
                alpha_ptr,
                beta_ptr,
                x_desc.raw,
                x_ptr,
                y_desc.raw,
                y_ptr,
                scale_bias_mean_var_desc.raw,
                scale_ptr,
                bias_ptr,
                mean_ptr,
                variance_ptr,
                epsilon,
            )
            .into_result()
        }
    }

    /// Performs the forward batch normalization pass in training mode, computing the batch
    /// statistics and updating the running (population) statistics.
    ///
    /// The `save_mean` and `save_inv_variance` buffers cache the batch statistics so they can be
    /// reused to speed up the backward pass; pass them again to
    /// [`batch_norm_backward`](CudnnContext::batch_norm_backward).
    ///
    /// # Arguments
    ///
    /// * `mode` - batch normalization mode.
    /// * `alpha` - scaling factor for the result. Must be stored in host memory.
    /// * `beta` - scaling factor for the destination tensor. Must be stored in host memory.
    /// * `x_desc` - tensor descriptor for the operand.
    /// * `x` - operand data in device memory.
    /// * `y_desc` - tensor descriptor for the result.
    /// * `y` - output data in device memory.
    /// * `scale_bias_mean_var_desc` - descriptor derived with
    ///   [`derive_batch_norm_descriptor`](CudnnContext::derive_batch_norm_descriptor).
    /// * `scale` - scale (`gamma`) data in device memory.
    /// * `bias` - bias (`beta`) data in device memory.
    /// * `exponential_average_factor` - factor used to update the running statistics. Use `1.0` on
    ///   the first call of a training cycle and `1/(1+n)` on the n-th call for a cumulative moving
    ///   average.
    /// * `running_mean` - running mean, updated in place.
    /// * `running_variance` - running variance, updated in place.
    /// * `epsilon` - epsilon value used in the batch normalization formula. Must be the same value
    ///   in the forward and backward passes.
    /// * `save_mean` - device memory that will hold the saved batch mean.
    /// * `save_inv_variance` - device memory that will hold the saved batch inverse variance.
    ///
    /// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnBatchNormalizationForwardTraining)
    /// may offer additional information about the API behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if the configuration in input is not supported, the tensor shapes differ or
    /// the data types of the input and destination tensor are not the same.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_norm_forward_training<T, CompT>(
        &self,
        mode: BatchNormMode,
        alpha: CompT,
        beta: CompT,
        x_desc: &TensorDescriptor<T>,
        x: &impl GpuBuffer<T>,
        y_desc: &TensorDescriptor<T>,
        y: &mut impl GpuBuffer<T>,
        scale_bias_mean_var_desc: &TensorDescriptor<T>,
        scale: &impl GpuBuffer<T>,
        bias: &impl GpuBuffer<T>,
        exponential_average_factor: f64,
        running_mean: &mut impl GpuBuffer<T>,
        running_variance: &mut impl GpuBuffer<T>,
        epsilon: f64,
        save_mean: &mut impl GpuBuffer<T>,
        save_inv_variance: &mut impl GpuBuffer<T>,
    ) -> Result<(), CudnnError>
    where
        T: DataType,
        CompT: SupportedBatchNorm<T>,
    {
        let alpha_ptr = &alpha as *const CompT as *const _;
        let beta_ptr = &beta as *const CompT as *const _;

        let x_ptr = x.as_device_ptr().as_ptr() as *const _;
        let y_ptr = y.as_device_ptr().as_mut_ptr() as *mut _;

        let scale_ptr = scale.as_device_ptr().as_ptr() as *const _;
        let bias_ptr = bias.as_device_ptr().as_ptr() as *const _;

        let running_mean_ptr = running_mean.as_device_ptr().as_mut_ptr() as *mut _;
        let running_variance_ptr = running_variance.as_device_ptr().as_mut_ptr() as *mut _;

        let save_mean_ptr = save_mean.as_device_ptr().as_mut_ptr() as *mut _;
        let save_inv_variance_ptr = save_inv_variance.as_device_ptr().as_mut_ptr() as *mut _;

        unsafe {
            cudnn_sys::cudnnBatchNormalizationForwardTraining(
                self.raw,
                mode.into(),
                alpha_ptr,
                beta_ptr,
                x_desc.raw,
                x_ptr,
                y_desc.raw,
                y_ptr,
                scale_bias_mean_var_desc.raw,
                scale_ptr,
                bias_ptr,
                exponential_average_factor,
                running_mean_ptr,
                running_variance_ptr,
                epsilon,
                save_mean_ptr,
                save_inv_variance_ptr,
            )
            .into_result()
        }
    }

    /// Performs the backward batch normalization pass, computing the gradients with respect to the
    /// input (`dx`), the scale (`d_scale`) and the bias (`d_bias`).
    ///
    /// The `saved_mean` and `saved_inv_variance` buffers must be the ones produced by the matching
    /// [`batch_norm_forward_training`](CudnnContext::batch_norm_forward_training) call.
    ///
    /// # Arguments
    ///
    /// * `mode` - batch normalization mode.
    /// * `alpha_data_diff` / `beta_data_diff` - blend factors for the `dx` gradient. Host memory.
    /// * `alpha_param_diff` / `beta_param_diff` - blend factors for the `d_scale` and `d_bias`
    ///   gradients. Host memory.
    /// * `x_desc` - tensor descriptor shared by `x`, `dy` and `dx`.
    /// * `x` - operand data in device memory.
    /// * `dy_desc` - tensor descriptor for the incoming gradient.
    /// * `dy` - incoming gradient data in device memory.
    /// * `dx_desc` - tensor descriptor for the resulting gradient.
    /// * `dx` - resulting gradient data in device memory.
    /// * `scale_bias_diff_desc` - descriptor derived with
    ///   [`derive_batch_norm_descriptor`](CudnnContext::derive_batch_norm_descriptor).
    /// * `scale` - scale (`gamma`) data in device memory.
    /// * `result_scale_diff` - resulting gradient with respect to the scale.
    /// * `result_bias_diff` - resulting gradient with respect to the bias.
    /// * `epsilon` - epsilon value. Must be the same value used in the forward pass.
    /// * `saved_mean` - saved batch mean produced by the forward training pass.
    /// * `saved_inv_variance` - saved batch inverse variance produced by the forward training pass.
    ///
    /// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnBatchNormalizationBackward)
    /// may offer additional information about the API behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if the configuration in input is not supported, the tensor shapes differ or
    /// the data types of the involved tensors are not the same.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_norm_backward<T, CompT>(
        &self,
        mode: BatchNormMode,
        alpha_data_diff: CompT,
        beta_data_diff: CompT,
        alpha_param_diff: CompT,
        beta_param_diff: CompT,
        x_desc: &TensorDescriptor<T>,
        x: &impl GpuBuffer<T>,
        dy_desc: &TensorDescriptor<T>,
        dy: &impl GpuBuffer<T>,
        dx_desc: &TensorDescriptor<T>,
        dx: &mut impl GpuBuffer<T>,
        scale_bias_diff_desc: &TensorDescriptor<T>,
        scale: &impl GpuBuffer<T>,
        result_scale_diff: &mut impl GpuBuffer<T>,
        result_bias_diff: &mut impl GpuBuffer<T>,
        epsilon: f64,
        saved_mean: &impl GpuBuffer<T>,
        saved_inv_variance: &impl GpuBuffer<T>,
    ) -> Result<(), CudnnError>
    where
        T: DataType,
        CompT: SupportedBatchNorm<T>,
    {
        let alpha_data_ptr = &alpha_data_diff as *const CompT as *const _;
        let beta_data_ptr = &beta_data_diff as *const CompT as *const _;
        let alpha_param_ptr = &alpha_param_diff as *const CompT as *const _;
        let beta_param_ptr = &beta_param_diff as *const CompT as *const _;

        let x_ptr = x.as_device_ptr().as_ptr() as *const _;
        let dy_ptr = dy.as_device_ptr().as_ptr() as *const _;
        let dx_ptr = dx.as_device_ptr().as_mut_ptr() as *mut _;

        let scale_ptr = scale.as_device_ptr().as_ptr() as *const _;
        let d_scale_ptr = result_scale_diff.as_device_ptr().as_mut_ptr() as *mut _;
        let d_bias_ptr = result_bias_diff.as_device_ptr().as_mut_ptr() as *mut _;

        let saved_mean_ptr = saved_mean.as_device_ptr().as_ptr() as *const _;
        let saved_inv_variance_ptr = saved_inv_variance.as_device_ptr().as_ptr() as *const _;

        unsafe {
            cudnn_sys::cudnnBatchNormalizationBackward(
                self.raw,
                mode.into(),
                alpha_data_ptr,
                beta_data_ptr,
                alpha_param_ptr,
                beta_param_ptr,
                x_desc.raw,
                x_ptr,
                dy_desc.raw,
                dy_ptr,
                dx_desc.raw,
                dx_ptr,
                scale_bias_diff_desc.raw,
                scale_ptr,
                d_scale_ptr,
                d_bias_ptr,
                epsilon,
                saved_mean_ptr,
                saved_inv_variance_ptr,
            )
            .into_result()
        }
    }
}

/// Supported data type configurations for batch normalization operations.
pub trait SupportedBatchNorm<T>: DataType + private::Sealed {}

impl SupportedBatchNorm<f32> for f32 {}
impl SupportedBatchNorm<f64> for f64 {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScalarC;
    use cust::memory::{CopyDestination, DeviceBuffer};

    // Batch norm in SPATIAL mode normalizes each channel across N,H,W to mean 0 / variance 1.
    // With scale=1, bias=0 the output of each channel [a, b] becomes [-1, 1].
    #[test]
    fn batch_norm_end_to_end() {
        let _ctx = cust::quick_init().unwrap();
        let cudnn = CudnnContext::new().unwrap();

        // N=1, C=2, H=1, W=2. Channel 0 = [1, 2], channel 1 = [10, 20].
        let shape = [1, 2, 1, 2];
        let x_desc = TensorDescriptor::<f32>::new_format(&shape, ScalarC::Nchw).unwrap();
        let y_desc = TensorDescriptor::<f32>::new_format(&shape, ScalarC::Nchw).unwrap();
        let bn_desc = cudnn
            .derive_batch_norm_descriptor(&x_desc, BatchNormMode::Spatial)
            .unwrap();

        let x = DeviceBuffer::from_slice(&[1.0f32, 2.0, 10.0, 20.0]).unwrap();
        let scale = DeviceBuffer::from_slice(&[1.0f32, 1.0]).unwrap();
        let bias = DeviceBuffer::from_slice(&[0.0f32, 0.0]).unwrap();
        let expected = [-1.0f32, 1.0, -1.0, 1.0];

        // ---- forward training ----
        let mut y = DeviceBuffer::from_slice(&[0.0f32; 4]).unwrap();
        let mut running_mean = DeviceBuffer::from_slice(&[0.0f32, 0.0]).unwrap();
        let mut running_var = DeviceBuffer::from_slice(&[1.0f32, 1.0]).unwrap();
        let mut save_mean = DeviceBuffer::from_slice(&[0.0f32, 0.0]).unwrap();
        let mut save_inv_var = DeviceBuffer::from_slice(&[0.0f32, 0.0]).unwrap();

        cudnn
            .batch_norm_forward_training::<f32, f32>(
                BatchNormMode::Spatial,
                1.0,
                0.0,
                &x_desc,
                &x,
                &y_desc,
                &mut y,
                &bn_desc,
                &scale,
                &bias,
                1.0,
                &mut running_mean,
                &mut running_var,
                1e-5,
                &mut save_mean,
                &mut save_inv_var,
            )
            .unwrap();

        let mut out = [0.0f32; 4];
        y.copy_to(&mut out).unwrap();
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!(
                (o - e).abs() < 1e-2,
                "forward_training output {out:?}, expected {expected:?}"
            );
        }

        // ---- forward inference (using the population stats we just derived) ----
        let est_mean = DeviceBuffer::from_slice(&[1.5f32, 15.0]).unwrap();
        let est_var = DeviceBuffer::from_slice(&[0.25f32, 25.0]).unwrap();
        let mut y_inf = DeviceBuffer::from_slice(&[0.0f32; 4]).unwrap();
        cudnn
            .batch_norm_forward_inference::<f32, f32>(
                BatchNormMode::Spatial,
                1.0,
                0.0,
                &x_desc,
                &x,
                &y_desc,
                &mut y_inf,
                &bn_desc,
                &scale,
                &bias,
                &est_mean,
                &est_var,
                1e-5,
            )
            .unwrap();
        let mut out_inf = [0.0f32; 4];
        y_inf.copy_to(&mut out_inf).unwrap();
        for (o, e) in out_inf.iter().zip(expected.iter()) {
            assert!(
                (o - e).abs() < 1e-2,
                "forward_inference output {out_inf:?}, expected {expected:?}"
            );
        }

        // ---- backward: d_bias equals the per-channel sum of dy ----
        let dy = DeviceBuffer::from_slice(&[1.0f32, -1.0, 0.5, -0.5]).unwrap();
        let mut dx = DeviceBuffer::from_slice(&[0.0f32; 4]).unwrap();
        let mut d_scale = DeviceBuffer::from_slice(&[0.0f32, 0.0]).unwrap();
        let mut d_bias = DeviceBuffer::from_slice(&[0.0f32, 0.0]).unwrap();
        cudnn
            .batch_norm_backward::<f32, f32>(
                BatchNormMode::Spatial,
                1.0,
                0.0,
                1.0,
                0.0,
                &x_desc,
                &x,
                &y_desc,
                &dy,
                &x_desc,
                &mut dx,
                &bn_desc,
                &scale,
                &mut d_scale,
                &mut d_bias,
                1e-5,
                &save_mean,
                &save_inv_var,
            )
            .unwrap();
        let mut d_bias_host = [0.0f32; 2];
        d_bias.copy_to(&mut d_bias_host).unwrap();
        // ch0: 1 + (-1) = 0 ; ch1: 0.5 + (-0.5) = 0
        assert!(
            d_bias_host.iter().all(|v| v.abs() < 1e-2),
            "backward d_bias {d_bias_host:?}, expected ~[0, 0]"
        );
    }
}
