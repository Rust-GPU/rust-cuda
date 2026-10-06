/// Specifies the mode of operation for batch normalization.
///
/// The mode determines how the mean and variance statistics are computed and the required
/// dimensions of the scale, bias, mean and variance tensors.
///
/// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnBatchNormMode_t)
/// may offer additional information about the API behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BatchNormMode {
    /// Normalization is performed per-activation. The scale, bias, mean and variance tensors are
    /// expected to have dimensions `1xCxHxW` and normalization is performed across the batch (`N`)
    /// dimension only. This mode is intended for use after non-convolutional layers.
    PerActivation,
    /// Normalization is performed over the `N`, `H` and `W` dimensions. The scale, bias, mean and
    /// variance tensors are expected to have dimensions `1xCx1x1`. This mode is intended for use
    /// after convolutional layers.
    Spatial,
    /// A faster variant of [`Spatial`](BatchNormMode::Spatial) that may impose limits on the range
    /// of representable values. It requires the saved intermediate results from the forward pass to
    /// be provided to the backward pass.
    SpatialPersistent,
}

impl From<BatchNormMode> for cudnn_sys::cudnnBatchNormMode_t {
    fn from(mode: BatchNormMode) -> Self {
        match mode {
            BatchNormMode::PerActivation => Self::CUDNN_BATCHNORM_PER_ACTIVATION,
            BatchNormMode::Spatial => Self::CUDNN_BATCHNORM_SPATIAL,
            BatchNormMode::SpatialPersistent => Self::CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
        }
    }
}
