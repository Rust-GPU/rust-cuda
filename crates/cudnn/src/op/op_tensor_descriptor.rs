use std::{marker::PhantomData, mem::MaybeUninit};

use crate::{BinaryOp, CudnnError, DataType, IntoResult, NanPropagation, UnaryOp};

/// Initializes an op tensor descriptor.
///
/// # Arguments
///
/// * `raw` - raw pointer to the descriptor.
/// * `op` - raw  operation type.
/// * `nan_opt` - raw nan propagation policy.
unsafe fn init_raw_op_descriptor<T: DataType>(
    op: cudnn_sys::cudnnOpTensorOp_t,
    nan_opt: cudnn_sys::cudnnNanPropagation_t,
) -> Result<cudnn_sys::cudnnOpTensorDescriptor_t, CudnnError> {
    let mut raw = MaybeUninit::uninit();

    unsafe {
        cudnn_sys::cudnnCreateOpTensorDescriptor(raw.as_mut_ptr()).into_result()?;
        let raw = raw.assume_init();
        cudnn_sys::cudnnSetOpTensorDescriptor(raw, op, T::into_raw(), nan_opt).into_result()?;
        Ok(raw)
    }
}

/// The description of a unary Tensor Core operation.
///
/// As specified in the cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#scaling-parameters),
/// admissible types for scaling parameters are `f32` and `f64` for `f32` and `f64` tensors
/// respectively.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct UnaryOpTensorDescriptor<T: DataType> {
    pub(crate) raw: cudnn_sys::cudnnOpTensorDescriptor_t,
    comp_type: PhantomData<T>,
    op: UnaryOp,
}

impl<T> UnaryOpTensorDescriptor<T>
where
    T: DataType,
{
    /// Creates a unary tensor point-wise math descriptor.
    ///
    /// # Arguments
    ///
    /// * `op` - the unary tensor point-wise math operation to be performed.
    ///
    /// * `nan_opt` - a NaN propagation policy.
    ///
    /// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetOpTensorDescriptor)
    /// may offer additional information about the APi behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{UnaryOp, UnaryOpTensorDescriptor, NanPropagation};
    ///
    /// let op = UnaryOp::Sqrt;
    /// let nan_policy = NanPropagation::PropagateNaN;
    ///
    /// // We are stating that the computation must be done in f32.
    /// let desc = UnaryOpTensorDescriptor::<f32>::new(op, nan_policy)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(op: UnaryOp, nan_opt: NanPropagation) -> Result<Self, CudnnError> {
        unsafe {
            let raw = init_raw_op_descriptor::<T>(op.into(), nan_opt.into())?;

            Ok(Self {
                raw,
                comp_type: PhantomData,
                op,
            })
        }
    }
}

impl<T: DataType> Drop for UnaryOpTensorDescriptor<T> {
    fn drop(&mut self) {
        unsafe {
            cudnn_sys::cudnnDestroyOpTensorDescriptor(self.raw);
        }
    }
}

/// The description of a binary Tensor Core operation.
///
///
/// As specified in the cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#scaling-parameters),
/// admissible types for scaling parameters are `f32` and `f64` for `f32` and `f64` tensors
/// respectively.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct BinaryOpTensorDescriptor<T: DataType> {
    pub(crate) raw: cudnn_sys::cudnnOpTensorDescriptor_t,
    comp_type: PhantomData<T>,
    op: BinaryOp,
}

impl<T> BinaryOpTensorDescriptor<T>
where
    T: DataType,
{
    /// Creates a binary tensor point-wise math descriptor.
    ///
    /// # Arguments
    ///
    /// * `op` - the unary tensor point-wise math operation to be performed.
    /// * `nan_opt` - a NaN propagation policy.
    ///
    /// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetOpTensorDescriptor)
    /// may offer additional information about the APi behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{BinaryOp, BinaryOpTensorDescriptor, NanPropagation};
    ///
    /// let op = BinaryOp::Add;
    /// let nan_policy = NanPropagation::PropagateNaN;
    ///
    /// // We are stating that the computation must be done in f32.
    /// let desc = BinaryOpTensorDescriptor::<f32>::new(op, nan_policy)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(op: BinaryOp, nan_opt: NanPropagation) -> Result<Self, CudnnError> {
        unsafe {
            let raw = init_raw_op_descriptor::<T>(op.into(), nan_opt.into())?;

            Ok(Self {
                raw,
                comp_type: PhantomData,
                op,
            })
        }
    }
}

impl<T: DataType> Drop for BinaryOpTensorDescriptor<T> {
    fn drop(&mut self) {
        unsafe {
            cudnn_sys::cudnnDestroyOpTensorDescriptor(self.raw);
        }
    }
}

/// Supported data type configurations for point-wise tensor core operations.
///
/// | CompType | Atype    | Btype    | Ctype    |
/// |----------|----------|----------|----------|
/// | f32      | f32      | f32      | f32      |
/// | f32      | i8       | i8       | f32      |
/// | f64      | f64      | f64      | f64      |
/// | f32      | i8       | i8       | i8       |
/// | f32      | f32      | f32      | i8       |
///
/// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnOpTensor).
pub trait SupportedOp<AType, BType, CType>
where
    Self: DataType,
    AType: DataType,
    BType: DataType,
    CType: DataType,
{
}

macro_rules! impl_supported_types {
    ($comp_type:ty, $a_type:ty, $b_type:ty, $c_type:ty) => {
        impl SupportedOp<$a_type, $b_type, $c_type> for $comp_type {}
    };
}

impl_supported_types!(f32, f32, f32, f32);
impl_supported_types!(f32, i8, i8, f32);
impl_supported_types!(f64, f64, f64, f64);
impl_supported_types!(f32, i8, i8, i8);
impl_supported_types!(f32, f32, f32, i8);
