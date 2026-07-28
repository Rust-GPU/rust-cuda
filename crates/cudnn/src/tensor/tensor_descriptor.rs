use std::{marker::PhantomData, mem::MaybeUninit};

use crate::{CudnnError, DataType, IntoResult, ScalarC, TensorFormat, VecType};

/// A generic description of an n-dimensional dataset.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct TensorDescriptor<T>
where
    T: DataType,
{
    pub(crate) raw: cudnn_sys::cudnnTensorDescriptor_t,
    data_type: PhantomData<T>,
}

impl<T> TensorDescriptor<T>
where
    T: DataType,
{
    /// Creates a tensor descriptor with the given shape and strides.
    ///
    /// # Arguments
    ///
    /// * `shape` - slice containing the size of the tensor for every dimension.
    ///
    /// * `strides` - strides for the tensor descriptor.
    ///
    /// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetTensorNdDescriptor)
    /// may offer additional information about the APi behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::TensorDescriptor;
    ///
    /// let shape = &[5, 5, 10, 25];
    /// let strides = &[1250, 250, 25, 1];
    ///
    /// let desc = TensorDescriptor::<f32>::new_strides(shape, strides)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_strides(shape: &[i32], strides: &[i32]) -> Result<Self, CudnnError> {
        let mut raw = MaybeUninit::uninit();

        let ndims = shape.len();

        assert_eq!(
            ndims,
            strides.len(),
            "shape and strides length do not match."
        );

        unsafe {
            cudnn_sys::cudnnCreateTensorDescriptor(raw.as_mut_ptr()).into_result()?;
            let raw = raw.assume_init();

            cudnn_sys::cudnnSetTensorNdDescriptor(
                raw,
                T::into_raw(),
                ndims as i32,
                shape.as_ptr(),
                strides.as_ptr(),
            )
            .into_result()?;

            Ok(Self {
                raw,
                data_type: PhantomData,
            })
        }
    }

    /// Creates a tensor descriptor with the given shape and format.
    ///
    /// # Arguments
    ///
    /// * `shape` - slice containing the size of the tensor for every dimension.
    ///
    /// * `format` - format for the tensor descriptor.
    ///
    /// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetTensorNdDescriptorEx)
    /// may offer additional information about the APi behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{TensorDescriptor, ScalarC};
    ///
    /// let shape = &[5, 5, 10, 25];
    /// let format = ScalarC::Nchw;
    ///
    /// let desc = TensorDescriptor::<f32>::new_format(shape, format)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_format(shape: &[i32], format: ScalarC) -> Result<Self, CudnnError> {
        let mut raw = MaybeUninit::uninit();

        let ndims = shape.len();

        unsafe {
            cudnn_sys::cudnnCreateTensorDescriptor(raw.as_mut_ptr()).into_result()?;
            let raw = raw.assume_init();

            cudnn_sys::cudnnSetTensorNdDescriptorEx(
                raw,
                format.into(),
                T::into_raw(),
                ndims as i32,
                shape.as_ptr(),
            )
            .into_result()?;

            Ok(TensorDescriptor {
                raw,
                data_type: PhantomData,
            })
        }
    }

    /// Creates a tensor descriptor with the given shape and vectorized format.
    ///
    /// # Arguments
    ///
    /// `shape` - slice containing the size of the tensor for every dimension.
    ///
    /// **Do note** that the actual vectorized data type must be specified with the associated
    /// generic.
    ///
    /// Check [cuDNN docs](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#nc32hw32-layout-x32) for more details.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{TensorDescriptor, Vec4};
    ///
    /// let shape = &[4, 32, 32, 32];
    ///
    /// let desc = TensorDescriptor::<i8>::new_vectorized::<Vec4>(shape)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_vectorized<V: VecType<T>>(shape: &[i32]) -> Result<Self, CudnnError> {
        let mut raw = MaybeUninit::uninit();

        let ndims = shape.len();
        let format = TensorFormat::NchwVectC;

        unsafe {
            cudnn_sys::cudnnCreateTensorDescriptor(raw.as_mut_ptr()).into_result()?;
            let raw = raw.assume_init();

            cudnn_sys::cudnnSetTensorNdDescriptorEx(
                raw,
                format.into(),
                V::into_raw(),
                ndims as i32,
                shape.as_ptr(),
            )
            .into_result()?;

            Ok(TensorDescriptor {
                raw,
                data_type: PhantomData,
            })
        }
    }
}

impl<T> TensorDescriptor<T>
where
    T: DataType,
{
    /// Wraps a raw cuDNN tensor descriptor created and configured elsewhere in the crate
    /// (e.g. by `cudnnDeriveBNTensorDescriptor`).
    pub(crate) fn from_raw(raw: cudnn_sys::cudnnTensorDescriptor_t) -> Self {
        Self {
            raw,
            data_type: PhantomData,
        }
    }
}

impl<T> Drop for TensorDescriptor<T>
where
    T: DataType,
{
    fn drop(&mut self) {
        unsafe {
            cudnn_sys::cudnnDestroyTensorDescriptor(self.raw);
        }
    }
}
