use std::{marker::PhantomData, mem::MaybeUninit};

use crate::{CudnnError, DataType, IntoResult, RnnDataLayout, private};

/// Specifies the allowed types for the recurrent neural network inputs and outputs.
///
/// As stated in the [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetRNNDataDescriptor)
/// the supported types are `f32` and `f64`.
pub trait RnnDataType: DataType + private::Sealed {}

impl RnnDataType for f32 {}
impl RnnDataType for f64 {}

/// Descriptor of a recurrent neural network data container.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct RnnDataDescriptor<T>
where
    T: RnnDataType,
{
    pub(crate) raw: cudnn_sys::cudnnRNNDataDescriptor_t,
    data_type: PhantomData<T>,
}

impl<T> RnnDataDescriptor<T>
where
    T: RnnDataType,
{
    /// Initializes a recurrent neural network data descriptor object.
    ///
    /// This data structure is intended to support the unpacked (padded) layout for
    /// input and output of extended RNN inference and training functions.
    ///
    /// **Do note** that packed (un-padded) layout is also supported for backward
    /// compatibility.
    ///
    /// # Arguments
    ///
    ///   * `layout` - memory layout of the RNN data tensor.
    ///   * `max_seq_length` - maximum sequence length within this RNN data tensor. In
    ///     the unpacked (padded) layout, this should include the padding vectors in
    ///     each sequence. In the packed (un-padded) layout, this should be equal to the
    ///     greatest element in `seq_lengths`.
    ///   * `batch_size` - number of sequences within the mini-batch.
    ///   * `seq_lengths` - an integer slice with `batch_size` number of elements.
    ///     Describes the length (number of time-steps) of each sequence. Each element
    ///     in the slice must be greater than or equal to 0 but less than or equal to
    ///     `max_seq_length`. In the packed layout, the elements should be sorted in
    ///     descending order.
    ///   * `padding_fill` - user-defined constant for filling the padding position in
    ///     RNN output. This is only effective when the descriptor is describing the RNN
    ///     output, and the unpacked layout is specified. The symbol should be in the
    ///     host memory, and if a `None` is passed in, then the padding position in the
    ///     output will be undefined.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetRNNDataDescriptor)
    /// may offer additional information about the APi behavior.
    ///
    /// # Panics
    ///
    /// If the length of `seq_lengths` doesn't match `batch_size`.
    ///
    /// # Errors
    ///
    /// Returns errors if an element of `seq_lengths` is less than zero or greater than
    /// `max_seq_length` or if the allocation of internal array storage has failed.
    ///
    /// # Examples
    ///
    /// A recurrent neural network data descriptor can be used to represent both input
    /// and output sequences for such model.
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{CudnnContext, RnnDataDescriptor, RnnDataLayout};
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let layout = RnnDataLayout::SeqMajorUnpacked;
    /// let max_seq_length = 3;
    /// let batch_size = 5;
    /// let vector_size = 10;
    /// let seq_lengths = [1, 2, 3, 2, 1];
    /// let padding_fill = None; // Should only be set for output sequences.
    ///
    /// let rnn_data_desc = RnnDataDescriptor::<f32>::new(
    ///     layout,
    ///     max_seq_length,
    ///     batch_size,
    ///     vector_size,
    ///     &seq_lengths,
    ///     padding_fill,
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        layout: RnnDataLayout,
        max_seq_length: i32,
        batch_size: i32,
        vector_size: i32,
        seq_lengths: &[i32],
        padding_fill: impl Into<Option<T>>,
    ) -> Result<Self, CudnnError> {
        assert_eq!(
            seq_lengths.len(),
            batch_size as usize,
            "Found {} sequence lengths, but batch size is {}",
            seq_lengths.len(),
            batch_size
        );

        let mut raw = MaybeUninit::uninit();

        unsafe {
            cudnn_sys::cudnnCreateRNNDataDescriptor(raw.as_mut_ptr()).into_result()?;

            let raw = raw.assume_init();

            let fill: *mut T = padding_fill
                .into()
                .map_or(std::ptr::null_mut(), |mut el| &mut el as *mut T);

            cudnn_sys::cudnnSetRNNDataDescriptor(
                raw,
                T::into_raw(),
                layout.into(),
                max_seq_length,
                batch_size,
                vector_size,
                seq_lengths.as_ptr(),
                fill as *mut std::ffi::c_void,
            )
            .into_result()?;

            Ok(Self {
                raw,
                data_type: PhantomData,
            })
        }
    }
}

impl<T> Drop for RnnDataDescriptor<T>
where
    T: RnnDataType,
{
    fn drop(&mut self) {
        unsafe {
            cudnn_sys::cudnnDestroyRNNDataDescriptor(self.raw);
        }
    }
}
