use std::{marker::PhantomData, mem::MaybeUninit};

use crate::{CudnnError, DataType, IntoResult, SeqDataAxis, private};

/// Specifies the allowed types for the sequential data buffer.
///
/// As stated in the [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetSeqataDescriptor)
/// the supported types are `f32` and `f64`.
pub trait SeqDataType: DataType + private::Sealed {}

impl SeqDataType for f32 {}
impl SeqDataType for f64 {}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SeqDataDescriptor<T>
where
    T: SeqDataType,
{
    pub(crate) raw: cudnn_sys::cudnnSeqDataDescriptor_t,
    data_type: PhantomData<T>,
}

/// A sequence data descriptor.
impl<T> SeqDataDescriptor<T>
where
    T: SeqDataType,
{
    /// Creates a new sequential data descriptor. In the most simplified view, this
    /// struct defines the dimensions and the data layout of a 4d tensor. All four
    /// dimensions of the sequence data have unique identifiers that can be used to
    /// index the `dims` array, they are:
    ///
    ///   * [`SeqDataAxis::TimeDim`]
    ///   * [`SeqDataAxis::BatchDim`]
    ///   * [`SeqDataAxis::BeamDim`]
    ///   * [`SeqDataAxis::VectDim`]
    ///
    /// For example, to express that vectors in the sequence data buffer are five
    /// elements long, we need to assign `dims[SeqDataAxis::VectDim] = 5` in the `dims`
    /// array.
    ///
    /// The underlying container is treated as a collection of fixed length vectors that
    /// form sequences, similarly to words constructing sentences.
    ///
    /// The **time** dimension spans the sequence length. Different sequences are
    /// bundled together in a batch. A **batch** may be a group of individual sequences
    /// or beams. A **beam** is a cluster of alternative sequences or candidates. When
    /// thinking about the beam, consider a translation task from one language to
    /// another. You may want to keep around and experiment with several translated
    /// versions of the original sentence before selecting the best one. The number of
    /// candidates kept around is the `dims[SeqDataAxis::BeamDim]` value.
    ///
    /// Every sequence can have a different length, even within the same beam, so
    /// vectors toward the end of the sequence can be just padding.
    ///
    /// It is assumed that a non-empty sequence always starts from the time index zero.
    /// The `seq_lengths` slice must specify all sequence lengths in the container so
    /// the total size of this array should be `dims[SeqDataAxis::BatchDim]` *
    /// `dims[SeqDataAxis::BeamDim]`.
    ///
    /// Each element of the `seq_lengths` slice should have a non-negative value, less
    /// than or equal to `dims[SeqDataAxis::TimeDim`]; the maximum sequence length.
    ///
    /// The `axes` array specifies the actual data layout in the GPU memory.
    ///
    /// # Arguments
    ///
    ///   * `dims` - shape of the sequential data 4d tensor. Use the [`SeqDataAxis`]
    ///     enum to index its elements.
    ///   * `axes` - actual layout of the sequential data in GPU memory. `axes[0]` is
    ///     the outermost dimension and `axes[3]` is the innermost.
    ///   * `seq_lengths` - array that defines all sequence lengths of the underlying
    ///     container.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetSeqDataDescriptor)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if the innermost dimension as specified in the `axes` array is
    /// not `SeqDataAxis::VectDim` or unsupported configurations of arguments are
    /// detected.
    ///
    /// # Examples
    ///
    /// The following example is equivalent to this layout:
    ///
    /// ![](https://docs.nvidia.com/deeplearning/cudnn/api/graphics/cudnnSetSeqDataDescriptor.PNG)
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{CudnnContext, SeqDataAxis, SeqDataDescriptor};
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let mut dims = [0; 4];
    ///
    /// dims[SeqDataAxis::TimeDim] = 4;
    /// dims[SeqDataAxis::BatchDim] = 3;
    /// dims[SeqDataAxis::BeamDim] = 2;
    /// dims[SeqDataAxis::VectDim] = 5;
    ///
    /// let axes = [
    ///     SeqDataAxis::TimeDim,
    ///     SeqDataAxis::BatchDim,
    ///     SeqDataAxis::BeamDim,
    ///     SeqDataAxis::VectDim
    /// ];
    ///
    /// let seq_lengths = &[3, 4, 4, 4, 2, 3];
    ///
    /// let seq_data_desc = SeqDataDescriptor::<f32>::new(dims, axes, seq_lengths)?;
    ///
    /// # Ok(())
    /// # }
    pub fn new(
        dims: [i32; 4],
        axes: [SeqDataAxis; 4],
        seq_lengths: &[i32],
    ) -> Result<Self, CudnnError> {
        let mut raw = MaybeUninit::uninit();

        unsafe {
            cudnn_sys::cudnnCreateSeqDataDescriptor(raw.as_mut_ptr()).into_result()?;

            let raw = raw.assume_init();

            let raw_axes = axes.map(SeqDataAxis::into);

            cudnn_sys::cudnnSetSeqDataDescriptor(
                raw,
                T::into_raw(),
                4_i32,
                dims.as_ptr(),
                raw_axes.as_ptr(),
                seq_lengths.len(),
                seq_lengths.as_ptr(),
                std::ptr::null_mut(),
            )
            .into_result()?;

            Ok(Self {
                raw,
                data_type: PhantomData,
            })
        }
    }
}

impl<T> Drop for SeqDataDescriptor<T>
where
    T: SeqDataType,
{
    fn drop(&mut self) {
        unsafe {
            cudnn_sys::cudnnDestroySeqDataDescriptor(self.raw);
        }
    }
}
