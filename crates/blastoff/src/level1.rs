//! Scalar and Vector-based operations

use crate::{
    BlasDatatype, Float,
    context::CublasContext,
    error::{Error, ToResult},
    raw::{ComplexLevel1, FloatLevel1, Level1},
};
use cust::memory::{GpuBox, GpuBuffer};
use cust::stream::Stream;

type Result<T = (), E = Error> = std::result::Result<T, E>;

fn check_stride<T: BlasDatatype>(x: &impl GpuBuffer<T>, n: usize, stride: Option<usize>) {
    let raw_len = x.len();
    let needed_len = n * stride.unwrap_or(1);
    assert!(
        raw_len >= needed_len,
        "Buffer is not long enough! required_len is {} ({} stride * {} n) but the buffer length is {}",
        needed_len,
        stride.unwrap_or(1),
        n,
        raw_len
    );
}

/// Scalar and Vector-based operations such as `min`, `max`, `axpy`, `copy`, `dot`,
/// `nrm2`, `rot`, `rotg`, `rotm`, `rotmg`, `scal`, and `swap`.
impl CublasContext {
    /// Same as [`CublasContext::amin`] but with an explicit stride.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is not long enough for the stride and length requested.
    pub fn amin_strided<T: Level1>(
        &mut self,
        stream: &Stream,
        x: &impl GpuBuffer<T>,
        n: usize,
        stride: Option<usize>,
        result: &mut impl GpuBox<i32>,
    ) -> Result {
        check_stride(x, n, stride);

        self.with_stream(stream, |ctx| unsafe {
            Ok(T::amin(
                ctx.raw,
                n as i32,
                x.as_device_ptr().as_ptr(),
                stride.unwrap_or(1) as i32,
                result.as_device_ptr().as_mut_ptr(),
            )
            .to_result()?)
        })
    }

    /// Finds the index of the smallest element inside of the GPU buffer by absolute value, writing the resulting
    /// index into `result`. The index is 1-based, not 0-based.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _a = cust::quick_init()?;
    /// # use blastoff::CublasContext;
    /// # use cust::prelude::*;
    /// # use cust::memory::DeviceBox;
    /// # use cust::util::SliceExt;
    /// # let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    /// let mut ctx = CublasContext::new()?;
    /// let data = [0.0f32, 1.0, 0.5, 5.0].as_dbuf()?;
    /// let mut result = DeviceBox::new(&0)?;
    ///
    /// ctx.amin(&stream, &data, &mut result)?;
    ///
    /// stream.synchronize()?;
    ///
    /// assert_eq!(result.as_host_value()?, 1);
    /// drop((result, data, ctx, stream));
    /// # Ok(())
    /// # }
    /// ```
    pub fn amin<T: Level1>(
        &mut self,
        stream: &Stream,
        x: &impl GpuBuffer<T>,
        result: &mut impl GpuBox<i32>,
    ) -> Result {
        self.amin_strided(stream, x, x.len(), None, result)
    }

    /// Same as [`CublasContext::amax`] but with an explicit stride.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is not long enough for the stride and length requested.
    pub fn amax_strided<T: Level1>(
        &mut self,
        stream: &Stream,
        x: &impl GpuBuffer<T>,
        n: usize,
        stride: Option<usize>,
        result: &mut impl GpuBox<i32>,
    ) -> Result {
        check_stride(x, n, stride);

        self.with_stream(stream, |ctx| unsafe {
            Ok(T::amax(
                ctx.raw,
                n as i32,
                x.as_device_ptr().as_ptr(),
                stride.unwrap_or(1) as i32,
                result.as_device_ptr().as_mut_ptr(),
            )
            .to_result()?)
        })
    }

    /// Finds the index of the smallest element inside of the GPU buffer by absolute value, writing the resulting
    /// index into `result`. The index is 1-based, not 0-based.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _a = cust::quick_init()?;
    /// # use blastoff::CublasContext;
    /// # use cust::prelude::*;
    /// # use cust::memory::DeviceBox;
    /// # use cust::util::SliceExt;
    /// # let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    /// let mut ctx = CublasContext::new()?;
    /// let data = [0.0f32, 1.0, 0.5, 5.0].as_dbuf()?;
    /// let mut result = DeviceBox::new(&0)?;
    ///
    /// ctx.amax(&stream, &data, &mut result)?;
    ///
    /// stream.synchronize()?;
    ///
    /// assert_eq!(result.as_host_value()?, 4);
    /// # Ok(())
    /// # }
    /// ```
    pub fn amax<T: Level1>(
        &mut self,
        stream: &Stream,
        x: &impl GpuBuffer<T>,
        result: &mut impl GpuBox<i32>,
    ) -> Result {
        self.amax_strided(stream, x, x.len(), None, result)
    }

    /// Same as [`CublasContext::axpy`] but with an explicit stride.
    ///
    /// # Panics
    ///
    /// Panics if the buffers are not long enough for the stride and length requested.
    pub fn axpy_strided<T: Level1>(
        &mut self,
        stream: &Stream,
        alpha: &impl GpuBox<T>,
        n: usize,
        x: &impl GpuBuffer<T>,
        x_stride: Option<usize>,
        y: &mut impl GpuBuffer<T>,
        y_stride: Option<usize>,
    ) -> Result {
        check_stride(x, n, x_stride);
        check_stride(y, n, y_stride);

        self.with_stream(stream, |ctx| unsafe {
            Ok(T::axpy(
                ctx.raw,
                n as i32,
                alpha.as_device_ptr().as_ptr(),
                x.as_device_ptr().as_ptr(),
                x_stride.unwrap_or(1) as i32,
                y.as_device_ptr().as_mut_ptr(),
                y_stride.unwrap_or(1) as i32,
            )
            .to_result()?)
        })
    }

    /// Multiplies `n` elements in `x` by `alpha`, then adds the result to `y`, overwriting
    /// `y` with the result.
    ///
    /// # Panics
    ///
    /// Panics if `x` or `y` are not long enough for the requested length `n`.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _a = cust::quick_init()?;
    /// # use blastoff::CublasContext;
    /// # use cust::prelude::*;
    /// # use cust::memory::DeviceBox;
    /// # use cust::util::SliceExt;
    /// # let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    /// let mut ctx = CublasContext::new()?;
    /// let alpha = DeviceBox::new(&2.0)?;
    /// let x = [1.0, 2.0, 3.0, 4.0].as_dbuf()?;
    /// let mut y = [1.0; 4].as_dbuf()?;
    ///
    /// ctx.axpy(&stream, &alpha, x.len(), &x, &mut y)?;
    ///
    /// stream.synchronize()?;
    ///
    /// let result = y.as_host_vec()?;
    /// assert_eq!(&result, &[3.0, 5.0, 7.0, 9.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn axpy<T: Level1>(
        &mut self,
        stream: &Stream,
        alpha: &impl GpuBox<T>,
        n: usize,
        x: &impl GpuBuffer<T>,
        y: &mut impl GpuBuffer<T>,
    ) -> Result {
        self.axpy_strided(stream, alpha, n, x, None, y, None)
    }

    /// Same as [`CublasContext::copy`] but with an explicit stride.
    ///
    /// # Panics
    ///
    /// Panics if the buffers are not long enough for the stride and length requested.
    pub fn copy_strided<T: Level1>(
        &mut self,
        stream: &Stream,
        n: usize,
        x: &impl GpuBuffer<T>,
        x_stride: Option<usize>,
        y: &mut impl GpuBuffer<T>,
        y_stride: Option<usize>,
    ) -> Result {
        check_stride(x, n, x_stride);
        check_stride(y, n, y_stride);

        self.with_stream(stream, |ctx| unsafe {
            Ok(T::copy(
                ctx.raw,
                n as i32,
                x.as_device_ptr().as_ptr(),
                x_stride.unwrap_or(1) as i32,
                y.as_device_ptr().as_mut_ptr(),
                y_stride.unwrap_or(1) as i32,
            )
            .to_result()?)
        })
    }

    /// Copies `n` elements from `x` into `y`, overriding any previous data inside `y`.
    ///
    /// # Panics
    ///
    /// Panics if `x` or `y` are not large enough for the requested amount of elements.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _a = cust::quick_init()?;
    /// # use blastoff::CublasContext;
    /// # use cust::prelude::*;
    /// # use cust::memory::DeviceBox;
    /// # use cust::util::SliceExt;
    /// # let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    /// let mut ctx = CublasContext::new()?;
    /// let x = [1.0f32, 2.0, 3.0, 4.0].as_dbuf()?;
    /// let mut y = [0.0; 4].as_dbuf()?;
    ///
    /// ctx.copy(&stream, x.len(), &x, &mut y)?;
    ///
    /// stream.synchronize()?;
    ///
    /// assert_eq!(x.as_host_vec()?, y.as_host_vec()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn copy<T: Level1>(
        &mut self,
        stream: &Stream,
        n: usize,
        x: &impl GpuBuffer<T>,
        y: &mut impl GpuBuffer<T>,
    ) -> Result {
        self.copy_strided(stream, n, x, None, y, None)
    }

    /// Same as [`CublasContext::dot`] but with an explicit stride.
    ///
    /// # Panics
    ///
    /// Panics if the buffers are not long enough for the stride and length requested.
    pub fn dot_strided<T: FloatLevel1>(
        &mut self,
        stream: &Stream,
        n: usize,
        x: &impl GpuBuffer<T>,
        x_stride: Option<usize>,
        y: &impl GpuBuffer<T>,
        y_stride: Option<usize>,
        result: &mut impl GpuBox<T>,
    ) -> Result {
        check_stride(x, n, x_stride);
        check_stride(y, n, y_stride);

        self.with_stream(stream, |ctx| unsafe {
            Ok(T::dot(
                ctx.raw,
                n as i32,
                x.as_device_ptr().as_ptr(),
                x_stride.unwrap_or(1) as i32,
                y.as_device_ptr().as_ptr(),
                y_stride.unwrap_or(1) as i32,
                result.as_device_ptr().as_mut_ptr(),
            )
            .to_result()?)
        })
    }

    /// Computes the dot product of two vectors:
    ///
    /// $$
    /// \sum^n_{i=1} \boldsymbol{x}_i * \boldsymbol{y}_i
    /// $$
    ///
    /// # Panics
    ///
    /// Panics if the buffers are not long enough for the length requested.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _a = cust::quick_init()?;
    /// # use blastoff::CublasContext;
    /// # use cust::prelude::*;
    /// # use cust::memory::DeviceBox;
    /// # use cust::util::SliceExt;
    /// # let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    /// let mut ctx = CublasContext::new()?;
    /// let x = [1.0f32, 2.0, 3.0, 4.0].as_dbuf()?;
    /// let y = [1.0f32, 2.0, 3.0, 4.0].as_dbuf()?;
    /// let mut result = DeviceBox::new(&0.0)?;
    ///
    /// ctx.dot(&stream, x.len(), &x, &y, &mut result)?;
    ///
    /// stream.synchronize()?;
    ///
    /// assert_eq!(result.as_host_value()?, 30.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn dot<T: FloatLevel1>(
        &mut self,
        stream: &Stream,
        n: usize,
        x: &impl GpuBuffer<T>,
        y: &impl GpuBuffer<T>,
        result: &mut impl GpuBox<T>,
    ) -> Result {
        self.dot_strided(stream, n, x, None, y, None, result)
    }

    /// Same as [`CublasContext::dotu`] but with an explicit stride.
    ///
    /// # Panics
    ///
    /// Panics if the buffers are not long enough for the stride and length requested.
    pub fn dotu_strided<T: ComplexLevel1>(
        &mut self,
        stream: &Stream,
        n: usize,
        x: &impl GpuBuffer<T>,
        x_stride: Option<usize>,
        y: &impl GpuBuffer<T>,
        y_stride: Option<usize>,
        result: &mut impl GpuBox<T>,
    ) -> Result {
        check_stride(x, n, x_stride);
        check_stride(y, n, y_stride);

        self.with_stream(stream, |ctx| unsafe {
            Ok(T::dotu(
                ctx.raw,
                n as i32,
                x.as_device_ptr().as_ptr(),
                x_stride.unwrap_or(1) as i32,
                y.as_device_ptr().as_ptr(),
                y_stride.unwrap_or(1) as i32,
                result.as_device_ptr().as_mut_ptr(),
            )
            .to_result()?)
        })
    }

    /// Computes the unconjugated dot product of two vectors of complex numbers.
    ///
    /// # Panics
    ///
    /// Panics if the buffers are not long enough for the length requested.
    pub fn dotu<T: ComplexLevel1>(
        &mut self,
        stream: &Stream,
        n: usize,
        x: &impl GpuBuffer<T>,
        y: &impl GpuBuffer<T>,
        result: &mut impl GpuBox<T>,
    ) -> Result {
        self.dotu_strided(stream, n, x, None, y, None, result)
    }

    /// Same as [`CublasContext::dotc`] but with an explicit stride.
    ///
    /// # Panics
    ///
    /// Panics if the buffers are not long enough for the stride and length requested.
    pub fn dotc_strided<T: ComplexLevel1>(
        &mut self,
        stream: &Stream,
        n: usize,
        x: &impl GpuBuffer<T>,
        x_stride: Option<usize>,
        y: &impl GpuBuffer<T>,
        y_stride: Option<usize>,
        result: &mut impl GpuBox<T>,
    ) -> Result {
        check_stride(x, n, x_stride);
        check_stride(y, n, y_stride);

        self.with_stream(stream, |ctx| unsafe {
            Ok(T::dotc(
                ctx.raw,
                n as i32,
                x.as_device_ptr().as_ptr(),
                x_stride.unwrap_or(1) as i32,
                y.as_device_ptr().as_ptr(),
                y_stride.unwrap_or(1) as i32,
                result.as_device_ptr().as_mut_ptr(),
            )
            .to_result()?)
        })
    }

    /// Computes the conjugated dot product of two vectors of complex numbers.
    ///
    /// # Panics
    ///
    /// Panics if the buffers are not long enough for the length requested.
    pub fn dotc<T: ComplexLevel1>(
        &mut self,
        stream: &Stream,
        n: usize,
        x: &impl GpuBuffer<T>,
        y: &impl GpuBuffer<T>,
        result: &mut impl GpuBox<T>,
    ) -> Result {
        self.dotc_strided(stream, n, x, None, y, None, result)
    }

    /// Same as [`CublasContext::nrm2`] but with an explicit stride.
    ///
    /// # Panics
    ///
    /// Panics if the buffers are not long enough for the stride and length requested.
    pub fn nrm2_strided<T: Level1>(
        &mut self,
        stream: &Stream,
        n: usize,
        x: &impl GpuBuffer<T>,
        x_stride: Option<usize>,
        result: &mut impl GpuBox<T::FloatTy>,
    ) -> Result {
        check_stride(x, n, x_stride);

        self.with_stream(stream, |ctx| unsafe {
            Ok(T::nrm2(
                ctx.raw,
                n as i32,
                x.as_device_ptr().as_ptr(),
                x_stride.unwrap_or(1) as i32,
                result.as_device_ptr().as_mut_ptr(),
            )
            .to_result()?)
        })
    }

    /// Computes the euclidian norm of a vector, in other words, the square root of
    /// the sum of the squares of each element in `x`:
    ///
    /// $$
    /// \sqrt{\sum_{i=1}^n (\boldsymbol{x}_i^2)}
    /// $$
    ///
    /// # Panics
    ///
    /// Panics if `x` is not large enough for the requested length `n`.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _a = cust::quick_init()?;
    /// # use blastoff::CublasContext;
    /// # use cust::prelude::*;
    /// # use cust::memory::DeviceBox;
    /// # use cust::util::SliceExt;
    /// # let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    /// let mut ctx = CublasContext::new()?;
    /// let x = [2.0f32; 4].as_dbuf()?;
    /// let mut result = DeviceBox::new(&0.0f32)?;
    ///
    /// ctx.nrm2(&stream, x.len(), &x, &mut result)?;
    ///
    /// stream.synchronize()?;
    ///
    /// let result = result.as_host_value()?;
    /// // float weirdness
    /// assert!(result >= 3.9 && result <= 4.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn nrm2<T: Level1>(
        &mut self,
        stream: &Stream,
        n: usize,
        x: &impl GpuBuffer<T>,
        result: &mut impl GpuBox<T::FloatTy>,
    ) -> Result {
        self.nrm2_strided(stream, n, x, None, result)
    }

    /// Same as [`CublasContext::rot`] but with an explicit stride.
    ///
    /// # Panics
    ///
    /// Panics if the buffers are not long enough for the stride and length requested.
    pub fn rot_strided<T: Level1>(
        &mut self,
        stream: &Stream,
        n: usize,
        x: &mut impl GpuBuffer<T>,
        x_stride: Option<usize>,
        y: &mut impl GpuBuffer<T>,
        y_stride: Option<usize>,
        c: &impl GpuBox<T::FloatTy>,
        s: &impl GpuBox<T::FloatTy>,
    ) -> Result {
        check_stride(x, n, x_stride);
        check_stride(y, n, y_stride);

        self.with_stream(stream, |ctx| unsafe {
            Ok(T::rot(
                ctx.raw,
                n as i32,
                x.as_device_ptr().as_mut_ptr(),
                x_stride.unwrap_or(1) as i32,
                y.as_device_ptr().as_mut_ptr(),
                y_stride.unwrap_or(1) as i32,
                c.as_device_ptr().as_ptr(),
                s.as_device_ptr().as_ptr(),
            )
            .to_result()?)
        })
    }

    /// Rotates points in the xy-plane using a Givens rotation matrix.
    ///
    /// Rotation matrix:
    ///
    /// <p>
    /// $$
    /// \begin{pmatrix}
    ///    c & s \\
    ///    -s & c
    /// \end{pmatrix}
    /// $$
    /// </p>
    ///
    /// Therefore:
    ///
    /// $$
    /// \boldsymbol{x}_i = \boldsymbol{x}_ic + \boldsymbol{y}_is
    /// $$
    ///
    /// And:
    ///
    /// $$
    /// \boldsymbol{y}_i = -\boldsymbol{x}_is + \boldsymbol{y}_ic
    /// $$
    ///
    /// Where $c$ and $s$ are usually
    ///
    /// <p>
    /// $$
    /// c = cos(\theta) \\
    /// s = sin(\theta)
    /// $$
    /// </p>
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _a = cust::quick_init()?;
    /// # use blastoff::CublasContext;
    /// # use cust::prelude::*;
    /// # use cust::memory::DeviceBox;
    /// # use cust::util::SliceExt;
    /// # let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    /// let mut ctx = CublasContext::new()?;
    /// let mut x = [1.0f32].as_dbuf()?;
    /// let mut y = [0.0].as_dbuf()?;
    /// let c = DeviceBox::new(&1.0)?;
    /// let s = DeviceBox::new(&0.0)?;
    ///
    /// ctx.rot(&stream, x.len(), &mut x, &mut y, &c, &s)?;
    ///
    /// stream.synchronize()?;
    ///
    /// // identity matrix
    /// assert_eq!(&x.as_host_vec()?, &[1.0]);
    /// assert_eq!(&y.as_host_vec()?, &[0.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn rot<T: Level1>(
        &mut self,
        stream: &Stream,
        n: usize,
        x: &mut impl GpuBuffer<T>,
        y: &mut impl GpuBuffer<T>,
        c: &impl GpuBox<T::FloatTy>,
        s: &impl GpuBox<T::FloatTy>,
    ) -> Result {
        self.rot_strided(stream, n, x, None, y, None, c, s)
    }

    /// Constructs the givens rotation matrix that zeros out the second entry of a 2x1 vector.
    pub fn rotg<T: Level1>(
        &mut self,
        stream: &Stream,
        a: &mut impl GpuBox<T>,
        b: &mut impl GpuBox<T>,
        c: &mut impl GpuBox<T::FloatTy>,
        s: &mut impl GpuBox<T>,
    ) -> Result {
        self.with_stream(stream, |ctx| unsafe {
            Ok(T::rotg(
                ctx.raw,
                a.as_device_ptr().as_mut_ptr(),
                b.as_device_ptr().as_mut_ptr(),
                c.as_device_ptr().as_mut_ptr(),
                s.as_device_ptr().as_mut_ptr(),
            )
            .to_result()?)
        })
    }

    /// Same as [`CublasContext::rotm`] but with an explicit stride.
    pub fn rotm_strided<T: Level1 + Float>(
        &mut self,
        stream: &Stream,
        n: usize,
        x: &mut impl GpuBuffer<T>,
        x_stride: Option<usize>,
        y: &mut impl GpuBuffer<T>,
        y_stride: Option<usize>,
        param: &impl GpuBox<T::FloatTy>,
    ) -> Result {
        check_stride(x, n, x_stride);
        check_stride(y, n, y_stride);

        self.with_stream(stream, |ctx| unsafe {
            Ok(T::rotm(
                ctx.raw,
                n as i32,
                x.as_device_ptr().as_mut_ptr(),
                x_stride.unwrap_or(1) as i32,
                y.as_device_ptr().as_mut_ptr(),
                y_stride.unwrap_or(1) as i32,
                param.as_device_ptr().as_ptr(),
            )
            .to_result()?)
        })
    }

    /// Applies the modified givens transformation to vectors `x` and `y`.
    pub fn rotm<T: Level1 + Float>(
        &mut self,
        stream: &Stream,
        n: usize,
        x: &mut impl GpuBuffer<T>,
        y: &mut impl GpuBuffer<T>,
        param: &impl GpuBox<T::FloatTy>,
    ) -> Result {
        self.rotm_strided(stream, n, x, None, y, None, param)
    }

    /// Same as [`CublasContext::rotmg`] but with an explicit stride.
    pub fn rotmg_strided<T: Level1 + Float>(
        &mut self,
        stream: &Stream,
        d1: &mut impl GpuBox<T>,
        d2: &mut impl GpuBox<T>,
        x1: &mut impl GpuBox<T>,
        y1: &mut impl GpuBox<T>,
        param: &mut impl GpuBox<T>,
    ) -> Result {
        self.with_stream(stream, |ctx| unsafe {
            Ok(T::rotmg(
                ctx.raw,
                d1.as_device_ptr().as_mut_ptr(),
                d2.as_device_ptr().as_mut_ptr(),
                x1.as_device_ptr().as_mut_ptr(),
                y1.as_device_ptr().as_ptr(),
                param.as_device_ptr().as_mut_ptr(),
            )
            .to_result()?)
        })
    }

    /// Constructs the modified givens transformation that zeros out the second entry of a 2x1 vector.
    pub fn rotmg<T: Level1 + Float>(
        &mut self,
        stream: &Stream,
        d1: &mut impl GpuBox<T>,
        d2: &mut impl GpuBox<T>,
        x1: &mut impl GpuBox<T>,
        y1: &mut impl GpuBox<T>,
        param: &mut impl GpuBox<T>,
    ) -> Result {
        self.rotmg_strided(stream, d1, d2, x1, y1, param)
    }

    /// Same as [`CublasContext::scal`] but with an explicit stride.
    pub fn scal_strided<T: Level1>(
        &mut self,
        stream: &Stream,
        n: usize,
        alpha: &impl GpuBox<T>,
        x: &mut impl GpuBuffer<T>,
        x_stride: Option<usize>,
    ) -> Result {
        check_stride(x, n, x_stride);

        self.with_stream(stream, |ctx| unsafe {
            Ok(T::scal(
                ctx.raw,
                n as i32,
                alpha.as_device_ptr().as_ptr(),
                x.as_device_ptr().as_mut_ptr(),
                x_stride.unwrap_or(1) as i32,
            )
            .to_result()?)
        })
    }

    /// Scales vector `x` by `alpha` and overrides it with the result.
    pub fn scal<T: Level1>(
        &mut self,
        stream: &Stream,
        n: usize,
        alpha: &impl GpuBox<T>,
        x: &mut impl GpuBuffer<T>,
    ) -> Result {
        self.scal_strided(stream, n, alpha, x, None)
    }

    /// Same as [`CublasContext::swap`] but with an explicit stride.
    pub fn swap_strided<T: Level1>(
        &mut self,
        stream: &Stream,
        n: usize,
        x: &mut impl GpuBuffer<T>,
        x_stride: Option<usize>,
        y: &mut impl GpuBuffer<T>,
        y_stride: Option<usize>,
    ) -> Result {
        check_stride(x, n, x_stride);
        check_stride(y, n, y_stride);

        self.with_stream(stream, |ctx| unsafe {
            Ok(T::swap(
                ctx.raw,
                n as i32,
                x.as_device_ptr().as_mut_ptr(),
                x_stride.unwrap_or(1) as i32,
                y.as_device_ptr().as_mut_ptr(),
                y_stride.unwrap_or(1) as i32,
            )
            .to_result()?)
        })
    }

    /// Swaps vectors `x` and `y`.
    pub fn swap<T: Level1>(
        &mut self,
        stream: &Stream,
        n: usize,
        x: &mut impl GpuBuffer<T>,
        y: &mut impl GpuBuffer<T>,
    ) -> Result {
        self.swap_strided(stream, n, x, None, y, None)
    }
}
