//! Trait for float intrinsics for making floats work in no_std gpu environments.
//!
//! Float functions are mapped directly to libdevice intrinsics on nvptx and
//! their std counterparts on non-nvptx.

/// std float intrinsics implemented using libdevice intrinsics so they can be used
/// from GPU no_std crates. Falls back to stdlib implementation on non-nvptx.
pub trait GpuFloat: Copy + PartialOrd + private::Sealed {
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn floor(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn ceil(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn round(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn trunc(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn fract(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn abs(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn signum(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn copysign(self, sign: Self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn mul_add(self, a: Self, b: Self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn div_euclid(self, rhs: Self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn rem_euclid(self, rhs: Self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn powi(self, n: i32) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn powf(self, n: Self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn sqrt(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn exp(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn exp2(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn ln(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn log(self, base: Self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn log2(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn log10(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn cbrt(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn hypot(self, other: Self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn sin(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn cos(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn tan(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn asin(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn acos(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn atan(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn atan2(self, other: Self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn sin_cos(self) -> (Self, Self);
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn exp_m1(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn ln_1p(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn sinh(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn cosh(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn tanh(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn asinh(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn acosh(self) -> Self;
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn atanh(self) -> Self;
}

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

macro_rules! f32_intrinsic {
    ($self:expr, $func:ident($($param:expr),*)) => {{
        #[cfg(not(target_arch = "nvptx64"))]
        let val = $self.$func($($param),*);
        #[cfg(target_arch = "nvptx64")]
        let val = paste::paste! { unsafe { intrinsics::[<$func f>]($self, $($param),*)} };
        val
    }};
}

macro_rules! f64_intrinsic {
    ($self:expr, $func:ident($($param:expr),*)) => {{
        #[cfg(not(target_arch = "nvptx64"))]
        let val = $self.$func($($param),*);
        #[cfg(target_arch = "nvptx64")]
        let val = unsafe { intrinsics::$func($self, $($param),*)};
        val
    }};
}

#[cfg(target_arch = "nvptx64")]
use crate::intrinsics;

impl GpuFloat for f32 {
    /// Returns the largest integer less than or equal to a number.
    #[inline]
    fn floor(self) -> f32 {
        f32_intrinsic!(self, floor())
    }

    /// Returns the smallest integer greater than or equal to a number.
    #[inline]
    fn ceil(self) -> f32 {
        f32_intrinsic!(self, ceil())
    }

    /// Returns the nearest integer to a number. Round half-way cases away from
    /// `0.0`.
    #[inline]
    fn round(self) -> f32 {
        f32_intrinsic!(self, round())
    }

    /// Returns the integer part of a number.
    #[inline]
    fn trunc(self) -> f32 {
        f32_intrinsic!(self, trunc())
    }

    /// Returns the fractional part of a number.
    #[inline]
    fn fract(self) -> f32 {
        self - self.trunc()
    }

    /// Computes the absolute value of `self`. Returns `NAN` if the
    /// number is `NAN`.
    #[inline]
    fn abs(self) -> f32 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.abs();
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::fabsf(self) } };
        val
    }

    /// Returns a number that represents the sign of `self`.
    ///
    /// - `1.0` if the number is positive, `+0.0` or `INFINITY`
    /// - `-1.0` if the number is negative, `-0.0` or `NEG_INFINITY`
    /// - `NAN` if the number is `NAN`intrinsics
    #[inline]
    fn signum(self) -> f32 {
        if self.is_nan() {
            Self::NAN
        } else {
            1.0_f32.copysign(self)
        }
    }

    /// Returns a number composed of the magnitude of `self` and the sign of
    /// `sign`.
    ///
    /// Equal to `self` if the sign of `self` and `sign` are the same, otherwise
    /// equal to `-self`. If `self` is a `NAN`, then a `NAN` with the sign of
    /// `sign` is returned.
    #[inline]
    fn copysign(self, sign: f32) -> f32 {
        f32_intrinsic!(self, copysign(sign))
    }

    /// Fused multiply-add. Computes `(self * a) + b` with only one rounding
    /// error, yielding a more accurate result than an unfused multiply-add.
    ///
    /// Using `mul_add` *may* be more performant than an unfused multiply-add if
    /// the target architecture has a dedicated `fma` CPU instruction. However,
    /// this is not always true, and will be heavily dependant on designing
    /// algorithms with specific target hardware in mind.
    #[inline]
    fn mul_add(self, a: f32, b: f32) -> f32 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.mul_add(a, b);
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::fmaf(self, a, b) } };
        val
    }

    /// Calculates Euclidean division, the matching method for `rem_euclid`.
    ///
    /// This computes the integer `n` such that
    /// `self = n * rhs + self.rem_euclid(rhs)`.
    /// In other words, the result is `self / rhs` rounded to the integer `n`
    /// such that `self >= n * rhs`.
    #[inline]
    fn div_euclid(self, rhs: f32) -> f32 {
        let q = (self / rhs).trunc();
        if self % rhs < 0.0 {
            return if rhs > 0.0 { q - 1.0 } else { q + 1.0 };
        }
        q
    }

    /// Calculates the least nonnegative remainder of `self (mod rhs)`.
    ///
    /// In particular, the return value `r` satisfies `0.0 <= r < rhs.abs()` in
    /// most cases. However, due to a floating point round-off error it can
    /// result in `r == rhs.abs()`, violating the mathematical definition, if
    /// `self` is much smaller than `rhs.abs()` in magnitude and `self < 0.0`.
    /// This result is not an element of the function's codomain, but it is the
    /// closest floating point number in the real numbers and thus fulfills the
    /// property `self == self.div_euclid(rhs) * rhs + self.rem_euclid(rhs)`
    /// approximatively.
    #[inline]
    fn rem_euclid(self, rhs: f32) -> f32 {
        let r = self % rhs;
        if r < 0.0 { r + rhs.abs() } else { r }
    }

    /// Raises a number to an integer power.
    ///
    /// Using this function is generally faster than using `powf`intrinsics
    #[inline]
    fn powi(self, n: i32) -> f32 {
        f32_intrinsic!(self, powi(n))
    }

    /// Raises a number to a floating point power.
    #[inline]
    fn powf(self, n: f32) -> f32 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.powf(n);
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::powf(self, n) } };
        val
    }

    /// Returns the square root of a number.
    ///
    /// Returns NaN if `self` is a negative number other than `-0.0`.
    #[inline]
    fn sqrt(self) -> f32 {
        f32_intrinsic!(self, sqrt())
    }

    /// Returns `e^(self)`, (the exponential function).
    #[inline]
    fn exp(self) -> f32 {
        f32_intrinsic!(self, exp())
    }

    /// Returns `2^(self)`.
    #[inline]
    fn exp2(self) -> f32 {
        f32_intrinsic!(self, exp2())
    }

    /// Returns the natural logarithm of the number.
    #[inline]
    fn ln(self) -> f32 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.ln();
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::logf(self) } };
        val
    }

    /// Returns the logarithm of the number with respect to an arbitrary base.
    ///
    /// The result might not be correctly rounded owing to implementation details;
    /// `self.log2()` can produce more accurate results for base 2, and
    /// `self.log10()` can produce more accurate results for base 10.
    #[inline]
    fn log(self, base: f32) -> f32 {
        self.ln() / base.ln()
    }

    /// Returns the base 2 logarithm of the number.
    #[inline]
    fn log2(self) -> f32 {
        f32_intrinsic!(self, log10())
    }

    /// Returns the base 10 logarithm of the number.
    #[inline]
    fn log10(self) -> f32 {
        f32_intrinsic!(self, log10())
    }

    /// Returns the cube root of a number.
    #[inline]
    fn cbrt(self) -> f32 {
        f32_intrinsic!(self, cbrt())
    }

    /// Calculates the length of the hypotenuse of a right-angle triangle given
    /// legs of length `x` and `y`.
    #[inline]
    fn hypot(self, other: f32) -> f32 {
        f32_intrinsic!(self, hypot(other))
    }

    /// Computes the sine of a number (in radians).
    #[inline]
    fn sin(self) -> f32 {
        f32_intrinsic!(self, sin())
    }

    /// Computes the cosine of a number (in radians).
    #[inline]
    fn cos(self) -> f32 {
        f32_intrinsic!(self, cos())
    }

    /// Computes the tangent of a number (in radians).
    #[inline]
    fn tan(self) -> f32 {
        f32_intrinsic!(self, tan())
    }

    /// Computes the arcsine of a number. Return value is in radians in
    /// the range [-pi/2, pi/2] or NaN if the number is outside the range
    /// [-1, 1].
    #[inline]
    fn asin(self) -> f32 {
        f32_intrinsic!(self, asin())
    }

    /// Computes the arccosine of a number. Return value is in radians in
    /// the range [0, pi] or NaN if the number is outside the range
    /// [-1, 1].
    #[inline]
    fn acos(self) -> f32 {
        f32_intrinsic!(self, acos())
    }

    /// Computes the arctangent of a number. Return value is in radians in the
    /// range [-pi/2, pi/2];intrinsics
    #[inline]
    fn atan(self) -> f32 {
        f32_intrinsic!(self, atan())
    }

    /// Computes the four quadrant arctangent of `self` (`y`) and `other` (`x`) in
    /// radians.
    ///
    ///   * `x = 0`, `y = 0`: `0`
    ///   * `x >= 0`: `arctan(y/x)` -> `[-pi/2, pi/2]`
    ///   * `y >= 0`: `arctan(y/x) + pi` -> `(pi/2, pi]`
    ///   * `y < 0`: `arctan(y/x) - pi` -> `(-pi, -pi/2)`intrinsics
    #[inline]
    fn atan2(self, other: f32) -> f32 {
        f32_intrinsic!(self, atan2(other))
    }

    /// Simultaneously computes the sine and cosine of the number, `x`. Returns
    /// `(sin(x), cos(x))`.
    #[inline]
    fn sin_cos(self) -> (f32, f32) {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.sin_cos();
        #[cfg(target_arch = "nvptx64")]
        let val = {
            let mut sptr = 0.0;
            let mut cptr = 0.0;
            unsafe {
                intrinsics::sincosf(self, &mut sptr as *mut _, &mut cptr as *mut _);
            }
            (sptr, cptr)
        };
        val
    }

    /// Returns `e^(self) - 1` in a way that is accurate even if the
    /// number is close to zero.
    #[inline]
    fn exp_m1(self) -> f32 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.exp_m1();
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::expm1f(self) } };
        val
    }

    /// Returns `ln(1+n)` (natural logarithm) more accurately than if
    /// the operations were performed separately.
    #[inline]
    fn ln_1p(self) -> f32 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.ln_1p();
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::log1pf(self) } };
        val
    }

    /// Hyperbolic sine function.
    #[inline]
    fn sinh(self) -> f32 {
        f32_intrinsic!(self, sinh())
    }

    /// Hyperbolic cosine function.
    #[inline]
    fn cosh(self) -> f32 {
        f32_intrinsic!(self, cosh())
    }

    /// Hyperbolic tangent function.
    #[inline]
    fn tanh(self) -> f32 {
        f32_intrinsic!(self, tanh())
    }

    /// Inverse hyperbolic sine function.
    #[inline]
    fn asinh(self) -> f32 {
        f32_intrinsic!(self, asinh())
    }

    /// Inverse hyperbolic cosine function.
    #[inline]
    fn acosh(self) -> f32 {
        f32_intrinsic!(self, acosh())
    }

    /// Inverse hyperbolic tangent function.
    #[inline]
    fn atanh(self) -> f32 {
        f32_intrinsic!(self, atanh())
    }
}

impl GpuFloat for f64 {
    /// Returns the largest integer less than or equal to a number.
    #[inline]
    fn floor(self) -> f64 {
        f64_intrinsic!(self, floor())
    }

    /// Returns the smallest integer greater than or equal to a number.
    #[inline]
    fn ceil(self) -> f64 {
        f64_intrinsic!(self, ceil())
    }

    /// Returns the nearest integer to a number. Round half-way cases away from
    /// `0.0`.
    #[inline]
    fn round(self) -> f64 {
        f64_intrinsic!(self, round())
    }

    /// Returns the integer part of a number.
    #[inline]
    fn trunc(self) -> f64 {
        f64_intrinsic!(self, trunc())
    }

    /// Returns the fractional part of a number.
    #[inline]
    fn fract(self) -> f64 {
        self - self.trunc()
    }

    /// Computes the absolute value of `self`. Returns `NAN` if the
    /// number is `NAN`.
    #[inline]
    fn abs(self) -> f64 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.abs();
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::fabs(self) } };
        val
    }

    /// Returns a number that represents the sign of `self`.
    ///
    /// - `1.0` if the number is positive, `+0.0` or `INFINITY`
    /// - `-1.0` if the number is negative, `-0.0` or `NEG_INFINITY`
    /// - `NAN` if the number is `NAN`intrinsics
    #[inline]
    fn signum(self) -> f64 {
        if self.is_nan() {
            Self::NAN
        } else {
            1.0_f64.copysign(self)
        }
    }

    /// Returns a number composed of the magnitude of `self` and the sign of
    /// `sign`.
    ///
    /// Equal to `self` if the sign of `self` and `sign` are the same, otherwise
    /// equal to `-self`. If `self` is a `NAN`, then a `NAN` with the sign of
    /// `sign` is returned.
    #[inline]
    fn copysign(self, sign: f64) -> f64 {
        f64_intrinsic!(self, copysign(sign))
    }

    /// Fused multiply-add. Computes `(self * a) + b` with only one rounding
    /// error, yielding a more accurate result than an unfused multiply-add.
    ///
    /// Using `mul_add` *may* be more performant than an unfused multiply-add if
    /// the target architecture has a dedicated `fma` CPU instruction. However,
    /// this is not always true, and will be heavily dependant on designing
    /// algorithms with specific target hardware in mind.
    #[inline]
    fn mul_add(self, a: f64, b: f64) -> f64 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.mul_add(a, b);
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::fma(self, a, b) } };
        val
    }

    /// Calculates Euclidean division, the matching method for `rem_euclid`.
    ///
    /// This computes the integer `n` such that
    /// `self = n * rhs + self.rem_euclid(rhs)`.
    /// In other words, the result is `self / rhs` rounded to the integer `n`
    /// such that `self >= n * rhs`.
    #[inline]
    fn div_euclid(self, rhs: f64) -> f64 {
        let q = (self / rhs).trunc();
        if self % rhs < 0.0 {
            return if rhs > 0.0 { q - 1.0 } else { q + 1.0 };
        }
        q
    }

    /// Calculates the least nonnegative remainder of `self (mod rhs)`.
    ///
    /// In particular, the return value `r` satisfies `0.0 <= r < rhs.abs()` in
    /// most cases. However, due to a floating point round-off error it can
    /// result in `r == rhs.abs()`, violating the mathematical definition, if
    /// `self` is much smaller than `rhs.abs()` in magnitude and `self < 0.0`.
    /// This result is not an element of the function's codomain, but it is the
    /// closest floating point number in the real numbers and thus fulfills the
    /// property `self == self.div_euclid(rhs) * rhs + self.rem_euclid(rhs)`
    /// approximatively.
    #[inline]
    fn rem_euclid(self, rhs: f64) -> f64 {
        let r = self % rhs;
        if r < 0.0 { r + rhs.abs() } else { r }
    }

    /// Raises a number to an integer power.
    ///
    /// Using this function is generally faster than using `powf`intrinsics
    #[inline]
    fn powi(self, n: i32) -> f64 {
        f64_intrinsic!(self, powi(n))
    }

    /// Raises a number to a floating point power.
    #[inline]
    fn powf(self, n: f64) -> f64 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.powf(n);
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::pow(self, n) } };
        val
    }

    /// Returns the square root of a number.
    ///
    /// Returns NaN if `self` is a negative number other than `-0.0`.
    #[inline]
    fn sqrt(self) -> f64 {
        f64_intrinsic!(self, sqrt())
    }

    /// Returns `e^(self)`, (the exponential function).
    #[inline]
    fn exp(self) -> f64 {
        f64_intrinsic!(self, exp())
    }

    /// Returns `2^(self)`.
    #[inline]
    fn exp2(self) -> f64 {
        f64_intrinsic!(self, exp2())
    }

    /// Returns the natural logarithm of the number.
    #[inline]
    fn ln(self) -> f64 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.ln();
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::log(self) } };
        val
    }

    /// Returns the logarithm of the number with respect to an arbitrary base.
    ///
    /// The result might not be correctly rounded owing to implementation details;
    /// `self.log2()` can produce more accurate results for base 2, and
    /// `self.log10()` can produce more accurate results for base 10.
    #[inline]
    fn log(self, base: f64) -> f64 {
        self.ln() / base.ln()
    }

    /// Returns the base 2 logarithm of the number.
    #[inline]
    fn log2(self) -> f64 {
        f64_intrinsic!(self, log10())
    }

    /// Returns the base 10 logarithm of the number.
    #[inline]
    fn log10(self) -> f64 {
        f64_intrinsic!(self, log10())
    }

    /// Returns the cube root of a number.
    #[inline]
    fn cbrt(self) -> f64 {
        f64_intrinsic!(self, cbrt())
    }

    /// Calculates the length of the hypotenuse of a right-angle triangle given
    /// legs of length `x` and `y`.
    #[inline]
    fn hypot(self, other: f64) -> f64 {
        f64_intrinsic!(self, hypot(other))
    }

    /// Computes the sine of a number (in radians).
    #[inline]
    fn sin(self) -> f64 {
        f64_intrinsic!(self, sin())
    }

    /// Computes the cosine of a number (in radians).
    #[inline]
    fn cos(self) -> f64 {
        f64_intrinsic!(self, cos())
    }

    /// Computes the tangent of a number (in radians).
    #[inline]
    fn tan(self) -> f64 {
        f64_intrinsic!(self, tan())
    }

    /// Computes the arcsine of a number. Return value is in radians in
    /// the range [-pi/2, pi/2] or NaN if the number is outside the range
    /// [-1, 1].
    #[inline]
    fn asin(self) -> f64 {
        f64_intrinsic!(self, asin())
    }

    /// Computes the arccosine of a number. Return value is in radians in
    /// the range [0, pi] or NaN if the number is outside the range
    /// [-1, 1].
    #[inline]
    fn acos(self) -> f64 {
        f64_intrinsic!(self, acos())
    }

    /// Computes the arctangent of a number. Return value is in radians in the
    /// range [-pi/2, pi/2];intrinsics
    #[inline]
    fn atan(self) -> f64 {
        f64_intrinsic!(self, atan())
    }

    /// Computes the four quadrant arctangent of `self` (`y`) and `other` (`x`) in
    /// radians.
    ///
    ///   * `x = 0`, `y = 0`: `0`
    ///   * `x >= 0`: `arctan(y/x)` -> `[-pi/2, pi/2]`
    ///   * `y >= 0`: `arctan(y/x) + pi` -> `(pi/2, pi]`
    ///   * `y < 0`: `arctan(y/x) - pi` -> `(-pi, -pi/2)`intrinsics
    #[inline]
    fn atan2(self, other: f64) -> f64 {
        f64_intrinsic!(self, atan2(other))
    }

    /// Simultaneously computes the sine and cosine of the number, `x`. Returns
    /// `(sin(x), cos(x))`.
    #[inline]
    fn sin_cos(self) -> (f64, f64) {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.sin_cos();
        #[cfg(target_arch = "nvptx64")]
        let val = {
            let mut sptr = 0.0;
            let mut cptr = 0.0;
            unsafe {
                intrinsics::sincos(self, &mut sptr as *mut _, &mut cptr as *mut _);
            }
            (sptr, cptr)
        };
        val
    }

    /// Returns `e^(self) - 1` in a way that is accurate even if the
    /// number is close to zero.
    #[inline]
    fn exp_m1(self) -> f64 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.exp_m1();
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::expm1(self) } };
        val
    }

    /// Returns `ln(1+n)` (natural logarithm) more accurately than if
    /// the operations were performed separately.
    #[inline]
    fn ln_1p(self) -> f64 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.ln_1p();
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::log1p(self) } };
        val
    }

    /// Hyperbolic sine function.
    #[inline]
    fn sinh(self) -> f64 {
        f64_intrinsic!(self, sinh())
    }

    /// Hyperbolic cosine function.
    #[inline]
    fn cosh(self) -> f64 {
        f64_intrinsic!(self, cosh())
    }

    /// Hyperbolic tangent function.
    #[inline]
    fn tanh(self) -> f64 {
        f64_intrinsic!(self, tanh())
    }

    /// Inverse hyperbolic sine function.
    #[inline]
    fn asinh(self) -> f64 {
        f64_intrinsic!(self, asinh())
    }

    /// Inverse hyperbolic cosine function.
    #[inline]
    fn acosh(self) -> f64 {
        f64_intrinsic!(self, acosh())
    }

    /// Inverse hyperbolic tangent function.
    #[inline]
    fn atanh(self) -> f64 {
        f64_intrinsic!(self, atanh())
    }
}
