use std::fmt;

use crate::names::Precision;

pub(crate) mod sealed {
    use std::borrow::Cow;

    /// The conversions of the element types, private to the crate.
    pub trait Sealed {
        /// The machine epsilon.
        const EPSILON: f64;
        /// The value as f64.
        fn f64(self) -> f64;
        /// The nearest value to `value`.
        fn from_f64(value: f64) -> Self;
        /// The C ABI storage dtype code (`SDCA_DTYPE_*`).
        const DTYPE_CODE: i32;
        /// The values as f64, borrowed for f64.
        fn to_f64s(values: &[Self]) -> Cow<'_, [f64]>
        where
            Self: Sized;
    }

    impl Sealed for f32 {
        const EPSILON: f64 = f32::EPSILON as f64;
        const DTYPE_CODE: i32 = 0;
        fn f64(self) -> f64 {
            f64::from(self)
        }
        fn from_f64(value: f64) -> Self {
            value as f32
        }
        fn to_f64s(values: &[Self]) -> Cow<'_, [f64]> {
            Cow::Owned(values.iter().map(|&v| f64::from(v)).collect())
        }
    }

    impl Sealed for f64 {
        const EPSILON: f64 = f64::EPSILON;
        const DTYPE_CODE: i32 = 1;
        fn f64(self) -> f64 {
            self
        }
        fn from_f64(value: f64) -> Self {
            value
        }
        fn to_f64s(values: &[Self]) -> Cow<'_, [f64]> {
            Cow::Borrowed(values)
        }
    }
}

/// The element types of the data, f32 or f64; sealed.
///
/// The element type selects the storage of inputs and results; see
/// [`crate::Precision`] for the working arithmetic.
pub trait Real:
    Copy + Default + PartialOrd + fmt::Debug + Send + Sync + 'static + sealed::Sealed
{
    /// The working arithmetic of `precision: None` in [`crate::Settings`]: that
    /// of the data type (Float32 for f32, Float64 for f64).
    ///
    /// ```
    /// use libsdca::{Precision, Real};
    ///
    /// assert_eq!(f64::DEFAULT_PRECISION, Precision::Float64);
    /// assert_eq!(f32::DEFAULT_PRECISION, Precision::Float32);
    /// ```
    const DEFAULT_PRECISION: Precision;
}

impl Real for f32 {
    const DEFAULT_PRECISION: Precision = Precision::Float32;
}

impl Real for f64 {
    const DEFAULT_PRECISION: Precision = Precision::Float64;
}
