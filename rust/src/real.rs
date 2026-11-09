use std::fmt;

use crate::error::{invalid, Result};
use crate::names::Precision;

pub(crate) mod sealed {
    use crate::names::Precision;

    /// The conversions of the element types, private to the crate.
    pub trait Sealed {
        /// The machine epsilon.
        const EPSILON: f64;
        /// The smallest positive normal number.
        const MIN_POSITIVE: f64;
        /// The largest finite number.
        const MAX: f64;
        /// The value as f64.
        fn f64(self) -> f64;
        /// The nearest value to `value`.
        fn from_f64(value: f64) -> Self;
        /// The C ABI precision code for `precision` on this storage type, if supported.
        fn abi_code(precision: Precision) -> Option<i32>;
        /// The C ABI storage dtype code (`SDCA_DTYPE_*`).
        const DTYPE_CODE: i32;
    }

    impl Sealed for f32 {
        const EPSILON: f64 = f32::EPSILON as f64;
        const MIN_POSITIVE: f64 = f32::MIN_POSITIVE as f64;
        const MAX: f64 = f32::MAX as f64;
        const DTYPE_CODE: i32 = 0;
        fn f64(self) -> f64 {
            f64::from(self)
        }
        fn from_f64(value: f64) -> Self {
            value as f32
        }
        fn abi_code(precision: Precision) -> Option<i32> {
            match precision {
                Precision::Float32 => Some(0),
                Precision::Float64 => Some(1),
            }
        }
    }

    impl Sealed for f64 {
        const EPSILON: f64 = f64::EPSILON;
        const MIN_POSITIVE: f64 = f64::MIN_POSITIVE;
        const MAX: f64 = f64::MAX;
        const DTYPE_CODE: i32 = 1;
        fn f64(self) -> f64 {
            self
        }
        fn from_f64(value: f64) -> Self {
            value
        }
        fn abi_code(precision: Precision) -> Option<i32> {
            match precision {
                Precision::Float32 => None,
                Precision::Float64 => Some(2),
            }
        }
    }
}

/// The element types of the data, f32 or f64; sealed.
pub trait Real: Copy + Default + PartialOrd + fmt::Debug + sealed::Sealed {
    /// The precision of an option `precision: None`: Float64 for both f64 and f32.
    ///
    /// ```
    /// use libsdca::{Precision, Real};
    ///
    /// assert_eq!(f64::DEFAULT_PRECISION, Precision::Float64);
    /// assert_eq!(f32::DEFAULT_PRECISION, Precision::Float64);
    /// ```
    const DEFAULT_PRECISION: Precision;
}

impl Real for f32 {
    const DEFAULT_PRECISION: Precision = Precision::Float64;
}

impl Real for f64 {
    const DEFAULT_PRECISION: Precision = Precision::Float64;
}

/// The C ABI precision code for `requested` (or the default of `T`) on storage `T`.
pub(crate) fn precision_code<T: Real>(requested: Option<Precision>) -> Result<i32> {
    let precision = requested.unwrap_or(T::DEFAULT_PRECISION);
    let Some(code) = T::abi_code(precision) else {
        return invalid(format!(
            "precision {precision} does not match {} data: f64 takes float64, f32 takes float32 or float64",
            std::any::type_name::<T>()
        ));
    };
    Ok(code)
}
