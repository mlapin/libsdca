//! Rust interface to libsdca: SDCA solvers, losses and coordinate steps.
//!
//! The API mirrors the Python `libsdca` module. [`fit`] trains the
//! [`Objective`]s into a [`Model`], [`loss_and_grad`] evaluates the
//! [`Loss`]es and [`coordinate_update`] takes one coordinate step. Matrices
//! are row-major [`Matrix`] views: features X (n, d), scores (n, classes),
//! dual variables A (n, classes) and weights W (d, classes); targets are
//! labels or label sets ([`Targets`]). The objective is
//! `0.5 ||W||^2 + C sum_i loss(X[i] W, y[i])`, without an implicit intercept.
//!
//! ```
//! use libsdca::{fit, loss_and_grad, FitOptions, LossOptions, Matrix, Status};
//!
//! let x = [1.0, 0.0, 0.0, 1.0, -1.0, -1.0];
//! let x = Matrix::new(&x, 3, 2)?;
//! let model = fit(x, &[0, 1, 2], &FitOptions::default())?;
//! assert_eq!(model.status(), Status::Solved);
//! assert_eq!(model.predict(x)?, [0, 1, 2]);
//!
//! let scores = model.decision_function(x)?;
//! let scores = Matrix::new(&scores, 3, 3)?;
//! let (values, gradients) = loss_and_grad(scores, &[0, 1, 2], &LossOptions::default())?;
//! assert_eq!((values.len(), gradients.len()), (3, 9));
//! # Ok::<(), libsdca::Error>(())
//! ```
//!
//! # Precision
//!
//! The element type of the data, f64 or f32, selects the storage; the
//! `precision` option selects the working arithmetic of [`fit`],
//! [`coordinate_update`] and [`loss_and_grad`]. Its default, `None`, is
//! [`Real::DEFAULT_PRECISION`] ([`Precision::Float64`] for both f64 and f32).
//! `Some(Precision::Float32)` computes f32 data in float, accurate to float
//! rounding of the scores: the entropy steps are faster, the hinge steps are
//! not. Requesting `Precision::Float32` on f64 data is an
//! [`Error::InvalidArgument`].

mod error;
mod ffi;
mod matrix;
mod model;
mod names;
mod options;
mod real;
mod solver;
mod targets;

pub use error::{Error, Result};
pub use matrix::Matrix;
pub use model::{History, Model, HISTORY};
pub use names::{Loss, Objective, Precision, ProjectionPrecision, Shrinking, Status};
pub use options::{DistillationOptions, FitOptions, LossOptions, UpdateOptions};
pub use real::Real;
pub use solver::{
    coordinate_update, coordinate_update_distillation, distillation_loss_and_grad, fit,
    fit_distillation, loss_and_grad, proximal_distillation_loss_and_grad,
};
pub use targets::Targets;

/// Compiles the README examples as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

/// The version of the native library.
///
/// ```
/// assert_eq!(libsdca::version(), env!("CARGO_PKG_VERSION"));
/// ```
#[must_use]
pub fn version() -> &'static str {
    // SAFETY: sdca_version returns a static NUL-terminated string.
    unsafe { std::ffi::CStr::from_ptr(ffi::sdca_version()) }
        .to_str()
        .unwrap_or("unknown")
}
