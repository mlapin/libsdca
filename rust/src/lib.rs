//! Rust interface to libsdca: accurate and fast classification losses for SDCA
//! and neural networks.
//!
//! A [`Loss`] carries its parameters. [`Solver`] trains it into a [`Model`] by
//! stochastic dual coordinate ascent, minimizing
//! `0.5 ||W||^2 + c sum_i loss(W x_i, y_i)` (no intercept) until the relative
//! duality gap reaches `tol`, and takes single coordinate steps;
//! [`Loss::loss_and_grad`] evaluates losses and their gradients. Matrices are
//! row-major [`Matrix`] views: features X (n, d) or a Gram matrix ([`Input`]),
//! scores (n, classes), duals A (n, classes) and weights W (classes, d), one
//! weight vector per class. The native library reads and writes these buffers
//! in place: no array is copied or reordered.
//! [`Targets`] are labels, label sets, target distributions or teacher logits.
//!
//! ```
//! use libsdca::{Loss, Matrix, Solver, Status};
//!
//! let x = Matrix::new(&[1.0, 0.0, 0.0, 1.0, -1.0, -1.0], 3, 2)?; // 3 examples, 2 features
//! let y = [0, 1, 2];
//!
//! let model = Solver::new(Loss::cross_entropy()).fit(x, &y)?;
//! assert_eq!((model.status(), model.predict(x)?), (Status::Solved, vec![0, 1, 2]));
//!
//! let model = Solver::new(Loss::topk_hinge(2)).fit(x, &y)?; // top-2 SVM
//! assert_eq!((model.status(), model.predict(x)?), (Status::Solved, vec![0, 1, 2]));
//!
//! let scores = model.decision_function(x)?;
//! let (losses, gradients, report) = Loss::topk_hinge(2).loss_and_grad(Matrix::new(&scores, 3, 3)?, &y)?;
//! assert_eq!((losses.len(), gradients.len(), report.invalid_rows), (3, 9, 0));
//! # Ok::<(), libsdca::Error>(())
//! ```
//!
//! # Precision
//!
//! The element type of the data, f32 or f64 ([`Real`]), is the storage of the
//! inputs and results; [`Settings::precision`] selects the working arithmetic.
//! Its default, `None`, is [`Real::DEFAULT_PRECISION`]: float for f32 and
//! double for f64. [`Precision::Float64`] on f32 data keeps a fit's iterate in
//! double for tighter convergence; [`Precision::Float32`] on f64 data is an
//! [`Error::InvalidArgument`]. Only [`Status::Solved`] guarantees the duality
//! gap, which every precision checks in double on the returned arrays.

mod error;
mod ffi;
mod loss;
mod matrix;
mod model;
mod names;
mod native;
mod options;
mod real;
mod settings;
mod solver;
mod targets;

pub use error::{Error, Result};
pub use loss::{Loss, LossReport};
pub use matrix::Matrix;
pub use model::{Evaluation, Model};
pub use names::{Precision, Shrinking, Status, TopKVariant};
pub use options::Options;
pub use real::Real;
pub use settings::{Settings, Threads};
pub use solver::{Input, Solver, Start};
pub use targets::Targets;

/// Compiles the README examples as doctests.
#[cfg(doctest)]
#[doc = include_str!(concat!(env!("OUT_DIR"), "/guide.md"))]
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
