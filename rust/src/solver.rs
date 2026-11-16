use std::ffi::c_int;
use std::ptr;

use crate::error::{invalid, Result};
use crate::ffi;
use crate::loss::Loss;
use crate::matrix::Matrix;
use crate::model::{Evaluation, Model};
use crate::names::{Precision, ProjectionPrecision, Shrinking, Status};
use crate::native::{self, call, check_logits, void};
use crate::real::Real;
use crate::settings::{Settings, Threads};
use crate::targets::{self, Targets};

/// The training input: features, or a Gram matrix of a kernel.
///
/// A [`Matrix`] converts into `Features`.
///
/// ```
/// use libsdca::{Input, Matrix};
///
/// let x = Matrix::new(&[1.0, 0.0, 0.0, 1.0], 2, 2)?;
/// assert_eq!(Input::from(x), Input::Features(x));
/// # Ok::<(), libsdca::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Input<'a, T> {
    /// Features X (n, d), one example per row: a linear model W (d, num_classes).
    Features(Matrix<'a, T>),
    /// A symmetric positive semidefinite Gram matrix K (n, n) of the
    /// examples: a kernel model of its duals.
    Kernel(Matrix<'a, T>),
}

impl<'a, T> From<Matrix<'a, T>> for Input<'a, T> {
    fn from(features: Matrix<'a, T>) -> Self {
        Input::Features(features)
    }
}

/// The starting point of [`Solver::fit_from`]; [`Start::default`] is a cold
/// start from zero duals.
///
/// ```
/// use libsdca::{Matrix, Solver, Start};
///
/// let x = Matrix::new(&[1.0, 0.0, 0.0, 1.0, -1.0, -1.0], 3, 2)?;
/// let solver = Solver::default().with_max_epoch(5);
/// let first = solver.fit(x, &[0, 1, 2])?;
/// let start = Start::default().with_initial_dual(first.dual());
/// let model = solver.with_max_epoch(1000).fit_from(x, &[0, 1, 2], &start)?;
/// assert_eq!(model.status(), libsdca::Status::Solved);
/// # Ok::<(), libsdca::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Start<'a, T> {
    /// Duals A (n, num_classes), row-major, usually those of an earlier fit:
    /// checked against the feasible set of the loss to the rounding of `T`,
    /// with the label entry of single-label rows rebuilt so that rows sum to
    /// zero exactly.
    pub initial_dual: Option<&'a [T]>,
    /// The center W0 (d, num_classes), row-major as [`Model::weights`], of the
    /// regularizer `0.5 ||W - W0||^2`; features only.
    pub center: Option<&'a [T]>,
}

impl<T> Default for Start<'_, T> {
    fn default() -> Self {
        Start {
            initial_dual: None,
            center: None,
        }
    }
}

impl<'a, T> Start<'a, T> {
    /// From the duals `initial_dual`.
    #[must_use]
    pub fn with_initial_dual(mut self, initial_dual: &'a [T]) -> Self {
        self.initial_dual = Some(initial_dual);
        self
    }

    /// With the regularizer center `center`.
    #[must_use]
    pub fn with_center(mut self, center: &'a [T]) -> Self {
        self.center = Some(center);
        self
    }
}

/// An SDCA solver of `0.5 ||W||^2 + c sum_i loss(x_i W, y_i)`, without an
/// intercept, and its coordinate steps.
///
/// [`Solver::new`] takes the loss; the `with_*` setters change the defaults of
/// every interface (c = 1, tol = 1e-3, max_epoch = 1000, eval_epoch = 10,
/// no evaluation on start, recompute_primal on, [`Shrinking::Auto`], seed 0,
/// the classes implied by the labels, [`Settings::default`]). The settings are
/// checked by [`Solver::fit`] and [`Solver::coordinate_update`].
///
/// The training losses are the single-label and multilabel losses;
/// `CrossEntropy` against [`Targets::Probabilities`] is the distillation
/// objective. The proximal losses and `Distillation` (teacher logits) are
/// losses of [`Loss::loss_and_grad`] only.
///
/// ```
/// use libsdca::{Loss, Matrix, Solver, Status};
///
/// let x = Matrix::new(&[1.0, 0.0, 0.0, 1.0, -1.0, -1.0], 3, 2)?;
/// let solver = Solver::new(Loss::topk_cross_entropy(2)).with_c(10.0).with_tol(1e-6);
/// let model = solver.fit(x, &[0, 1, 2])?;
/// assert_eq!((model.status(), model.c()), (Status::Solved, 10.0));
/// assert_eq!(model.predict(x)?, [0, 1, 2]);
/// # Ok::<(), libsdca::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Solver {
    loss: Loss,
    c: f64,
    tol: f64,
    max_epoch: usize,
    eval_epoch: usize,
    eval_on_start: bool,
    recompute_primal: bool,
    shrinking: Shrinking,
    seed: u64,
    num_classes: Option<usize>,
    settings: Settings,
}

impl Default for Solver {
    /// A cross-entropy solver.
    fn default() -> Self {
        Solver::new(Loss::CrossEntropy)
    }
}

impl Solver {
    /// A solver of `loss` with the default settings.
    #[must_use]
    pub fn new(loss: Loss) -> Self {
        Solver {
            loss,
            c: 1.0,
            tol: 1e-3,
            max_epoch: 1000,
            eval_epoch: 10,
            eval_on_start: false,
            recompute_primal: true,
            shrinking: Shrinking::Auto,
            seed: 0,
            num_classes: None,
            settings: Settings::default(),
        }
    }

    /// With the loss weight `c`, finite and positive; `c`, every nonzero
    /// squared example norm and their products must be normal in the data type.
    #[must_use]
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// With the relative duality gap of success `tol`, finite and >= 0; 0
    /// stops at `max_epoch` (or with no progress or a failure).
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// With the epoch limit `max_epoch`.
    #[must_use]
    pub fn with_max_epoch(mut self, max_epoch: usize) -> Self {
        self.max_epoch = max_epoch;
        self
    }

    /// Evaluate after `eval_epoch` full passes of work (epochs without
    /// shrinking, at most 50 epochs apart with it), and up to twice between
    /// where the trend of the gap reaches `tol`; 0 evaluates only at the end.
    #[must_use]
    pub fn with_eval_epoch(mut self, eval_epoch: usize) -> Self {
        self.eval_epoch = eval_epoch;
        self
    }

    /// Whether to also evaluate before the first epoch.
    #[must_use]
    pub fn with_eval_on_start(mut self, eval_on_start: bool) -> Self {
        self.eval_on_start = eval_on_start;
        self
    }

    /// Whether to rebuild W = center + X' A at each evaluation of a feature
    /// model, so that the drift of the updates of W does not enter the gap.
    #[must_use]
    pub fn with_recompute_primal(mut self, recompute_primal: bool) -> Self {
        self.recompute_primal = recompute_primal;
        self
    }

    /// With what the epochs of the hinge losses leave out; the entropy losses
    /// take [`Shrinking::Auto`] only.
    #[must_use]
    pub fn with_shrinking(mut self, shrinking: Shrinking) -> Self {
        self.shrinking = shrinking;
        self
    }

    /// With the seed `seed` of the order of the examples in the epochs; seeds
    /// equal modulo 2^31 - 2 give the same order.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// With `num_classes` classes (>= 2) instead of the largest label plus one
    /// (or the columns of target probabilities).
    #[must_use]
    pub fn with_num_classes(mut self, num_classes: usize) -> Self {
        self.num_classes = Some(num_classes);
        self
    }

    /// With the arithmetic, threads and projection solver of `settings`.
    #[must_use]
    pub fn with_settings(mut self, settings: Settings) -> Self {
        self.settings = settings;
        self
    }

    /// With the working arithmetic `precision`; see [`Settings::with_precision`].
    #[must_use]
    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.settings = self.settings.with_precision(precision);
        self
    }

    /// With the threads `threads`; see [`Settings::with_threads`].
    #[must_use]
    pub fn with_threads(mut self, threads: Threads) -> Self {
        self.settings = self.settings.with_threads(threads);
        self
    }

    /// With the projection solver `projection_precision`; see
    /// [`Settings::with_projection_precision`].
    #[must_use]
    pub fn with_projection_precision(mut self, projection_precision: ProjectionPrecision) -> Self {
        self.settings = self
            .settings
            .with_projection_precision(projection_precision);
        self
    }

    /// The loss.
    #[must_use]
    pub fn loss(&self) -> Loss {
        self.loss
    }

    /// The loss weight C.
    #[must_use]
    pub fn c(&self) -> f64 {
        self.c
    }

    /// The relative duality gap of success.
    #[must_use]
    pub fn tol(&self) -> f64 {
        self.tol
    }

    /// The epoch limit.
    #[must_use]
    pub fn max_epoch(&self) -> usize {
        self.max_epoch
    }

    /// The full passes of work between evaluations (0: only at the end).
    #[must_use]
    pub fn eval_epoch(&self) -> usize {
        self.eval_epoch
    }

    /// Whether the objective is also evaluated before the first epoch.
    #[must_use]
    pub fn eval_on_start(&self) -> bool {
        self.eval_on_start
    }

    /// Whether evaluations rebuild W = center + X' A.
    #[must_use]
    pub fn recompute_primal(&self) -> bool {
        self.recompute_primal
    }

    /// What the epochs of the hinge losses leave out.
    #[must_use]
    pub fn shrinking(&self) -> Shrinking {
        self.shrinking
    }

    /// The seed of the order of the examples in the epochs.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The number of classes; `None` for the classes implied by the targets.
    #[must_use]
    pub fn num_classes(&self) -> Option<usize> {
        self.num_classes
    }

    /// The arithmetic, threads and projection solver.
    #[must_use]
    pub fn settings(&self) -> Settings {
        self.settings
    }

    /// Trains on `input` with `targets` (one per example) from zero duals:
    /// [`Solver::fit_from`] with [`Start::default`].
    ///
    /// # Errors
    /// As [`Solver::fit_from`].
    ///
    /// ```
    /// use libsdca::{Loss, Matrix, Solver, Status, Targets};
    ///
    /// let x = Matrix::new(&[1.0, 0.0, 0.0, 1.0, -1.0, -1.0], 3, 2)?;
    /// let q = Matrix::new(&[0.8, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.8], 3, 3)?;
    /// let model = Solver::default().fit(x, Targets::Probabilities(q))?;
    /// assert!(model.has_probability_targets() && model.status() == Status::Solved);
    /// # Ok::<(), libsdca::Error>(())
    /// ```
    pub fn fit<'x, 'y, T: Real>(
        &self,
        input: impl Into<Input<'x, T>>,
        targets: impl Into<Targets<'y, T>>,
    ) -> Result<Model<T>> {
        self.fit_from(input, targets, &Start::default())
    }

    /// Trains on `input` with `targets` (one per example) from `start`.
    ///
    /// The single-label losses take [`Targets::Labels`], the multilabel ones
    /// label sets of between 1 and num_classes - 1 labels, and `CrossEntropy`
    /// also [`Targets::Probabilities`]. The model keeps the storage type `T`;
    /// see [`crate::Precision`] for the arithmetic.
    ///
    /// # Errors
    /// [`crate::Error::InvalidArgument`] for an empty or nonfinite input, a
    /// kernel that is not square, symmetric and positive semidefinite (to
    /// tolerance), a loss that is not a training objective, targets of a kind
    /// the loss does not take or of the wrong size, invalid labels, loss
    /// parameters or solver settings (such as shrinking other than
    /// [`Shrinking::Auto`] for an entropy loss), a center with [`Input::Kernel`],
    /// an infeasible initial dual, a precision the data type does not take, or
    /// sizes beyond the native BLAS integers; [`crate::Error::Native`] for a
    /// native failure or a library that does not match the crate. Not
    /// converging or a numerical failure of training is no error: see
    /// [`Model::status`].
    pub fn fit_from<'x, 'y, T: Real>(
        &self,
        input: impl Into<Input<'x, T>>,
        targets: impl Into<Targets<'y, T>>,
        start: &Start<'_, T>,
    ) -> Result<Model<T>> {
        native::handshake()?;
        let execution = native::execution::<T>(&self.settings)?;
        let (x, kernel) = match input.into() {
            Input::Features(x) => (x, false),
            Input::Kernel(x) => (x, true),
        };
        let (n, d) = (x.rows(), x.cols());
        if n == 0 || d == 0 {
            return invalid("input must be a nonempty matrix");
        }
        if kernel && n != d {
            return invalid(format!(
                "Input::Kernel must be a square Gram matrix, not {n}x{d}"
            ));
        }
        let targets = targets.into();
        let classes = match self.num_classes {
            Some(classes) => classes,
            None => implied_classes(&targets)?,
        };
        if classes < 2 {
            return invalid(format!("num_classes must be at least 2, not {classes}"));
        }
        // Before the buffers of (n, classes) and (d, classes) are allocated.
        let sizes = [
            n,
            d,
            classes,
            d.saturating_mul(classes),
            n.saturating_mul(classes),
        ];
        let limit = native::max_extent();
        if sizes.iter().any(|&size| size > limit) {
            return invalid("Dimensions exceed the native BLAS integer limit");
        }
        self.check_objective(classes)?;
        let targets = targets::native(self.loss, &targets, n, classes, true)?;
        let options = ffi::FitOptions {
            c: self.c,
            tol: self.tol,
            max_epoch: self.max_epoch,
            eval_epoch: self.eval_epoch,
            eval_on_start: c_int::from(self.eval_on_start),
            recompute_primal: c_int::from(self.recompute_primal),
            shrinking: self.shrinking.code(),
            seed: self.seed,
            ..Default::default()
        };
        let rows = native::history_rows(&options)?;
        let mut history = Vec::new();
        if history
            .try_reserve_exact(rows * ffi::HISTORY_FIELDS)
            .is_err()
        {
            return invalid("max_epoch / eval_epoch is too large for the history buffer");
        }
        history.resize(rows * ffi::HISTORY_FIELDS, 0.0);
        let center = match start.center {
            None => None,
            Some(_) if kernel => {
                return invalid("Start::center is for Input::Features, not Input::Kernel");
            }
            Some(center) => Some(column_major(center, d, classes)?),
        };
        if let Some(initial) = start.initial_dual {
            if initial.len() != n * classes {
                return invalid(format!(
                    "initial_dual must have the {n}x{classes} entries of (n, num_classes), not {}",
                    initial.len()
                ));
            }
        }
        let mut weights = vec![T::default(); if kernel { 0 } else { d * classes }];
        let mut dual = vec![T::default(); n * classes];
        let input = ffi::Input {
            kind: if kernel {
                ffi::INPUT_KERNEL
            } else {
                ffi::INPUT_FEATURES
            },
            dtype: T::DTYPE_CODE,
            n,
            d,
            x: void(x.data()),
            ..Default::default()
        };
        let mut result = ffi::FitResult::default();
        // SAFETY: the buffers have the sizes of the C ABI documentation, the
        // targets point into storage that outlives the call, and the weights
        // and center are NULL for a kernel.
        call(unsafe {
            ffi::sdca_model_fit(
                &self.loss.abi(self.settings.projection_precision),
                &options,
                &execution,
                &input,
                classes,
                &targets.abi(),
                center.as_deref().map_or(ptr::null(), <[T]>::as_ptr).cast(),
                start.initial_dual.map_or(ptr::null(), <[T]>::as_ptr).cast(),
                if kernel {
                    ptr::null_mut()
                } else {
                    weights.as_mut_ptr().cast()
                },
                dual.as_mut_ptr().cast(),
                history.as_mut_ptr(),
                rows,
                &mut result,
            )
        })?;
        let status = Status::from_code(result.status)?;
        // W (d, classes) column-major to row-major.
        let weights = (!kernel).then(|| {
            (0..d * classes)
                .map(|i| weights[(i % classes) * d + i / classes])
                .collect()
        });
        Ok(Model {
            weights,
            dual,
            history: Evaluation::history(&history, result.history_rows),
            status,
            epochs: result.epochs,
            loss: self.loss,
            c: self.c,
            num_classes: classes,
            probabilities: matches!(targets, targets::Native::Probabilities(_)),
        })
    }

    /// New dual blocks (n, classes) of one coordinate step of the loss from
    /// the blocks `previous` (zeros if `None`) at the scores `logits` (n,
    /// classes), for examples of squared norm `norm2`, with the loss weight
    /// and settings of the solver.
    ///
    /// The targets are those of [`Solver::fit_from`]; `previous` must be
    /// feasible for the loss (as the duals of a fit are), which the library
    /// checks to the rounding of `T`.
    ///
    /// # Errors
    /// [`crate::Error::InvalidArgument`] for logits that are not finite or have
    /// fewer than one row or two columns (or not `num_classes` columns if
    /// set), `previous` of another shape, not finite or infeasible, `norm2`
    /// not finite and positive, `c`, `norm2` or `c * norm2` not normal in the
    /// data type, and as [`Solver::fit_from`] for the loss and targets;
    /// [`crate::Error::Native`] for a native failure.
    ///
    /// ```
    /// use libsdca::{Matrix, Solver};
    ///
    /// let logits = Matrix::new(&[2.0, -0.5, 1.0], 1, 3)?;
    /// let a = Solver::default().with_c(3.0).coordinate_update(logits, &[2], 1.0, None)?;
    /// assert!(a[2] > 0.0 && a.iter().sum::<f64>().abs() < 1e-12);
    /// # Ok::<(), libsdca::Error>(())
    /// ```
    pub fn coordinate_update<'y, T: Real>(
        &self,
        logits: Matrix<'_, T>,
        targets: impl Into<Targets<'y, T>>,
        norm2: f64,
        previous: Option<&[T]>,
    ) -> Result<Vec<T>> {
        native::handshake()?;
        let execution = native::execution::<T>(&self.settings)?;
        check_logits(logits, "logits")?;
        let (n, classes) = (logits.rows(), logits.cols());
        if let Some(expected) = self.num_classes {
            if classes != expected {
                return invalid(format!(
                    "logits must have num_classes = {expected} columns, not {classes}"
                ));
            }
        }
        self.check_objective(classes)?;
        let targets = targets::native(self.loss, &targets.into(), n, classes, true)?;
        if let Some(previous) = previous {
            if previous.len() != n * classes {
                return invalid(format!(
                    "previous must have the {n}x{classes} entries of logits, not {}",
                    previous.len()
                ));
            }
        }
        let norms = vec![norm2; n];
        let mut out = vec![T::default(); n * classes];
        // SAFETY: the buffers have the sizes of the C ABI documentation, and
        // the targets point into storage that outlives the call.
        call(unsafe {
            ffi::sdca_dual_update(
                &self.loss.abi(self.settings.projection_precision),
                self.c,
                &execution,
                T::DTYPE_CODE,
                n,
                classes,
                void(logits.data()),
                &targets.abi(),
                previous.map_or(ptr::null(), <[T]>::as_ptr).cast(),
                norms.as_ptr(),
                out.as_mut_ptr().cast(),
            )
        })?;
        Ok(out)
    }

    /// Rejects a loss that is not a training objective, and invalid loss
    /// parameters for `classes` classes.
    fn check_objective(&self, classes: usize) -> Result<()> {
        if !self.loss.is_objective() {
            let hint = if self.loss.is_distillation() {
                "; train on target distributions with Loss::CrossEntropy and Targets::Probabilities"
            } else {
                ""
            };
            return invalid(format!(
                "{} is a loss, not a training objective{hint}",
                self.loss.name()
            ));
        }
        self.loss.check(classes)
    }
}

/// The classes implied by the targets: the largest label plus one, or the
/// columns of a matrix.
fn implied_classes<T>(targets: &Targets<'_, T>) -> Result<usize> {
    let largest = match targets {
        Targets::Labels(labels) => labels.iter().max().copied(),
        Targets::Sets(sets) => sets.iter().flatten().max().copied(),
        Targets::SetSlices(sets) => sets.iter().copied().flatten().max().copied(),
        Targets::Probabilities(q) | Targets::Teacher(q) => return Ok(q.cols()),
    };
    match largest {
        None => Ok(0),
        Some(label) => label.checked_add(1).map_or_else(
            || invalid("Dimensions exceed the native BLAS integer limit"),
            Ok,
        ),
    }
}

/// The center (d, classes), row-major, in the column-major layout of the C ABI.
fn column_major<T: Real>(center: &[T], d: usize, classes: usize) -> Result<Vec<T>> {
    if center.len() != d * classes {
        return invalid(format!(
            "Start::center must have the {d}x{classes} entries of (d, num_classes), not {}",
            center.len()
        ));
    }
    Ok((0..d * classes)
        .map(|i| center[(i % d) * classes + i / d])
        .collect())
}
