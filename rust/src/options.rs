use crate::names::{Loss, Objective, Precision, ProjectionPrecision, Shrinking};

/// Options of [`crate::fit`], with the defaults of every interface.
///
/// ```
/// use libsdca::{FitOptions, Objective};
///
/// let options = FitOptions::<f64> {
///     c: 10.0,
///     objective: Objective::TopKCrossEntropy,
///     k: 2,
///     ..Default::default()
/// };
/// assert_eq!((options.max_epoch, options.tol), (1000, 1e-3));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitOptions<'a, T> {
    /// Loss weight C, finite and positive.
    pub c: f64,
    /// The objective; cross_entropy on label sets trains multilabel_cross_entropy.
    pub objective: Objective,
    /// Top-k parameter in [1, num_classes-1].
    pub k: usize,
    /// Hinge smoothing, finite and >= 0 (0: nonsmooth).
    pub gamma: f64,
    /// The number of classes (>= 2); the largest label plus one if None.
    pub num_classes: Option<usize>,
    /// The epoch limit.
    pub max_epoch: usize,
    /// Epochs between evaluations (>= 1); up to two more fall between where
    /// the gap's trend reaches tol.
    pub eval_epoch: usize,
    /// Relative duality gap of success; 0 disables gap-based success.
    pub tol: f64,
    /// Evaluate the objective before the first epoch.
    pub eval_on_start: bool,
    /// Rebuild W = center + X' A at each evaluation (feature models).
    pub recompute_primal: bool,
    /// How the hinge projections solve; see [`ProjectionPrecision`].
    pub projection_precision: ProjectionPrecision,
    /// Whether x is a Gram matrix (n, n).
    pub kernel: bool,
    /// Center (d, num_classes), row-major as [`crate::Model::weights`], of the
    /// regularizer 0.5 ||W - center||^2 (features only).
    pub center: Option<&'a [T]>,
    /// Warm start A (n, num_classes), usually the dual of an earlier fit; checked
    /// against the feasible set, the label entry of single-label rows rebuilt.
    pub initial_dual: Option<&'a [T]>,
    /// The arithmetic; None for [`crate::Real::DEFAULT_PRECISION`].
    pub precision: Option<Precision>,
    /// What the epochs of the hinge objectives leave out; see [`Shrinking`].
    pub shrinking: Shrinking,
}

impl<T> Default for FitOptions<'_, T> {
    fn default() -> Self {
        FitOptions {
            c: 1.0,
            objective: Objective::CrossEntropy,
            k: 1,
            gamma: 0.0,
            num_classes: None,
            max_epoch: 1000,
            eval_epoch: 10,
            tol: 1e-3,
            eval_on_start: false,
            recompute_primal: true,
            projection_precision: ProjectionPrecision::Float64,
            kernel: false,
            center: None,
            initial_dual: None,
            precision: None,
            shrinking: Shrinking::Auto,
        }
    }
}

/// Options of [`crate::loss_and_grad`].
///
/// ```
/// use libsdca::{Loss, LossOptions};
///
/// let options = LossOptions {
///     loss: Loss::TopKHingeBeta,
///     k: 2,
///     gamma: 0.5,
///     ..Default::default()
/// };
/// assert_eq!(options.alpha, 1.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LossOptions {
    /// The loss.
    pub loss: Loss,
    /// Top-k parameter in [1, classes-1].
    pub k: usize,
    /// Hinge smoothing, finite and >= 0 (0: nonsmooth).
    pub gamma: f64,
    /// Step of proximal_cross_entropy, finite and positive.
    pub alpha: f64,
    /// The arithmetic; None for [`crate::Real::DEFAULT_PRECISION`].
    pub precision: Option<Precision>,
    /// Threads for the rows: None takes `SDCA_NUM_THREADS`, else automatic, as
    /// Some(0) (the available CPUs, as the work allows); Some(1) is serial.
    /// The results do not depend on it.
    pub num_threads: Option<usize>,
    /// How the smooth multilabel_hinge projects; see [`ProjectionPrecision`].
    pub projection_precision: ProjectionPrecision,
}

impl Default for LossOptions {
    fn default() -> Self {
        LossOptions {
            loss: Loss::CrossEntropy,
            k: 1,
            gamma: 0.0,
            alpha: 1.0,
            precision: None,
            num_threads: None,
            projection_precision: ProjectionPrecision::Float64,
        }
    }
}

/// Options of [`crate::coordinate_update`].
///
/// ```
/// use libsdca::{Objective, UpdateOptions};
///
/// let previous = [0.0f32; 6];
/// let options = UpdateOptions {
///     previous: Some(&previous),
///     objective: Objective::TopKHingeAlpha,
///     ..Default::default()
/// };
/// assert_eq!(options.c, 1.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UpdateOptions<'a, T> {
    /// Current dual blocks (n, classes); zeros if None.
    pub previous: Option<&'a [T]>,
    /// Loss weight C, finite and positive.
    pub c: f64,
    /// Squared norm of the example (the same for every row).
    pub norm2: f64,
    /// The objective.
    pub objective: Objective,
    /// Top-k parameter in [1, classes-1].
    pub k: usize,
    /// Hinge smoothing, finite and >= 0 (0: nonsmooth).
    pub gamma: f64,
    /// The arithmetic; None for [`crate::Real::DEFAULT_PRECISION`].
    pub precision: Option<Precision>,
    /// Threads for the rows, as [`LossOptions::num_threads`].
    pub num_threads: Option<usize>,
    /// How the multilabel_hinge steps project; see [`ProjectionPrecision`].
    pub projection_precision: ProjectionPrecision,
}

impl<T> Default for UpdateOptions<'_, T> {
    fn default() -> Self {
        UpdateOptions {
            previous: None,
            c: 1.0,
            norm2: 1.0,
            objective: Objective::CrossEntropy,
            k: 1,
            gamma: 0.0,
            precision: None,
            num_threads: None,
            projection_precision: ProjectionPrecision::Float64,
        }
    }
}

/// Options of [`crate::distillation_loss_and_grad`] and
/// [`crate::proximal_distillation_loss_and_grad`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistillationOptions {
    /// Positive temperature T.
    pub temperature: f64,
    /// Proximal step >= 0 (0: plain distillation, > 0: proximal distillation).
    pub alpha: f64,
    /// The arithmetic; None for [`crate::Real::DEFAULT_PRECISION`].
    pub precision: Option<Precision>,
    /// Threads for the rows, as [`LossOptions::num_threads`].
    pub num_threads: Option<usize>,
}

impl Default for DistillationOptions {
    fn default() -> Self {
        DistillationOptions {
            temperature: 1.0,
            alpha: 0.0,
            precision: None,
            num_threads: None,
        }
    }
}
