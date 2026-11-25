use std::fmt;
use std::str::FromStr;

use crate::error::{invalid, Error, Result, TWO_CLASSES};
use crate::ffi;
use crate::matrix::Matrix;
use crate::names::TopKVariant;
use crate::native::{self, call, void};
use crate::options::Options;
use crate::real::Real;
use crate::settings::Settings;
use crate::targets::{self, Targets};

/// A loss with its parameters: a training objective of [`crate::Solver`] or
/// a loss of [`Loss::loss_and_grad`].
///
/// The constructors give the default parameters (k = 1 where it is not an
/// argument, [`TopKVariant::Beta`], gamma = 0: the nonsmooth hinges, gamma = 1
/// for the proximal losses, temperature = 1); write other parameters as the
/// variant, as in `Loss::TopKHinge { k: 2, variant: TopKVariant::Alpha,
/// gamma: 0.5 }`. The native library checks the parameters where the loss is
/// used: k in [1, num_classes - 1], gamma finite and >= 0 for the hinges and
/// finite and > 0 for the proximal losses, temperature finite and > 0. The
/// losses are defined in docs/API.md; [`Targets`] lists the targets each takes.
///
/// ```
/// use libsdca::{Loss, TopKVariant};
///
/// let beta = Loss::TopKHinge { k: 2, variant: TopKVariant::Beta, gamma: 0.0 };
/// assert_eq!(Loss::topk_hinge(2), beta);
/// assert_eq!("topk_hinge".parse::<Loss>()?, Loss::topk_hinge(1));
/// let alpha = Loss::TopKHinge { k: 3, variant: TopKVariant::Alpha, gamma: 0.5 };
/// assert_eq!(alpha.name(), "topk_hinge");
/// assert!(Loss::multilabel_hinge().is_multilabel());
/// # Ok::<(), libsdca::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Loss {
    /// Softmax cross-entropy; against [`Targets::Probabilities`], the
    /// distillation objective of [`crate::Solver`].
    CrossEntropy,
    /// Top-k cross-entropy (top-k entropy); cross-entropy at k = 1.
    TopKCrossEntropy {
        /// The k of top-k.
        k: usize,
    },
    /// Multiclass hinge of Crammer and Singer.
    MulticlassHinge {
        /// The smoothing; 0 is the nonsmooth hinge.
        gamma: f64,
    },
    /// Top-k hinge: the average of the k largest hinges
    /// ([`TopKVariant::Beta`]) or the hinge of the average of the k largest
    /// margins ([`TopKVariant::Alpha`]).
    TopKHinge {
        /// The k of top-k.
        k: usize,
        /// The variant.
        variant: TopKVariant,
        /// The smoothing; 0 is the nonsmooth hinge.
        gamma: f64,
    },
    /// Multilabel hinge: the largest non-label score against the smallest
    /// label score.
    MultilabelHinge {
        /// The smoothing; 0 is the nonsmooth hinge.
        gamma: f64,
    },
    /// Cross-entropy with a uniform target over the labels.
    MultilabelCrossEntropy,
    /// Moreau envelope of the cross-entropy; a loss only, not a training
    /// objective.
    ProximalCrossEntropy {
        /// The step of the envelope.
        gamma: f64,
    },
    /// Cross-entropy of the softmax of the logits at a temperature against
    /// that of teacher logits ([`Targets::Teacher`]); a loss only. Train on
    /// target distributions with `CrossEntropy` and [`Targets::Probabilities`].
    Distillation {
        /// The temperature T of both softmaxes.
        temperature: f64,
    },
    /// Moreau envelope of distillation in logits / T, against teacher logits;
    /// a loss only, computed in float64 arithmetic.
    ProximalDistillation {
        /// The temperature T.
        temperature: f64,
        /// The step of the envelope.
        gamma: f64,
    },
}

impl Loss {
    /// Softmax cross-entropy.
    #[must_use]
    pub const fn cross_entropy() -> Self {
        Loss::CrossEntropy
    }

    /// Top-k cross-entropy.
    #[must_use]
    pub const fn topk_cross_entropy(k: usize) -> Self {
        Loss::TopKCrossEntropy { k }
    }

    /// The nonsmooth multiclass hinge.
    #[must_use]
    pub const fn multiclass_hinge() -> Self {
        Loss::MulticlassHinge { gamma: 0.0 }
    }

    /// The nonsmooth top-k hinge, the beta variant.
    #[must_use]
    pub const fn topk_hinge(k: usize) -> Self {
        Loss::TopKHinge {
            k,
            variant: TopKVariant::Beta,
            gamma: 0.0,
        }
    }

    /// The nonsmooth multilabel hinge.
    #[must_use]
    pub const fn multilabel_hinge() -> Self {
        Loss::MultilabelHinge { gamma: 0.0 }
    }

    /// Multilabel cross-entropy.
    #[must_use]
    pub const fn multilabel_cross_entropy() -> Self {
        Loss::MultilabelCrossEntropy
    }

    /// Proximal cross-entropy with gamma = 1.
    #[must_use]
    pub const fn proximal_cross_entropy() -> Self {
        Loss::ProximalCrossEntropy { gamma: 1.0 }
    }

    /// Distillation at temperature 1.
    #[must_use]
    pub const fn distillation() -> Self {
        Loss::Distillation { temperature: 1.0 }
    }

    /// Proximal distillation at temperature 1 with gamma = 1.
    #[must_use]
    pub const fn proximal_distillation() -> Self {
        Loss::ProximalDistillation {
            temperature: 1.0,
            gamma: 1.0,
        }
    }

    /// The name shared with the other interfaces, which `Display` writes and
    /// `str::parse` reads.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Loss::CrossEntropy => "cross_entropy",
            Loss::TopKCrossEntropy { .. } => "topk_cross_entropy",
            Loss::MulticlassHinge { .. } => "multiclass_hinge",
            Loss::TopKHinge { .. } => "topk_hinge",
            Loss::MultilabelHinge { .. } => "multilabel_hinge",
            Loss::MultilabelCrossEntropy => "multilabel_cross_entropy",
            Loss::ProximalCrossEntropy { .. } => "proximal_cross_entropy",
            Loss::Distillation { .. } => "distillation",
            Loss::ProximalDistillation { .. } => "proximal_distillation",
        }
    }

    /// Whether the loss takes label sets (`MultilabelHinge` and
    /// `MultilabelCrossEntropy`) rather than one label per example.
    #[must_use]
    pub const fn is_multilabel(self) -> bool {
        matches!(
            self,
            Loss::MultilabelHinge { .. } | Loss::MultilabelCrossEntropy
        )
    }

    /// The struct sdca_loss of the loss: its own parameters, the defaults of
    /// SDCA_LOSS_INIT(kind) for the others.
    pub(crate) fn abi(self) -> ffi::Loss {
        let kind = match self {
            Loss::CrossEntropy => ffi::LOSS_CROSS_ENTROPY,
            Loss::TopKCrossEntropy { .. } => ffi::LOSS_TOPK_CROSS_ENTROPY,
            Loss::MulticlassHinge { .. } => ffi::LOSS_MULTICLASS_HINGE,
            Loss::TopKHinge {
                variant: TopKVariant::Alpha,
                ..
            } => ffi::LOSS_TOPK_HINGE_ALPHA,
            Loss::TopKHinge { .. } => ffi::LOSS_TOPK_HINGE_BETA,
            Loss::MultilabelHinge { .. } => ffi::LOSS_MULTILABEL_HINGE,
            Loss::MultilabelCrossEntropy => ffi::LOSS_MULTILABEL_CROSS_ENTROPY,
            Loss::ProximalCrossEntropy { .. } => ffi::LOSS_PROXIMAL_CROSS_ENTROPY,
            Loss::Distillation { .. } => ffi::LOSS_DISTILLATION,
            Loss::ProximalDistillation { .. } => ffi::LOSS_PROXIMAL_DISTILLATION,
        };
        let base = ffi::Loss::init(kind);
        match self {
            Loss::CrossEntropy => base,
            Loss::TopKCrossEntropy { k } => ffi::Loss { k, ..base },
            Loss::MulticlassHinge { gamma } => ffi::Loss { gamma, ..base },
            Loss::TopKHinge { k, gamma, .. } => ffi::Loss { k, gamma, ..base },
            Loss::MultilabelHinge { gamma } => ffi::Loss { gamma, ..base },
            Loss::MultilabelCrossEntropy => base,
            Loss::ProximalCrossEntropy { gamma } => ffi::Loss { gamma, ..base },
            Loss::Distillation { temperature } => ffi::Loss {
                temperature,
                ..base
            },
            Loss::ProximalDistillation { temperature, gamma } => ffi::Loss {
                temperature,
                gamma,
                ..base
            },
        }
    }

    /// Per-example losses (n) of the scores `logits` (n, classes) and their
    /// gradients (n, classes) with respect to the scores (a subgradient at the
    /// kinks of the nonsmooth hinges), with the default [`Settings`]: see
    /// [`Loss::loss_and_grad_with`].
    ///
    /// # Errors
    /// As [`Loss::loss_and_grad_with`].
    ///
    /// ```
    /// use libsdca::{Loss, Matrix};
    ///
    /// let logits = Matrix::new(&[2.0, -0.5, 1.0, 0.0, 1.0, 3.0], 2, 3)?;
    /// let (values, gradients, report) = Loss::topk_hinge(2).loss_and_grad(logits, &[0, 2])?;
    /// assert_eq!((values.len(), gradients.len()), (2, 6));
    /// assert!(report.is_valid());
    ///
    /// let teacher = Matrix::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 3)?;
    /// let (values, _, _) = Loss::distillation().loss_and_grad(logits, libsdca::Targets::Teacher(teacher))?;
    /// assert!(values.iter().all(|&v| v > 0.0));
    /// # Ok::<(), libsdca::Error>(())
    /// ```
    pub fn loss_and_grad<'y, T: Real>(
        &self,
        logits: Matrix<'_, T>,
        targets: impl Into<Targets<'y, T>>,
    ) -> Result<(Vec<f64>, Vec<T>, LossReport)> {
        self.loss_and_grad_with(logits, targets, &Settings::default())
    }

    /// Per-example losses (n) of the scores `logits` (n, classes), with
    /// classes >= 2, and their gradients (n, classes) with respect to the
    /// scores, in the arithmetic and on the threads of `settings`.
    ///
    /// The single-label losses take [`Targets::Labels`] (`CrossEntropy` also
    /// [`Targets::Probabilities`], taken as given), the multilabel ones label
    /// sets, and `Distillation` and `ProximalDistillation` take
    /// [`Targets::Teacher`] logits of the shape of `logits` and give the
    /// gradient (q - p) / T with no T^2 factor. `ProximalDistillation`
    /// computes in float64 arithmetic for every data type. The losses carry no
    /// loss weight C.
    ///
    /// A logit of -inf is a masked class, at the limit of the loss: the class
    /// is removed and its gradient is +0.0 (docs/API.md, Masked classes).
    ///
    /// A label out of range is [`crate::Error::InvalidArgument`], as in
    /// PyTorch. An invalid row (a logit of NaN or +inf, an empty label set or
    /// one of every class for the multilabel hinge, a row of masked classes
    /// that is invalid for the loss) gets a NaN value and NaN
    /// gradients; the other rows are as computed alone. The [`LossReport`]
    /// counts the invalid rows. [`Settings::with_check_rows`] refuses such rows
    /// instead.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for logits with fewer than two columns,
    /// targets of a kind the loss does not take or of the wrong size, label
    /// sets with duplicates, invalid loss parameters, a precision the data
    /// type does not take, or with [`Settings::check_rows`] an invalid row
    /// (the native library's message, but for the sizes of the targets);
    /// [`Error::Native`] for a native failure or a library that does not match
    /// the crate.
    ///
    /// ```
    /// use libsdca::{Loss, Matrix, Precision, Settings};
    ///
    /// let logits = Matrix::new(&[2.0f32, -0.5, 1.0, 0.0, 1.0, 3.0], 2, 3)?;
    /// let settings = Settings::default().with_precision(Precision::Float64);
    /// let sets = [vec![0, 1], vec![2]];
    /// let (values, gradients, _) = Loss::multilabel_hinge().loss_and_grad_with(logits, &sets, &settings)?;
    /// assert_eq!((values.len(), gradients.len()), (2, 6));
    /// # Ok::<(), libsdca::Error>(())
    /// ```
    pub fn loss_and_grad_with<'y, T: Real>(
        &self,
        logits: Matrix<'_, T>,
        targets: impl Into<Targets<'y, T>>,
        settings: &Settings,
    ) -> Result<(Vec<f64>, Vec<T>, LossReport)> {
        self.loss_and_grad_with_options(logits, targets, &Options::default(), settings)
    }

    /// [`Loss::loss_and_grad_with`] with the class weights, label smoothing
    /// and ignored label of `options`, as PyTorch's `cross_entropy` takes them.
    ///
    /// # Errors
    /// As [`Loss::loss_and_grad_with`], and [`Error::InvalidArgument`] for an
    /// option the loss does not take, class weights of another length than
    /// the classes or not finite and nonnegative, label smoothing outside
    /// [0, 1], or an ignored label with probabilities.
    pub fn loss_and_grad_with_options<'y, T: Real>(
        &self,
        logits: Matrix<'_, T>,
        targets: impl Into<Targets<'y, T>>,
        options: &Options<'_>,
        settings: &Settings,
    ) -> Result<(Vec<f64>, Vec<T>, LossReport)> {
        native::handshake()?;
        let (n, classes) = (logits.rows(), logits.cols());
        if classes < 2 {
            return invalid(TWO_CLASSES);
        }
        let mut loss = self.abi();
        if let Some(weight) = options.weight() {
            if weight.len() != classes {
                return invalid(format!(
                    "weight must have shape ({classes},), not ({},)",
                    weight.len()
                ));
            }
            loss.weight = weight.as_ptr();
        }
        loss.label_smoothing = options.label_smoothing();
        if let Some(index) = options.ignore_index() {
            // The bits of the C ABI's int64 labels (targets::native).
            loss.ignore = 1;
            loss.ignore_index = index as i64;
        }
        let targets = targets::native(&targets.into(), n, classes, targets::Rows::Logits)?;
        let (mut values, mut gradients) = (vec![0.0; n], vec![T::default(); n * classes]);
        let mut report = ffi::LossReport::default();
        let execution = ffi::Execution {
            report: &mut report,
            ..native::execution(settings)
        };
        // SAFETY: the buffers have the sizes of the C ABI documentation, and
        // the targets, the weights and the report outlive the call.
        let code = unsafe {
            ffi::sdca_loss_evaluate(
                &loss,
                &execution,
                T::DTYPE_CODE,
                n,
                classes,
                void(logits.data()),
                &targets.abi(),
                values.as_mut_ptr(),
                gradients.as_mut_ptr().cast(),
            )
        };
        if code != ffi::INVALID_ROWS {
            call(code)?;
        }
        let report = LossReport {
            invalid_rows: report.invalid_rows,
            first_invalid_row: (report.invalid_rows > 0).then_some(report.first_invalid_row),
        };
        Ok((values, gradients, report))
    }
}

/// The rows of [`Loss::loss_and_grad`] it could not compute, whose values and
/// gradients are NaN (a logit of NaN or +inf, an invalid row of masked
/// classes); the other rows hold their results.
///
/// ```
/// use libsdca::{Loss, Matrix};
///
/// let logits = Matrix::new(&[2.0, -0.5, 1.0, f64::NAN, 1.0, 3.0], 2, 3)?;
/// let (values, _, report) = Loss::cross_entropy().loss_and_grad(logits, &[0, 2])?;
/// assert_eq!((report.invalid_rows, report.first_invalid_row), (1, Some(1)));
/// assert!(values[0].is_finite() && values[1].is_nan());
/// # Ok::<(), libsdca::Error>(())
/// ```
#[must_use = "rows the loss could not compute hold NaN"]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct LossReport {
    /// The number of invalid rows.
    pub invalid_rows: usize,
    /// The first invalid row; `None` if every row is valid.
    pub first_invalid_row: Option<usize>,
}

impl LossReport {
    /// Whether every row is valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.invalid_rows == 0
    }
}

impl fmt::Display for Loss {
    /// Writes [`Loss::name`].
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Loss {
    type Err = Error;

    /// The loss of a [`Loss::name`] with the default parameters; the top-k
    /// names give k = 1 (write the enum variant for another k).
    fn from_str(text: &str) -> Result<Self> {
        Ok(match text {
            "cross_entropy" => Loss::cross_entropy(),
            "topk_cross_entropy" => Loss::topk_cross_entropy(1),
            "multiclass_hinge" => Loss::multiclass_hinge(),
            "topk_hinge" => Loss::topk_hinge(1),
            "multilabel_hinge" => Loss::multilabel_hinge(),
            "multilabel_cross_entropy" => Loss::multilabel_cross_entropy(),
            "proximal_cross_entropy" => Loss::proximal_cross_entropy(),
            "distillation" => Loss::distillation(),
            "proximal_distillation" => Loss::proximal_distillation(),
            _ => return invalid(format!("Unknown loss: {text}")),
        })
    }
}
