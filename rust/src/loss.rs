use std::fmt;
use std::str::FromStr;

use crate::error::{invalid, Error, Result};
use crate::ffi;
use crate::matrix::Matrix;
use crate::names::{Precision, ProjectionPrecision};
use crate::native::{self, call, check_logits, void};
use crate::real::Real;
use crate::settings::Settings;
use crate::targets::{self, Targets};

/// A loss with its parameters: a training objective of [`crate::Solver`] or
/// a loss of [`Loss::loss_and_grad`].
///
/// The constructors give the default parameters (k = 1 where it is not an
/// argument, gamma = 0: the nonsmooth hinges, alpha = 1, temperature = 1);
/// write other parameters as the variant, as in
/// `Loss::TopKHingeBeta { k: 2, gamma: 0.5 }`. The parameters are checked
/// where the loss is used: k in [1, num_classes - 1], gamma finite and >= 0,
/// alpha and temperature finite and > 0. The losses are defined in
/// docs/API.md; [`Targets`] lists the targets each takes.
///
/// ```
/// use libsdca::Loss;
///
/// assert_eq!(Loss::topk_hinge(2), Loss::TopKHingeBeta { k: 2, gamma: 0.0 });
/// assert_eq!("topk_hinge_beta".parse::<Loss>()?, Loss::topk_hinge(1));
/// assert_eq!(Loss::TopKHingeAlpha { k: 3, gamma: 0.5 }.name(), "topk_hinge_alpha");
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
    /// Hinge of the average of the k largest margins.
    TopKHingeAlpha {
        /// The k of top-k.
        k: usize,
        /// The smoothing; 0 is the nonsmooth hinge.
        gamma: f64,
    },
    /// Average of the k largest hinges: the top-k hinge of [`Loss::topk_hinge`].
    TopKHingeBeta {
        /// The k of top-k.
        k: usize,
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
        alpha: f64,
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
        alpha: f64,
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

    /// The nonsmooth top-k hinge, the beta variant: as [`Loss::topk_hinge_beta`].
    #[must_use]
    pub const fn topk_hinge(k: usize) -> Self {
        Loss::topk_hinge_beta(k)
    }

    /// The nonsmooth top-k hinge, alpha variant.
    #[must_use]
    pub const fn topk_hinge_alpha(k: usize) -> Self {
        Loss::TopKHingeAlpha { k, gamma: 0.0 }
    }

    /// The nonsmooth top-k hinge, beta variant.
    #[must_use]
    pub const fn topk_hinge_beta(k: usize) -> Self {
        Loss::TopKHingeBeta { k, gamma: 0.0 }
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

    /// Proximal cross-entropy with alpha = 1.
    #[must_use]
    pub const fn proximal_cross_entropy() -> Self {
        Loss::ProximalCrossEntropy { alpha: 1.0 }
    }

    /// Distillation at temperature 1.
    #[must_use]
    pub const fn distillation() -> Self {
        Loss::Distillation { temperature: 1.0 }
    }

    /// Proximal distillation at temperature 1 with alpha = 1.
    #[must_use]
    pub const fn proximal_distillation() -> Self {
        Loss::ProximalDistillation {
            temperature: 1.0,
            alpha: 1.0,
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
            Loss::TopKHingeAlpha { .. } => "topk_hinge_alpha",
            Loss::TopKHingeBeta { .. } => "topk_hinge_beta",
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

    /// Whether [`crate::Solver`] trains the loss.
    pub(crate) const fn is_objective(self) -> bool {
        !matches!(
            self,
            Loss::ProximalCrossEntropy { .. }
                | Loss::Distillation { .. }
                | Loss::ProximalDistillation { .. }
        )
    }

    /// The message of targets of a kind the loss does not take.
    pub(crate) fn wrong_targets<T>(self, targets: &Targets<'_, T>) -> String {
        let (name, kind) = (self.name(), targets.kind());
        match self {
            Loss::Distillation { .. } | Loss::ProximalDistillation { .. } => {
                format!("{name} takes teacher logits (Targets::Teacher), not {kind}")
            }
            Loss::MultilabelHinge { .. } | Loss::MultilabelCrossEntropy => {
                format!("{name} takes label sets, not {kind}")
            }
            Loss::CrossEntropy if targets.is_sets() => format!(
                "{name} takes one label per example (Targets::Labels), not label sets; \
                 use Loss::MultilabelCrossEntropy for label sets"
            ),
            _ => format!("{name} takes one label per example (Targets::Labels), not {kind}"),
        }
    }

    /// Whether the loss takes teacher logits.
    pub(crate) const fn is_distillation(self) -> bool {
        matches!(
            self,
            Loss::Distillation { .. } | Loss::ProximalDistillation { .. }
        )
    }

    /// The struct sdca_loss of the loss: its own parameters, the defaults of
    /// SDCA_LOSS_INIT for the others; `projection` for multilabel_hinge only.
    pub(crate) fn abi(self, projection: ProjectionPrecision) -> ffi::Loss {
        let base = ffi::Loss::default();
        match self {
            Loss::CrossEntropy => ffi::Loss { kind: 0, ..base },
            Loss::TopKCrossEntropy { k } => ffi::Loss { kind: 1, k, ..base },
            Loss::MulticlassHinge { gamma } => ffi::Loss {
                kind: 2,
                gamma,
                ..base
            },
            Loss::TopKHingeAlpha { k, gamma } => ffi::Loss {
                kind: 3,
                k,
                gamma,
                ..base
            },
            Loss::TopKHingeBeta { k, gamma } => ffi::Loss {
                kind: 4,
                k,
                gamma,
                ..base
            },
            Loss::MultilabelHinge { gamma } => ffi::Loss {
                kind: 5,
                gamma,
                projection_precision: projection.code(),
                ..base
            },
            Loss::MultilabelCrossEntropy => ffi::Loss { kind: 6, ..base },
            Loss::ProximalCrossEntropy { alpha } => ffi::Loss {
                kind: 7,
                alpha,
                ..base
            },
            Loss::Distillation { temperature } => ffi::Loss {
                kind: 8,
                temperature,
                ..base
            },
            Loss::ProximalDistillation { temperature, alpha } => ffi::Loss {
                kind: 9,
                temperature,
                alpha,
                ..base
            },
        }
    }

    /// Checks the parameters for `classes` classes.
    pub(crate) fn check(self, classes: usize) -> Result<()> {
        let name = self.name();
        let k = match self {
            Loss::TopKCrossEntropy { k }
            | Loss::TopKHingeAlpha { k, .. }
            | Loss::TopKHingeBeta { k, .. } => k,
            _ => 1,
        };
        if k < 1 || k >= classes {
            return invalid(format!(
                "{name}: k must be in [1, {}] for {classes} classes, not {k}",
                classes.saturating_sub(1)
            ));
        }
        match self {
            Loss::MulticlassHinge { gamma }
            | Loss::TopKHingeAlpha { gamma, .. }
            | Loss::TopKHingeBeta { gamma, .. }
            | Loss::MultilabelHinge { gamma } => {
                if !(gamma.is_finite() && gamma >= 0.0) {
                    return invalid(format!("{name}: gamma must be finite and nonnegative"));
                }
            }
            Loss::ProximalCrossEntropy { alpha } => {
                native::positive(alpha, &format!("{name}: alpha"))?;
            }
            Loss::Distillation { temperature } => {
                native::positive(temperature, &format!("{name}: temperature"))?;
            }
            Loss::ProximalDistillation { temperature, alpha } => {
                native::positive(temperature, &format!("{name}: temperature"))?;
                native::positive(alpha, &format!("{name}: alpha"))?;
            }
            Loss::CrossEntropy | Loss::TopKCrossEntropy { .. } | Loss::MultilabelCrossEntropy => {}
        }
        Ok(())
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
    /// let (values, gradients) = Loss::topk_hinge(2).loss_and_grad(logits, &[0, 2])?;
    /// assert_eq!((values.len(), gradients.len()), (2, 6));
    ///
    /// let teacher = Matrix::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 3)?;
    /// let (values, _) = Loss::distillation().loss_and_grad(logits, libsdca::Targets::Teacher(teacher))?;
    /// assert!(values.iter().all(|&v| v > 0.0));
    /// # Ok::<(), libsdca::Error>(())
    /// ```
    pub fn loss_and_grad<'y, T: Real>(
        &self,
        logits: Matrix<'_, T>,
        targets: impl Into<Targets<'y, T>>,
    ) -> Result<(Vec<f64>, Vec<T>)> {
        self.loss_and_grad_with(logits, targets, &Settings::default())
    }

    /// Per-example losses (n) of the scores `logits` (n, classes), with
    /// classes >= 2, and their gradients (n, classes) with respect to the
    /// scores, in the arithmetic and on the threads of `settings`.
    ///
    /// The single-label losses take [`Targets::Labels`], the multilabel ones
    /// label sets, and `Distillation` and `ProximalDistillation` take
    /// [`Targets::Teacher`] logits of the shape of `logits` and give the
    /// gradient (q - p) / T with no T^2 factor. `ProximalDistillation`
    /// computes in float64 arithmetic for every data type, and so rejects
    /// `Precision::Float32`. The losses carry no loss weight C.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for logits that are not finite or have fewer
    /// than one row or two columns, targets of a kind the loss does not take
    /// or of the wrong size, labels out of range, invalid loss parameters, or
    /// a precision the data type or loss does not take; [`Error::Native`] for
    /// a native failure or a library that does not match the crate.
    ///
    /// ```
    /// use libsdca::{Loss, Matrix, Precision, Settings};
    ///
    /// let logits = Matrix::new(&[2.0f32, -0.5, 1.0, 0.0, 1.0, 3.0], 2, 3)?;
    /// let settings = Settings::default().with_precision(Precision::Float64);
    /// let sets = [vec![0, 1], vec![2]];
    /// let (values, gradients) = Loss::multilabel_hinge().loss_and_grad_with(logits, &sets, &settings)?;
    /// assert_eq!((values.len(), gradients.len()), (2, 6));
    /// # Ok::<(), libsdca::Error>(())
    /// ```
    pub fn loss_and_grad_with<'y, T: Real>(
        &self,
        logits: Matrix<'_, T>,
        targets: impl Into<Targets<'y, T>>,
        settings: &Settings,
    ) -> Result<(Vec<f64>, Vec<T>)> {
        native::handshake()?;
        let loss = *self;
        if matches!(loss, Loss::ProximalDistillation { .. })
            && settings.precision == Some(Precision::Float32)
        {
            return invalid(
                "proximal_distillation computes in float64 arithmetic: Precision::Float32 is not supported",
            );
        }
        let execution = native::execution::<T>(settings)?;
        check_logits(logits, "logits")?;
        let (n, classes) = (logits.rows(), logits.cols());
        loss.check(classes)?;
        let targets = targets::native(loss, &targets.into(), n, classes, false)?;
        let (mut values, mut gradients) = (vec![0.0; n], vec![T::default(); n * classes]);
        // SAFETY: the buffers have the sizes of the C ABI documentation, and
        // the targets point into storage that outlives the call.
        call(unsafe {
            ffi::sdca_loss_evaluate(
                &loss.abi(settings.projection_precision),
                &execution,
                T::DTYPE_CODE,
                n,
                classes,
                void(logits.data()),
                &targets.abi(),
                values.as_mut_ptr(),
                gradients.as_mut_ptr().cast(),
            )
        })?;
        Ok((values, gradients))
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
    /// names give k = 1 (write the variant for another k).
    fn from_str(text: &str) -> Result<Self> {
        Ok(match text {
            "cross_entropy" => Loss::cross_entropy(),
            "topk_cross_entropy" => Loss::topk_cross_entropy(1),
            "multiclass_hinge" => Loss::multiclass_hinge(),
            "topk_hinge_alpha" => Loss::topk_hinge_alpha(1),
            "topk_hinge_beta" => Loss::topk_hinge_beta(1),
            "multilabel_hinge" => Loss::multilabel_hinge(),
            "multilabel_cross_entropy" => Loss::multilabel_cross_entropy(),
            "proximal_cross_entropy" => Loss::proximal_cross_entropy(),
            "distillation" => Loss::distillation(),
            "proximal_distillation" => Loss::proximal_distillation(),
            _ => return invalid(format!("Unknown loss: {text}")),
        })
    }
}
