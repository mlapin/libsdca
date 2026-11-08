//! The shared vocabulary: the same names and ids as the C ABI and Python.
use std::fmt;
use std::str::FromStr;

use crate::error::Error;

macro_rules! named {
    ($(#[$doc:meta])* $name:ident { $($(#[$vdoc:meta])* $variant:ident = $id:literal, $text:literal),+ $(,)? }) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum $name { $($(#[$vdoc])* $variant = $id),+ }

        impl $name {
            /// Every value, in id order.
            pub const ALL: &'static [$name] = &[$($name::$variant),+];

            /// The name shared with the other interfaces, which `str::parse` reads.
            #[must_use]
            pub fn name(self) -> &'static str {
                match self { $($name::$variant => $text),+ }
            }

            #[allow(dead_code)]
            pub(crate) fn id(self) -> i32 {
                self as i32
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.name())
            }
        }
    };
}

macro_rules! parse {
    ($name:ident, $what:literal $(, $alias:literal => $value:expr)*) => {
        impl FromStr for $name {
            type Err = Error;
            fn from_str(text: &str) -> Result<Self, Error> {
                $(if text == $alias { return Ok($value); })*
                $name::ALL.iter().copied().find(|v| v.name() == text).ok_or_else(|| {
                    Error::InvalidArgument(format!(concat!("Unknown ", $what, ": {}"), text))
                })
            }
        }
    };
}

named! {
    /// The working arithmetic of the solver, coordinate steps and losses; see
    /// [`crate::Real::DEFAULT_PRECISION`]. `str::parse` also takes "fp32" and
    /// "single" for Float32, "fp64" and "double" for Float64.
    #[non_exhaustive]
    Precision {
        /// Float working arithmetic on f32 data: faster entropy steps.
        Float32 = 0, "float32",
        /// Double working arithmetic on f32 or f64 data.
        Float64 = 2, "float64",
    }
}
parse!(
    Precision,
    "precision",
    "fp32" => Precision::Float32,
    "single" => Precision::Float32,
    "fp64" => Precision::Float64,
    "double" => Precision::Float64
);

named! {
    /// How the hinge projections solve (multilabel_hinge's two-simplex steps
    /// and smooth loss values; the top-k hinges run the extended solver in
    /// both). See "Projection precision" in docs/API.md.
    #[non_exhaustive]
    ProjectionPrecision {
        /// Double-precision solve with a per-call check; a failed check
        /// re-solves with the extended solver. The default.
        Float64 = 0, "float64",
        /// Always the extended-precision solver.
        Extended = 1, "extended",
    }
}
parse!(ProjectionPrecision, "projection_precision");

named! {
    /// What the epochs of [`crate::fit`] leave out for the hinge objectives; the
    /// entropy objectives update every class of every example whatever the
    /// value, and every evaluation covers every example. See "Shrinking" in
    /// docs/API.md.
    #[non_exhaustive]
    Shrinking {
        /// Both kinds below. The default.
        Auto = 0, "auto",
        /// Every class of every example in every epoch.
        Off = 1, "off",
        /// Between full passes (every tenth epoch), skip the examples whose
        /// last step left their duals unchanged and step an example on its
        /// classes of nonzero dual.
        Heuristic = 2, "heuristic",
        /// Leave out for good the zero duals that the certified gap proves
        /// zero at the optimum (gap-safe screening; single-label hinges on
        /// features).
        Safe = 3, "safe",
        /// Heuristic and safe.
        Both = 4, "both",
    }
}
parse!(Shrinking, "shrinking");

named! {
    /// Training objectives of [`crate::fit`]; the last two take label sets. The
    /// hinges are smoothed by `gamma > 0` (`gamma = 0`: nonsmooth).
    #[non_exhaustive]
    Objective {
        /// Softmax cross-entropy.
        CrossEntropy = 0, "cross_entropy",
        /// Top-k cross-entropy (top-k entropy); cross-entropy at k = 1.
        TopKCrossEntropy = 1, "topk_cross_entropy",
        /// Multiclass hinge of Crammer and Singer (k = 1).
        MulticlassHinge = 2, "multiclass_hinge",
        /// Hinge of the average of the k largest margins.
        TopKHingeAlpha = 3, "topk_hinge_alpha",
        /// Average of the k largest hinges.
        TopKHingeBeta = 4, "topk_hinge_beta",
        /// Multilabel hinge.
        MultilabelHinge = 5, "multilabel_hinge",
        /// Cross-entropy with a uniform target over the labels.
        MultilabelCrossEntropy = 6, "multilabel_cross_entropy",
    }
}
parse!(Objective, "objective", "topk_hinge" => Objective::TopKHingeBeta);

impl Objective {
    /// Alias for [`Objective::TopKHingeBeta`], the default top-k hinge variant.
    #[allow(non_upper_case_globals)]
    pub const TopKHinge: Self = Self::TopKHingeBeta;

    /// Whether the objective takes label sets.
    #[must_use]
    pub fn is_multilabel(self) -> bool {
        self.id() >= Objective::MultilabelHinge.id()
    }
}

named! {
    /// Losses of [`crate::loss_and_grad`]: the objectives and
    /// proximal_cross_entropy.
    #[non_exhaustive]
    Loss {
        /// Softmax cross-entropy (uniform target over the labels of a set).
        CrossEntropy = 0, "cross_entropy",
        /// Top-k cross-entropy (top-k entropy); cross-entropy at k = 1.
        TopKCrossEntropy = 1, "topk_cross_entropy",
        /// Multiclass hinge of Crammer and Singer (k = 1).
        MulticlassHinge = 2, "multiclass_hinge",
        /// Hinge of the average of the k largest margins.
        TopKHingeAlpha = 3, "topk_hinge_alpha",
        /// Average of the k largest hinges.
        TopKHingeBeta = 4, "topk_hinge_beta",
        /// Multilabel hinge.
        MultilabelHinge = 5, "multilabel_hinge",
        /// Cross-entropy with a uniform target over the labels.
        MultilabelCrossEntropy = 6, "multilabel_cross_entropy",
        /// Moreau envelope of the cross-entropy with step alpha.
        ProximalCrossEntropy = 7, "proximal_cross_entropy",
    }
}
parse!(Loss, "loss", "topk_hinge" => Loss::TopKHingeBeta);

impl Loss {
    /// Alias for [`Loss::TopKHingeBeta`], the default top-k hinge variant.
    #[allow(non_upper_case_globals)]
    pub const TopKHinge: Self = Self::TopKHingeBeta;
}

named! {
    /// Solver status; only Solved certifies the duality gap.
    #[non_exhaustive]
    Status {
        /// Not started.
        None = 0, "none",
        /// Running.
        Solving = 1, "solving",
        /// The relative duality gap reached the tolerance.
        Solved = 2, "solved",
        /// The objectives stopped improving.
        NoProgress = 3, "no_progress",
        /// The epoch limit was reached.
        MaxEpoch = 4, "max_epoch",
        /// The CPU time limit was reached (MATLAB only: [`crate::fit`] sets none).
        MaxCpuTime = 5, "max_cpu_time",
        /// The wall time limit was reached (MATLAB only: [`crate::fit`] sets none).
        MaxWallTime = 6, "max_wall_time",
        /// A numerical failure.
        Failed = 7, "failed",
    }
}
parse!(Status, "status");

impl Status {
    pub(crate) fn from_id(id: i64) -> Status {
        Status::ALL
            .iter()
            .copied()
            .find(|s| i64::from(s.id()) == id)
            .unwrap_or(Status::Failed)
    }
}
