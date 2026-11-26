//! The named settings shared with the C ABI and the other interfaces.
use std::fmt;
use std::str::FromStr;

use crate::error::{Error, Result};

macro_rules! named {
    ($(#[$doc:meta])* $name:ident { $($(#[$vdoc:meta])* $variant:ident = $id:literal, $text:literal),+ $(,)? }) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum $name { $($(#[$vdoc])* $variant = $id),+ }

        impl $name {
            /// Every value, in the order of the C ABI codes.
            pub const ALL: &'static [$name] = &[$($name::$variant),+];

            /// The name shared with the other interfaces, which `str::parse` reads.
            #[must_use]
            pub fn name(self) -> &'static str {
                match self { $($name::$variant => $text),+ }
            }

            /// The C ABI code.
            #[allow(dead_code)]
            pub(crate) fn code(self) -> i32 {
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
    ($name:ident, $what:literal) => {
        impl FromStr for $name {
            type Err = Error;

            /// Parses [`Self::name`].
            fn from_str(text: &str) -> Result<Self> {
                $name::ALL
                    .iter()
                    .copied()
                    .find(|v| v.name() == text)
                    .ok_or_else(|| {
                        Error::InvalidArgument(format!(concat!("Unknown ", $what, ": {}"), text))
                    })
            }
        }
    };
}

named! {
    /// The working arithmetic of training, coordinate steps and losses (the
    /// codes of enum sdca_precision).
    ///
    /// `None` in [`crate::Settings::precision`] is [`crate::Real::DEFAULT_PRECISION`]
    /// of the data.
    ///
    /// ```
    /// use libsdca::Precision;
    ///
    /// assert_eq!("float32".parse::<Precision>()?, Precision::Float32);
    /// assert!("double".parse::<Precision>().is_err());
    /// assert_eq!(Precision::Float64.to_string(), "float64");
    /// # Ok::<(), libsdca::Error>(())
    /// ```
    Precision {
        /// Float work on f32 data (f64 data rejects it): faster entropy steps,
        /// accurate to float rounding of the scores.
        Float32 = 1, "float32",
        /// Double work on f32 or f64 data.
        Float64 = 2, "float64",
    }
}
parse!(Precision, "precision");

named! {
    /// The variant of [`crate::Loss::TopKHinge`]; the codes order the C ABI's
    /// SDCA_LOSS_TOPK_HINGE_ALPHA and SDCA_LOSS_TOPK_HINGE_BETA.
    ///
    /// ```
    /// use libsdca::TopKVariant;
    ///
    /// assert_eq!(TopKVariant::default(), TopKVariant::Beta);
    /// assert_eq!("alpha".parse::<TopKVariant>()?, TopKVariant::Alpha);
    /// # Ok::<(), libsdca::Error>(())
    /// ```
    #[derive(Default)]
    TopKVariant {
        /// The hinge of the average of the k largest margins.
        Alpha = 0, "alpha",
        /// The average of the k largest hinges. The default.
        #[default]
        Beta = 1, "beta",
    }
}
parse!(TopKVariant, "variant");

named! {
    /// What the epochs of [`crate::Solver::fit`] leave out for the hinge
    /// losses. The entropy losses update every class of every example and take
    /// `Auto` only (another value is an [`crate::Error::InvalidArgument`]).
    /// Every evaluation covers every example, so [`Status::Solved`] means the
    /// same in every mode. See "Shrinking" in docs/API.md.
    ///
    /// ```
    /// use libsdca::Shrinking;
    ///
    /// assert_eq!(Shrinking::default(), Shrinking::Auto);
    /// assert_eq!("safe".parse::<Shrinking>()?, Shrinking::Safe);
    /// # Ok::<(), libsdca::Error>(())
    /// ```
    #[derive(Default)]
    Shrinking {
        /// The library's choice, currently `Both`; it may change. The default.
        #[default]
        Auto = 0, "auto",
        /// Every class of every example in every epoch.
        Off = 1, "off",
        /// Between full passes (after `eval_epoch` full passes of work, at most
        /// 50 epochs), skip the examples whose last step left their duals
        /// unchanged and step an example on its classes of nonzero dual.
        Heuristic = 2, "heuristic",
        /// Leave out for good the zero duals that the duality gap proves zero
        /// at the optimum (gap-safe screening; single-label hinges on features).
        Safe = 3, "safe",
        /// `Heuristic` and `Safe`.
        Both = 4, "both",
    }
}
parse!(Shrinking, "shrinking");

named! {
    /// Why [`crate::Solver::fit`] stopped; only `Solved` guarantees the duality gap.
    ///
    /// ```
    /// use libsdca::Status;
    ///
    /// assert_eq!("max_epoch".parse::<Status>()?, Status::MaxEpoch);
    /// assert_eq!(Status::Solved.to_string(), "solved");
    /// # Ok::<(), libsdca::Error>(())
    /// ```
    Status {
        /// The relative duality gap reached the tolerance.
        Solved = 0, "solved",
        /// The objectives stopped improving.
        NoProgress = 1, "no_progress",
        /// The epoch limit was reached.
        MaxEpoch = 2, "max_epoch",
        /// A numerical failure: the model holds the last consistent iterate.
        Failed = 3, "failed",
    }
}
parse!(Status, "status");

impl Status {
    /// The status of a C ABI code; a code that a finished fit cannot report
    /// is a native error.
    pub(crate) fn from_code(code: i32) -> Result<Status> {
        Status::ALL
            .iter()
            .copied()
            .find(|status| status.code() == code)
            .ok_or_else(|| Error::Native(format!("The native solver returned status {code}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_codes_map_to_the_finished_states_only() {
        let known = [
            (0, Status::Solved),
            (1, Status::NoProgress),
            (2, Status::MaxEpoch),
            (3, Status::Failed),
        ];
        for (code, status) in known {
            assert_eq!(Status::from_code(code), Ok(status));
        }
        // Codes beyond enum sdca_status.
        for code in [-1, 4, 7] {
            assert!(matches!(Status::from_code(code), Err(Error::Native(_))));
        }
    }
}
