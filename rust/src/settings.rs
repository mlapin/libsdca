use std::num::NonZeroUsize;

use crate::names::Precision;

/// The threads of the native row loops: loss evaluations, coordinate steps and
/// the evaluations and checks of [`crate::Solver::fit`] (whose epochs are
/// sequential).
///
/// The results do not depend on the count. A call adds threads only when its
/// work would take at least 1 ms on one thread; see "Threads" in docs/API.md.
///
/// ```
/// use std::num::NonZeroUsize;
/// use libsdca::Threads;
///
/// assert_eq!(Threads::default(), Threads::FromEnv);
/// let serial = Threads::Count(NonZeroUsize::MIN);
/// # let _ = serial;
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Threads {
    /// `SDCA_NUM_THREADS` if set to a count below 4096 (0 is automatic), else
    /// automatic: the CPUs the process may run on (at most `OMP_NUM_THREADS`
    /// if set), split among concurrent calls. The default.
    #[default]
    FromEnv,
    /// At most this many threads; 1 runs on the calling thread.
    Count(NonZeroUsize),
}

impl Threads {
    /// The C ABI num_threads: 0 for the default (`SDCA_NUM_THREADS`, else
    /// automatic), else the count.
    pub(crate) fn code(self) -> usize {
        match self {
            Threads::FromEnv => 0,
            // The library reads SIZE_MAX as its default too.
            Threads::Count(count) => count.get().min(usize::MAX - 1),
        }
    }
}

/// The arithmetic and threads of a call: of [`crate::Loss::loss_and_grad_with`],
/// and through [`crate::Solver::with_settings`] of training and coordinate steps.
///
/// Build it from [`Settings::default`] with the `with_*` setters.
///
/// ```
/// use libsdca::{Precision, Settings, Threads};
///
/// let settings = Settings::default()
///     .with_precision(Precision::Float64)
///     .with_threads(Threads::FromEnv);
/// assert_eq!(settings.precision, Some(Precision::Float64));
/// assert_eq!(Settings::default().precision, None);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Settings {
    /// The working arithmetic; `None` (the default) is
    /// [`crate::Real::DEFAULT_PRECISION`] of the data.
    pub precision: Option<Precision>,
    /// The threads of the native row loops; [`Threads::FromEnv`] by default.
    pub threads: Threads,
    /// Whether [`crate::Loss::loss_and_grad_with`] refuses invalid rows (NaN or
    /// +inf scores, invalid rows of masked classes) with
    /// [`crate::Error::InvalidArgument`]; `false` (the default) gives them NaN
    /// values and gradients.
    pub check_rows: bool,
}

impl Settings {
    /// With the working arithmetic `precision`.
    #[must_use]
    pub const fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = Some(precision);
        self
    }

    /// With the threads `threads`.
    #[must_use]
    pub const fn with_threads(mut self, threads: Threads) -> Self {
        self.threads = threads;
        self
    }

    /// With invalid rows refused (`true`) or given NaN outputs (`false`, the
    /// default); see [`Settings::check_rows`].
    #[must_use]
    pub const fn with_check_rows(mut self, check_rows: bool) -> Self {
        self.check_rows = check_rows;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threads_map_to_the_c_abi_encoding() {
        assert_eq!(Threads::FromEnv.code(), 0);
        assert_eq!(Threads::Count(NonZeroUsize::MIN).code(), 1);
        let eight = NonZeroUsize::new(8).unwrap();
        assert_eq!(Threads::Count(eight).code(), 8);
        assert_eq!(Threads::Count(NonZeroUsize::MAX).code(), usize::MAX - 1);
    }
}
