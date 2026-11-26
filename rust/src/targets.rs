use std::borrow::Cow;

use crate::error::{invalid, Result};
use crate::ffi;
use crate::matrix::Matrix;
use crate::real::Real;

/// The targets of the examples: labels, label sets, target distributions or
/// teacher logits.
///
/// The methods take `impl Into<Targets>`, so labels and label sets pass as
/// slices, arrays or vectors (of `Vec<usize>` or `&[usize]` sets); the two
/// matrix kinds are written out. Each [`crate::Loss`] takes some kinds:
///
/// | Kind | Losses |
/// | --- | --- |
/// | [`Targets::Labels`] | `CrossEntropy`, `TopKCrossEntropy`, `MulticlassHinge`, `TopKHinge`, `ProximalCrossEntropy` |
/// | label sets | `MultilabelHinge`, `MultilabelCrossEntropy` |
/// | [`Targets::Probabilities`] | `CrossEntropy` (in [`crate::Solver`], the distillation objective) |
/// | [`Targets::Teacher`] | `Distillation`, `ProximalDistillation` in [`crate::Loss::loss_and_grad`] |
///
/// ```
/// use libsdca::{Matrix, Targets};
///
/// assert!(matches!(Targets::<f64>::from(&[0, 2, 1]), Targets::Labels(_)));
/// let sets = vec![vec![0, 1], vec![2]];
/// assert_eq!(Targets::<f64>::from(&sets).len(), 2);
/// let rows: [&[usize]; 2] = [&[0, 1], &[2]];
/// assert_eq!(Targets::<f64>::from(&rows).len(), 2);
/// let q = Matrix::new(&[0.9, 0.1, 0.2, 0.8], 2, 2)?;
/// assert_eq!(Targets::Probabilities(q).len(), 2);
/// # Ok::<(), libsdca::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Targets<'a, T> {
    /// One label in [0, num_classes) per example.
    Labels(&'a [usize]),
    /// One set of distinct labels in [0, num_classes) per example, in any order.
    Sets(&'a [Vec<usize>]),
    /// Label sets as [`Targets::Sets`], borrowed as slices.
    SetSlices(&'a [&'a [usize]]),
    /// Target distributions (n, num_classes): finite, nonnegative rows, such
    /// as the softmax of a teacher; for training they sum to one within
    /// the rounding of a row normalized in `T` (gamma_{m+1}), and the library
    /// renormalizes them.
    Probabilities(Matrix<'a, T>),
    /// Teacher logits of the shape of the logits.
    Teacher(Matrix<'a, T>),
}

impl<'a, T> From<&'a [usize]> for Targets<'a, T> {
    fn from(labels: &'a [usize]) -> Self {
        Targets::Labels(labels)
    }
}

impl<'a, T, const N: usize> From<&'a [usize; N]> for Targets<'a, T> {
    fn from(labels: &'a [usize; N]) -> Self {
        Targets::Labels(labels)
    }
}

impl<'a, T> From<&'a Vec<usize>> for Targets<'a, T> {
    fn from(labels: &'a Vec<usize>) -> Self {
        Targets::Labels(labels)
    }
}

impl<'a, T> From<&'a [Vec<usize>]> for Targets<'a, T> {
    fn from(sets: &'a [Vec<usize>]) -> Self {
        Targets::Sets(sets)
    }
}

impl<'a, T, const N: usize> From<&'a [Vec<usize>; N]> for Targets<'a, T> {
    fn from(sets: &'a [Vec<usize>; N]) -> Self {
        Targets::Sets(sets)
    }
}

impl<'a, T> From<&'a Vec<Vec<usize>>> for Targets<'a, T> {
    fn from(sets: &'a Vec<Vec<usize>>) -> Self {
        Targets::Sets(sets)
    }
}

impl<'a, 'b: 'a, T> From<&'a [&'b [usize]]> for Targets<'a, T> {
    fn from(sets: &'a [&'b [usize]]) -> Self {
        Targets::SetSlices(sets)
    }
}

impl<'a, 'b: 'a, T, const N: usize> From<&'a [&'b [usize]; N]> for Targets<'a, T> {
    fn from(sets: &'a [&'b [usize]; N]) -> Self {
        Targets::SetSlices(sets)
    }
}

impl<'a, 'b: 'a, T> From<&'a Vec<&'b [usize]>> for Targets<'a, T> {
    fn from(sets: &'a Vec<&'b [usize]>) -> Self {
        Targets::SetSlices(sets)
    }
}

impl<T> Targets<'_, T> {
    /// The number of examples.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Targets::Labels(labels) => labels.len(),
            Targets::Sets(sets) => sets.len(),
            Targets::SetSlices(sets) => sets.len(),
            Targets::Probabilities(matrix) | Targets::Teacher(matrix) => matrix.rows(),
        }
    }

    /// Whether there are no examples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The labels of example `i` (one for [`Targets::Labels`]).
    pub(crate) fn set(&self, i: usize) -> &[usize] {
        match self {
            Targets::Labels(labels) => std::slice::from_ref(&labels[i]),
            Targets::Sets(sets) => &sets[i],
            Targets::SetSlices(sets) => sets[i],
            Targets::Probabilities(_) | Targets::Teacher(_) => &[],
        }
    }

    /// One entry per example.
    pub(crate) fn check_count(&self, n: usize) -> Result<()> {
        if self.len() != n {
            return invalid(format!(
                "targets must have one entry per example ({n}), not {}",
                self.len()
            ));
        }
        Ok(())
    }
}

/// Targets in the layout of the C ABI, with the storage the struct points into.
pub(crate) enum Native<'a, T> {
    /// One label per example (n).
    Labels(Cow<'a, [i64]>),
    /// Sorted label sets: their concatenated labels and offsets (n + 1).
    Sets { labels: Vec<i64>, offsets: Vec<i64> },
    /// Target distributions (n, classes), row-major.
    Probabilities(Cow<'a, [f64]>),
    /// Teacher logits (n, classes), row-major.
    Logits(&'a [T]),
}

impl<T> Native<'_, T> {
    /// The struct sdca_targets of the targets, valid while `self` lives.
    pub(crate) fn abi(&self) -> ffi::Targets {
        let empty = ffi::Targets::default();
        match self {
            Native::Labels(labels) => ffi::Targets {
                kind: ffi::TARGET_LABELS,
                labels: labels.as_ptr(),
                ..empty
            },
            Native::Sets { labels, offsets } => ffi::Targets {
                kind: ffi::TARGET_LABEL_SETS,
                labels: labels.as_ptr(),
                offsets: offsets.as_ptr(),
                ..empty
            },
            Native::Probabilities(q) => ffi::Targets {
                kind: ffi::TARGET_PROBABILITIES,
                probabilities: q.as_ptr(),
                ..empty
            },
            Native::Logits(logits) => ffi::Targets {
                kind: ffi::TARGET_LOGITS,
                logits: logits.as_ptr().cast(),
                ..empty
            },
        }
    }
}

/// What the targets go with: the logits of a loss evaluation, whose size checks
/// say what PyTorch says (docs/API.md, Errors: one text per error), or the
/// examples of a fit or a coordinate step.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Rows {
    Logits,
    Examples,
}

/// The targets of `n` examples and `classes` classes in the layout of the C
/// ABI, which checks their values and whether the loss takes them. The sizes
/// are checked here: the library reads as many entries as they imply.
pub(crate) fn native<'a, T: Real>(
    targets: &Targets<'a, T>,
    n: usize,
    classes: usize,
    rows: Rows,
) -> Result<Native<'a, T>> {
    Ok(match *targets {
        Targets::Labels(labels) => {
            check_count(targets, n, rows)?;
            Native::Labels(label_codes(labels))
        }
        Targets::Sets(_) | Targets::SetSlices(_) => {
            check_count(targets, n, rows)?;
            let (labels, offsets) = label_sets(targets);
            Native::Sets { labels, offsets }
        }
        Targets::Probabilities(q) => {
            check_shape(q, n, classes, rows, "Targets::Probabilities")?;
            Native::Probabilities(T::to_f64s(q.data()))
        }
        Targets::Teacher(teacher) => {
            check_shape(teacher, n, classes, rows, "Targets::Teacher")?;
            Native::Logits(teacher.data())
        }
    })
}

/// One label or label set per example.
fn check_count<T>(targets: &Targets<'_, T>, n: usize, rows: Rows) -> Result<()> {
    let len = targets.len();
    if len == n {
        return Ok(());
    }
    if rows == Rows::Logits {
        return invalid(format!(
            "target must have shape ({n},) (the logits' shape without the class dimension), \
             not ({len},)"
        ));
    }
    targets.check_count(n)
}

/// A matrix of targets of the shape of the logits (n, classes).
fn check_shape<T>(
    matrix: Matrix<'_, T>,
    n: usize,
    classes: usize,
    rows: Rows,
    what: &str,
) -> Result<()> {
    let (r, c) = (matrix.rows(), matrix.cols());
    if (r, c) == (n, classes) {
        return Ok(());
    }
    match (rows, what) {
        (Rows::Logits, "Targets::Teacher") => invalid(format!(
            "student and teacher must have one shape, not ({n}, {classes}) and ({r}, {c})"
        )),
        (Rows::Logits, _) => invalid(format!(
            "class probabilities must have the logits' shape ({n}, {classes}), not ({r}, {c})"
        )),
        (Rows::Examples, _) => invalid(format!(
            "{what} must have the shape of the logits ({n}x{classes}), not {r}x{c}"
        )),
    }
}

/// The C ABI's int64 labels: the same memory on 64-bit targets. A label of
/// 2^63 or more reads as a negative one, out of range for the library.
#[cfg(target_pointer_width = "64")]
fn label_codes(labels: &[usize]) -> Cow<'_, [i64]> {
    const _: () = assert!(std::mem::size_of::<usize>() == std::mem::size_of::<i64>());
    // SAFETY: usize and i64 have the same size and alignment on 64-bit targets,
    // and every bit pattern is an i64.
    Cow::Borrowed(unsafe {
        std::slice::from_raw_parts(labels.as_ptr().cast::<i64>(), labels.len())
    })
}

/// The C ABI's int64 labels.
#[cfg(not(target_pointer_width = "64"))]
fn label_codes(labels: &[usize]) -> Cow<'_, [i64]> {
    Cow::Owned(labels.iter().map(|&label| label as i64).collect())
}

/// Concatenated sorted label sets and their offsets (n + 1), as the C ABI
/// takes them; the library checks the labels.
fn label_sets<T>(targets: &Targets<'_, T>) -> (Vec<i64>, Vec<i64>) {
    let n = targets.len();
    let (mut labels, mut offsets) = (Vec::new(), Vec::with_capacity(n + 1));
    offsets.push(0i64);
    for i in 0..n {
        let start = labels.len();
        labels.extend(targets.set(i).iter().map(|&label| label as i64));
        labels[start..].sort_unstable();
        offsets.push(labels.len() as i64);
    }
    (labels, offsets)
}
