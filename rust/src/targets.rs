use std::borrow::Cow;

use crate::error::{invalid, Result};
use crate::ffi;
use crate::loss::Loss;
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
/// | [`Targets::Labels`] | `CrossEntropy`, `TopKCrossEntropy`, `MulticlassHinge`, `TopKHingeAlpha`, `TopKHingeBeta`, `ProximalCrossEntropy` |
/// | label sets | `MultilabelHinge`, `MultilabelCrossEntropy` |
/// | [`Targets::Probabilities`] | `CrossEntropy` in [`crate::Solver`] (the distillation objective) |
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
    /// Target distributions (n, num_classes): finite, nonnegative rows that sum
    /// to one to the rounding of `T`, such as the softmax of a teacher.
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

    /// Whether the targets are label sets.
    pub(crate) fn is_sets(&self) -> bool {
        matches!(self, Targets::Sets(_) | Targets::SetSlices(_))
    }

    /// The kind, as the variant is written.
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Targets::Labels(_) => "Targets::Labels",
            Targets::Sets(_) | Targets::SetSlices(_) => "label sets",
            Targets::Probabilities(_) => "Targets::Probabilities",
            Targets::Teacher(_) => "Targets::Teacher",
        }
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

/// The checked targets of `loss` for `n` examples and `classes` classes, in
/// the layout of the C ABI; `training` also takes target distributions.
pub(crate) fn native<'a, T: Real>(
    loss: Loss,
    targets: &Targets<'a, T>,
    n: usize,
    classes: usize,
    training: bool,
) -> Result<Native<'a, T>> {
    if !matches!(targets, Targets::Teacher(_)) {
        targets.check_count(n)?;
    }
    Ok(match *targets {
        Targets::Labels(labels) if !loss.is_multilabel() && !loss.is_distillation() => {
            check_labels(labels, classes)?;
            Native::Labels(label_codes(labels))
        }
        // The library needs a class outside every label set.
        Targets::Sets(_) | Targets::SetSlices(_) if loss.is_multilabel() => {
            let (labels, offsets) = label_sets(targets, classes, classes - 1)?;
            Native::Sets { labels, offsets }
        }
        Targets::Probabilities(q) if training && loss == Loss::CrossEntropy => {
            Native::Probabilities(distributions(q, classes)?)
        }
        Targets::Probabilities(_) if training => {
            return invalid(format!(
                "{} does not take Targets::Probabilities: target distributions train \
                 Loss::CrossEntropy",
                loss.name()
            ));
        }
        Targets::Teacher(teacher) if loss.is_distillation() => {
            if (teacher.rows(), teacher.cols()) != (n, classes) {
                return invalid(format!(
                    "teacher logits must have the shape of the logits ({n}x{classes}), not {}x{}",
                    teacher.rows(),
                    teacher.cols()
                ));
            }
            Native::Logits(teacher.data())
        }
        _ => return invalid(loss.wrong_targets(targets)),
    })
}

/// Labels in [0, classes).
fn check_labels(labels: &[usize], classes: usize) -> Result<()> {
    if labels.iter().any(|&label| label >= classes) {
        return invalid(format!("labels must lie in [0, {classes})"));
    }
    Ok(())
}

/// The C ABI's int64 labels of labels in [0, classes) (checked): the same
/// memory on 64-bit targets.
#[cfg(target_pointer_width = "64")]
fn label_codes(labels: &[usize]) -> Cow<'_, [i64]> {
    const _: () = assert!(std::mem::size_of::<usize>() == std::mem::size_of::<i64>());
    // SAFETY: usize and i64 have the same size and alignment on 64-bit targets,
    // and every bit pattern is an i64. Labels below 2^63 read as the same
    // values; the callers check them against the number of classes first.
    Cow::Borrowed(unsafe {
        std::slice::from_raw_parts(labels.as_ptr().cast::<i64>(), labels.len())
    })
}

/// The C ABI's int64 labels of labels in [0, classes) (checked).
#[cfg(not(target_pointer_width = "64"))]
fn label_codes(labels: &[usize]) -> Cow<'_, [i64]> {
    Cow::Owned(labels.iter().map(|&label| label as i64).collect())
}

/// Concatenated sorted label sets of between 1 and `largest` labels in [0,
/// classes) each, and their offsets (n + 1), as the C ABI takes them.
fn label_sets<T>(
    targets: &Targets<'_, T>,
    classes: usize,
    largest: usize,
) -> Result<(Vec<i64>, Vec<i64>)> {
    let n = targets.len();
    let (mut labels, mut offsets) = (Vec::new(), Vec::with_capacity(n + 1));
    offsets.push(0i64);
    for i in 0..n {
        let set = targets.set(i);
        let start = labels.len();
        for &label in set {
            if label >= classes {
                return invalid(format!("labels must lie in [0, {classes})"));
            }
            labels.push(label as i64);
        }
        let row = &mut labels[start..];
        row.sort_unstable();
        if row.windows(2).any(|pair| pair[0] == pair[1]) {
            return invalid("label sets must not contain duplicates");
        }
        if row.is_empty() || row.len() > largest {
            return invalid(format!(
                "every label set needs between 1 and {largest} labels"
            ));
        }
        offsets.push(labels.len() as i64);
    }
    Ok((labels, offsets))
}

/// Target distributions (n, classes) as f64. The library checks f64 rows to
/// its double tolerance; f32 rows sum to one only to f32 rounding, so they are
/// checked here to that of f32 and renormalized in f64.
fn distributions<'a, T: Real>(q: Matrix<'a, T>, classes: usize) -> Result<Cow<'a, [f64]>> {
    if q.cols() != classes {
        return invalid(format!(
            "Targets::Probabilities must have num_classes = {classes} columns, not {}",
            q.cols()
        ));
    }
    let mut q = T::to_f64s(q.data());
    if let Cow::Owned(rows) = &mut q {
        let tol = 64.0 * T::EPSILON * classes as f64;
        for row in rows.chunks_exact_mut(classes) {
            if !row.iter().all(|&p| p.is_finite() && p >= 0.0) {
                return invalid("Targets::Probabilities must be finite and nonnegative");
            }
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > tol {
                return invalid("Targets::Probabilities rows must sum to one");
            }
            row.iter_mut().for_each(|p| *p /= sum);
        }
    }
    Ok(q)
}
