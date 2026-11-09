use crate::error::{invalid, Result};

/// One label per example, or one label set per example (multilabel).
///
/// The entry points take `impl Into<Targets>`: label slices, arrays and
/// vectors, or slices, arrays and vectors of label sets (`Vec<usize>` or
/// `&[usize]`).
///
/// ```
/// use libsdca::Targets;
///
/// assert!(matches!(Targets::from(&[0, 2, 1]), Targets::Labels(_)));
/// let sets = vec![vec![0, 1], vec![2]];
/// assert_eq!(Targets::from(&sets).len(), 2);
/// let rows: [&[usize]; 2] = [&[0, 1], &[2]];
/// assert_eq!(Targets::from(&rows).len(), 2);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Targets<'a> {
    /// Labels in [0, num_classes).
    Labels(&'a [usize]),
    /// Label sets of between 1 and num_classes-1 distinct labels.
    Sets(&'a [Vec<usize>]),
    /// Label sets as [`Targets::Sets`], borrowed as slices.
    SetSlices(&'a [&'a [usize]]),
}

impl<'a> From<&'a [usize]> for Targets<'a> {
    fn from(labels: &'a [usize]) -> Self {
        Targets::Labels(labels)
    }
}

impl<'a, const N: usize> From<&'a [usize; N]> for Targets<'a> {
    fn from(labels: &'a [usize; N]) -> Self {
        Targets::Labels(labels)
    }
}

impl<'a> From<&'a Vec<usize>> for Targets<'a> {
    fn from(labels: &'a Vec<usize>) -> Self {
        Targets::Labels(labels)
    }
}

impl<'a> From<&'a [Vec<usize>]> for Targets<'a> {
    fn from(sets: &'a [Vec<usize>]) -> Self {
        Targets::Sets(sets)
    }
}

impl<'a, const N: usize> From<&'a [Vec<usize>; N]> for Targets<'a> {
    fn from(sets: &'a [Vec<usize>; N]) -> Self {
        Targets::Sets(sets)
    }
}

impl<'a> From<&'a Vec<Vec<usize>>> for Targets<'a> {
    fn from(sets: &'a Vec<Vec<usize>>) -> Self {
        Targets::Sets(sets)
    }
}

impl<'a, 'b: 'a> From<&'a [&'b [usize]]> for Targets<'a> {
    fn from(sets: &'a [&'b [usize]]) -> Self {
        Targets::SetSlices(sets)
    }
}

impl<'a, 'b: 'a, const N: usize> From<&'a [&'b [usize]; N]> for Targets<'a> {
    fn from(sets: &'a [&'b [usize]; N]) -> Self {
        Targets::SetSlices(sets)
    }
}

impl<'a, 'b: 'a> From<&'a Vec<&'b [usize]>> for Targets<'a> {
    fn from(sets: &'a Vec<&'b [usize]>) -> Self {
        Targets::SetSlices(sets)
    }
}

impl Targets<'_> {
    /// The number of examples.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Targets::Labels(labels) => labels.len(),
            Targets::Sets(sets) => sets.len(),
            Targets::SetSlices(sets) => sets.len(),
        }
    }

    /// Whether there are no examples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the targets are label sets.
    pub(crate) fn is_sets(&self) -> bool {
        !matches!(self, Targets::Labels(_))
    }

    /// The labels of example `i`.
    fn row(&self, i: usize) -> &[usize] {
        match self {
            Targets::Labels(labels) => std::slice::from_ref(&labels[i]),
            Targets::Sets(sets) => &sets[i],
            Targets::SetSlices(sets) => sets[i],
        }
    }

    /// The largest label plus one.
    pub(crate) fn implied_classes(&self) -> Result<usize> {
        let largest = (0..self.len()).flat_map(|i| self.row(i)).max().copied();
        let Some(label) = largest else {
            return Ok(0);
        };
        label.checked_add(1).map_or_else(
            || invalid("Dimensions exceed the native BLAS integer limit"),
            Ok,
        )
    }

    fn check_count(&self, n: usize) -> Result<()> {
        if self.len() != n {
            return invalid(format!("y must have one target per example ({n})"));
        }
        Ok(())
    }

    /// One label per example as int64.
    pub(crate) fn labels(&self, n: usize, classes: usize) -> Result<Vec<i64>> {
        let Targets::Labels(labels) = self else {
            return invalid("This objective takes one integer label per example");
        };
        self.check_count(n)?;
        if labels.iter().any(|&label| label >= classes) {
            return invalid("Labels must lie in [0, num_classes)");
        }
        Ok(labels.iter().map(|&label| label as i64).collect())
    }

    /// Concatenated sorted label sets (single labels are sets of one) of
    /// between 1 and `largest` labels, and their offsets (n + 1).
    pub(crate) fn label_sets(
        &self,
        n: usize,
        classes: usize,
        largest: usize,
    ) -> Result<(Vec<i64>, Vec<i64>)> {
        self.check_count(n)?;
        let (mut labels, mut offsets) = (Vec::new(), vec![0i64]);
        for i in 0..n {
            let mut row = self.row(i).to_vec();
            row.sort_unstable();
            if row.iter().any(|&label| label >= classes) {
                return invalid("Labels must lie in [0, num_classes)");
            }
            if row.windows(2).any(|pair| pair[0] == pair[1]) {
                return invalid("Label sets must not contain duplicates");
            }
            if row.is_empty() || row.len() > largest {
                return invalid(format!("Each example needs between 1 and {largest} labels"));
            }
            labels.extend(row.iter().map(|&label| label as i64));
            offsets.push(labels.len() as i64);
        }
        Ok((labels, offsets))
    }

    /// The (n, classes) 0/1 mask of the labels of a loss, which may mark
    /// every class (cross-entropy with a uniform target).
    pub(crate) fn positive(&self, n: usize, classes: usize) -> Result<Vec<u8>> {
        let (labels, offsets) = self.label_sets(n, classes, classes)?;
        let mut mask = vec![0u8; n * classes];
        for i in 0..n {
            for &label in &labels[offsets[i] as usize..offsets[i + 1] as usize] {
                mask[i * classes + label as usize] = 1;
            }
        }
        Ok(mask)
    }

    /// The (n, classes) mask of a single-label or label-set target.
    pub(crate) fn is_positive(labels: &[i64], offsets: Option<&[i64]>, i: usize, j: usize) -> bool {
        match offsets {
            None => labels[i] as usize == j,
            Some(o) => labels[o[i] as usize..o[i + 1] as usize].contains(&(j as i64)),
        }
    }
}
