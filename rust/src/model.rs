use crate::error::{invalid, Result};
use crate::matrix::Matrix;
use crate::names::{Objective, Status};
use crate::real::Real;

/// The history fields, as in the C ABI and Python (accuracy is the rank loss
/// of the multilabel objectives).
pub const HISTORY: &[&str] = &[
    "epoch",
    "primal",
    "dual",
    "primal_loss",
    "dual_loss",
    "regularizer",
    "relative_gap",
    "accuracy",
    "cpu_time",
    "wall_time",
];

/// One value per evaluation of each [`HISTORY`] field.
///
/// ```
/// use libsdca::{fit, FitOptions, Matrix, HISTORY};
///
/// let x = Matrix::new(&[1.0, 0.0, 0.0, 1.0, -1.0, -1.0], 3, 2)?;
/// let model = fit(x, &[0, 1, 2], &FitOptions::default())?;
/// let history = model.history();
/// assert_eq!(history.get("relative_gap").unwrap().len(), history.len());
/// assert_eq!(history.last().unwrap().count(), HISTORY.len());
/// # Ok::<(), libsdca::Error>(())
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct History {
    rows: Vec<[f64; 10]>,
}

impl History {
    pub(crate) fn new(buffer: &[f64], rows: usize) -> History {
        let rows = buffer.chunks_exact(HISTORY.len()).take(rows);
        History {
            rows: rows.map(|row| row.try_into().unwrap()).collect(),
        }
    }

    /// The number of evaluations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the objective was never evaluated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The values of a [`HISTORY`] field ("rank_loss" names "accuracy"); None
    /// for another name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Vec<f64>> {
        let name = if name == "rank_loss" {
            "accuracy"
        } else {
            name
        };
        let j = HISTORY.iter().position(|&field| field == name)?;
        Some(self.rows.iter().map(|row| row[j]).collect())
    }

    /// The last evaluation as (field, value) pairs; None if there is none.
    #[must_use]
    pub fn last(&self) -> Option<impl Iterator<Item = (&'static str, f64)> + '_> {
        self.rows
            .last()
            .map(|row| HISTORY.iter().copied().zip(row.iter().copied()))
    }
}

/// The result of [`crate::fit`]; read-only, so that its shapes stay
/// consistent.
///
/// ```
/// use libsdca::{fit, FitOptions, Matrix, Objective};
///
/// let x = Matrix::new(&[1.0, 0.0, 0.0, 1.0, -1.0, -1.0], 3, 2)?;
/// let model = fit(x, &[0, 1, 2], &FitOptions::default())?;
/// assert_eq!((model.num_classes(), model.objective()), (3, Objective::CrossEntropy));
/// assert_eq!((model.weights().unwrap().len(), model.dual().len()), (6, 9));
/// assert!(!model.kernel() && model.epochs() > 0);
/// # Ok::<(), libsdca::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Model<T> {
    pub(crate) weights: Option<Vec<T>>,
    pub(crate) dual: Vec<T>,
    pub(crate) history: History,
    pub(crate) status: Status,
    pub(crate) epochs: usize,
    pub(crate) kernel: bool,
    pub(crate) objective: Objective,
    pub(crate) num_classes: usize,
}

impl<T: Real> Model<T> {
    /// Primal weights (d, num_classes), row-major; None for kernel input.
    #[must_use]
    pub fn weights(&self) -> Option<&[T]> {
        self.weights.as_deref()
    }

    /// Dual variables A (n, num_classes), row-major.
    #[must_use]
    pub fn dual(&self) -> &[T] {
        &self.dual
    }

    /// The evaluations of the objective.
    #[must_use]
    pub fn history(&self) -> &History {
        &self.history
    }

    /// Why the solver stopped; only Solved certifies the duality gap.
    #[must_use]
    pub fn status(&self) -> Status {
        self.status
    }

    /// The number of epochs run.
    #[must_use]
    pub fn epochs(&self) -> usize {
        self.epochs
    }

    /// Whether the model was trained on a Gram matrix.
    #[must_use]
    pub fn kernel(&self) -> bool {
        self.kernel
    }

    /// The objective trained (cross_entropy on label sets becomes
    /// multilabel_cross_entropy).
    #[must_use]
    pub fn objective(&self) -> Objective {
        self.objective
    }

    /// The number of classes.
    #[must_use]
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Scores (n_test, num_classes) of features (n_test, d), or of the
    /// cross-kernel (n_test, n_train) of a kernel model, accumulated in f64.
    ///
    /// # Errors
    /// [`crate::Error::InvalidArgument`] if the columns of `x` do not match the
    /// model.
    pub fn decision_function(&self, x: Matrix<'_, T>) -> Result<Vec<T>> {
        let classes = self.num_classes;
        // Kernel scores are K A; feature scores X W.
        let right = if self.kernel {
            &self.dual
        } else {
            self.weights.as_ref().expect("weights")
        };
        if x.cols() != right.len() / classes {
            return invalid(format!("x must have {} columns", right.len() / classes));
        }
        let mut scores = vec![T::default(); x.rows() * classes];
        let mut sums = vec![0.0f64; classes];
        for (i, row) in scores.chunks_exact_mut(classes).enumerate() {
            sums.fill(0.0);
            for (l, &v) in x.row(i).iter().enumerate() {
                let vf = v.f64();
                let w_row = &right[l * classes..(l + 1) * classes];
                for (s, &w) in sums.iter_mut().zip(w_row) {
                    *s += vf * w.f64();
                }
            }
            for (score, &s) in row.iter_mut().zip(&sums) {
                *score = T::from_f64(s);
            }
        }
        Ok(scores)
    }

    /// Predicted labels: the first largest score of each row.
    ///
    /// # Errors
    /// As [`Model::decision_function`].
    pub fn predict(&self, x: Matrix<'_, T>) -> Result<Vec<usize>> {
        let scores = self.decision_function(x)?;
        Ok(scores
            .chunks_exact(self.num_classes)
            .map(|row| (0..row.len()).fold(0, |best, j| if row[j] > row[best] { j } else { best }))
            .collect())
    }
}
