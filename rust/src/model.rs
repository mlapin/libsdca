use crate::error::{invalid, Result};
use crate::ffi::HISTORY_FIELDS;
use crate::loss::Loss;
use crate::matrix::Matrix;
use crate::names::Status;
use crate::real::Real;

/// One evaluation of the objective during [`crate::Solver::fit`].
///
/// The primal objective is `0.5 ||W - center||^2 + c sum_i loss_i` at the
/// weights W; the dual is evaluated at the duals A.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct Evaluation {
    /// The epochs run before the evaluation.
    pub epoch: usize,
    /// The primal objective: `regularizer + primal_loss`.
    pub primal: f64,
    /// The dual objective.
    pub dual: f64,
    /// `c` times the sum of the losses.
    pub primal_loss: f64,
    /// The loss term of the dual objective.
    pub dual_loss: f64,
    /// `0.5 ||W - center||^2` (center 0 if none).
    pub regularizer: f64,
    /// `(primal - dual) / max(|primal|, |dual|)`.
    pub relative_gap: f64,
    /// The fraction of training examples whose largest score is their label
    /// (top-1 accuracy); for the multilabel losses, the rank loss instead.
    pub accuracy: f64,
    /// CPU seconds of training and evaluation so far.
    pub cpu_time: f64,
    /// Wall-clock seconds of training and evaluation so far.
    pub wall_time: f64,
}

impl Evaluation {
    /// An evaluation of a row of the C ABI history.
    pub(crate) fn from_row(row: &[f64]) -> Evaluation {
        Evaluation {
            epoch: row[0] as usize,
            primal: row[1],
            dual: row[2],
            primal_loss: row[3],
            dual_loss: row[4],
            regularizer: row[5],
            relative_gap: row[6],
            accuracy: row[7],
            cpu_time: row[8],
            wall_time: row[9],
        }
    }

    /// The evaluations of the first `rows` rows of a C ABI history.
    pub(crate) fn history(buffer: &[f64], rows: usize) -> Vec<Evaluation> {
        buffer
            .chunks_exact(HISTORY_FIELDS)
            .take(rows)
            .map(Evaluation::from_row)
            .collect()
    }
}

/// A trained model, the result of [`crate::Solver::fit`]; read-only, so that
/// its shapes stay consistent.
///
/// ```
/// use libsdca::{Loss, Matrix, Solver};
///
/// let x = Matrix::new(&[1.0, 0.0, 0.0, 1.0, -1.0, -1.0], 3, 2)?;
/// let model = Solver::new(Loss::topk_hinge(2)).fit(x, &[0, 1, 2])?;
/// assert_eq!(model.loss(), Loss::TopKHingeBeta { k: 2, gamma: 0.0 });
/// assert_eq!((model.num_classes(), model.num_features()), (3, Some(2)));
/// assert_eq!((model.weights().map(<[f64]>::len), model.dual().len()), (Some(6), 9));
/// let last = model.history().last().expect("an evaluation at the end");
/// assert!(last.relative_gap <= 1e-3);
/// # Ok::<(), libsdca::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Model<T> {
    pub(crate) weights: Option<Vec<T>>,
    pub(crate) dual: Vec<T>,
    pub(crate) history: Vec<Evaluation>,
    pub(crate) status: Status,
    pub(crate) epochs: usize,
    pub(crate) loss: Loss,
    pub(crate) c: f64,
    pub(crate) num_classes: usize,
    pub(crate) probabilities: bool,
}

impl<T: Real> Model<T> {
    /// The weights W (d, num_classes), row-major; `None` for a kernel model.
    #[must_use]
    pub fn weights(&self) -> Option<&[T]> {
        self.weights.as_deref()
    }

    /// The dual variables A (n, num_classes), row-major; W = center + X' A.
    #[must_use]
    pub fn dual(&self) -> &[T] {
        &self.dual
    }

    /// The evaluations of the objective, in order; the last one decided the
    /// status.
    #[must_use]
    pub fn history(&self) -> &[Evaluation] {
        &self.history
    }

    /// Why the solver stopped; only [`Status::Solved`] guarantees the duality gap.
    #[must_use]
    pub fn status(&self) -> Status {
        self.status
    }

    /// The number of epochs run.
    #[must_use]
    pub fn epochs(&self) -> usize {
        self.epochs
    }

    /// The loss trained, with its parameters: the loss of the solver.
    #[must_use]
    pub fn loss(&self) -> Loss {
        self.loss
    }

    /// The loss weight C.
    #[must_use]
    pub fn c(&self) -> f64 {
        self.c
    }

    /// Whether the model was trained on [`crate::Input::Kernel`].
    #[must_use]
    pub fn is_kernel(&self) -> bool {
        self.weights.is_none()
    }

    /// Whether the model was trained on [`crate::Targets::Probabilities`] (the
    /// distillation objective); its [`Model::loss`] is `CrossEntropy`.
    #[must_use]
    pub fn has_probability_targets(&self) -> bool {
        self.probabilities
    }

    /// The number of classes.
    #[must_use]
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// The number of features d; `None` for a kernel model.
    #[must_use]
    pub fn num_features(&self) -> Option<usize> {
        self.weights
            .as_ref()
            .map(|weights| weights.len() / self.num_classes)
    }

    /// Scores (n_test, num_classes), accumulated in f64, of features
    /// (n_test, d), or for a kernel model of the kernel (n_test, n_train)
    /// between the test and the training examples.
    ///
    /// # Errors
    /// [`crate::Error::InvalidArgument`] if the columns of `input` do not match
    /// the model.
    ///
    /// ```
    /// use libsdca::{Matrix, Solver};
    ///
    /// let x = Matrix::new(&[1.0, 0.0, 0.0, 1.0, -1.0, -1.0], 3, 2)?;
    /// let model = Solver::default().fit(x, &[0, 1, 2])?;
    /// assert_eq!(model.decision_function(x)?.len(), 9);
    /// # Ok::<(), libsdca::Error>(())
    /// ```
    pub fn decision_function(&self, input: Matrix<'_, T>) -> Result<Vec<T>> {
        let classes = self.num_classes;
        // Feature scores X W, kernel scores K A.
        let right = self.weights.as_deref().unwrap_or(&self.dual);
        let inner = right.len() / classes;
        if input.cols() != inner {
            let what = if self.is_kernel() {
                "training examples"
            } else {
                "features"
            };
            return invalid(format!(
                "input must have {inner} columns (the {what} of the model), not {}",
                input.cols()
            ));
        }
        let mut scores = vec![T::default(); input.rows() * classes];
        let mut sums = vec![0.0f64; classes];
        for (i, row) in scores.chunks_exact_mut(classes).enumerate() {
            sums.fill(0.0);
            for (&v, w_row) in input.row(i).iter().zip(right.chunks_exact(classes)) {
                let v = v.f64();
                for (s, &w) in sums.iter_mut().zip(w_row) {
                    *s += v * w.f64();
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
    pub fn predict(&self, input: Matrix<'_, T>) -> Result<Vec<usize>> {
        let scores = self.decision_function(input)?;
        Ok(scores
            .chunks_exact(self.num_classes)
            .map(|row| (0..row.len()).fold(0, |best, j| if row[j] > row[best] { j } else { best }))
            .collect())
    }
}
