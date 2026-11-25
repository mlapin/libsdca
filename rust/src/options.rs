//! The options of a loss evaluation beside a loss's parameters.

/// The options of PyTorch's `cross_entropy` for [`crate::Loss::loss_and_grad_with_options`]
/// (docs/API.md, Options): class weights and an ignored label for the single-label losses,
/// label smoothing for `CrossEntropy`. A loss that does not take an option it is given is
/// an error, as in every interface.
///
/// A row of label y has the loss of the target weights
/// u_j = w_j ((1 - eps) [j = y] + eps / classes) for `CrossEntropy` (with
/// [`crate::Targets::Probabilities`] p: u_j = w_j ((1 - eps) p_j + eps / classes)), and w_y times
/// its loss for the other single-label losses; a row of the ignored label has the loss 0 and a
/// gradient of +0.0.
///
/// ```
/// use libsdca::{Loss, Matrix, Options, Settings};
///
/// let logits = Matrix::new(&[2.0, -0.5, 1.0, 0.0, 1.0, 3.0], 2, 3)?;
/// let weight = [1.0, 2.0, 0.5];
/// let options = Options::default().with_weight(&weight).with_label_smoothing(0.1);
/// let (values, _, _) = Loss::cross_entropy().loss_and_grad_with_options(
///     logits,
///     &[0, 2],
///     &options,
///     &Settings::default(),
/// )?;
/// assert_eq!(values.len(), 2);
/// # Ok::<(), libsdca::Error>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Options<'a> {
    weight: Option<&'a [f64]>,
    label_smoothing: f64,
    ignore_index: Option<usize>,
}

impl<'a> Options<'a> {
    /// Class weights (classes), finite and nonnegative.
    #[must_use]
    pub fn with_weight(self, weight: &'a [f64]) -> Self {
        Self {
            weight: Some(weight),
            ..self
        }
    }

    /// Label smoothing eps in [0, 1].
    #[must_use]
    pub fn with_label_smoothing(self, label_smoothing: f64) -> Self {
        Self {
            label_smoothing,
            ..self
        }
    }

    /// The label whose rows are ignored.
    #[must_use]
    pub fn with_ignore_index(self, ignore_index: usize) -> Self {
        Self {
            ignore_index: Some(ignore_index),
            ..self
        }
    }

    /// The class weights, if any.
    #[must_use]
    pub fn weight(&self) -> Option<&'a [f64]> {
        self.weight
    }

    /// The label smoothing (0: none).
    #[must_use]
    pub fn label_smoothing(&self) -> f64 {
        self.label_smoothing
    }

    /// The ignored label, if any.
    #[must_use]
    pub fn ignore_index(&self) -> Option<usize> {
        self.ignore_index
    }
}
