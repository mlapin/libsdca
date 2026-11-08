use std::ffi::{c_int, c_void, CStr};
use std::ptr;
use std::sync::OnceLock;

use crate::error::{invalid, Error, Result};
use crate::ffi;
use crate::matrix::Matrix;
use crate::model::{History, Model, HISTORY};
use crate::names::{Objective, Precision, Status};
use crate::options::{DistillationOptions, FitOptions, LossOptions, UpdateOptions};
use crate::real::{self, Real};
use crate::targets::Targets;

/// SDCA_ABI_VERSION of include/sdca/c_api.h.
const ABI_VERSION: c_int = 5;

/// SDCA_INVALID_ARGUMENT of include/sdca/c_api.h.
const INVALID_ARGUMENT: c_int = -2;

/// Whether the native library has the wrapper's C ABI and history layout.
fn check_abi(abi: c_int, history_fields: usize) -> Result<()> {
    if abi != ABI_VERSION {
        return Err(Error::Native(
            "Library and wrapper disagree on the C ABI".into(),
        ));
    }
    if history_fields != HISTORY.len() {
        return Err(Error::Native(
            "Library and wrapper disagree on the history layout".into(),
        ));
    }
    Ok(())
}

/// The ABI handshake, done once, then the C ABI precision code of an entry point.
fn ready<T: Real>(precision: Option<Precision>) -> Result<c_int> {
    static HANDSHAKE: OnceLock<Result<()>> = OnceLock::new();
    HANDSHAKE
        .get_or_init(|| {
            // SAFETY: plain queries of the native library.
            let (abi, fields) = unsafe { (ffi::sdca_abi_version(), ffi::sdca_history_fields()) };
            check_abi(abi, fields)
        })
        .clone()?;
    real::precision_code::<T>(precision)
}

fn native(call: impl FnOnce() -> c_int) -> Result<()> {
    let code = call();
    if code == 0 {
        return Ok(());
    }
    // SAFETY: sdca_last_error returns a NUL-terminated thread-local string.
    let message = unsafe { CStr::from_ptr(ffi::sdca_last_error()) };
    let message = message.to_string_lossy().into_owned();
    Err(if code == INVALID_ARGUMENT {
        Error::InvalidArgument(message)
    } else {
        Error::Native(message)
    })
}

/// The native num_threads: SDCA_THREADS_DEFAULT for None.
fn threads(num_threads: Option<usize>) -> usize {
    num_threads.unwrap_or(usize::MAX)
}

fn positive(value: f64, name: &str) -> Result<()> {
    if !(value.is_finite() && value > 0.0) {
        return invalid(format!("{name} must be finite and positive"));
    }
    Ok(())
}

/// A finite gamma >= 0: 0 is the nonsmooth hinge.
fn smoothing(gamma: f64) -> Result<()> {
    if !(gamma.is_finite() && gamma >= 0.0) {
        return invalid("gamma must be finite and nonnegative");
    }
    Ok(())
}

fn check_k(k: usize, classes: usize) -> Result<()> {
    if k < 1 || k >= classes {
        return invalid("k must be an integer in [1, classes-1]");
    }
    Ok(())
}

fn finite<T: Real>(values: &[T]) -> Result<()> {
    if !values.iter().all(|v| v.f64().is_finite()) {
        return invalid("Inputs must be finite");
    }
    Ok(())
}

fn check_logits<T: Real>(z: Matrix<'_, T>) -> Result<()> {
    if z.rows() == 0 || z.cols() < 2 {
        return invalid("logits must have shape (n>=1,k>=2)");
    }
    finite(z.data())
}

fn check_scale<T: Real>(c: f64, norm2: f64) -> Result<()> {
    let normal = |v: f64| (T::MIN_POSITIVE..=T::MAX).contains(&v);
    if !(normal(c) && normal(norm2) && normal(c * norm2)) {
        return invalid("C, norm2, and C*norm2 must be normal in the data dtype");
    }
    Ok(())
}

fn void<T>(values: &[T]) -> *const c_void {
    values.as_ptr().cast()
}

fn optional<T>(values: Option<&[T]>) -> *const T {
    values.map_or(ptr::null(), <[T]>::as_ptr)
}

/// Cross-entropy on label sets is multilabel_cross_entropy.
fn resolve(objective: Objective, y: &Targets<'_>) -> Objective {
    if objective == Objective::CrossEntropy && y.is_sets() {
        Objective::MultilabelCrossEntropy
    } else {
        objective
    }
}

/// The labels, and the offsets of the multilabel objectives.
fn targets(
    y: &Targets<'_>,
    n: usize,
    classes: usize,
    objective: Objective,
) -> Result<(Vec<i64>, Option<Vec<i64>>)> {
    if objective.is_multilabel() {
        let (labels, offsets) = y.label_sets(n, classes, classes - 1)?;
        Ok((labels, Some(offsets)))
    } else {
        Ok((y.labels(n, classes)?, None))
    }
}

/// Unregularized per-row loss of the scores `logits` (n, classes) and its
/// gradient (n, classes) with respect to them (a subgradient at the kinks of
/// the hinges).
///
/// `target` holds a label or a label set per row; the entropy loss takes a
/// uniform target over the set, which may hold every class.
///
/// # Errors
/// [`Error::InvalidArgument`] for invalid scores, targets or options (or a
/// precision that does not store `T`), [`Error::Native`] for a native
/// failure or a library that does not match the wrapper.
///
/// ```
/// use libsdca::{loss_and_grad, Loss, LossOptions, Matrix};
///
/// let logits = Matrix::new(&[2.0, -0.5, 1.0, 0.0, 1.0, 3.0], 2, 3)?;
/// let options = LossOptions { loss: Loss::TopKHingeAlpha, k: 2, ..Default::default() };
/// let (values, gradients) = loss_and_grad(logits, &[0, 2], &options)?;
/// assert_eq!((values.len(), gradients.len()), (2, 6));
/// # Ok::<(), libsdca::Error>(())
/// ```
pub fn loss_and_grad<'y, T: Real>(
    logits: Matrix<'_, T>,
    target: impl Into<Targets<'y>>,
    options: &LossOptions,
) -> Result<(Vec<f64>, Vec<T>)> {
    let code = ready::<T>(options.precision)?;
    check_logits(logits)?;
    let (n, classes) = (logits.rows(), logits.cols());
    check_k(options.k, classes)?;
    smoothing(options.gamma)?;
    positive(options.alpha, "alpha")?;
    let mask = target.into().positive(n, classes)?;
    let (mut values, mut gradients) = (vec![0.0; n], vec![T::default(); n * classes]);
    let native_options = ffi::Options {
        precision: code,
        loss: options.loss.id(),
        k: options.k,
        gamma: options.gamma,
        alpha: options.alpha,
        projection_precision: options.projection_precision.id(),
        num_threads: threads(options.num_threads),
        ..Default::default()
    };
    // SAFETY: the buffers have the sizes of the C ABI documentation.
    native(|| unsafe {
        ffi::sdca_loss_gradient(
            &native_options,
            n,
            classes,
            void(logits.data()),
            mask.as_ptr(),
            values.as_mut_ptr(),
            gradients.as_mut_ptr().cast(),
        )
    })?;
    Ok((values, gradients))
}

/// Unregularized per-row distillation loss (or proximal distillation if
/// `options.alpha > 0`) and gradient with respect to `student`.
pub fn distillation_loss_and_grad<T: Real>(
    student: Matrix<'_, T>,
    teacher: Matrix<'_, T>,
    options: &DistillationOptions,
) -> Result<(Vec<f64>, Vec<T>)> {
    let code = ready::<T>(options.precision)?;
    check_logits(student)?;
    check_logits(teacher)?;
    if (teacher.rows(), teacher.cols()) != (student.rows(), student.cols()) {
        return invalid("teacher must have the same shape as student");
    }
    positive(options.temperature, "temperature")?;
    if !(options.alpha.is_finite() && options.alpha >= 0.0) {
        return invalid("alpha must be finite and nonnegative");
    }
    let (n, classes) = (student.rows(), student.cols());
    let dtype = T::DTYPE_CODE;
    let (mut values, mut gradients) = (vec![0.0; n], vec![T::default(); n * classes]);
    if options.alpha > 0.0 {
        let native_options = ffi::Options {
            precision: 2,
            num_threads: threads(options.num_threads),
            ..Default::default()
        };
        // SAFETY: the buffers have the sizes of the C ABI documentation.
        native(|| unsafe {
            ffi::sdca_proximal_distillation_rows(
                &native_options,
                dtype,
                n,
                classes,
                void(student.data()),
                void(teacher.data()),
                options.temperature,
                options.alpha,
                values.as_mut_ptr(),
                gradients.as_mut_ptr().cast(),
            )
        })?;
        return Ok((values, gradients));
    }
    let native_options = ffi::Options {
        precision: if code == 0 { 0 } else { 2 },
        num_threads: threads(options.num_threads),
        ..Default::default()
    };
    let mut stats = vec![0.0; n * 7];
    let scale = [1.0f64];
    // SAFETY: the buffers have the sizes of the C ABI documentation.
    native(|| unsafe {
        ffi::sdca_distillation_forward(
            &native_options,
            dtype,
            n,
            classes,
            void(student.data()),
            void(teacher.data()),
            options.temperature,
            values.as_mut_ptr(),
            stats.as_mut_ptr(),
        )
    })?;
    // SAFETY: the buffers have the sizes of the C ABI documentation.
    native(|| unsafe {
        ffi::sdca_distillation_backward(
            &native_options,
            dtype,
            n,
            classes,
            void(student.data()),
            void(teacher.data()),
            options.temperature,
            stats.as_ptr(),
            scale.as_ptr(),
            0,
            gradients.as_mut_ptr().cast(),
        )
    })?;
    Ok((values, gradients))
}

/// Unregularized per-row proximal distillation loss (`options.alpha > 0`) and
/// gradient with respect to `student`.
pub fn proximal_distillation_loss_and_grad<T: Real>(
    student: Matrix<'_, T>,
    teacher: Matrix<'_, T>,
    options: &DistillationOptions,
) -> Result<(Vec<f64>, Vec<T>)> {
    positive(options.alpha, "alpha")?;
    distillation_loss_and_grad(student, teacher, options)
}

/// New dual blocks (n, classes) of one coordinate step from the blocks
/// `options.previous` at the scores `logits` (n, classes) (not gradients).
///
/// # Errors
/// As [`loss_and_grad`]; `previous` must match the shape of `logits`.
///
/// ```
/// use libsdca::{coordinate_update, Matrix, UpdateOptions};
///
/// let logits = Matrix::new(&[2.0, -0.5, 1.0], 1, 3)?;
/// let a = coordinate_update(logits, &[2], &UpdateOptions { c: 3.0, ..Default::default() })?;
/// assert!(a[2] > 0.0 && a.iter().sum::<f64>().abs() < 1e-12);
/// # Ok::<(), libsdca::Error>(())
/// ```
pub fn coordinate_update<'y, T: Real>(
    logits: Matrix<'_, T>,
    y: impl Into<Targets<'y>>,
    options: &UpdateOptions<'_, T>,
) -> Result<Vec<T>> {
    let code = ready::<T>(options.precision)?;
    check_logits(logits)?;
    let (n, classes) = (logits.rows(), logits.cols());
    let y = y.into();
    let objective = resolve(options.objective, &y);
    let (labels, offsets) = targets(&y, n, classes, objective)?;
    positive(options.c, "C")?;
    positive(options.norm2, "norm2")?;
    smoothing(options.gamma)?;
    check_k(options.k, classes)?;
    check_scale::<T>(options.c, options.norm2)?;
    let zeros = vec![T::default(); n * classes];
    let previous = options.previous.unwrap_or(&zeros);
    if previous.len() != n * classes {
        return invalid("previous must match logits.shape");
    }
    finite(previous)?;
    let mut out = vec![T::default(); n * classes];
    let native_options = ffi::Options {
        precision: code,
        loss: objective.id(),
        c: options.c,
        k: options.k,
        gamma: options.gamma,
        projection_precision: options.projection_precision.id(),
        num_threads: threads(options.num_threads),
        ..Default::default()
    };
    // SAFETY: the buffers have the sizes of the C ABI documentation.
    native(|| unsafe {
        ffi::sdca_update_objective(
            &native_options,
            n,
            classes,
            void(logits.data()),
            labels.as_ptr(),
            optional(offsets.as_deref()),
            void(previous),
            options.norm2,
            out.as_mut_ptr().cast(),
        )
    })?;
    Ok(out)
}

/// Train an objective on features `x` (n, d), or on a Gram matrix (n, n) if
/// `options.kernel`, with labels or label sets `y`.
///
/// # Errors
/// [`Error::InvalidArgument`] for invalid inputs or options (or a precision
/// that does not store `T`), [`Error::Native`] for a native failure or a
/// library that does not match the wrapper. Not converging is no error:
/// see [`Model::status`].
///
/// ```
/// use libsdca::{fit, FitOptions, Matrix, Precision, Status};
///
/// let x = [1.0f32, 0.0, 0.0, 1.0, -1.0, -1.0];
/// let x = Matrix::new(&x, 3, 2)?;
/// let options = FitOptions { c: 10.0, precision: Some(Precision::Float32), ..Default::default() };
/// let model = fit(x, &[0, 1, 2], &options)?;
/// assert_eq!(model.status(), Status::Solved);
/// assert_eq!(model.predict(x)?, [0, 1, 2]);
/// # Ok::<(), libsdca::Error>(())
/// ```
pub fn fit<'y, T: Real>(
    x: Matrix<'_, T>,
    y: impl Into<Targets<'y>>,
    options: &FitOptions<'_, T>,
) -> Result<Model<T>> {
    let code = ready::<T>(options.precision)?;
    let problem = problem(x, &y.into(), options)?;
    positive(options.c, "C")?;
    smoothing(options.gamma)?;
    check_k(options.k, problem.classes)?;
    check_norms(&problem, options)?;
    if options.eval_epoch < 1 {
        return invalid("eval_epoch must be >=1");
    }
    if !(options.tol.is_finite() && options.tol >= 0.0) {
        return invalid("tol must be finite and nonnegative");
    }
    let center = center(&problem, options)?;
    let dual = dual(&problem, options)?;
    solve(code, &problem, options, center, dual)
}

fn check_distribution(targets: Matrix<'_, f64>, n: usize, classes: Option<usize>) -> Result<usize> {
    if targets.rows() != n || targets.cols() < 2 {
        return invalid("distillation targets must have shape (n, num_classes>=2)");
    }
    if let Some(c) = classes {
        if targets.cols() != c {
            return invalid("distillation targets must match num_classes");
        }
    }
    let m = targets.cols();
    let tol = 64.0 * f64::EPSILON * m as f64;
    for i in 0..n {
        let row = targets.row(i);
        if row.iter().any(|&q| !q.is_finite() || q < 0.0) {
            return invalid("Target probabilities must be finite and nonnegative");
        }
        let sum: f64 = row.iter().sum();
        if (sum - 1.0).abs() > tol {
            return invalid("Target probability rows must sum to one");
        }
    }
    Ok(m)
}

/// New dual blocks (n, classes) of one soft-target (distillation) coordinate step.
pub fn coordinate_update_distillation<T: Real>(
    logits: Matrix<'_, T>,
    targets: Matrix<'_, f64>,
    options: &UpdateOptions<'_, T>,
) -> Result<Vec<T>> {
    let code = ready::<T>(options.precision)?;
    check_logits(logits)?;
    let (n, classes) = (logits.rows(), logits.cols());
    check_distribution(targets, n, Some(classes))?;
    positive(options.c, "C")?;
    positive(options.norm2, "norm2")?;
    smoothing(options.gamma)?;
    check_k(options.k, classes)?;
    check_scale::<T>(options.c, options.norm2)?;
    let zeros = vec![T::default(); n * classes];
    let previous = options.previous.unwrap_or(&zeros);
    if previous.len() != n * classes {
        return invalid("previous must match logits.shape");
    }
    finite(previous)?;
    let mut out = vec![T::default(); n * classes];
    let native_options = ffi::Options {
        precision: code,
        c: options.c,
        projection_precision: options.projection_precision.id(),
        num_threads: threads(options.num_threads),
        ..Default::default()
    };
    // SAFETY: the buffers have the sizes of the C ABI documentation.
    native(|| unsafe {
        ffi::sdca_update_distillation(
            &native_options,
            n,
            classes,
            void(logits.data()),
            targets.data().as_ptr(),
            void(previous),
            options.norm2,
            out.as_mut_ptr().cast(),
        )
    })?;
    Ok(out)
}

/// Train a soft-target (distillation) model on features `x` (n, d), or on a
/// Gram matrix (n, n) if `options.kernel`, with target probability matrix
/// `targets` (n, num_classes).
pub fn fit_distillation<T: Real>(
    x: Matrix<'_, T>,
    targets: Matrix<'_, f64>,
    options: &FitOptions<'_, T>,
) -> Result<Model<T>> {
    let code = ready::<T>(options.precision)?;
    if x.rows() == 0 || x.cols() == 0 {
        return invalid("X must be a nonempty matrix");
    }
    finite(x.data())?;
    if options.kernel && x.rows() != x.cols() {
        return invalid("Training kernel must be square");
    }
    if let Some(c) = options.num_classes {
        if c < 2 {
            return invalid("num_classes must be >=2");
        }
    }
    let (n, d) = (x.rows(), x.cols());
    let classes = check_distribution(targets, n, options.num_classes)?;
    if [
        n,
        d,
        classes,
        d.saturating_mul(classes),
        n.saturating_mul(classes),
    ]
    .iter()
    .any(|&size| size > i32::MAX as usize)
    {
        return invalid("Dimensions exceed the native BLAS integer limit");
    }
    let problem = Problem {
        x,
        labels: Vec::new(),
        offsets: None,
        objective: Objective::CrossEntropy,
        classes,
    };
    positive(options.c, "C")?;
    smoothing(options.gamma)?;
    check_k(options.k, classes)?;
    check_norms(&problem, options)?;
    if options.eval_epoch < 1 {
        return invalid("eval_epoch must be >=1");
    }
    if !(options.tol.is_finite() && options.tol >= 0.0) {
        return invalid("tol must be finite and nonnegative");
    }
    let center = center(&problem, options)?;
    let dual = match options.initial_dual {
        None => None,
        Some(initial) => {
            if initial.len() != n * classes {
                return invalid("initial_dual must have shape (n, num_classes)");
            }
            finite(initial)?;
            let eps = 32.0 * T::EPSILON;
            let slack = eps * options.c;
            for i in 0..n {
                let row = &initial[i * classes..(i + 1) * classes];
                let q = targets.row(i);
                // Relative to the largest entry: near the f64 maximum, the sum of |v| overflows.
                let largest = row.iter().fold(0.0f64, |m, v| m.max(v.f64().abs()));
                let unit = if largest > 0.0 { largest } else { 1.0 };
                let sum: f64 = row.iter().map(|v| v.f64() / unit).sum();
                let size: f64 = row.iter().map(|v| (v.f64() / unit).abs()).sum();
                if sum.abs() > eps * size {
                    return invalid("initial_dual rows must sum to zero to storage precision");
                }
                if (0..classes).any(|j| {
                    let v = row[j].f64();
                    v > options.c * q[j] + slack || v < options.c * (q[j] - 1.0) - slack
                }) {
                    return invalid("initial_dual is outside the distillation feasible set");
                }
            }
            Some(initial.to_vec())
        }
    };
    let warm = dual.is_some();
    let mut w = match (&dual, options.kernel) {
        (_, true) => Vec::new(),
        (Some(a), false) => warm_primal(x, a, classes, center.as_deref()),
        (None, false) => vec![T::default(); d * classes],
    };
    let mut a = dual.unwrap_or_else(|| vec![T::default(); n * classes]);
    let rows = history_rows(options)?;
    let mut history = Vec::new();
    if history.try_reserve_exact(rows * HISTORY.len()).is_err() {
        return invalid("max_epoch / eval_epoch is too large for the history buffer");
    }
    history.resize(rows * HISTORY.len(), 0.0);
    let native_options = ffi::Options {
        precision: code,
        c: options.c,
        projection_precision: options.projection_precision.id(),
        max_epoch: options.max_epoch,
        eval_epoch: options.eval_epoch,
        tol: options.tol,
        eval_on_start: c_int::from(options.eval_on_start),
        recompute_primal: c_int::from(options.recompute_primal),
        kernel: c_int::from(options.kernel),
        warm: c_int::from(warm),
        shrinking: options.shrinking.id(),
        ..Default::default()
    };
    let mut result = ffi::FitResult::default();
    // SAFETY: the buffers have the sizes of the C ABI documentation.
    native(|| unsafe {
        ffi::sdca_fit_distillation(
            &native_options,
            n,
            d,
            classes,
            void(x.data()),
            targets.data().as_ptr(),
            optional(center.as_deref()).cast(),
            if options.kernel {
                ptr::null_mut()
            } else {
                w.as_mut_ptr().cast()
            },
            a.as_mut_ptr().cast(),
            history.as_mut_ptr(),
            rows,
            &mut result,
        )
    })?;
    let weights = (!options.kernel).then(|| {
        (0..d * classes)
            .map(|i| w[(i % classes) * d + i / classes])
            .collect()
    });
    Ok(Model {
        weights,
        dual: a,
        history: History::new(&history, result.history_used),
        status: Status::from_id(result.status.into()),
        epochs: result.epochs,
        kernel: options.kernel,
        objective: Objective::CrossEntropy,
        num_classes: classes,
    })
}

/// Validated training input in the layout of the native solver.
struct Problem<'a, T> {
    x: Matrix<'a, T>,
    labels: Vec<i64>,
    offsets: Option<Vec<i64>>,
    objective: Objective,
    classes: usize,
}

fn problem<'x, T: Real>(
    x: Matrix<'x, T>,
    y: &Targets<'_>,
    options: &FitOptions<'_, T>,
) -> Result<Problem<'x, T>> {
    if x.rows() == 0 || x.cols() == 0 {
        return invalid("X must be a nonempty matrix");
    }
    finite(x.data())?;
    // The library checks a kernel's symmetry and Cauchy-Schwarz bounds.
    if options.kernel && x.rows() != x.cols() {
        return invalid("Training kernel must be square");
    }
    let objective = resolve(options.objective, y);
    let classes = match options.num_classes {
        Some(c) => c,
        None => y.implied_classes()?,
    };
    if classes < 2 {
        return invalid("num_classes must be >=2");
    }
    let (n, d) = (x.rows(), x.cols());
    if [
        n,
        d,
        classes,
        d.saturating_mul(classes),
        n.saturating_mul(classes),
    ]
    .iter()
    .any(|&size| size > i32::MAX as usize)
    {
        return invalid("Dimensions exceed the native BLAS integer limit");
    }
    let (labels, offsets) = targets(y, n, classes, objective)?;
    Ok(Problem {
        x,
        labels,
        offsets,
        objective,
        classes,
    })
}

/// Nonzero squared norms must keep C*norm2 normal; zero rows have zero scores.
fn check_norms<T: Real>(problem: &Problem<'_, T>, options: &FitOptions<'_, T>) -> Result<()> {
    let x = problem.x;
    let norms: Vec<f64> = (0..x.rows())
        .map(|i| {
            if options.kernel {
                x.row(i)[i].f64()
            } else {
                x.row(i).iter().map(|v| v.f64() * v.f64()).sum()
            }
        })
        .collect();
    if norms.iter().any(|&norm| norm < 0.0) {
        return invalid("A kernel diagonal must be nonnegative");
    }
    // Zero rows take a block of size C, so C itself must fit the data type.
    if !(T::MIN_POSITIVE..=T::MAX).contains(&options.c) {
        return invalid("C, norm2, and C*norm2 must be normal in the data dtype");
    }
    let (lo, hi) = norms
        .iter()
        .filter(|&&v| v > 0.0)
        .fold((f64::MAX, 0.0f64), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    if hi == 0.0 {
        return Ok(());
    }
    check_scale::<T>(options.c, lo)?;
    check_scale::<T>(options.c, hi)
}

/// The center (d, classes) in the column-major layout of the native solver.
fn center<T: Real>(
    problem: &Problem<'_, T>,
    options: &FitOptions<'_, T>,
) -> Result<Option<Vec<T>>> {
    let Some(center) = options.center else {
        return Ok(None);
    };
    if options.kernel {
        return invalid("center is only supported for feature input");
    }
    let (d, classes) = (problem.x.cols(), problem.classes);
    if center.len() != d * classes || !center.iter().all(|v| v.f64().is_finite()) {
        return invalid("center must be finite with shape (d, num_classes)");
    }
    Ok(Some(
        (0..d * classes)
            .map(|i| center[(i % d) * classes + i / d])
            .collect(),
    ))
}

/// A validated copy of the warm start, the label entry of single-label
/// rows rebuilt so that rows sum to zero exactly.
fn dual<T: Real>(problem: &Problem<'_, T>, options: &FitOptions<'_, T>) -> Result<Option<Vec<T>>> {
    let Some(initial) = options.initial_dual else {
        return Ok(None);
    };
    let (n, classes) = (problem.x.rows(), problem.classes);
    if initial.len() != n * classes {
        return invalid("initial_dual must have shape (n, num_classes)");
    }
    finite(initial)?;
    let mut a = initial.to_vec();
    for i in 0..n {
        check_initial_row(problem, i, &a[i * classes..(i + 1) * classes], options)?;
        if problem.offsets.is_none() {
            let label = problem.labels[i] as usize;
            let others: f64 = (0..classes)
                .filter(|&j| j != label)
                .map(|j| a[i * classes + j].f64())
                .sum();
            a[i * classes + label] = T::from_f64(-others);
        }
    }
    Ok(Some(a))
}

fn solve<T: Real>(
    code: c_int,
    problem: &Problem<'_, T>,
    options: &FitOptions<'_, T>,
    center: Option<Vec<T>>,
    dual: Option<Vec<T>>,
) -> Result<Model<T>> {
    let (n, d, classes) = (problem.x.rows(), problem.x.cols(), problem.classes);
    let warm = dual.is_some();
    // A cold start fills W itself (zero or the center).
    let mut w = match (&dual, options.kernel) {
        (_, true) => Vec::new(),
        (Some(a), false) => warm_primal(problem.x, a, classes, center.as_deref()),
        (None, false) => vec![T::default(); d * classes],
    };
    let mut a = dual.unwrap_or_else(|| vec![T::default(); n * classes]);
    let rows = history_rows(options)?;
    let mut history = Vec::new();
    if history.try_reserve_exact(rows * HISTORY.len()).is_err() {
        return invalid("max_epoch / eval_epoch is too large for the history buffer");
    }
    history.resize(rows * HISTORY.len(), 0.0);
    let native_options = ffi::Options {
        precision: code,
        loss: problem.objective.id(),
        c: options.c,
        k: options.k,
        gamma: options.gamma,
        projection_precision: options.projection_precision.id(),
        max_epoch: options.max_epoch,
        eval_epoch: options.eval_epoch,
        tol: options.tol,
        eval_on_start: c_int::from(options.eval_on_start),
        recompute_primal: c_int::from(options.recompute_primal),
        kernel: c_int::from(options.kernel),
        warm: c_int::from(warm),
        shrinking: options.shrinking.id(),
        ..Default::default()
    };
    let mut result = ffi::FitResult::default();
    // SAFETY: the buffers have the sizes of the C ABI documentation.
    native(|| unsafe {
        ffi::sdca_fit(
            &native_options,
            n,
            d,
            classes,
            void(problem.x.data()),
            problem.labels.as_ptr(),
            optional(problem.offsets.as_deref()),
            optional(center.as_deref()).cast(),
            if options.kernel {
                ptr::null_mut()
            } else {
                w.as_mut_ptr().cast()
            },
            a.as_mut_ptr().cast(),
            history.as_mut_ptr(),
            rows,
            &mut result,
        )
    })?;
    let weights = (!options.kernel).then(|| {
        (0..d * classes)
            .map(|i| w[(i % classes) * d + i / classes])
            .collect()
    });
    Ok(Model {
        weights,
        dual: a,
        history: History::new(&history, result.history_used),
        status: Status::from_id(result.status.into()),
        epochs: result.epochs,
        kernel: options.kernel,
        objective: problem.objective,
        num_classes: classes,
    })
}

/// Rows of the history: at most max_epoch / eval_epoch + 2 evaluations
/// (every eval_epoch epochs, up to two between, the start and the end),
/// checked against overflow.
fn history_rows<T>(options: &FitOptions<'_, T>) -> Result<usize> {
    (options.max_epoch / options.eval_epoch)
        .checked_add(2)
        .filter(|rows| rows.checked_mul(HISTORY.len()).is_some())
        .map_or_else(
            || invalid("max_epoch / eval_epoch is too large for the history buffer"),
            Ok,
        )
}

/// W = center + X' A (d, classes) column-major, accumulated in f64.
fn warm_primal<T: Real>(x: Matrix<'_, T>, a: &[T], classes: usize, center: Option<&[T]>) -> Vec<T> {
    let d = x.cols();
    let mut w: Vec<f64> = center.map_or_else(
        || vec![0.0; d * classes],
        |c| c.iter().map(|v| v.f64()).collect(),
    );
    for i in 0..x.rows() {
        let x_row = x.row(i);
        for j in 0..classes {
            let aij = a[i * classes + j].f64();
            if aij == 0.0 {
                continue;
            }
            let col = &mut w[j * d..(j + 1) * d];
            for (wl, &v) in col.iter_mut().zip(x_row) {
                *wl += v.f64() * aij;
            }
        }
    }
    w.into_iter().map(T::from_f64).collect()
}

/// Reject a dual block outside the objective's feasible set (to rounding).
fn check_initial_row<T: Real>(
    problem: &Problem<'_, T>,
    i: usize,
    row: &[T],
    options: &FitOptions<'_, T>,
) -> Result<()> {
    let eps = 32.0 * T::EPSILON;
    let slack = eps * options.c;
    let v: Vec<f64> = row.iter().map(|x| x.f64()).collect();
    // Relative to the largest entry: near the f64 maximum, the sum of |v| overflows.
    let largest = v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let unit = if largest > 0.0 { largest } else { 1.0 };
    let (sum, size) = (
        v.iter().map(|x| x / unit).sum::<f64>(),
        v.iter().map(|x| (x / unit).abs()).sum::<f64>(),
    );
    if sum.abs() > eps * size {
        return invalid("initial_dual rows must sum to zero to storage precision");
    }
    let is_label =
        |j: usize| Targets::is_positive(&problem.labels, problem.offsets.as_deref(), i, j);
    // Exact: a positive non-label entry is outside the domain of every dual.
    if (0..v.len()).any(|j| !is_label(j) && v[j] > 0.0) {
        return invalid("initial_dual: non-label entries must be nonpositive");
    }
    let labels: Vec<f64> = (0..v.len())
        .filter(|&j| is_label(j))
        .map(|j| v[j])
        .collect();
    let mass: f64 = labels.iter().sum();
    let c = options.c;
    // multiclass_hinge has no top-k, as cross_entropy.
    let topk = if problem.objective == Objective::MulticlassHinge {
        1
    } else {
        options.k
    };
    let k = topk as f64;
    match problem.objective {
        Objective::CrossEntropy | Objective::MultilabelCrossEntropy => {
            if labels.iter().any(|&x| x > c / labels.len() as f64 + slack) {
                return invalid("initial_dual is outside the entropy feasible set");
            }
        }
        objective => {
            if labels.iter().any(|&x| x < -slack) || mass > c + slack {
                return invalid("initial_dual is outside the feasible set");
            }
            let cone = matches!(
                objective,
                Objective::TopKCrossEntropy
                    | Objective::MulticlassHinge
                    | Objective::TopKHingeAlpha
            );
            let cap = objective == Objective::TopKHingeBeta;
            let cone_slack = (v.len() / topk).max(1) as f64 * 64.0 * T::EPSILON * c;
            if cone && v.iter().any(|&x| -x > mass / k + cone_slack) {
                return invalid("initial_dual violates the top-k cone");
            }
            if cap && v.iter().any(|&x| -x > c / k + slack) {
                return invalid("initial_dual violates the top-k cap");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_handshake_rejects_another_abi_or_history_layout() {
        assert_eq!(check_abi(ABI_VERSION, HISTORY.len()), Ok(()));
        // ABI 4 builds take positional options.
        assert!(check_abi(4, HISTORY.len())
            .unwrap_err()
            .to_string()
            .contains("C ABI"));
        let abi = check_abi(ABI_VERSION + 1, HISTORY.len()).unwrap_err();
        assert!(abi.to_string().contains("C ABI"));
        let layout = check_abi(ABI_VERSION, HISTORY.len() + 1).unwrap_err();
        assert!(layout.to_string().contains("history layout"));
    }
}
