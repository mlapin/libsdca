//! A downstream crate: trains and scores through the BLAS that SDCA_BLAS linked and
//! checks the duality gap, W = A' X and the scores X W' with plain loops, independently
//! of that BLAS.
use libsdca::{Loss, Matrix, Real, Solver, Status};

/// Deterministic features in [-1, 1) and labels of (n, d) data with `classes` classes.
fn data(n: usize, d: usize, classes: usize) -> (Vec<f64>, Vec<usize>) {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    };
    let x: Vec<f64> = (0..n * d).map(|_| next()).collect();
    let y = (0..n).map(|i| (i * 7 + i / 3) % classes).collect();
    (x, y)
}

fn check<T: Real + Into<f64>>(loss: Loss, x: &[T], y: &[usize], d: usize) -> String {
    // Real is sealed to f32/f64. Each defaults to working arithmetic of its own
    // type, and primal/dual storage rounding follows the element type.
    let wide = std::mem::size_of::<T>() == std::mem::size_of::<f64>();
    let (n, classes) = (y.len(), 4);
    let tol = if wide { 1e-8 } else { 1e-4 };
    let solver = Solver::new(loss)
        .with_c(10.0)
        .with_tol(tol)
        .with_max_epoch(2000);
    let features = Matrix::new(x, n, d).unwrap();
    let model = solver.fit(features, y).unwrap();
    let gap = model.history().last().unwrap().relative_gap;
    assert_eq!(model.status(), Status::Solved, "{loss}");
    assert!(gap <= tol, "{loss}: relative gap {gap} above {tol}");
    // W (classes, d) = A' X, both row-major.
    let (w, a) = (model.weights().unwrap(), model.dual());
    let mut error = 0.0_f64;
    for l in 0..d {
        for j in 0..classes {
            let xa: f64 = (0..n)
                .map(|i| x[i * d + l].into() * a[i * classes + j].into())
                .sum();
            error = error.max((w[j * d + l].into() - xa).abs());
        }
    }
    let limit = if wide { 1e-10 } else { 1e-4 };
    assert!(error <= limit, "{loss}: max |W - A'X| {error}");
    // Scores (n, classes) = X W': one gemm of the linked BLAS.
    let scores = model.decision_function(features).unwrap();
    let mut score_error = 0.0_f64;
    for i in 0..n {
        for j in 0..classes {
            let xw: f64 = (0..d)
                .map(|l| x[i * d + l].into() * w[j * d + l].into())
                .sum();
            score_error = score_error.max((scores[i * classes + j].into() - xw).abs());
        }
    }
    assert!(
        score_error <= limit,
        "{loss}: max |scores - XW'| {score_error}"
    );
    format!(
        "{loss} {}: solved in {} epochs, gap {gap:.1e}, |W - A'X| {error:.1e}, |scores - XW'| {score_error:.1e}",
        std::any::type_name::<T>(),
        model.epochs()
    )
}

fn main() {
    let (n, d) = (120, 9);
    let (x, y) = data(n, d, 4);
    let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let alpha = Loss::TopKHingeAlpha { k: 2, gamma: 1.0 };
    for loss in [Loss::cross_entropy(), alpha] {
        println!("{}", check(loss, &x, &y, d));
        println!("{}", check(loss, &x32, &y, d));
    }
}
