//! A downstream crate: trains through the BLAS that SDCA_BLAS linked and checks
//! the certificate and W = X' A with plain loops, independently of that BLAS.
use libsdca::{fit, FitOptions, Matrix, Objective, Real, Status};

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

fn check<T: Real + Into<f64>>(objective: Objective, x: &[T], y: &[usize], d: usize) -> String {
    // Real is sealed to f32/f64. Both default to f64 working arithmetic,
    // but primal/dual storage rounding still follows the element type.
    let wide = std::mem::size_of::<T>() == std::mem::size_of::<f64>();
    let (n, classes) = (y.len(), 4);
    let tol = if wide { 1e-8 } else { 1e-4 };
    let options = FitOptions {
        c: 10.0,
        objective,
        k: 2,
        gamma: 1.0,
        tol,
        max_epoch: 2000,
        ..Default::default()
    };
    let model = fit(Matrix::new(x, n, d).unwrap(), y, &options).unwrap();
    let gap = model.history().get("relative_gap").unwrap();
    let gap = *gap.last().unwrap();
    assert_eq!(model.status(), Status::Solved, "{objective}");
    assert!(gap <= tol, "{objective}: relative gap {gap} above {tol}");
    // W (d, classes) = X' A, both row-major.
    let (w, a) = (model.weights().unwrap(), model.dual());
    let mut error = 0.0_f64;
    for l in 0..d {
        for j in 0..classes {
            let xa: f64 = (0..n)
                .map(|i| x[i * d + l].into() * a[i * classes + j].into())
                .sum();
            error = error.max((w[l * classes + j].into() - xa).abs());
        }
    }
    let limit = if wide { 1e-10 } else { 1e-4 };
    assert!(error <= limit, "{objective}: max |W - X'A| {error}");
    format!(
        "{objective} {}: solved in {} epochs, gap {gap:.1e}, |W - X'A| {error:.1e}",
        std::any::type_name::<T>(),
        model.epochs()
    )
}

fn main() {
    let (n, d) = (120, 9);
    let (x, y) = data(n, d, 4);
    let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    for objective in [Objective::CrossEntropy, Objective::TopKHingeAlpha] {
        println!("{}", check(objective, &x, &y, d));
        println!("{}", check(objective, &x32, &y, d));
    }
}
