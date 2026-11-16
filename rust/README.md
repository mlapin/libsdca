# libsdca for Rust

Rust bindings to [libsdca](https://github.com/mlapin/libsdca/blob/v0.3.1/README.md),
accurate and fast classification losses for SDCA and neural networks: SDCA solvers for
linear and kernel SVMs and softmax classifiers, plus loss, gradient and coordinate-step
evaluation through the shared C ABI (`include/sdca/c_api.h`;
[technical report](https://mlapin.github.io/libsdca/report/)).

```rust
use libsdca::{fit, FitOptions, Matrix, Objective};

fn main() -> libsdca::Result<()> {
    let x = Matrix::new(&[1.0, 0.0, 0.0, 1.0, -1.0, -1.0], 3, 2)?; // (n, d) row-major
    let options = FitOptions { c: 10.0, objective: Objective::CrossEntropy, ..Default::default() };
    let model = fit(x, &[0, 1, 2], &options)?;
    println!("{} {:?}", model.status(), model.predict(x)?);
    Ok(())
}
```

- **Targets:** Single-label objectives take 0-based class slices (`&[usize]`); multilabel
  objectives take label sets (`&[Vec<usize>]` or `&[&[usize]]`), and `fit_distillation`
  takes a target probability `Matrix`.
- **Errors and convergence:** Invalid inputs return `Error::InvalidArgument` and failures
  of the native library return `Error::Native`. A fit that fails numerically or stops at
  `max_epoch` still returns `Ok(model)`: check `model.status()` (`Status::Failed`,
  `Status::MaxEpoch`; only `Status::Solved` guarantees the duality gap).
- **Losses and coordinate steps:** `loss_and_grad`, `distillation_loss_and_grad`,
  `proximal_distillation_loss_and_grad`, and `coordinate_update` evaluate rows in parallel
  across `num_threads` threads (`None` uses `SDCA_NUM_THREADS` or automatic CPU detection;
  `Some(1)` runs serially; [threads](https://github.com/mlapin/libsdca/blob/v0.3.1/docs/API.md#threads)):

```rust
use libsdca::{loss_and_grad, Loss, LossOptions, Matrix};

fn main() -> libsdca::Result<()> {
    let scores = Matrix::new(&[2.0, -0.5, 1.0, 0.0, 1.0, 3.0], 2, 3)?;
    let options = LossOptions { loss: Loss::TopKHingeBeta, k: 2, gamma: 0.5, ..Default::default() };
    let (values, gradients) = loss_and_grad(scores, &[1, 0], &options)?;
    println!("{values:?} {gradients:?}");
    Ok(())
}
```

## Building

Install from crates.io with `cargo add libsdca`, or from git with
`cargo add libsdca --git https://github.com/mlapin/libsdca --tag v0.3.1`.

By default, `build.rs` compiles `src/c_api.cpp` with the system C++11 compiler (`CXX`,
default `c++`) and the built-in BLAS—no external dependencies are required. To link an
external CBLAS instead, set `SDCA_BLAS` (and optionally `SDCA_BLAS_BACKEND=EXTERNAL`):

| `SDCA_BLAS` entry | Meaning |
| --- | --- |
| `-L<dir>` | Add `<dir>` to the native library search path |
| `-l<name>` | Link `lib<name>` (e.g. `-lopenblas`) |
| `-l:<file>` | Link the exact library filename (e.g. `-l:libblas.so.3`) |
| `/abs/path/<file>.so[.N]` or `.a` | Search `/abs/path` and link `<file>` verbatim |

When linking static OpenBLAS on x86-64 Linux with Rust $\ge 1.90$ (where `lld` is default),
pass `RUSTFLAGS="-C link-arg=-fuse-ld=bfd"` so OpenBLAS's `.ctors` initializer runs. Cross
compilation requires `CXX` and `AR` set for the target.

```bash
cargo test --manifest-path rust/Cargo.toml
cargo run --manifest-path rust/Cargo.toml --example train
rust/package.sh                     # builds rust/target/package/libsdca-<version>.crate
```

## Precision

Working precision follows the input type: `Precision::Float32` for `f32` and
`Precision::Float64` for `f64`. Set `precision: Some(Precision::Float64)` to keep an
`f32` fit's internal iterate in `double` for tighter convergence; the returned
`Model<f32>` arrays retain their storage type. This costs `8(d + n)m` extra bytes
and can slow fitting. Both modes check the returned model's duality gap in `double`
([precision](https://github.com/mlapin/libsdca/blob/v0.3.1/docs/API.md#precision)).

```rust
use libsdca::{fit, FitOptions, Matrix, Precision};

fn main() -> libsdca::Result<()> {
    let x = Matrix::new(&[1.0f32, 0.0, 0.0, 1.0, -1.0, -1.0], 3, 2)?;
    let options = FitOptions { precision: Some(Precision::Float64), ..Default::default() };
    let model = fit(x, &[0, 1, 2], &options)?;
    println!("{:?}", model.weights());
    Ok(())
}
```

See [API.md](https://github.com/mlapin/libsdca/blob/v0.3.1/docs/API.md) for all options,
[projection precision](https://github.com/mlapin/libsdca/blob/v0.3.1/docs/API.md#projection-precision),
and [shrinking](https://github.com/mlapin/libsdca/blob/v0.3.1/docs/API.md#shrinking).
