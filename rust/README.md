# libsdca for Rust

Rust bindings to [libsdca](https://github.com/mlapin/libsdca/blob/v0.4.0/README.md),
accurate and fast classification losses for SDCA and neural networks: SDCA solvers for
linear and kernel SVMs and softmax classifiers, plus loss, gradient and coordinate-step
evaluation through the shared C ABI (`include/sdca/c_api.h`;
[technical report](https://mlapin.github.io/libsdca/report/)).

```rust
use libsdca::{Loss, Matrix, Solver};

fn main() -> libsdca::Result<()> {
    let x = Matrix::new(&[1.0, 0.0, 0.0, 1.0, -1.0, -1.0], 3, 2)?; // 3 examples, 2 features
    let y = [0, 1, 2];

    let model = Solver::new(Loss::cross_entropy()).fit(x, &y)?;
    println!("{} {:?}", model.status(), model.predict(x)?);

    let model = Solver::new(Loss::topk_hinge(2)).fit(x, &y)?; // top-2 SVM
    println!("{} {:?}", model.status(), model.predict(x)?);
    Ok(())
}
```

- **Losses:** `Loss` carries its parameters: `Loss::topk_hinge(2)` is the nonsmooth top-2
  hinge, `Loss::TopKHingeBeta { k: 2, gamma: 0.5 }` its smoothed form. `Solver` sets the
  training options (`with_c`, `with_tol`, `with_max_epoch`, ...) and `Settings` the
  precision, threads and projection solver.
- **Inputs and targets:** A `Matrix` of features (n, d), or `Input::Kernel` of a Gram
  matrix (n, n). `Targets` are 0-based labels (`&[usize]`), label sets for the multilabel
  losses (`&[Vec<usize>]` or `&[&[usize]]`), `Targets::Probabilities` (training
  cross-entropy against target distributions) or `Targets::Teacher` (distillation losses).
- **Errors and convergence:** Invalid inputs return `Error::InvalidArgument` and failures
  of the native library `Error::Native`. A fit that stops at `max_epoch` or fails
  numerically still returns `Ok(model)`: check `model.status()`; only `Status::Solved`
  guarantees the duality gap.
- **Losses and coordinate steps:** `Loss::loss_and_grad` and `Solver::coordinate_update`
  compute the rows on several threads with the same results for every count
  (`Threads::FromEnv` by default: `SDCA_NUM_THREADS`, else all available CPUs;
  [threads](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#threads)):

```rust
use libsdca::{Loss, Matrix};

fn main() -> libsdca::Result<()> {
    let scores = Matrix::new(&[2.0, -0.5, 1.0, 0.0, 1.0, 3.0], 2, 3)?;
    let (values, gradients) = Loss::topk_hinge(2).loss_and_grad(scores, &[1, 0])?;
    println!("{values:?} {gradients:?}");
    Ok(())
}
```

## Building

Install from crates.io with `cargo add libsdca`, or from git with
`cargo add libsdca --git https://github.com/mlapin/libsdca --tag v0.4.0`.

By default, `build.rs` compiles `src/c_api.cpp` with the system C++11 compiler (`CXX`,
default `c++`) and the built-in BLAS—no external dependencies are required. To link an
external CBLAS instead, set `SDCA_BLAS` (and optionally `SDCA_BLAS_BACKEND=EXTERNAL`):

| `SDCA_BLAS` entry | Meaning |
| --- | --- |
| `-L<dir>` | Add `<dir>` to the native library search path |
| `-l<name>` | Link `lib<name>` (e.g. `-lopenblas`) |
| `-l:<file>` | Link the exact library filename (e.g. `-l:libblas.so.3`) |
| `/abs/path/<file>.so[.N]` or `.a` | Search `/abs/path` and link `<file>` verbatim |

When linking static OpenBLAS on x86-64 Linux with Rust 1.90 or later (where `lld` is default),
pass `RUSTFLAGS="-C link-arg=-fuse-ld=bfd"` so OpenBLAS's `.ctors` initializer runs. Cross
compilation requires `CXX` and `AR` set for the target.

```bash
cargo test --manifest-path rust/Cargo.toml
cargo run --manifest-path rust/Cargo.toml --example train
rust/package.sh                     # builds rust/target/package/libsdca-<version>.crate
```

## Precision

The working precision follows the data type: `Precision::Float32` for `f32` and
`Precision::Float64` for `f64`. `with_precision(Precision::Float64)` keeps an `f32` fit's
iterate in `double` for tighter convergence, at `8(d + n)m` extra bytes; the model keeps
`f32` arrays. Both check the returned model's duality gap in `double`
([precision](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#precision)).

```rust
use libsdca::{Loss, Matrix, Precision, Solver};

fn main() -> libsdca::Result<()> {
    let x = Matrix::new(&[1.0f32, 0.0, 0.0, 1.0, -1.0, -1.0], 3, 2)?;
    let solver = Solver::new(Loss::cross_entropy()).with_precision(Precision::Float64);
    let model = solver.fit(x, &[0, 1, 2])?;
    println!("{:?}", model.weights());
    Ok(())
}
```

See [API.md](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md) for all options,
[projection precision](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#projection-precision),
and [shrinking](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#shrinking).
