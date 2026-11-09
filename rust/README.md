# libsdca for Rust

Rust bindings to [libsdca](https://github.com/mlapin/libsdca/blob/v0.3.0/README.md): SDCA solvers for multiclass and
multilabel classification (cross-entropy, top-k cross-entropy, multiclass
and top-k hinge, and multilabel losses) and their losses, gradients and
coordinate steps.

The API mirrors the [Python interface](https://github.com/mlapin/libsdca/blob/v0.3.0/python/README.md): the same
objective, loss, status and history names, the same options and defaults,
and the same errors, through the shared C ABI (`include/sdca/c_api.h`).

For feature-space training with a center, history and the stopping certificate
use the full regularizer `0.5*||W-center||^2` and its matching dual objective.
No center-dependent constant is omitted. Large centers can still make small
updates unrepresentable in float storage; inspect the returned status and use
double storage when necessary.

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

Targets are labels in `[0, num_classes)` (`&[usize]`) or, for the multilabel
objectives, label sets (`&[Vec<usize>]`, `&[&[usize]]`); see `Targets`.
Invalid inputs return `Error::InvalidArgument`, native failures
`Error::Native`; not converging is no error (`Model::status`).

`loss_and_grad` evaluates a loss and its gradient on scores (n, classes),
for example as the loss layer of a model trained elsewhere, and
`coordinate_update` takes one dual coordinate step:

```rust
use libsdca::{loss_and_grad, Loss, LossOptions, Matrix};

fn main() -> libsdca::Result<()> {
    let scores = Matrix::new(&[2.0, -0.5, 1.0, 0.0, 1.0, 3.0], 2, 3)?; // (n, classes)
    let options = LossOptions { loss: Loss::TopKHingeBeta, k: 2, gamma: 0.5, ..Default::default() };
    let (values, gradients) = loss_and_grad(scores, &[1, 0], &options)?; // (n), (n, classes)
    println!("{values:?} {gradients:?}");
    Ok(())
}
```

`LossOptions::num_threads` and `UpdateOptions::num_threads` can split the
rows among native threads: the default `None` takes `SDCA_NUM_THREADS`, else
automatic threads (the available CPUs, as the work allows, as `Some(0)`);
`Some(1)` runs on the calling thread, and the results are identical for every count
([threads](https://github.com/mlapin/libsdca/blob/v0.3.0/docs/API.md#threads)).

`projection_precision` (in `FitOptions`, `LossOptions` and `UpdateOptions`)
selects how the `multilabel_hinge` projections solve:
`ProjectionPrecision::Float64` (default) solves in double, checks the result
and solves again with the extended-precision solver where the check fails;
`ProjectionPrecision::Extended` always runs that solver
([projection precision](https://github.com/mlapin/libsdca/blob/v0.3.0/docs/API.md#projection-precision)).

## Building

`cargo build` compiles `src/c_api.cpp` with the system C++ compiler (`CXX`,
default `c++`) and the built-in BLAS; no other dependency is needed. To link
an external CBLAS instead, set `SDCA_BLAS` to whitespace-separated entries,
optionally with `SDCA_BLAS_BACKEND=EXTERNAL`:

| Entry | Meaning |
| --- | --- |
| `-L<dir>` | search `<dir>` for libraries |
| `-l<name>` | link `lib<name>`, for example `-lopenblas` |
| `-l:<file>` | link the library file named exactly `<file>`, for example `-l:libblas.so.3` |
| `/abs/path/<file>.so[.N]`, `/abs/path/<file>.a` | search `/abs/path` and link `<file>` by its exact name |

The libraries propagate to the crates that depend on `libsdca`. Any other
entry is a build error; pass other linker flags in `RUSTFLAGS`.

Any CBLAS works. CI builds a downstream crate (`rust/consumer`) against
Ubuntu's reference BLAS (`libblas-dev`), OpenBLAS (`libopenblas-dev`) and
BLIS (`libblis-dev`), each as `-l<name>`, `-l:<file>`, an absolute `.so`, an
absolute static `.a` and `-L<dir> -l<name>`, and against the built-in BLAS.
Static OpenBLAS registers its initializer in `.ctors`, which lld (Rust's
default linker on x86-64 Linux since 1.90) never runs; link it with GNU ld:
`RUSTFLAGS="-C link-arg=-fuse-ld=bfd"`. MKL and Apple Accelerate are untested. The build
uses the host's tools, so cross compilation needs `CXX` and `AR` set to the
target's.

Install from crates.io with `cargo add libsdca` (or from git with
`cargo add libsdca --git https://github.com/mlapin/libsdca --tag v0.3.0`).
Published crates vendor the C++ sources in `native/`; in a repository checkout
or git dependency, `build.rs` compiles the surrounding C++ sources (or
`SDCA_SOURCE_DIR`). `rust/package.sh` vendors them into `rust/native/` to build
a self-contained `.crate` archive.

```bash
cargo test --manifest-path rust/Cargo.toml
cargo run --manifest-path rust/Cargo.toml --example train
rust/package.sh                     # rust/target/package/libsdca-<version>.crate
```

## Precision

The element type of the data selects the storage, and the `precision`
field of the options selects the working arithmetic of `fit`, `loss_and_grad`, and
`coordinate_update`, and `fit`'s iterate (an `f32` fit in `Precision::Float64` keeps `W`
and the duals in double and returns them rounded once, certified as returned).
`f64` data computes in double (`Precision::Float64`).
For `f32` data, the default `Precision::Float64` computes in double (entropy
blocks measured within a few float ulps of `f64` on the tested corpus, not a
bound; [limits](https://github.com/mlapin/libsdca/blob/v0.3.0/docs/numerics.md#known-limits)), and `precision: Some(Precision::Float32)` in
float, accurate to float rounding of the scores; requesting `Precision::Float32`
on `f64` data is an `Error::InvalidArgument`. Float arithmetic speeds up the entropy steps
and `loss_and_grad` (which computes in double where float cannot hold
`gamma`, `alpha` or a row's score spread); the hinge steps are no faster.

```rust
use libsdca::{fit, FitOptions, Matrix, Precision};

fn main() -> libsdca::Result<()> {
    let x = Matrix::new(&[1.0f32, 0.0, 0.0, 1.0, -1.0, -1.0], 3, 2)?;
    let options = FitOptions { precision: Some(Precision::Float32), ..Default::default() };
    let model = fit(x, &[0, 1, 2], &options)?; // Model<f32>
    println!("{:?}", model.weights());
    Ok(())
}
```
