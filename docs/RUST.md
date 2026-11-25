# libsdca for Rust

[All guides](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/README.md) ·
[Shared API reference](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md)

Rust bindings to [libsdca](https://github.com/mlapin/libsdca/blob/v0.4.0/README.md),
accurate and fast classification losses for SDCA and neural networks: SDCA solvers for
linear and kernel SVMs and softmax classifiers, plus loss, gradient and coordinate-step
evaluation through the shared C ABI (`include/sdca/c_api.h`;
[technical report](https://mlapin.github.io/libsdca/report/)).

## Installation

Install from crates.io with `cargo add libsdca`, or from git with
`cargo add libsdca --git https://github.com/mlapin/libsdca --tag v0.4.0`.

### Building

`build.rs` compiles `src/c_api.cpp` with the system C++11 compiler (`CXX`, default `c++`)
and the flags of `src/native_build.txt`, and links the BLAS of the rule that every build
shares ([API.md](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#blas)): with `SDCA_BLAS_BACKEND=AUTO` (the default), the first
system CBLAS that links (Accelerate on macOS, then OpenBLAS, CBLAS or BLAS), else the
built-in products with a Cargo warning. `BUILTIN` takes the built-in products, `EXTERNAL`
requires a CBLAS, and `SDCA_BLAS` names one:

| `SDCA_BLAS` entry | Meaning |
| --- | --- |
| `-L<dir>` | Add `<dir>` to the native library search path |
| `-l<name>` | Link `lib<name>` (e.g. `-lopenblas`) |
| `-l:<file>` | Link the exact library filename (e.g. `-l:libblas.so.3`) |
| `/abs/path/<file>.so[.N]` or `.a` | Search `/abs/path` and link `<file>` verbatim |

When linking static OpenBLAS on x86-64 Linux with Rust 1.90 or later (where `lld` is default),
pass `RUSTFLAGS="-C link-arg=-fuse-ld=bfd"` so OpenBLAS's `.ctors` initializer runs. Cross
compilation requires `CXX` and `AR` set for the target.

From a source checkout, `cargo run --manifest-path rust/Cargo.toml --example train` runs the
[example](https://github.com/mlapin/libsdca/blob/v0.4.0/rust/examples/train.rs).

## Quick start

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

## Losses and options

`Loss` carries its parameters: `Loss::topk_hinge(2)` is the nonsmooth top-2
hinge, `Loss::TopKHinge { k: 2, variant: TopKVariant::Beta, gamma: 0.5 }` its smoothed
form (`TopKVariant::Alpha` the other variant).

`Solver::new(loss)` trains with `fit(input, targets)` or `fit_from(input, targets, &start)`
(`Start::center`, `Start::initial_dual`), and takes one SDCA dual block step with
`coordinate_update(logits, targets, norm2, previous)`. Its builder methods:

| Method | Default | Meaning |
| --- | --- | --- |
| `with_c(c)` | `1.0` | Loss weight, finite and positive |
| `with_tol(tol)` | `1e-3` | Relative duality gap of success (`0`: run to `max_epoch`) |
| `with_max_epoch(n)` | `1000` | Epoch limit |
| `with_eval_epoch(n)` | `10` | Evaluation period (`0`: only at the end) |
| `with_eval_on_start(b)` | `false` | Also evaluate before the first epoch |
| `with_recompute_primal(b)` | `true` | Rebuild the weights from the duals at each evaluation |
| `with_shrinking(s)` | `Shrinking::Auto` | Active-set shrinking of the hinges ([shrinking](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#shrinking)) |
| `with_seed(seed)` | `0` | Order of the examples in the epochs |
| `with_num_classes(m)` | largest label + 1 | Number of classes |
| `with_settings(s)`, `with_precision(p)`, `with_threads(t)` | `Settings::default()` | Arithmetic and threads |

`Settings` holds `precision` (`None`: the data type's), `threads` (`Threads::FromEnv`) and
`check_rows` (`false`), set with `with_precision`, `with_threads` and `with_check_rows`.

## Inputs and targets

- **Input:** a `Matrix` of features (n, d), or `Input::Kernel` of a Gram matrix (n, n).
- **Targets:** 0-based labels (`&[usize]`), label sets for the multilabel losses
  (`&[Vec<usize>]` or `&[&[usize]]`), `Targets::Probabilities` (cross-entropy against
  target distributions; for training each row sums to one within the rounding of a row
  normalized in the data type, and the library renormalizes it) or `Targets::Teacher` (distillation
  losses).

## Array layouts

Every array is row-major and shared with the native library in place: nothing is copied or
reordered ([layouts](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#array-layouts)).

| Array | Shape |
| --- | --- |
| `Model::weights`, `Start::center` | (num_classes, d): one weight vector per class |
| `Model::dual`, `Start::initial_dual` | (n, num_classes) |
| `Model::decision_function` | (n_test, num_classes) |

`decision_function` is one matrix product (`gemm`) of the external BLAS (the built-in
products without one; [above](#building)).

## Errors and convergence

Invalid inputs return `Error::InvalidArgument` and failures
of the native library `Error::Native`. The native library checks the values (the
messages are the C ABI's); the crate checks the sizes of its slices. A fit that stops at `max_epoch` or fails
numerically still returns `Ok(model)`: check `model.status()`; only `Status::Solved`
guarantees the duality gap.

## Losses and gradients

`Loss::loss_and_grad` and `Solver::coordinate_update`
compute the rows on several threads with the same results for every count
(`Threads::FromEnv` by default: `SDCA_NUM_THREADS`, else all available CPUs;
[threads](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#threads)).
`loss_and_grad` takes scores of `f32::NEG_INFINITY` or `f64::NEG_INFINITY` as
[masked classes](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#masked-classes)
for every loss. A label out of range is `Error::InvalidArgument` (`Target 7 is out of
bounds.`, as in PyTorch). A row it cannot compute (`NaN` or `+inf` scores, an invalid row
of masked classes) gets `NaN` outputs, and the returned `LossReport` counts
these rows (`invalid_rows`, `first_invalid_row`); `Settings::with_check_rows(true)` returns
`Error::InvalidArgument` for them instead:

```rust
use libsdca::{Loss, Matrix};

fn main() -> libsdca::Result<()> {
    let scores = Matrix::new(&[2.0, -0.5, 1.0, 0.0, 1.0, 3.0], 2, 3)?;
    let (values, gradients, report) = Loss::topk_hinge(2).loss_and_grad(scores, &[1, 0])?;
    assert!(report.is_valid());
    println!("{values:?} {gradients:?}");
    Ok(())
}
```

`Loss::loss_and_grad_with_options` takes PyTorch's `cross_entropy` options as `Options`:
class weights and an ignored label for the single-label losses, label smoothing for
`CrossEntropy` ([options](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#loss-evaluation-options));
an option the loss does not take is `Error::InvalidArgument`:

```rust
use libsdca::{Loss, Matrix, Options, Settings};

fn main() -> libsdca::Result<()> {
    let scores = Matrix::new(&[2.0, -0.5, 1.0, 0.0, 1.0, 3.0], 2, 3)?;
    let weight = [1.0, 2.0, 0.5];
    let options = Options::default()
        .with_weight(&weight)
        .with_label_smoothing(0.1)
        .with_ignore_index(1);
    let settings = Settings::default();
    let (values, _, _) =
        Loss::cross_entropy().loss_and_grad_with_options(scores, &[1, 0], &options, &settings)?;
    assert_eq!(values[0], 0.0); // label 1 is ignored
    Ok(())
}
```

## Precision

The working precision follows the data type: `Precision::Float32` for `f32` and
`Precision::Float64` for `f64`.

`with_precision(Precision::Float64)` keeps an `f32` fit's iterate in `double` for tighter
convergence, at `8(d + n)m` extra bytes; the model keeps `f32` arrays. Both check the
returned model's duality gap in `double`
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

See [API.md](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md) for all options and
[shrinking](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#shrinking).
