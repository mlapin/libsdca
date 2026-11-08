# libsdca

[![CI](https://github.com/mlapin/libsdca/actions/workflows/ci.yml/badge.svg)](https://github.com/mlapin/libsdca/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/libsdca.svg)](https://pypi.org/project/libsdca/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**libsdca** trains linear and kernel classifiers with stochastic dual
coordinate ascent (SDCA) and provides the corresponding classification losses
as differentiable layers for PyTorch and JAX and as NumPy functions. It covers
multiclass and multilabel SVMs, softmax cross-entropy, knowledge distillation
to teacher logits, and the top-k losses of [Lapin et al.](#citation): smooth
and nonsmooth top-k hinge losses and the top-k entropy loss. Solver and layers
are numerically stable where standard implementations are not: the loss and
target gradient of a confident prediction stay nonzero to the subnormal limit
instead of collapsing to zero (softmax collapse [6]), and the distillation
gradient is exactly zero at equal student and teacher logits (softmax
cancellation [6]), in float16, bfloat16, float32 and float64 on CPU and CUDA.
The [known limits](docs/numerics.md#known-limits) list the exceptions: JAX on
the CPU flushes results below the smallest normal number to zero, and default
float32 work on a GPU can lose tails whose sum is representable.

The core is a header-only C++11 library with a CMake package and a C ABI. It
powers the NumPy, scikit-learn, Rust and MATLAB interfaces, and all of them
share one vocabulary and the same defaults. The PyTorch losses also have tensor implementations of the same
definitions for CPU and CUDA tensors.

```python
from libsdca import CrossEntropyLoss

criterion = CrossEntropyLoss()  # raw (N, C, ...) logits and integer targets
loss = criterion(logits, targets)
loss.backward()
```

For top-k training, use `TopKCrossEntropyLoss(k=5)` or `TopKHingeLoss(k=5)`.
[PyTorch compatibility](python/TORCH.md) explains weights, ignored targets,
precision and acceleration; [NumPy and scikit-learn](python/README.md) cover
training classifiers directly.

- **Dual solvers with a certificate.** Each coordinate step solves its block
  subproblem to working precision. Training evaluates the duality gap of the
  returned weights and duals in double precision, with a first-order rounding
  estimate `e`. The status `solved` requires
  `P(W) - D(A) + e <= tol * max(|P|, |D|)`. This estimates suboptimality under
  the [documented floating-point assumptions](docs/numerics.md); it is not
  a universal interval-arithmetic guarantee.
- **Numerically robust.** The softmax and top-k solvers stay accurate at
  extreme scores: confident predictions, large common offsets, underflowing
  probabilities and huge `C` ([numerical notes](docs/numerics.md)). The release
  validation passes 43,865 checks of the loss layers against 100-digit `mpmath`
  references on CPU and CUDA without a failure.
- **Precision control.** `precision="float32"` or `"float64"` selects working
  arithmetic independently of storage. Native training uses float32 or float64
  storage and defaults to float64 arithmetic. PyTorch and JAX also accept float16
  and bfloat16 logits. Gradients keep the logits dtype; every loss value has
  dtype `promote(logits dtype, float32)`, so float32 for 16-bit logits, as
  under `torch.autocast`. Default arithmetic depends on the
  loss and backend: in PyTorch, cross-entropy (integer targets, boolean masks
  and top-k cross-entropy) uses float32 exponentials with float64 row
  statistics, while the hinge paths use float64; the entropy losses of the
  jax backend compute narrower scores in float32
  ([JAX.md](python/JAX.md#backends)).
  See the [support matrix](docs/API.md#precision) and
  [numerical limits](docs/numerics.md#what-each-precision-guarantees).
- **Fits into existing code.** `SDCAClassifier` is a scikit-learn estimator
  that passes scikit-learn's own conformance suite (`check_estimator`), so it
  works in pipelines, grid search and cross-validation. The PyTorch losses
  run on CPU and CUDA tensors, and the JAX losses under `jit`, `grad` and
  `vmap`. `import libsdca` loads neither PyTorch, JAX nor scikit-learn until
  you use them.
- **No external BLAS required.** The C++ core uses an external CBLAS when one
  is available, and a built-in BLAS otherwise.

## Objectives

| Name | Loss | Labels |
| --- | --- | --- |
| `cross_entropy` | Softmax cross-entropy | one per example |
| `topk_cross_entropy` | Top-k entropy [5] | one per example |
| `multiclass_hinge` | Crammer-Singer SVM [1] | one per example |
| `topk_hinge_alpha`, `topk_hinge_beta` | Top-k hinge losses α [4] and β [5] | one per example |
| `multilabel_hinge` | Multilabel SVM [2] | label sets |
| `multilabel_cross_entropy` | Multilabel cross-entropy | label sets |

The hinges take `gamma`: 0 (the default) is the nonsmooth hinge, a positive
value its smoothed version [5].

Every objective minimizes `0.5‖W‖² + C Σᵢ loss(Wᵀxᵢ, yᵢ)` over features or a
precomputed kernel. The loss layers also offer `proximal_cross_entropy`, a Moreau
envelope of the entropy loss, `distillation_loss` between student and teacher logits
([distillation](python/TORCH.md#teacher-logits-and-distillation)), and
`linear_cross_entropy`, the cross-entropy of a language-model head computed without storing
its `(N, V)` logits ([linear layer](python/TORCH.md#linear-layer-and-cross-entropy-without-the-logits)).
The library further exposes the proximal
operators behind the solvers: projections onto the simplex [3], the top-k
simplex and cone, entropic projections and the Lambert W function of an
exponent.

## Installation

| Interface | Install | Guide |
| --- | --- | --- |
| Python (NumPy) | `pip install libsdca` | [python/README.md](python/README.md) |
| scikit-learn estimator | `pip install 'libsdca[sklearn]'` | [python/README.md](python/README.md#classifier) |
| PyTorch losses | `pip install 'libsdca[torch]'` | [python/TORCH.md](python/TORCH.md) |
| JAX losses | `pip install 'libsdca[jax]'` | [python/JAX.md](python/JAX.md) |
| Rust | `cargo add libsdca --git https://github.com/mlapin/libsdca --tag v0.3.0` (crates.io release to follow) | [rust/README.md](rust/README.md) |
| C++ (header-only) | CMake package `libsdca::libsdca` | [docs/CPP.md](docs/CPP.md) |
| C | CMake package `libsdca::sdca_c` (`-DSDCA_BUILD_C_API=ON`) | [docs/C.md](docs/C.md) |
| MATLAB | `run matlab/make.m` | [docs/MATLAB.md](docs/MATLAB.md) |
| GNU Octave | `OCTAVE_HOME=<prefix> tests/matlab/run_octave_tests.sh` | [docs/MATLAB.md](docs/MATLAB.md) |

Python wheels for Linux x86-64 (manylinux_2_28) and macOS 11+ on Apple
silicon are self-contained; elsewhere pip builds the sdist, which needs a
C++11 compiler. The PyTorch losses compile their fastest CPU and CUDA paths on
first use: install a C++17 compiler with OpenMP (C++20 for PyTorch 2.12 and
later; checked with GCC 11, Clang 14 and 18, Apple clang 21; Linux:
`build-essential` or `gcc-c++`; macOS: `xcode-select --install`) and the Python
headers (Debian/Ubuntu: `python3-dev`); ninja comes with `libsdca[torch]`. Without them the losses still
run, more slowly, with one warning ([build tools](python/TORCH.md#build-tools),
[environment variables](docs/API.md#environment-variables)). The native JAX
losses likewise compile a small XLA FFI handler on first use (C++17 compiler
only) and otherwise use a host callback. CMake 3.12 or newer is needed only for the CMake packages. See
[DEPENDENCIES.md](DEPENDENCIES.md) for the BLAS options. Linux x86-64 is
validated in full, macOS arm64 for the C++ tests and the core and PyTorch CPU suites
([tested configurations](DEPENDENCIES.md#tested-configurations)).

```bash
git clone https://github.com/mlapin/libsdca && cd libsdca
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/usr/local   # add -DSDCA_BUILD_C_API=ON for C
cmake --build build --target install
```

## Quick start

### Python

```python
import numpy as np
import libsdca

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 10))                        # (examples, features)
y = (X @ rng.normal(size=(10, 3))).argmax(axis=1)     # labels 0, 1, 2

# scikit-learn style (pip install 'libsdca[sklearn]')
clf = libsdca.SDCAClassifier("topk_cross_entropy", k=2, C=1.0).fit(X, y)
print(clf.score(X, y))

# or the functional API, with the full training history
model = libsdca.fit(X, y, "topk_cross_entropy", k=2, C=1.0)
print(model.status, model.history["relative_gap"][-1])
values, gradients = libsdca.loss_and_grad(X @ model.weights, y, "topk_hinge_alpha", k=2, gamma=1.0)

# float32 data is solved in double by default; float arithmetic is faster
fast = libsdca.fit(X.astype(np.float32), y, precision="float32")
```

### PyTorch

```python
import torch
from libsdca import CrossEntropyLoss

net = torch.nn.Linear(10, 5)
# Replace: criterion = torch.nn.CrossEntropyLoss()
criterion = CrossEntropyLoss()
loss = criterion(net(torch.randn(32, 10)), torch.randint(0, 5, (32,)))
loss.backward()
```

For ordinary multiclass softmax cross-entropy, that is the only change:
pass the same raw logits and integer labels, and keep your model and optimizer.
The loss follows the logits onto CPU or CUDA. It preserves tiny losses and
target gradients for confident correct predictions that standard cross-entropy
can round to zero. On an A100, at 8192×32000 and 8192×128256 bfloat16 or float32
logits, its forward+backward takes 0.33–0.41× the time of eager `F.cross_entropy` and
0.93–0.98× `torch.compile`'s, and keeps one fewer `(N, C)` tensor than eager. Small,
host-bound calls up to about 10⁶ logits take longer
([where libsdca is slower](docs/performance.md#where-libsdca-is-slower)).
See the [argument parity](python/TORCH.md#argument-parity) with `torch.nn.CrossEntropyLoss`.
To try a top-k loss instead, use `TopKHingeLoss(k=2, variant="beta", gamma=0.5)`
from `libsdca`. The generic `SDCALoss` interface remains available.

### Rust

```rust
use libsdca::{FitOptions, LossOptions, Loss, Matrix, Objective};

let x = Matrix::new(&[1.0, 0.0, 0.0, 1.0, -1.0, -1.0], 3, 2)?;   // (n, d) row-major
let options = FitOptions { c: 10.0, objective: Objective::CrossEntropy, ..Default::default() };
let model = libsdca::fit(x, &[0, 1, 2], &options)?;
println!("{} {:?}", model.status(), model.predict(x)?);

let scores = Matrix::new(&[0.3, -0.8, 1.2], 1, 3)?;
let loss = LossOptions { loss: Loss::TopKHingeBeta, k: 2, gamma: 1.0, ..Default::default() };
let (values, gradients) = libsdca::loss_and_grad(scores, &[0], &loss)?;
```

### C++

```cpp
#include "sdca/solver.h"
#include "sdca/utility/logging.cpp"   // in exactly one translation unit

// Column-major features (d x n), zero-based labels, caller-owned buffers.
auto context = sdca::make_context(
    sdca::make_input_feature(d, n, features), sdca::multiclass_output(m, labels),
    sdca::l2_entropy<double, double>(/* C */ 1.0), dual, weights);
sdca::make_solver(context).solve();
```

### MATLAB

```matlab
model = matsdca_fit(X, Y, struct('objective', 'cross_entropy', 'C', 1));   % X: d-by-n
[~, predicted] = max(model.W' * Xtest);
```

Complete, runnable programs are in [examples/](examples) (Python, C++ and C),
[rust/examples](rust/examples) and [matlab/demo.m](matlab/demo.m).

## Documentation

- [API reference](docs/API.md): choosing a loss, definitions, names, options,
  defaults, precision, results and errors
- Interface guides: [Python](python/README.md), [PyTorch](python/TORCH.md),
  [JAX](python/JAX.md), [Rust](rust/README.md), [C++](docs/CPP.md), [C](docs/C.md),
  [MATLAB](docs/MATLAB.md)
- [Numerical notes](docs/numerics.md): precision guarantees, verification and limits
- [Performance](docs/performance.md): runtime and memory against PyTorch, JAX and
  scikit-learn
- [Softmax collapse and cancellation](docs/softmax-collapse.md): the fixes of
  Barron et al. [6] and a reproduction of their experiments
- [Dependencies](DEPENDENCIES.md), [contributing](CONTRIBUTING.md) and the
  [changelog](CHANGELOG.md)

The [project page](https://mlapin.github.io/libsdca/) has the technical report with
the full benchmark tables.

The truncated top-k entropy loss [5] is available in release
[v0.2.0](https://github.com/mlapin/libsdca/releases/tag/v0.2.0) (`libsdca_gd`).

## Citation

If libsdca helps your research, please cite the papers that introduced the
losses you use:

```bibtex
@inproceedings{lapin2015topk,
  title     = {Top-k Multiclass {SVM}},
  author    = {Lapin, Maksim and Hein, Matthias and Schiele, Bernt},
  booktitle = {Advances in Neural Information Processing Systems (NIPS)},
  year      = {2015}
}

@inproceedings{lapin2016loss,
  title     = {Loss Functions for Top-k Error: Analysis and Insights},
  author    = {Lapin, Maksim and Hein, Matthias and Schiele, Bernt},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016}
}
```

## References

1. K. Crammer and Y. Singer. On the algorithmic implementation of multiclass
   kernel-based vector machines. *JMLR*, 2001.
2. K. Crammer and Y. Singer. A family of additive online algorithms for
   category ranking. *JMLR*, 2003.
3. K. C. Kiwiel. Variable fixing algorithms for the continuous quadratic
   knapsack problem. *JOTA*, 2008.
4. M. Lapin, M. Hein and B. Schiele. Top-k multiclass SVM. *NIPS*, 2015.
5. M. Lapin, M. Hein and B. Schiele. Loss functions for top-k error: analysis
   and insights. *CVPR*, 2016.
6. J. T. Barron, D. Gnanapragasam and M. Tschannen. A fix for softmax and
   cross-entropy. arXiv:2610.31642, 2026. libsdca follows the paper's
   formulations (the `log1p` normalizer and the centered `expm1` gradient of
   its code samples) and is independent of the framework patches and packages
   that implement them; its [reproduction](docs/softmax-collapse.md#reproduction)
   covers Sections 3, 6 and 7 with every libsdca loss.

## License

libsdca is released under the [MIT License](LICENSE).
