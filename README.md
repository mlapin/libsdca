# libsdca

[![CI](https://github.com/mlapin/libsdca/actions/workflows/ci.yml/badge.svg)](https://github.com/mlapin/libsdca/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/libsdca.svg)](https://pypi.org/project/libsdca/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/libsdca/)
[![Crates.io](https://img.shields.io/crates/v/libsdca.svg)](https://crates.io/crates/libsdca)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**libsdca** provides accurate and fast classification losses for SDCA and neural
networks. It implements multiclass, top-*k*, multilabel, and soft-target
classification losses in two forms: fused CPU and CUDA loss layers for PyTorch and
JAX (and NumPy functions), and stochastic dual coordinate ascent (SDCA) solvers for
linear and kernel models. It covers cross-entropy, the multiclass hinge, knowledge
distillation, the multilabel losses, and the top-*k* hinge and entropy losses of
[Lapin et al.](#citation).

Every loss and gradient is accurate to the limit of its floating-point format:
confident predictions keep nonzero losses and target gradients down to subnormal
underflow instead of collapsing to zero (softmax collapse [7], fixed in [6]), and
distillation gradients are exact zero at equal student and teacher logits (softmax
cancellation [6]), across `float16`, `bfloat16`, `float32` and `float64` on CPU
and CUDA ([known limits](docs/numerics.md#known-limits)).

```python
from libsdca import CrossEntropyLoss

criterion = CrossEntropyLoss()  # drop-in for torch.nn.CrossEntropyLoss(), except integer
                                # class labels only and no label smoothing
loss = criterion(logits, targets)
loss.backward()
```

- **Accurate to the limit of the format:** Every loss value and gradient entry is
  verified within `64` output-dtype eps against 100-digit `mpmath` references across
  `2` to `32,769` classes, subnormal tails, cancellation offsets and `±finfo.max`
  ([verification](docs/numerics.md#how-it-is-verified)).
- **Fast fused loss kernels:** On an A100 at `8192 × 32000` and `8192 × 128256`
  (`bfloat16` / `float32`), `CrossEntropyLoss` runs in `0.33–0.41×` eager
  `F.cross_entropy` time (`0.93–0.98×` `torch.compile`) and saves a few scalars
  per row instead of an `(N, C)` tensor ([performance](docs/performance.md)).
- **Certified SDCA:** Training evaluates the duality gap of the returned
  weights and the feasible projection of the returned duals in `double`, with a
  first-order rounding estimate $e$;
  `status == "solved"` certifies $P(W) - D(A) + e \le \text{tol} \cdot \max(|P|, |D|)$
  ([numerics](docs/numerics.md#the-solved-certificate)).
- **Decoupled storage and compute precision:** `precision="float32"` or `"float64"`
  selects working arithmetic independently of array/tensor storage
  ([support matrix](docs/API.md#precision)).
- **Zero-overhead imports & optional BLAS:** `import libsdca` loads neither
  PyTorch, JAX nor scikit-learn until used, and the C++ core includes a built-in
  BLAS fallback when no external CBLAS is installed.

## Objectives

| Name | Loss | Targets |
| --- | --- | --- |
| `cross_entropy` | Softmax cross-entropy | single class index |
| `topk_cross_entropy` | Top-*k* entropy [5] | single class index |
| `multiclass_hinge` | Multiclass hinge (Crammer–Singer SVM) [1] | single class index |
| `topk_hinge_alpha`, `topk_hinge_beta` | Top-*k* hinge $\alpha$ [4] and $\beta$ [5] | single class index |
| `multilabel_hinge` | Multilabel hinge (multilabel SVM) [2] | label sets / boolean mask |
| `multilabel_cross_entropy` | Multilabel cross-entropy | label sets / boolean mask |

Hinge losses accept `gamma >= 0` (`0` = nonsmooth hinge, `> 0` = Moreau-smoothed
hinge [5]). The loss layers additionally provide `proximal_cross_entropy`,
`distillation_loss` ([TORCH.md](python/TORCH.md#teacher-logits-and-distillation)),
`proximal_distillation_loss`, and `linear_cross_entropy`, which evaluates a
language-model linear head + cross-entropy in chunks without materializing `(N, V)`
logits ([TORCH.md](python/TORCH.md#linear-layer-and-cross-entropy-without-the-logits)).

## Installation

| Interface | Install | Guide |
| --- | --- | --- |
| Python (NumPy) | `pip install libsdca` | [python/README.md](python/README.md) |
| scikit-learn estimator | `pip install 'libsdca[sklearn]'` | [python/README.md](python/README.md#classifier) |
| PyTorch losses | `pip install 'libsdca[torch]'` | [python/TORCH.md](python/TORCH.md) |
| JAX losses | `pip install 'libsdca[jax]'` | [python/JAX.md](python/JAX.md) |
| Rust | `cargo add libsdca` | [rust/README.md](rust/README.md) |
| C++ (header-only) | CMake target `libsdca::libsdca` | [docs/CPP.md](docs/CPP.md) |
| C | CMake target `libsdca::sdca_c` (`-DSDCA_BUILD_C_API=ON`) | [docs/C.md](docs/C.md) |
| MATLAB / Octave | `run matlab/make.m` / `tests/matlab/run_octave_tests.sh` | [docs/MATLAB.md](docs/MATLAB.md) |

- **Python wheels:** Prebuilt wheels for Linux `x86_64` (`manylinux_2_28`) and
  macOS 11+ (`arm64`) are self-contained; other platforms build from the sdist with
  a C++11 compiler.
- **Runtime C++ extensions:** PyTorch and JAX compile small C++ extensions on first
  use when a C++17/C++20 compiler and Python headers are installed (`build-essential`
  and `python3-dev` on Debian/Ubuntu; `xcode-select --install` on macOS). Without a
  compiler, losses fall back automatically with a one-time warning
  ([build tools](python/TORCH.md#build-tools), [environment variables](docs/API.md#environment-variables)).
- **CMake and BLAS:** See [DEPENDENCIES.md](DEPENDENCIES.md) for CMake and BLAS options.

```bash
git clone https://github.com/mlapin/libsdca && cd libsdca
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/usr/local   # add -DSDCA_BUILD_C_API=ON for C
cmake --build build --target install
```

## Quick start

### Python (scikit-learn and NumPy)

```python
import numpy as np
import libsdca

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 10))
y = (X @ rng.normal(size=(10, 3))).argmax(axis=1)

clf = libsdca.SDCAClassifier("topk_cross_entropy", k=2, C=1.0).fit(X, y)
model = libsdca.fit(X, y, "topk_cross_entropy", k=2, C=1.0)
values, gradients = libsdca.loss_and_grad(X @ model.weights, y, "topk_hinge_alpha", k=2, gamma=1.0)
```

### PyTorch

```python
import torch
from libsdca import CrossEntropyLoss, TopKHingeLoss

net = torch.nn.Linear(10, 5)
criterion = CrossEntropyLoss()  # or TopKHingeLoss(k=2, variant="beta", gamma=0.5)
loss = criterion(net(torch.randn(32, 10)), torch.randint(0, 5, (32,)))
loss.backward()
```

### JAX

```python
import jax
import libsdca.jax as sjax

loss_and_grad = jax.jit(jax.value_and_grad(
    lambda w, x, y: sjax.sdca_loss(x @ w, y, "topk_hinge_beta", k=3, gamma=0.5)
))
```

### Rust

```rust
use libsdca::{fit, loss_and_grad, FitOptions, Loss, LossOptions, Matrix, Objective};

let x = Matrix::new(&[1.0, 0.0, 0.0, 1.0, -1.0, -1.0], 3, 2)?;
let fit_opts = FitOptions { c: 10.0, objective: Objective::CrossEntropy, ..Default::default() };
let model = fit(x, &[0, 1, 2], &fit_opts)?;
let scores = Matrix::new(&[0.3, -0.8, 1.2], 1, 3)?;
let loss_opts = LossOptions { loss: Loss::TopKHingeBeta, k: 2, gamma: 1.0, ..Default::default() };
let (values, gradients) = loss_and_grad(scores, &[0], &loss_opts)?;
```

### C++

```cpp
#include "sdca/solver.h"
#include "sdca/utility/logging.cpp"   // in exactly one translation unit

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

Runnable examples are in [examples/](examples), [rust/examples](rust/examples) and
[matlab/demo.m](matlab/demo.m).

## Documentation

- [API reference](docs/API.md): loss formulas, cross-language names, options,
  precision, results and environment variables
- Interface guides: [Python](python/README.md), [PyTorch](python/TORCH.md),
  [JAX](python/JAX.md), [Rust](rust/README.md), [C++](docs/CPP.md), [C](docs/C.md),
  [MATLAB](docs/MATLAB.md)
- [Numerical notes](docs/numerics.md): precision guarantees, verification and known limits
- [Performance](docs/performance.md): CUDA, CPU and solver benchmarks
- [Softmax collapse and cancellation](docs/softmax-collapse.md): formulations and
  reproduction of Barron et al. [6]
- [Dependencies](DEPENDENCIES.md), [contributing](CONTRIBUTING.md),
  [AI coding assistant guide](AGENTS.md) and [changelog](CHANGELOG.md)
- [Project page](https://mlapin.github.io/libsdca/) and technical report,
  [*libsdca: Accurate and Fast Classification Losses for SDCA and Neural Networks*](https://mlapin.github.io/libsdca/report/)

## Citation

If `libsdca` is useful in your research, please cite the paper for the losses you use
and the technical report for the library:

```bibtex
@inproceedings{lapin2015topk,
  title     = {Top-k Multiclass {SVM}},
  author    = {Lapin, Maksim and Hein, Matthias and Schiele, Bernt},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2015}
}

@inproceedings{lapin2016loss,
  title     = {Loss Functions for Top-k Error: Analysis and Insights},
  author    = {Lapin, Maksim and Hein, Matthias and Schiele, Bernt},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016}
}

@article{lapin2018analysis,
  title     = {Analysis and Optimization of Loss Functions for Multiclass, Top-k, and Multilabel Classification},
  author    = {Lapin, Maksim and Hein, Matthias and Schiele, Bernt},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  volume    = {40},
  number    = {7},
  pages     = {1533--1554},
  year      = {2018}
}

@misc{lapin2026libsdca,
  title        = {libsdca: Accurate and Fast Classification Losses for {SDCA} and Neural Networks},
  author       = {Lapin, Max},
  year         = {2026},
  howpublished = {Technical report, \url{https://mlapin.github.io/libsdca/report/}}
}
```

GitHub's *Cite this repository* button uses [CITATION.cff](CITATION.cff).

## References

1. K. Crammer and Y. Singer. On the algorithmic implementation of multiclass
   kernel-based vector machines. *JMLR*, 2001.
2. K. Crammer and Y. Singer. A family of additive online algorithms for
   category ranking. *JMLR*, 2003.
3. K. C. Kiwiel. Variable fixing algorithms for the continuous quadratic
   knapsack problem. *JOTA*, 2008.
4. M. Lapin, M. Hein and B. Schiele. Top-k multiclass SVM. *NeurIPS*, 2015.
5. M. Lapin, M. Hein and B. Schiele. Loss functions for top-k error: analysis
   and insights. *CVPR*, 2016.
6. J. T. Barron, D. Gnanapragasam and M. Tschannen. A fix for softmax and
   cross-entropy. arXiv:2610.31642, 2026 ([reproduction](docs/softmax-collapse.md#reproduction)).
7. L. Prieto, M. Barsbey, P. A. M. Mediano and T. Birdal. Grokking at the edge of
   numerical stability. *ICLR*, 2025.

## License

`libsdca` is released under the [MIT License](LICENSE).
