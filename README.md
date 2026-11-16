# libsdca

[![CI](https://github.com/mlapin/libsdca/actions/workflows/ci.yml/badge.svg)](https://github.com/mlapin/libsdca/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/libsdca.svg)](https://pypi.org/project/libsdca/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/libsdca/)
[![Crates.io](https://img.shields.io/crates/v/libsdca.svg)](https://crates.io/crates/libsdca)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**libsdca** is a fast and accurate implementation of classification losses: cross-entropy,
multiclass and multilabel SVM hinges, top-*k* losses and knowledge distillation. The losses
are available as PyTorch and JAX layers on CPU and CUDA and as NumPy, Rust, C++, C and MATLAB
functions, and a stochastic dual coordinate ascent (SDCA) solver trains linear and kernel
classifiers with them.

- **Fast:** The losses are fused CPU and CUDA kernels, not chains of tensor operations. On an
  A100 with 32k–128k classes, cross-entropy matches `torch.compile` and is 2.4–3× faster than
  eager PyTorch, and the multiclass hinge is 4–8× faster than compiled and 8–14× faster than
  eager PyTorch implementations ([benchmarks](python/TORCH.md#benchmarks)).
- **Accurate:** Losses and gradients of confident predictions are kept down to the subnormal
  range of the output dtype: a naive `float32` cross-entropy rounds to zero from a margin of
  about 17 (softmax collapse [7]), and `libsdca` stays accurate up to about 104. Distillation
  gradients are exactly zero at equal student and teacher logits [6]. The entropy and
  distillation losses are tested on CPU and CUDA against 100-digit `mpmath` references
  ([known limits](docs/API.md#known-limits)).
- **Many losses:** Cross-entropy, the multiclass hinge (Crammer–Singer SVM [1]), the top-*k*
  hinge and entropy losses [4, 5], the multilabel hinge [2] and cross-entropy, logit
  distillation and proximal variants, with nonsmooth or smoothed hinges, and
  `linear_cross_entropy` for a language-model head without its `(N, V)` logits
  ([losses](#losses)).
- **Many interfaces:** PyTorch modules and functions (with `torch.compile` support), JAX
  functions for `jit`, `grad` and `vmap`, NumPy, a scikit-learn estimator, Rust, header-only
  C++, C and MATLAB/Octave, with the same loss and option names in each
  ([API reference](docs/API.md)).
- **SDCA solver with a duality-gap guarantee:** SDCA trains linear and kernel classifiers
  (multiclass, top-*k* and multilabel SVMs, softmax regression). The duality gap bounds the
  distance to the optimum, so `status == "solved"` guarantees that the objective of the
  returned model is within the relative tolerance `tol` of the optimal value
  ([results](docs/API.md#results)). Its steps use efficient projections onto the simplex and
  the top-*k* cone [3, 4], which C++ and MATLAB also provide directly.

## Installation

Create a fresh environment with [uv](https://docs.astral.sh/uv/) and install `libsdca` with
PyTorch support:

<!-- smoke: env-uv -->
```bash
uv venv --seed --managed-python --python 3.12 .venv
source .venv/bin/activate
pip install 'libsdca[torch]'
```

Then use the losses in place of the PyTorch ones:

<!-- smoke: torch -->
```python
import torch
from libsdca import CrossEntropyLoss, TopKHingeLoss

device = "cuda" if torch.cuda.is_available() else "cpu"
logits = torch.randn(32, 5, device=device, requires_grad=True)
targets = torch.randint(5, (32,), device=device)

ce = CrossEntropyLoss()(logits, targets)       # replaces torch.nn.CrossEntropyLoss()
hinge = TopKHingeLoss(k=2)(logits, targets)    # top-2 SVM loss
(ce + hinge).backward()
```

### Other interfaces

Every interface installs with one command or CMake target, and its guide documents all
options:

| Interface | Install | Guide |
| --- | --- | --- |
| Python (NumPy) | `pip install libsdca` | [python/README.md](python/README.md) |
| scikit-learn estimator | `pip install 'libsdca[sklearn]'` | [python/README.md](python/README.md#classifier) |
| PyTorch losses | `pip install 'libsdca[torch]'` | [python/TORCH.md](python/TORCH.md) |
| JAX losses (CPU) | `pip install 'libsdca[jax]'` | [python/JAX.md](python/JAX.md) |
| JAX losses (Linux NVIDIA GPU) | `pip install 'libsdca[jax-cuda]'` | [python/JAX.md](python/JAX.md) |
| Rust | `cargo add libsdca` | [rust/README.md](rust/README.md) |
| C++ (header-only) | CMake target `libsdca::libsdca` | [docs/CPP.md](docs/CPP.md) |
| C | CMake target `libsdca::sdca_c` (`-DSDCA_BUILD_C_API=ON`) | [docs/C.md](docs/C.md) |
| MATLAB / Octave | `run matlab/make.m` / `tests/matlab/run_octave_tests.sh` | [docs/MATLAB.md](docs/MATLAB.md) |

<details>
<summary>conda environment</summary>

<!-- smoke: env-conda -->
```bash
conda create --yes --override-channels --channel conda-forge --prefix .venv python=3.12 pip
conda activate ./.venv
pip install 'libsdca[torch]'
```

</details>

<details>
<summary>C++ and C from source</summary>

```bash
git clone https://github.com/mlapin/libsdca && cd libsdca
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/usr/local   # add -DSDCA_BUILD_C_API=ON for C
cmake --build build --target install
```

CMake and BLAS options are in [DEPENDENCIES.md](DEPENDENCIES.md).

</details>

<details>
<summary>Wheels, GPU and compiler requirements</summary>

- **Wheels:** Prebuilt for Linux `x86_64` (`manylinux_2_28`) and macOS 11+ `arm64`; other
  platforms build from the sdist with a C++11 compiler.
- **GPU:** CUDA needs Linux, an NVIDIA GPU and a compatible driver. The PyTorch examples
  use CUDA when available; `jax-cuda` installs the CUDA 13 runtime and Triton (Python 3.11+,
  driver 580+). NumPy, scikit-learn, Rust, C, C++ and MATLAB run on CPU.
- **Compilers:** PyTorch and JAX compile small C++ extensions on first use when a C++17
  compiler and Python headers are present (`build-essential python3-dev` on Debian/Ubuntu,
  `xcode-select --install` on macOS); otherwise they fall back with a one-time warning
  ([build tools](python/TORCH.md#build-tools)). uv-managed and conda Pythons include headers.

</details>

## Quick start

Each example uses cross-entropy and the top-2 hinge (top-2 SVM) loss with the default
options; the PyTorch example is [above](#installation). The [API reference](docs/API.md) and
the interface guides document every parameter and option.

### JAX

The losses are plain functions that compose with `jax.grad`, `jit` and `vmap`:

<!-- smoke: jax -->
```python
import jax
import jax.numpy as jnp
from libsdca.jax import cross_entropy, topk_hinge_loss

logits = jnp.array([[1.0, 2.0, 0.5], [0.0, 1.0, 3.0]])
targets = jnp.array([0, 2])

value, gradient = jax.value_and_grad(cross_entropy)(logits, targets)
value, gradient = jax.value_and_grad(topk_hinge_loss)(logits, targets, k=2)
```

### Python (NumPy)

`fit` trains a linear classifier with the SDCA solver and returns the model with its
convergence status:

<!-- smoke: numpy -->
```python
import numpy as np
import libsdca

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 10))
y = (X @ rng.normal(size=(10, 3))).argmax(axis=1)

model = libsdca.fit(X, y)                            # cross-entropy (softmax regression)
model = libsdca.fit(X, y, "topk_hinge_beta", k=2)    # top-2 SVM
print(model.status, (model.predict(X) == y).mean())
```

### Python (scikit-learn)

`SDCAClassifier` wraps the solver as a scikit-learn estimator:

<!-- smoke: sklearn -->
```python
import numpy as np
from libsdca import SDCAClassifier

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 10))
y = (X @ rng.normal(size=(10, 3))).argmax(axis=1)

clf = SDCAClassifier().fit(X, y)                          # cross-entropy
clf = SDCAClassifier("topk_hinge_beta", k=2).fit(X, y)    # top-2 SVM
print(clf.status_, clf.score(X, y))
```

### Rust

In a Cargo project (Rust 1.74+ and a C++11 compiler), run `cargo add libsdca`, save this as
`src/main.rs`, and run `cargo run`:

<!-- smoke: rust -->
```rust
use libsdca::{fit, FitOptions, Matrix, Objective};

fn main() -> libsdca::Result<()> {
    let x = Matrix::new(&[1.0, 0.0, 0.0, 1.0, -1.0, -1.0], 3, 2)?; // 3 examples, 2 features
    let y = [0, 1, 2];

    let model = fit(x, &y, &FitOptions::default())?; // cross-entropy
    println!("{} {:?}", model.status(), model.predict(x)?);

    let options = FitOptions { objective: Objective::TopKHingeBeta, k: 2, ..Default::default() };
    let model = fit(x, &y, &options)?; // top-2 SVM
    println!("{} {:?}", model.status(), model.predict(x)?);
    Ok(())
}
```

### C++

After the [CMake installation](#other-interfaces) (*C++ and C from source*), compile this as
a C++11 program with the installed include directory and
`-DBLAS_BUILTIN -pthread -fno-fast-math -ffp-contract=off` (a CMake consumer is in
[examples/cpp](examples/cpp)):

<!-- smoke: cpp -->
```cpp
#include <utility>
#include <vector>
#include "sdca/solver.h"
#include "sdca/utility/logging.cpp"   // in exactly one translation unit

template <typename Objective>
bool train(Objective objective) {
    double features[] = {1, 0, 0, 1, -1, -1};   // 3 examples, 2 features
    double weights[6] = {}, dual[9] = {};
    std::vector<sdca::size_type> labels = {0, 1, 2};
    auto context = sdca::make_context(
        sdca::make_input_feature(2, 3, features), sdca::multiclass_output(3, labels),
        std::move(objective), dual, weights);
    sdca::make_solver(context).solve();
    return context.status == sdca::solver_status::solved;
}

int main() {
    bool solved = train(sdca::l2_entropy<double, double>(1.0))            // cross-entropy
                  && train(sdca::l2_topk_hinge<double, double>(1.0, 2));  // top-2 SVM
    return solved ? 0 : 1;
}
```

### C

Build with `-DSDCA_BUILD_C_API=ON` as for C++ and link `libsdca_c` (a CMake consumer is in
[examples/c](examples/c)); this program evaluates both losses and their gradients:

<!-- smoke: c -->
```c
#include <stdint.h>
#include "sdca/c_api.h"

int main(void) {
    const double scores[] = {0.3, -0.8, 1.2};
    const uint8_t positive[] = {1, 0, 0};
    double values[1], gradients[3];
    struct sdca_options options = SDCA_OPTIONS_DEFAULT;   /* cross-entropy */
    int status = sdca_loss_gradient(&options, 1, 3, scores, positive, values, gradients);
    options.loss = SDCA_TOPK_HINGE_BETA;                  /* top-2 SVM loss */
    options.k = 2;
    status |= sdca_loss_gradient(&options, 1, 3, scores, positive, values, gradients);
    return status != SDCA_OK;
}
```

### MATLAB / Octave

In MATLAB, `run matlab/make.m; addpath matlab` builds the gateways (with a configured MEX
C++ compiler). In Octave, run `tests/matlab/run_octave_tests.sh`, then `addpath build/octave`.

<!-- smoke: matlab -->
```matlab
X = [1 0 -1; 0 1 -1];    % 2 features, 3 examples
Y = [1 2 3];
model = matsdca_fit(X, Y);                                                  % cross-entropy
model = matsdca_fit(X, Y, struct('objective', 'topk_hinge_beta', 'k', 2));  % top-2 SVM
[~, predicted] = max(model.W' * X);
```

More examples: [examples/](examples), [rust/examples](rust/examples) and
[matlab/demo.m](matlab/demo.m).

## Losses

All interfaces share these loss names:

| Name | Loss | Targets |
| --- | --- | --- |
| `cross_entropy` | Softmax cross-entropy | class index |
| `topk_cross_entropy` | Top-*k* entropy [5] | class index |
| `multiclass_hinge` | Multiclass hinge (Crammer–Singer SVM) [1] | class index |
| `topk_hinge_alpha`, `topk_hinge_beta` | Top-*k* hinge $\alpha$ [4] and $\beta$ [5] | class index |
| `multilabel_hinge` | Multilabel hinge (multilabel SVM) [2] | label sets / boolean mask |
| `multilabel_cross_entropy` | Multilabel cross-entropy | label sets / boolean mask |
| `distillation`, `proximal_distillation` | Teacher-logit distillation and its Moreau envelope | teacher logits |
| `proximal_cross_entropy` | Moreau envelope of cross-entropy | class index |

Hinge losses take `gamma >= 0` (`0`: nonsmooth, `> 0`: smoothed [5]). PyTorch also provides
`linear_cross_entropy` (linear head + cross-entropy in chunks). Formulas:
[API.md](docs/API.md#loss-definitions).

## Documentation

An index of the guides is in [docs/](docs/README.md).

- [API reference](docs/API.md): loss formulas, names across languages, options, precision,
  results, threads, environment variables and known limits
- Interface guides: [Python](python/README.md), [PyTorch](python/TORCH.md),
  [JAX](python/JAX.md), [Rust](rust/README.md), [C++](docs/CPP.md), [C](docs/C.md),
  [MATLAB](docs/MATLAB.md)
- [Dependencies](DEPENDENCIES.md), [contributing](CONTRIBUTING.md),
  [guide for AI coding assistants](AGENTS.md) and [changelog](CHANGELOG.md)
- [Project page](https://mlapin.github.io/libsdca/) and technical report

## Citation

If `libsdca` is useful in your research, please cite the TPAMI paper; the earlier papers
introduce the top-*k* losses, and the technical report describes the library:

```bibtex
@article{lapin2018analysis,
  title     = {Analysis and Optimization of Loss Functions for Multiclass, Top-k, and Multilabel Classification},
  author    = {Lapin, Maksim and Hein, Matthias and Schiele, Bernt},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  volume    = {40},
  number    = {7},
  pages     = {1533--1554},
  year      = {2018}
}

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

@misc{lapin2026libsdca,
  title        = {libsdca: Accurate and Fast Classification Losses for {SDCA} and Neural Networks},
  author       = {Lapin, Max},
  year         = {2026},
  howpublished = {Technical report, \url{https://mlapin.github.io/libsdca/report/}}
}
```

GitHub's *Cite this repository* button uses [CITATION.cff](CITATION.cff).

## References

The numbered citations above refer to:

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
   cross-entropy. arXiv:2610.31642, 2026.
7. L. Prieto, M. Barsbey, P. A. M. Mediano and T. Birdal. Grokking at the edge of
   numerical stability. *ICLR*, 2025.

## License

`libsdca` is released under the [MIT License](LICENSE).
