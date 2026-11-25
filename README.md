# libsdca

[![CI](https://github.com/mlapin/libsdca/actions/workflows/ci.yml/badge.svg)](https://github.com/mlapin/libsdca/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/libsdca.svg)](https://pypi.org/project/libsdca/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/libsdca/)
[![Crates.io](https://img.shields.io/crates/v/libsdca.svg)](https://crates.io/crates/libsdca)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**libsdca** is an accurate and fast implementation of classification losses: cross-entropy,
multiclass and multilabel SVM hinges, top-*k* losses and knowledge distillation. The losses
are available as PyTorch and JAX layers on CPU and CUDA and as NumPy, Rust, C++ and C
functions, and a stochastic dual coordinate ascent (SDCA) solver trains linear and kernel
classifiers with the multiclass and multilabel training objectives.

- **Accurate:** For confident predictions, the cross-entropy kernels avoid softmax collapse
  [7] and can retain losses and gradient entries down to the subnormal range of the output
  dtype ([known limits](docs/API.md#known-limits)).
  - Distillation gradients are exactly zero at equal student and teacher logits [6].
  - Losses and gradients are checked against high-precision references
    ([technical report](https://mlapin.github.io/libsdca/report/)).
- **Fast:** Fused CPU and CUDA kernels implement the losses. In our benchmarks they match
  or outperform `torch.compile` and `jax.jit` in most comparisons
  ([technical report](https://mlapin.github.io/libsdca/report/)).
- **Many losses:** Cross-entropy, the multiclass hinge (Crammer–Singer SVM [1]), the top-*k*
  hinge and entropy losses [4, 5], the multilabel hinge [2] and cross-entropy, logit
  distillation and proximal variants, with nonsmooth or smoothed hinges, and
  `linear_cross_entropy` for a language-model head without its `(N, V)` logits
  ([losses](docs/API.md#loss-definitions)).
- **Many interfaces:** PyTorch modules and functions (with `torch.compile` support), JAX
  functions for `jit`, `grad` and `vmap`, NumPy, a scikit-learn estimator, Rust, header-only
  C++ and C, with shared loss definitions and interface-specific calling conventions
  ([API reference](docs/API.md)).
- **SDCA solver with a duality-gap guarantee:** SDCA trains linear and kernel classifiers
  (multiclass, top-*k* and multilabel SVMs, softmax regression).
  - When `status == "solved"`, the duality gap bounds the returned model’s objective
    suboptimality under the relative stopping criterion ([results](docs/API.md#results)).
  - Its steps use efficient projections onto the simplex and the top-*k* cone [3, 4], which
    C++ also provides directly.

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
from libsdca.torch import CrossEntropyLoss, TopKHingeLoss

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
| Python (NumPy) | `pip install libsdca` | [docs/PYTHON.md](docs/PYTHON.md) |
| scikit-learn estimator | `pip install 'libsdca[sklearn]'` | [docs/PYTHON.md](docs/PYTHON.md#classifier) |
| PyTorch losses | `pip install 'libsdca[torch]'` | [docs/TORCH.md](docs/TORCH.md) |
| JAX losses (CPU) | `pip install 'libsdca[jax]'` | [docs/JAX.md](docs/JAX.md) |
| JAX losses (Linux NVIDIA GPU) | `pip install 'libsdca[jax-cuda]'` | [docs/JAX.md](docs/JAX.md) |
| Rust | `cargo add libsdca` | [docs/RUST.md](docs/RUST.md) |
| C++ (header-only) | CMake target `libsdca::libsdca` | [docs/CPP.md](docs/CPP.md) |
| C | CMake target `libsdca::sdca_c` (`-DSDCA_BUILD_C_API=ON`) | [docs/C.md](docs/C.md) |

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

- **Wheels:** Prebuilt for Linux `x86_64` (`manylinux_2_28`, OpenBLAS from the
  `scipy-openblas32` package, which `pip` installs with it) and macOS 11+ `arm64`
  (Accelerate); other platforms build from the sdist with a C++11 compiler and link
  a system CBLAS when one is found ([BLAS](docs/API.md#blas)). The build uses `CC` and `CXX` when
  set, else the compiler of the Python installation, else `c++`, `g++` or `clang++` on `PATH`.
- **GPU:** CUDA needs Linux, an NVIDIA GPU and a compatible driver. The PyTorch examples
  use CUDA when available; `jax-cuda` installs the CUDA 13 runtime (Python 3.11+, driver
  580+) and Triton. The fused kernels launch through jax-triton 0.5 on Python 3.12+,
  or a C++17 XLA FFI handler built on first use. NumPy, scikit-learn, Rust, C and C++ run on CPU.
- **Compilers:** PyTorch and JAX compile their CPU kernels on first use, which needs a C++17
  compiler and Python headers (`build-essential python3-dev` on Debian/Ubuntu,
  `xcode-select --install` on macOS); without them PyTorch and JAX CPU calls run the same
  kernels in the prebuilt native library and PyTorch CUDA calls launch them through Python,
  with a one-time warning ([build tools](docs/TORCH.md#build-tools)). uv-managed and conda
  Pythons include headers.

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

model = libsdca.fit(X, y)                                  # cross-entropy (softmax regression)
print(model.status, model.score(X, y))
model = libsdca.fit(X, y, libsdca.TopKHinge(k=2))          # top-2 SVM
print(model.status, model.score(X, y))
```

### Python (scikit-learn)

`SDCAClassifier` wraps the solver as a scikit-learn estimator:

<!-- smoke: sklearn -->
```python
import numpy as np
from libsdca.sklearn import SDCAClassifier

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 10))
y = (X @ rng.normal(size=(10, 3))).argmax(axis=1)

clf = SDCAClassifier().fit(X, y)                          # cross-entropy
print(clf.status_, clf.score(X, y))
clf = SDCAClassifier("topk_hinge", k=2).fit(X, y)         # top-2 SVM
print(clf.status_, clf.score(X, y))
```

### Rust

In a Cargo project (Rust 1.74+ and a C++11 compiler), run `cargo add libsdca`, save this as
`src/main.rs`, and run `cargo run`:

<!-- smoke: rust -->
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

### C++

After the [CMake installation](#other-interfaces) (*C++ and C from source*), compile this as
a C++11 program with the installed include directory and
`-DBLAS_BUILTIN -pthread -fno-fast-math -ffp-contract=off` (or `-DBLAS_DEFAULT_LOCAL_HEADER`
and a CBLAS such as `-lopenblas`; a CMake consumer is in
[examples/cpp](examples/cpp)):

<!-- smoke: cpp -->
```cpp
#include <utility>
#include <vector>
#include "sdca/solver.h"

// Trains the objective it is given; solved is whether the gap reached tol.
struct train {
    bool& solved;
    template <typename Objective>
    void operator()(Objective objective) const {
        double features[] = {1, 0, 0, 1, -1, -1};   // 3 examples, 2 features
        double weights[6] = {}, dual[9] = {};       // weights (3, 2), dual (3, 3)
        std::vector<int> labels = {0, 1, 2};
        auto context = sdca::make_context(
            sdca::make_input_feature(3, 2, features),
            sdca::make_output_multiclass(labels.begin(), labels.end(), 3),
            std::move(objective), weights, dual);
        sdca::make_solver(context).solve();
        solved = context.status == sdca::solver_status::solved;
    }
};

int main() {
    bool entropy = false, hinge = false;
    sdca::visit_multiclass_objective<double, double>(
        sdca::cross_entropy_objective(), train{entropy});            // cross-entropy
    sdca::visit_multiclass_objective<double, double>(
        sdca::topk_hinge_beta_objective(1.0, 2), train{hinge});      // top-2 hinge
    return entropy && hinge ? 0 : 1;
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
    const int64_t label[] = {0};
    double values[1], gradients[3];
    struct sdca_loss loss = SDCA_LOSS_INIT(SDCA_LOSS_CROSS_ENTROPY);
    struct sdca_execution execution = SDCA_EXECUTION_INIT;
    struct sdca_targets targets = SDCA_TARGETS_INIT;  /* class labels */
    targets.labels = label;
    int status = sdca_loss_evaluate(&loss, &execution, SDCA_DTYPE_FLOAT64, 1, 3, scores,
                                    &targets, values, gradients);
    loss.kind = SDCA_LOSS_TOPK_HINGE_BETA;                /* top-2 SVM loss */
    loss.k = 2;
    status |= sdca_loss_evaluate(&loss, &execution, SDCA_DTYPE_FLOAT64, 1, 3, scores,
                                 &targets, values, gradients);
    return status != SDCA_OK;
}
```

More examples: [examples/](examples) and [rust/examples](rust/examples).

## Documentation

An index of the guides is in [docs/](docs/README.md).

- [API reference](docs/API.md): loss formulas, names across languages, options, precision,
  results, threads, environment variables and known limits
- Interface guides: [Python](docs/PYTHON.md), [PyTorch](docs/TORCH.md),
  [JAX](docs/JAX.md), [Rust](docs/RUST.md), [C++](docs/CPP.md), [C](docs/C.md)
- [Benchmarks and numerical validation](benchmarks/README.md) (summary; the
  [technical report](https://mlapin.github.io/libsdca/report/) has the details)
- [Dependencies](DEPENDENCIES.md), [contributing](CONTRIBUTING.md),
  [guide for AI coding assistants](AGENTS.md) and [changelog](CHANGELOG.md)
- [Project page](https://mlapin.github.io/libsdca/)

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
  author       = {Lapin, Maksim},
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
