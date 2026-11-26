# Using `libsdca` (Guide for AI Coding Assistants)

A short reference for AI coding assistants that integrate `libsdca` into user code. Usage
examples are in the [README](README.md#quick-start) and the linked guides; loss names,
options, defaults and errors are shared across languages and defined in
[docs/API.md](docs/API.md).

## Installation

```bash
pip install libsdca              # NumPy solver and losses (numpy>=1.23)
pip install 'libsdca[torch]'     # PyTorch losses (torch>=2.1)
pip install 'libsdca[jax]'       # JAX CPU losses (jax>=0.4.35)
pip install 'libsdca[jax-cuda]'  # JAX CUDA losses (Linux, NVIDIA driver 580+; Python 3.11+, C++17 compiler or jax-triton 0.5)
pip install 'libsdca[sklearn]'   # SDCAClassifier (scikit-learn>=1.6)
cargo add libsdca                # Rust crate (Rust>=1.74 and a C++11 compiler)
```

- **Wheels:** Linux `x86_64` (`manylinux_2_28`) and macOS `arm64` (11+). Other platforms
  build the sdist with a C++11 compiler and link a system CBLAS when one is found.
- **First-use builds:** PyTorch and JAX compile their CPU kernels on first use (a C++17
  compiler and Python headers: `build-essential python3-dev` on Debian/Ubuntu,
  `xcode-select --install` on macOS). Without them, CPU calls run the same kernels in the
  prebuilt native library and PyTorch CUDA calls launch them through Python (far more
  overhead per call), each with a one-time warning ([TORCH.md](docs/TORCH.md#build-tools)).
- **Check the selected path:** `libsdca.torch.explain_backend(cross_entropy, logits, target)`
  returns `(backend, reason)` for the call `cross_entropy(logits, target)`;
  `libsdca.jax.explain_backend` does the same for JAX.

### Building from source

| Interface | Build | Guide |
| --- | --- | --- |
| Python | `pip install .` from a checkout; BLAS through `SDCA_BLAS_BACKEND=AUTO\|BUILTIN\|EXTERNAL` or `SDCA_BLAS="-lopenblas"` | [DEPENDENCIES.md](DEPENDENCIES.md#python) |
| C++ (header-only) | `cmake -S . -B build && cmake --build build --target install`, then `find_package(libsdca CONFIG)` and `libsdca::libsdca` | [CPP.md](docs/CPP.md) |
| C | the same with `-DSDCA_BUILD_C_API=ON`; link `libsdca::sdca_c` | [C.md](docs/C.md) |
| Rust | `cargo build`; `build.rs` compiles the C ABI with `CXX` and the same BLAS rule | [RUST.md](docs/RUST.md#building) |

Without CMake, compile C++ with `-DBLAS_BUILTIN -pthread -fno-fast-math -ffp-contract=off`
(or `-DBLAS_DEFAULT_LOCAL_HEADER` and a CBLAS such as `-lopenblas`); BLAS selection is
described in [API.md](docs/API.md#blas).

## Which API to use

| User goal | API | Guide |
| --- | --- | --- |
| Replace `torch.nn.CrossEntropyLoss` (labels or probabilities, weights, label smoothing) | `libsdca.torch.CrossEntropyLoss`, `cross_entropy` | [TORCH.md](docs/TORCH.md#replace-pytorch-cross-entropy) |
| Large-vocabulary head + cross-entropy without `(N, V)` logits | `LinearCrossEntropyLoss`, `linear_cross_entropy` | [TORCH.md](docs/TORCH.md#linear-layer-and-cross-entropy-without-the-logits) |
| Top-*k*, hinge and multilabel PyTorch losses | `TopKCrossEntropyLoss`, `TopKHingeLoss`, `MulticlassHingeLoss`, `MultilabelHingeLoss`, `MultilabelCrossEntropyLoss` | [TORCH.md](docs/TORCH.md) |
| Logit distillation in PyTorch | `DistillationLoss`, `distillation_loss` | [TORCH.md](docs/TORCH.md#teacher-logits-and-distillation) |
| Losses in JAX (`jit`, `grad`, `vmap`) | `libsdca.jax` | [JAX.md](docs/JAX.md) |
| Replace optax's softmax cross-entropies or multiclass hinge (per-example, optax signatures) | `libsdca.optax` | [JAX.md](docs/JAX.md#optax-losses) |
| Linear or kernel SVM / softmax classifier (scikit-learn) | `libsdca.sklearn.SDCAClassifier` | [PYTHON.md](docs/PYTHON.md#classifier) |
| Same, with duality-gap history and warm starts | `libsdca.fit(X, y, loss)` with loss objects such as `libsdca.TopKHinge(k=2)` | [PYTHON.md](docs/PYTHON.md#training) |
| Losses and gradients on NumPy arrays | `libsdca.loss_and_grad` | [PYTHON.md](docs/PYTHON.md#losses-and-gradients) |
| Rust, C++, C | crate `libsdca`, headers `sdca/*.h`, library `libsdca_c` | [Rust](docs/RUST.md), [C++](docs/CPP.md), [C](docs/C.md) |

## Pitfalls

- **Class axis:** every PyTorch loss takes classes on `dim=1` (`dim=0` for 1-D logits),
  like `F.cross_entropy`; JAX losses take them on `axis=-1`.
- **Loss parameters** belong to their loss: `TopKHingeLoss(k=2)`,
  `libsdca.TopKHinge(k=2)`; passing a parameter another loss uses is an error, and top-*k*
  losses have no default `k`. PyTorch/JAX hyperparameters are keyword-only; NumPy loss
  objects also accept positional arguments.
- **Backends:** `"auto"` (default) and `"kernel"` run libsdca's kernels and raise
  `NotImplementedError` naming `backend="composed"` where no kernel computes a call (another
  device, a CUDA shape beyond the kernels); `"composed"` uses framework operations.
- **Distillation** takes raw logits for both inputs and applies no $T^2$ factor
  ([TORCH.md](docs/TORCH.md#teacher-logits-and-distillation)).
- **Memory:** a CPU sum or mean computes its gradient in the forward pass (peak: logits plus
  gradient). Other PyTorch backward passes (CUDA, `reduction="none"`, scaled upstreams)
  write the gradient over the logits when nothing else references them, and warn once when
  something does (`reuse_logits=False` turns it off; [TORCH.md](docs/TORCH.md#memory)).
- **Derivatives:** the kernels are first order; pass `backend="composed"` for higher
  derivatives or forward mode ([TORCH.md](docs/TORCH.md#compilation-and-transformations)).
- **Solver objective:** $\frac12\Vert W \Vert_2^2 + C\sum_i \ell_i$ with no intercept;
  standardize features; for $d \gg n$ pass a Gram matrix
  ([PYTHON.md](docs/PYTHON.md#training), [API.md](docs/API.md#options)).
- **Convergence:** only `status == "solved"` guarantees the duality gap
  ([API.md](docs/API.md#results)).
- **Precision:** `precision=None` follows the storage dtype (16-bit inputs use `float32`
  work); pass `precision="float64"` for double work ([API.md](docs/API.md#precision),
  [known limits](docs/API.md#known-limits)).

## Developing `libsdca`

Building, testing and the repository layout are in [CONTRIBUTING.md](CONTRIBUTING.md).
