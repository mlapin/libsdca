# Using `libsdca` (Guide for AI Coding Assistants)

A short reference for AI coding assistants integrating `libsdca` into user code; usage
examples are in the [README](README.md#quick-start) and the linked guides.

## Installation

```bash
pip install libsdca              # NumPy solver and losses (numpy>=1.23)
pip install 'libsdca[torch]'     # PyTorch losses (torch>=2.1, installs ninja)
pip install 'libsdca[jax]'       # JAX losses (jax>=0.4.35)
pip install 'libsdca[sklearn]'   # SDCAClassifier (scikit-learn>=1.6)
cargo add libsdca                # Rust crate (Rust>=1.74 and a C++11 compiler)
```

- Prebuilt wheels: Linux `x86_64` (`manylinux_2_28`) and macOS `arm64` (11+).
- PyTorch and JAX compile small C++ extensions on first use when a compiler and Python
  headers are present; otherwise they fall back with a one-time warning
  ([TORCH.md](python/TORCH.md#build-tools)).

## Which API to use

| User goal | API | Guide |
| --- | --- | --- |
| Replace `torch.nn.CrossEntropyLoss` (integer class labels only, no label smoothing) | `libsdca.CrossEntropyLoss`, `cross_entropy` | [TORCH.md](python/TORCH.md#replace-pytorch-cross-entropy) |
| Large-vocabulary head + cross-entropy without `(N, V)` logits | `LinearCrossEntropyLoss`, `linear_cross_entropy` | [TORCH.md](python/TORCH.md#linear-layer-and-cross-entropy-without-the-logits) |
| Top-*k*, hinge and multilabel PyTorch losses | `TopKCrossEntropyLoss`, `TopKHingeLoss`, `MulticlassHingeLoss`, `MultilabelHingeLoss`, `MultilabelCrossEntropyLoss` | [TORCH.md](python/TORCH.md) |
| Logit distillation in PyTorch | `DistillationLoss`, `distillation_loss` | [TORCH.md](python/TORCH.md#teacher-logits-and-distillation) |
| Losses in JAX (`jit`, `grad`, `vmap`) | `libsdca.jax` | [JAX.md](python/JAX.md) |
| Linear or kernel SVM / softmax classifier (scikit-learn) | `libsdca.SDCAClassifier` | [python/README.md](python/README.md#classifier) |
| Same, with duality-gap history and warm starts | `libsdca.fit` | [python/README.md](python/README.md#training) |
| Rust, C++, C, MATLAB | crate `libsdca`, headers, `libsdca_c`, `matsdca_*` | [Rust](rust/README.md), [C++](docs/CPP.md), [C](docs/C.md), [MATLAB](docs/MATLAB.md) |

## Pitfalls

- **Class axis:** `CrossEntropyLoss` takes classes on `dim=1` like PyTorch; every other
  loss defaults to `dim=-1`. Pass `dim=1` for `(N, C, H, W)` inputs to the others
  ([TORCH.md](python/TORCH.md#argument-parity)).
- **Label smoothing and probability targets:** use `distillation_loss` with
  log-probabilities as the teacher
  ([TORCH.md](python/TORCH.md#label-smoothing-and-probability-targets)).
- **Distillation** takes raw logits for both inputs and applies no $T^2$ factor
  ([TORCH.md](python/TORCH.md#teacher-logits-and-distillation)).
- **Memory:** CUDA losses can reuse the logits buffer for the gradient when nothing else
  references it ([TORCH.md](python/TORCH.md#memory)).
- **Solver objective:** $\frac12\Vert W \Vert_2^2 + C\sum_i \ell_i$ with no intercept; standardize
  features; for $d \gg n$ pass a Gram matrix ([python/README.md](python/README.md#training),
  [API.md](docs/API.md#options)).
- **Convergence:** only `status == "solved"` certifies the duality gap
  ([API.md](docs/API.md#results)).

## Further documentation

- Loss formulas, options and environment variables: [docs/API.md](docs/API.md)
- Numerics and known limits: [docs/numerics.md](docs/numerics.md)
- Performance: [docs/performance.md](docs/performance.md)
