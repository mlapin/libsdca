# Documentation

Reference guides and technical notes for `libsdca`:

| Guide | Contents |
| --- | --- |
| [API.md](API.md) | Loss definitions, cross-language names, options, precision, results, threads and environment variables |
| [CPP.md](CPP.md) | Header-only C++11 quick start, CMake integration, data layouts and proximal operators |
| [C.md](C.md) | Exception-safe C ABI (`include/sdca/c_api.h`, CMake target `libsdca::sdca_c`) |
| [MATLAB.md](MATLAB.md) | MATLAB and GNU Octave MEX gateways (`matsdca_fit`, `matsdca_loss`, `matsdca_prox`) |
| [numerics.md](numerics.md) | Precision guarantees, the `solved` certificate, representation limits, verification and known limits |
| [performance.md](performance.md) | PyTorch CUDA/CPU, JAX and solver benchmarks against PyTorch, Liger, JAX and scikit-learn |
| [softmax-collapse.md](softmax-collapse.md) | Softmax collapse and cancellation formulations and reproduction of Barron et al. (2026) |

Language guides that live with their packages:

- **Python & scikit-learn:** [python/README.md](../python/README.md) (`fit`, `SDCAClassifier`, `loss_and_grad`, `coordinate_update`)
- **PyTorch loss layers:** [python/TORCH.md](../python/TORCH.md) (`CrossEntropyLoss`, `LinearCrossEntropyLoss`, top-*k*, hinge and distillation losses)
- **JAX loss functions:** [python/JAX.md](../python/JAX.md) (`libsdca.jax` under `jit`, `grad` and `vmap`)
- **Rust crate:** [rust/README.md](../rust/README.md) (`libsdca` on crates.io)

Project guides in the repository root: [README.md](../README.md),
[DEPENDENCIES.md](../DEPENDENCIES.md), [CONTRIBUTING.md](../CONTRIBUTING.md),
[AGENTS.md](../AGENTS.md) and [CHANGELOG.md](../CHANGELOG.md).
