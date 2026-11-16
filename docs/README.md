# Documentation

Start with the [README](../README.md) for installation and a quick start.

| Guide | Covers |
| --- | --- |
| [API.md](API.md) | Loss definitions, objectives, options, precision, results, threads, environment variables, errors and [known limits](API.md#known-limits) |
| [python/README.md](../python/README.md) | NumPy solver (`fit`, `Model`, `Solver`), `SDCAClassifier`, losses and gradients |
| [python/TORCH.md](../python/TORCH.md) | PyTorch loss layers: the `CrossEntropyLoss` drop-in, `linear_cross_entropy`, devices, precision, memory, `torch.compile`, distillation, benchmarks |
| [python/JAX.md](../python/JAX.md) | JAX losses under `jit`, `grad` and `vmap`, backends, distillation |
| [rust/README.md](../rust/README.md) | Rust crate |
| [CPP.md](CPP.md) | Header-only C++: training, losses and projections |
| [C.md](C.md) | C ABI (`libsdca_c`): building, conventions, tensor entry points |
| [MATLAB.md](MATLAB.md) | MATLAB and Octave gateways |

Runnable programs for every interface are in [examples/](../examples/README.md).
[CONTRIBUTING.md](../CONTRIBUTING.md) covers building, testing and releasing;
[CHANGELOG.md](../CHANGELOG.md) lists the changes of each release.
