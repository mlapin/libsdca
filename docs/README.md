# Documentation

Start with the [README](../README.md) for installation and a quick start.

| Guide | Covers |
| --- | --- |
| [Shared API contract](API.md) | Loss definitions, solver defaults, precision, results, errors and [known limits](API.md#known-limits) |
| [Python: NumPy and scikit-learn](PYTHON.md) | Training, model results, losses and gradients |
| [PyTorch](TORCH.md) | Loss layers, linear cross-entropy, backends, memory and transforms |
| [JAX](JAX.md) | Loss functions, `jit`, `grad`, `vmap` and backends |
| [Rust](RUST.md) | Crate installation, training, losses, inputs and results |
| [C++](CPP.md) | Header-only training, losses and projections |
| [C](C.md) | C ABI: building, structs, buffer layouts and errors |

Runnable programs for every interface are in [examples/](../examples/README.md); precision
measurements and validation runs are in [benchmarks/](../benchmarks/README.md).
[CONTRIBUTING.md](../CONTRIBUTING.md) covers building, testing and releasing;
[CHANGELOG.md](../CHANGELOG.md) lists the changes of each release.
