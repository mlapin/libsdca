# Benchmarks and numerical validation

The [technical report](https://mlapin.github.io/libsdca/report/) describes how `libsdca` was
benchmarked and validated, and gives the results: runtime and memory against eager and
compiled PyTorch and JAX, and the numerical accuracy of the losses. In our benchmarks,
libsdca matches or outperforms `torch.compile` and `jax.jit` in most comparisons; the report
lists every comparison, including those where libsdca is slower.

This directory holds the precision measurements and the test runs behind the release.

## Precision measurements

[`precision.csv`](precision.csv) holds precision measurements on an NVIDIA A100: for each
setting, the error of the loss value and gradient of libsdca and of each reference
implementation, measured against high-precision `mpmath` references on the same inputs.

| Column | Meaning |
| --- | --- |
| `framework`, `device`, `loss`, `dtype`, `shape`, `rows`, `mode` | The setting (`rows`: `finite`, or `unaligned` for a strided view; `mode`: `train` or `fwd`) |
| `implementation` | `libsdca`; the framework's own loss (`PyTorch eager`, `torch.compile`, `JAX eager`, `jax.jit`); a framework formula where the framework has no such loss (`(formula)`); libsdca's `backend="composed"`; `Liger` |
| `value_err_median`, `value_err_max` | Relative error of the per-row loss, in units of the working dtype's epsilon |
| `grad_err_median`, `grad_err_max` | Gradient-row error relative to the row's largest entry, in units of the storage dtype's epsilon |
| `grad_entry_err_median`, `grad_entry_err_max` | Largest relative error of a gradient entry in a row (entries with a normal exact value), in units of the storage dtype's epsilon |
| `grad_correctly_rounded_pct`, `grad_faithful_pct` | Percentage of gradient entries equal to the correctly rounded value, or to either of the two floating-point numbers around the exact value (faithful) |

## Numerical tests

These runs of the test suite completed without a failure outside the
[documented limits](../docs/API.md#known-limits):

| Test | Checks | Result |
| --- | --- | --- |
| [`release_numerics.py`](../tests/python/release_numerics.py) `--device cpu` | Entropy and distillation losses and gradients against 100-digit `mpmath` references | 31,280 checks, 0 failures (Linux x86-64) |
| [`boundaries.py`](../tests/python/boundaries.py) `--device cpu` | Scans magnitude, spread, width, ties, masking and upstream gradients per loss and dtype (NumPy, PyTorch and JAX kernels) | 1,649,510 points, 0 failures outside the documented limits (macOS arm64) |

How to run them is in [CONTRIBUTING.md](../CONTRIBUTING.md).
