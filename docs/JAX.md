# JAX losses

[All guides](https://github.com/mlapin/libsdca/blob/master/docs/README.md) ·
[Shared API reference](https://github.com/mlapin/libsdca/blob/master/docs/API.md)

`libsdca.jax` provides classification and distillation losses as JAX functions that
compose with `jit`, `grad` and `vmap`.

## Installation

```bash
pip install 'libsdca[jax]'       # CPU
pip install 'libsdca[jax-cuda]'  # Linux NVIDIA GPUs
```

The CUDA extra includes CUDA 13 and Triton and needs Python 3.11+ and NVIDIA driver 580+.

## Quick start

```python
import jax
import jax.numpy as jnp
from libsdca.jax import cross_entropy

logits = jnp.array([[2., 0., -1.], [0., 1., 3.]])
labels = jnp.array([0, 2])
value, gradient = jax.jit(jax.value_and_grad(cross_entropy))(logits, labels)
print(value, gradient)
```

## Functions

Every function takes `reduction` (`"mean"`, `"sum"`, `"none"`), `axis=-1`, `backend`
(`"auto"`, `"jax"`, `"native"`) and `precision`; the hyperparameters are keyword-only:

| Function | Loss | Targets | Parameters |
| --- | --- | --- | --- |
| `cross_entropy` | Softmax cross-entropy | labels or class probabilities | `weight`, `ignore_index=None`, `label_smoothing=0.0` |
| `topk_cross_entropy` | Top-*k* entropy | labels | `k` (required), `weight`, `ignore_index` |
| `multiclass_hinge_loss` | Crammer–Singer multiclass hinge | labels | `gamma=0.0`, `weight`, `ignore_index` |
| `topk_hinge_loss` | Top-*k* hinge, $\beta$ (default) or $\alpha$ | labels | `k` (required), `variant="beta"`, `gamma=0.0`, `weight`, `ignore_index` |
| `proximal_cross_entropy` | Moreau envelope of cross-entropy | labels | `alpha=1.0`, `weight`, `ignore_index` |
| `multilabel_cross_entropy` | Cross-entropy to the uniform target on the mask | boolean mask | — |
| `multilabel_hinge_loss` | Multilabel hinge | boolean mask | `gamma=0.0` |
| `distillation_loss` | Logit distillation | teacher logits | `temperature=1.0` |
| `proximal_distillation_loss` | Its Moreau envelope | teacher logits | `temperature=1.0`, `alpha=1.0` |

Targets follow the table above. Labels are integers of any dtype with the shape of
`logits` without `axis`; masks are booleans with the shape of `logits`. A boolean target of
a single-label loss raises `TypeError`, as an integer target of a multilabel loss does. See
[API.md](API.md#loss-definitions) and
[parameter domains](API.md#parameter-domains-and-numerical-limits).

### Common options

| Option | Default | Behavior |
| --- | --- | --- |
| `axis` | `-1` | Class axis; labels have the logits' shape with this axis removed |
| `reduction` | `"mean"` | `"none"` returns per-example values; `"sum"` sums them |
| `weight` | `None` | Class weights for single-label losses; the label mean divides by the sum of unignored label weights |
| `ignore_index` | `None` | Matching labels contribute no loss or gradient; unavailable for probability targets |
| `label_smoothing` | `0.0` | Cross-entropy only; mixes the target with a uniform distribution |
| `precision` | `None` | Storage precision, with `float32` work for 16-bit logits; `"float64"` requires `jax_enable_x64` on every backend |

`precision` also accepts `jnp.float32` and `jnp.float64`. Kernel parameters `k`, `gamma`,
`alpha` and `temperature` must be Python numbers or concrete scalar arrays: close over
them or mark them static under `jit`. `weight` and `label_smoothing` may be traced.

### Weighted cross-entropy

For a label target, smoothing mixes the weighted label loss with a uniform-target term:

$$ (1-\varepsilon) w_y \ell + \frac{\varepsilon}{C} \sum_j w_j (\mathrm{lse}(z)-z_j). $$

Here $\mathrm{lse}(z)=\log\sum_j\exp(z_j)$. Probability targets instead give
$-\sum_j w_j t_j \log\mathrm{softmax}(z)_j$, with the mean divided by the number of
examples. These options compose `jax.numpy` operations around the per-example losses.

## Backends

On CUDA, `backend="auto"` takes the shared Triton kernels for every loss. Elsewhere it
selects:

| Loss | `backend="auto"` selection off CUDA | Derivatives |
| --- | --- | --- |
| `cross_entropy`, `multilabel_cross_entropy`, `topk_cross_entropy` ($k=1$) | `"jax"` on CPU/TPU | Any order |
| The other losses of labels and masks | `"jax"` on TPU, and on CPU for calls of $\le 16{,}384$ logit entries in total; otherwise `"native"` (see below) | First order (reverse) |
| `distillation_loss`, `proximal_distillation_loss` | The native CPU kernels through the FFI handlers; else `"jax"` | See [Distillation](#distillation) |

- **The other losses (`"auto"`):** two conditions narrow the `"jax"` choice of the table.
  - **Arithmetic:** `"jax"` needs the loss's arithmetic: `jax_enable_x64`, or
    `precision="float32"`. Without it the call is `"native"`.
  - **`proximal_cross_entropy`** takes `"native"` on CPU at every size.
- **CUDA (`"auto"`):** Runs the same fused Triton kernels as `libsdca.torch`, as one XLA FFI
  call under `jit`, `grad`, `vmap` and `jax.export`. Per-row results match PyTorch.
  - **Export:** Pass
    `disabled_checks=[jax.export.DisabledSafetyCheck.custom_call("libsdca_cuda_kernels")]`;
    the export holds kernels for the exporting GPU's architecture.
  - **Triton:** Linux wheels include the handler; without Triton 3.3–3.9 the losses keep
    `jax.numpy` (`SDCA_JAX_CUDA`; with a one-time warning when Triton is missing or older).
  - **Devices:** CPU-committed arguments stay on the CPU.
- **`"jax"`:** Pure `jax.numpy` / XLA, with no host round trip and full `jax.export` support.
  Losses that default to `float64` work need `jax_enable_x64=True` here: without it,
  `backend="auto"` takes the native library and `backend="jax"` raises.
- **`"native"`:** Calls the C++ library with a custom VJP on CPU threads.
  - On CPU it compiles a zero-copy XLA FFI handler on first use (`SDCA_JAX_FFI`, no PyTorch
    needed), else falls back to `jax.pure_callback`.
  - The distillation losses run natively only through the FFI handlers on the CPU
    (`NotImplementedError` elsewhere).
  - Probability targets have no native path (`NotImplementedError`).
- **Subnormals:** XLA on CPU flushes subnormal inputs and outputs to zero, so CPU losses and
  gradient entries below `exp(-87.3)` (`float32`) or `exp(-708.4)` (`float64`) are `0`. Sums of
  many individually underflowing exponentials are still kept (`float32` cross-entropy over
  1,000 classes 90 below the target returns `8.19e-37`).

JAX 0.4.35 is supported with `jit`, `grad` and `vmap`; the shared native kernels need a JAX
that exposes `jax.ffi`. See [environment variables](API.md#environment-variables) for
the switches above.

## CPU pass fusion

On a multithreaded CPU, the array paths of `cross_entropy`, `multilabel_cross_entropy`,
`topk_cross_entropy` ($k=1$) and both distillation losses above $16{,}384$ logit entries read the
scores in several passes, either fused into one XLA program or *fenced* into separate passes
that each run on every thread. `SDCA_JAX_CPU_FUSION` selects the variant:

- **`auto`** (default): the first trace of a loss times both variants once at 4096 × 1024
  (proximal distillation follows a fixed rule instead),
  keeps the fenced one unless the unfenced one takes at most 0.85 of its time, and caches the
  decision in `SDCA_CACHE_DIR` per CPU, thread setting, dtype and library version.
- **`fenced`** or **`unfenced`**: forces one variant.

The variants differ only in summation order. The setting is read when a loss is traced; call
`jax.clear_caches()` after changing it.

## Errors

- **Arguments:** Wrong types (a non-floating `logits` array, a target of the wrong kind, a
  non-string `reduction`, a non-integer `k`, a traced `gamma`, ...) raise `TypeError`; bad
  values (shapes, `k` outside $[1, C-1]$, a negative `gamma`, an unknown `reduction` or
  `backend`) raise `ValueError`, before any computation.
- **Data values, concrete calls:** Concrete (untraced) calls check them eagerly and raise
  `ValueError` for nonfinite logits, labels out of range or masks without a positive (or,
  for `multilabel_hinge_loss`, a negative) label.
- **Data values inside `jit` or `vmap`:** Rows with such values give `nan` losses and
  gradients on every backend, as do non-convergent `proximal_cross_entropy` rows: after a
  failed native call the native backend repeats the call row by row, so valid calls cost
  nothing extra.

## Distillation

`distillation_loss` and `proximal_distillation_loss` take student and teacher logits (a
different teacher dtype is promoted with the student's); the teacher receives no gradient and
equal logits give an **exact zero gradient**. No $T^2$ factor is applied. Ordinary
distillation supports reverse-over-reverse and `jax.hessian`; proximal distillation supports
`jax.hessian` (forward-over-reverse) only.

## Benchmarks

### CPU: multiclass cross-entropy and hinge

Forward and backward on **AMD EPYC 7J13**, Linux, **32 threads**, JAX/jaxlib 0.12.0.1,
logits **4096 × 32000**.

- **Setup:** These November 2026 pre-0.4.0 measurements use `backend="auto"` and `jax.jit`
  for the library and references.
- **Timing:** Timings include dispatch and synchronization after warm-up, excluding
  compilation: ten fresh processes per cell, each with ten interleaved timing rounds and no
  retry selection.
- **Evidence:** The [technical report](https://mlapin.github.io/libsdca/report/) provides
  the full grid and uncertainty.

| Loss | dtype | `libsdca` (ms) | Time / `jax.numpy` | Time / Optax |
| --- | --- | ---: | ---: | ---: |
| Multiclass cross-entropy | `bfloat16` | 112.05 | 0.83 | 0.97 |
| Multiclass cross-entropy | `float32` | 96.71 | 1.02 | 0.87 |
| Multiclass cross-entropy | `float64` | 233.65 | 1.24 | 0.98 |
| Multiclass hinge (`gamma=0`) | `bfloat16` | 95.38 | 0.46 | — |
| Multiclass hinge (`gamma=0`) | `float32` | 32.94 | 0.38 | — |
| Multiclass hinge (`gamma=0`) | `float64` | 55.08 | 0.27 | — |

- **Statistics:** Times are medians of process medians. Ratios are geometric means of
  paired process ratios (`libsdca / reference`); below `1` is faster.
- **References:** The `jax.numpy` references compute the same objectives; Optax is measured
  only for cross-entropy.
- **Results:** FP32 and FP64 cross-entropy take longer than the `jax.numpy` reference in
  this grid, while the hinge is faster in all three dtypes. Small point-estimate differences
  alone do not establish an advantage.

### CUDA: multilabel cross-entropy and distillation

Forward and backward on **NVIDIA A100-SXM4-40GB**, JAX/jaxlib 0.12.0.1, logits
**4096 × 32000**.

- **Setup:** These November 2026 pre-0.4.0 kernel measurements use `jax.jit` on both
  implementations, warmed up before timing and synchronized after execution.
- **Timing:** Times in **ms** are medians of process medians over ten fresh processes per
  cell; compilation is excluded.
- **Evidence:** The [technical report](https://mlapin.github.io/libsdca/report/) gives the
  methodology, the full evidence and uncertainty.

| Loss | dtype | `libsdca` | Plain `jax.numpy` |
| --- | --- | ---: | ---: |
| Multilabel cross-entropy | `float32` | 1.733 | 2.126 |
| Multilabel cross-entropy | `bfloat16` | 1.282 | 1.456 |
| Multilabel cross-entropy | `float64` | 4.888 | 5.391 |
| Distillation | `float32` | 3.206 | 3.438 |
| Distillation | `bfloat16` | 2.219 | 2.002 |
| Distillation | `float64` | 10.712 | 7.210 |

BF16 and FP64 distillation are slower than plain JAX in this grid.

These measurements use independent teachers at `temperature=2` and synchronized wall time,
so their absolute values cannot be compared directly with the PyTorch table's correlated
teachers and CUDA-event timing. Shared kernels alone do not guarantee identical end-to-end
performance.

## Tests

```bash
python -m pytest -n auto tests/python --group jax   # from a source checkout
```
