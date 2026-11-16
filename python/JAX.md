# JAX losses

`libsdca.jax` provides classification and distillation losses as JAX functions that
compose with `jit`, `grad` and `vmap`. Install `libsdca[jax]` for CPU, or
`libsdca[jax-cuda]` for Linux NVIDIA GPUs (CUDA 13 runtime and Triton included;
Python 3.11+ and NVIDIA driver 580+):

```python
import jax
from libsdca.jax import sdca_loss

def objective(w, x, y):
    return sdca_loss(x @ w, y, "topk_hinge_beta", k=2, gamma=0.5)

step = jax.jit(jax.grad(objective))
```

## Named functions

All functions accept `reduction` (`"mean"`, `"sum"`, `"none"`), `axis=-1` and `precision`,
and all except the distillation losses accept `backend` (`"auto"`, `"jax"`, `"native"`):

| Function | Loss | Parameters |
| --- | --- | --- |
| `cross_entropy` | Softmax cross-entropy (labels or mask) | — |
| `topk_cross_entropy` | Top-*k* entropy | `k` |
| `multiclass_hinge_loss` | Crammer–Singer multiclass hinge | `gamma=0.0` |
| `topk_hinge_loss` (`topk_hinge_beta_loss`, `topk_hinge_alpha_loss`) | Top-*k* hinge ($\beta$ default, or $\alpha$) | `k`, `variant="beta"`, `gamma=0.0` |
| `multilabel_cross_entropy` | Cross-entropy to uniform target on mask | — |
| `multilabel_hinge_loss` | Multilabel hinge | `gamma=0.0` |
| `proximal_cross_entropy` | Moreau envelope of cross-entropy | `alpha=1.0` |
| `distillation_loss`, `proximal_distillation_loss` | Logit distillation and Moreau envelope | `temperature=1.0`, `alpha=1.0` |
| `sdca_loss` | Generic wrapper for the eight `LOSSES` | `loss`, `k=1`, `gamma=0.0`, `alpha=1.0` |

See [API.md](../docs/API.md#loss-definitions) and
[parameter domains](../docs/API.md#parameter-domains-and-numerical-limits).

## Backends

| Loss | `backend="auto"` selection | Derivatives |
| --- | --- | --- |
| `cross_entropy`, `multilabel_cross_entropy`, `topk_cross_entropy` ($k=1$) | Shared Triton kernels on CUDA; `"jax"` on CPU/TPU | Any order |
| Other `LOSSES` | Shared Triton kernels on CUDA; `"jax"` on TPU and on CPU for calls of $\le 16{,}384$ logit entries in total when its arithmetic is available (`jax_enable_x64` or `precision="float32"`, except `proximal_cross_entropy`); otherwise `"native"` | First order (reverse) |

- **CUDA (`"auto"`):** Runs the same fused Triton kernels as `libsdca.torch`, as one XLA FFI
  call under `jit`, `grad`, `vmap` and `jax.export` (pass
  `disabled_checks=[jax.export.DisabledSafetyCheck.custom_call("libsdca_cuda_kernels")]`; the
  export holds kernels for the exporting GPU's architecture). Per-row results match PyTorch.
  Linux wheels include the handler; without Triton 3.3–3.9 the losses keep `jax.numpy`
  (`SDCA_JAX_CUDA`; with a one-time warning when Triton is missing or older). CPU-committed
  arguments stay on the CPU.
- **`"jax"`:** Pure `jax.numpy` / XLA, with no host round trip and full `jax.export` support.
  Losses that default to `float64` work need `jax_enable_x64=True` here: without it,
  `backend="auto"` takes the native library and `backend="jax"` raises.
- **`"native"`:** Calls the C++ library with a custom VJP on CPU threads. On CPU it compiles
  a zero-copy XLA FFI handler on first use (`SDCA_JAX_FFI`, no PyTorch needed), else falls
  back to `jax.pure_callback`. Native work can be `float64` without `jax_enable_x64`.
- **Subnormals:** XLA on CPU flushes subnormal inputs and outputs to zero, so CPU losses and
  gradient entries below `exp(-87.3)` (`float32`) or `exp(-708.4)` (`float64`) are `0`. Sums of
  many individually underflowing exponentials are still kept (`float32` cross-entropy over
  1,000 classes 90 below the target returns `8.19e-37`).

JAX 0.4.35 is supported with `jit`, `grad` and `vmap`; the shared native kernels need a JAX
that exposes `jax.ffi`. See [environment variables](../docs/API.md#environment-variables) for
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

## Targets and validation

Targets are integer class indices (shape of `logits` without `axis`) or a boolean mask of
the same shape as `logits`. Concrete (untraced) calls validate inputs eagerly and raise
`ValueError`. Inside `jit` or `vmap`:
- `"native"` validates values at run time and raises `JaxRuntimeError`.
- `"jax"` returns `nan` loss and gradients for invalid targets, nonfinite logits, or
  non-convergent `proximal_cross_entropy` rows.

## Distillation

`distillation_loss` and `proximal_distillation_loss` take student and teacher logits; the
teacher receives no gradient and equal logits give an **exact zero gradient**. `temperature`
and `alpha` are static under `jit`. Ordinary distillation supports reverse-over-reverse and
`jax.hessian`; proximal distillation supports `jax.hessian` (forward-over-reverse) only.

## Tests

```bash
python -m pytest -n auto tests/python --group jax   # from a source checkout
```
