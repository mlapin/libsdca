# JAX losses

`libsdca.jax` provides classification and distillation losses as JAX functions that
compose with `jit`, `grad` and `vmap`. Install `libsdca[jax]` for CPU, or
`libsdca[jax-cuda]` for Linux NVIDIA GPUs (CUDA 13 runtime and Triton included;
Python 3.11+ and NVIDIA driver 580+):

```python
import jax
from libsdca.jax import cross_entropy, topk_hinge_loss

def objective(w, x, y):
    return cross_entropy(x @ w, y)        # or topk_hinge_loss(x @ w, y, k=2)

step = jax.jit(jax.grad(objective))
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

Each loss takes one kind of target: labels are integers of any dtype with the shape of
`logits` without `axis`; masks are booleans with the shape of `logits`. A boolean target of
a single-label loss raises `TypeError`, as an integer target of a multilabel loss does. See
[API.md](../docs/API.md#loss-definitions) and
[parameter domains](../docs/API.md#parameter-domains-and-numerical-limits).

- **`weight`, `ignore_index`, `label_smoothing`** follow `torch.nn.functional.cross_entropy`:
  each example's loss is multiplied by its label's weight, examples whose label equals
  `ignore_index` add nothing (no loss, no gradient), and the mean divides by the sum of the
  weights of the other labels. Label smoothing mixes $(1 - \varepsilon)$ of the label with
  $\varepsilon$ of the uniform distribution: $(1-\varepsilon)\,w_y \ell + \frac{\varepsilon}{C}
  \sum_j w_j (\operatorname{lse}(z) - z_j)$. Probability targets give $-\sum_j w_j t_j \log
  \operatorname{softmax}(z)_j$, their mean divides by the number of examples, and they take no
  `ignore_index`. The calls without these options run unchanged; with them, the
  composition is `jax.numpy` around the same per-example losses.
- **Static hyperparameters:** `k`, `gamma`, `alpha` and `temperature` select kernels and
  working precision, so they must be Python numbers (or concrete 0-d arrays); a traced value
  raises `TypeError` (under `jit`, close over them or mark them static). `weight` and
  `label_smoothing` may be traced.
- **`precision`:** `None` (the storage dtype's work; 16-bit logits work in `float32`),
  `"float32"`, `"float64"`, or `jnp.float32` / `jnp.float64`. `"float64"` needs
  `jax_enable_x64` on every backend (`ValueError` otherwise).

## Backends

| Loss | `backend="auto"` selection | Derivatives |
| --- | --- | --- |
| `cross_entropy`, `multilabel_cross_entropy`, `topk_cross_entropy` ($k=1$) | Shared Triton kernels on CUDA; `"jax"` on CPU/TPU | Any order |
| The other losses of labels and masks | Shared Triton kernels on CUDA; `"jax"` on TPU and on CPU for calls of $\le 16{,}384$ logit entries in total when its arithmetic is available (`jax_enable_x64` or `precision="float32"`, except `proximal_cross_entropy`); otherwise `"native"` | First order (reverse) |
| `distillation_loss`, `proximal_distillation_loss` | Shared Triton kernels on CUDA; the native CPU kernels through the FFI handlers; else `"jax"` | See [Distillation](#distillation) |

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
  back to `jax.pure_callback`. The distillation losses run natively only through the FFI
  handlers on the CPU (`NotImplementedError` elsewhere); probability targets have no native
  path (`NotImplementedError`).
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

## Errors

- Wrong types (a non-floating `logits` array, a target of the wrong kind, a non-string
  `reduction`, a non-integer `k`, a traced `gamma`, ...) raise `TypeError`; bad values
  (shapes, `k` outside $[1, C-1]$, a negative `gamma`, an unknown `reduction` or `backend`)
  raise `ValueError`, before any computation.
- Data values: concrete (untraced) calls check them eagerly and raise `ValueError` for
  nonfinite logits, labels out of range or masks without a positive (or, for
  `multilabel_hinge_loss`, a negative) label. Inside `jit` or `vmap`, such rows give `nan`
  losses and gradients on every backend, as do non-convergent `proximal_cross_entropy` rows:
  after a failed native call the native backend repeats the call row by row, so valid calls
  cost nothing extra.

## Distillation

`distillation_loss` and `proximal_distillation_loss` take student and teacher logits (a
different teacher dtype is promoted with the student's); the teacher receives no gradient and
equal logits give an **exact zero gradient**. No $T^2$ factor is applied. Ordinary
distillation supports reverse-over-reverse and `jax.hessian`; proximal distillation supports
`jax.hessian` (forward-over-reverse) only.

## Tests

```bash
python -m pytest -n auto tests/python --group jax   # from a source checkout
```
