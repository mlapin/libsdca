# JAX losses

`libsdca.jax` provides the classification and distillation losses as JAX functions that
compose with `jit`, `grad` and `vmap` (`pip install 'libsdca[jax]'`):

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
| `cross_entropy` | Softmax cross-entropy (integer labels or boolean mask) | — |
| `topk_cross_entropy` | Top-$k$ entropy | `k` |
| `multiclass_hinge_loss` | Crammer–Singer multiclass hinge | `gamma=0.0` |
| `topk_hinge_loss` (`topk_hinge_beta_loss`, `topk_hinge_alpha_loss`) | Top-$k$ hinge ($\beta$ default, or $\alpha$) | `k`, `variant="beta"`, `gamma=0.0` |
| `multilabel_cross_entropy` | Cross-entropy with uniform target over positive mask entries | — |
| `multilabel_hinge_loss` | Multilabel hinge | `gamma=0.0` |
| `proximal_cross_entropy` | Moreau envelope of cross-entropy | `alpha=1.0` |
| `distillation_loss`, `proximal_distillation_loss` | Logit distillation and its Moreau envelope (`jax.numpy`) | `temperature=1.0`, `alpha=1.0` |
| `sdca_loss` | Generic wrapper for any of the eight `LOSSES` | `loss`, `k=1`, `gamma=0.0`, `alpha=1.0` |

See [API.md](../docs/API.md#loss-definitions) and
[parameter domains](../docs/API.md#parameter-domains-and-numerical-limits).

## Backends

| Loss | `backend="auto"` selection | Derivatives |
| --- | --- | --- |
| `cross_entropy`, `multilabel_cross_entropy`, `topk_cross_entropy` ($k=1$) | `"jax"` on all devices | Any order, forward and reverse |
| Other `LOSSES` | `"jax"` on GPU/TPU and on CPU for $\le 16{,}384$ scores when required arithmetic (`jax_enable_x64` or `precision="float32"`) is enabled (except `proximal_cross_entropy`); otherwise `"native"` | First order, reverse mode only |

- **`"jax"`:** Pure `jax.numpy` / XLA operations with no host round trip and full
  `jax.export` support. Entropy and distillation losses use fused variadic row reductions
  on CPU. Non-entropy losses use `float64` by default (requiring `jax_enable_x64=True`) or
  `float32` when `precision="float32"`.
- **`"native"`:** Calls the C++ library with a custom VJP on CPU threads
  (`SDCA_NUM_THREADS`). On CPU it compiles and caches a zero-copy XLA FFI custom call on
  first use (`SDCA_JAX_FFI` in [API.md](../docs/API.md#environment-variables)), falling
  back to `jax.pure_callback` when no C++ compiler is available or on GPU/TPU.
- **Subnormal and GPU tail limits:** XLA on CPU flushes subnormal inputs and outputs to
  zero across all backends, so CPU values vanish below the normal limit (`exp(-87.3)` in
  `float32`, `exp(-708.4)` in `float64`). On GPU, default `float32` work can lose
  individually underflowed exponentials before their sum is formed (`precision="float64"`
  retains them; [aggregate-tail limits](../docs/numerics.md#aggregate-tail-limitations)).

## Targets and validation

Targets are integer class indices (shape of `logits` without `axis`) or a boolean mask of
the same shape as `logits`. Concrete (untraced) calls validate inputs eagerly and raise
`ValueError`. Inside `jit` or `vmap`:
- `"native"` validates values at run time and raises `JaxRuntimeError`.
- `"jax"` returns `nan` loss and gradients for invalid targets, nonfinite logits, or
  non-convergent `proximal_cross_entropy` rows.

## Distillation

`distillation_loss` and `proximal_distillation_loss` evaluate soft-target cross-entropy
and its Moreau envelope on student and teacher logits in `jax.numpy`. The teacher receives
no gradient, and equal logits produce an **exact zero gradient**. `temperature` and `alpha`
are static under `jit`; loss values have dtype `promote(logits.dtype, float32)` and
gradients keep `logits.dtype`. Reverse-over-reverse and `jax.hessian` are supported.

## Tests

```bash
python -m pytest -n auto tests/python --group jax
```

See [performance.md](../docs/performance.md#jax) for CPU and A100 benchmarks.
