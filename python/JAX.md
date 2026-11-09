# JAX losses

`libsdca.jax` provides the losses of the PyTorch layers as JAX functions that
work under `jit`, `grad` and `vmap`:

```python
import jax
from libsdca.jax import sdca_loss

def objective(w, x, y):
    return sdca_loss(x @ w, y, "topk_hinge_beta", k=2, gamma=0.5)

step = jax.jit(jax.grad(objective))
```

Install it with `pip install 'libsdca[jax]'`. `import libsdca` does not
import JAX; `import libsdca.jax` does.

## Named functions

All named functions accept `reduction`, `axis=-1` and `precision`, and all but
the distillation losses `backend`. Those are wrappers over `sdca_loss`; canonical objective names remain useful
when switching among objectives programmatically.

| Function | Generic loss | Additional parameters |
| --- | --- | --- |
| `cross_entropy` | `cross_entropy` | none |
| `topk_cross_entropy` | `topk_cross_entropy` | `k` |
| `multiclass_hinge_loss` | `multiclass_hinge` | `gamma=0.0` |
| `topk_hinge_loss` | `topk_hinge_beta`, or `topk_hinge_alpha` with `variant="alpha"` | `k`, `variant="beta"`, `gamma=0.0` |
| `topk_hinge_alpha_loss`, `topk_hinge_beta_loss` | `topk_hinge_alpha`, `topk_hinge_beta` | `k`, `gamma=0.0` |
| `multilabel_cross_entropy` | `multilabel_cross_entropy`: cross-entropy with uniformly distributed positive labels | none |
| `multilabel_hinge_loss` | `multilabel_hinge` | `gamma=0.0` |
| `proximal_cross_entropy` | `proximal_cross_entropy` | `alpha` |
| `distillation_loss`, `proximal_distillation_loss` | none (`jax.numpy` only) | `temperature`, `alpha` |

`gamma=0.0` is the nonsmooth hinge; a positive `gamma` smooths it.
`multilabel_cross_entropy` is categorical
cross-entropy with uniform positive targets, not independent sigmoid binary
cross-entropy. JAX `cross_entropy(logits, target, axis=-1)` accepts integer
indices or boolean label masks and has no `weight` or `ignore_index` arguments.
PyTorch's named `cross_entropy(input, target, ...)` instead follows PyTorch's
fixed class axis (1 except for a one-dimensional input), integer targets,
weights and ignored labels. Use each frontend's generic `sdca_loss` to choose
an arbitrary class axis. [Shared parameter domains](../docs/API.md#parameter-domains-and-numerical-limits)
and the [top-k hinge convention](../docs/API.md#top-k-hinge-convention) apply.

## Backends

| Loss | `backend="auto"` runs | Derivatives |
| --- | --- | --- |
| `cross_entropy`, `multilabel_cross_entropy`, `topk_cross_entropy` with `k=1` | jax | any order, forward and reverse |
| the other `LOSSES` | jax where its arithmetic is available (below): on a GPU or TPU, and on the CPU for calls of at most 16,384 scores except `proximal_cross_entropy` (and without the FFI handlers also the nonsmooth `multiclass_hinge`, top-k hinges at `k=1` and `multilabel_hinge`); else native | first order, reverse mode |

- `"jax"` computes every loss in `jax.numpy` on any device: pure XLA operations, no
  host round trip, and `jax.export` support. Entropy losses use a composed
  `log1p` log-softmax (fused variadic row reductions on the CPU) and keep tiny
  losses and target gradients to rounding; masked entropy preserves small logit
  differences in the gradient. The other losses follow the PyTorch tensor
  algorithms (`include/sdca/loss.h`) in float64 (or float32 with
  `precision="float32"` where float32 holds `gamma`, `alpha` and the row spread),
  selecting the top competitors via `lax.top_k` and sorting full rows only when a
  smooth/multilabel hinge threshold extends past the top candidates. Float64 work
  requires `jax_enable_x64`; without it, `backend="jax"` raises for non-entropy
  losses unless `precision="float32"`, and `"auto"` routes them to `"native"`.
- `"native"` runs the C++ loss with a custom VJP on the available CPUs
  (or `SDCA_NUM_THREADS`). On the CPU it uses an XLA FFI custom call compiled on
  first use against the running `jaxlib`'s headers (`SDCA_JAX_FFI` in
  [API.md](../docs/API.md#environment-variables)), operating directly on XLA's
  buffers without a host round trip or copy. Where the FFI handler cannot be
  built (no C++ compiler, `SDCA_JAX_FFI=off`, or on a GPU/TPU), it falls back to
  `jax.pure_callback` with identical results. Float32 scores are solved in double
  (as in `loss_and_grad`), float64 in float64, and 16-bit scores in float32.
  `jax.export` does not support `"native"` calls.
- `"auto"` on the CPU uses `"jax"` for entropy (`cross_entropy`,
  `multilabel_cross_entropy`, and `topk_cross_entropy` at `k=1`). When the required
  `"jax"` arithmetic is enabled (`jax_enable_x64` or `precision="float32"`), `"auto"`
  also uses `"jax"` on accelerators and, on the CPU, for calls of at most 16,384
  scores (except `proximal_cross_entropy`, plus nonsmooth `k=1` / multilabel hinges
  when the FFI handler is unavailable); larger non-entropy CPU calls use `"native"`
  ([performance](../docs/performance.md#jax)).
- **Subnormal and GPU tail limits:** XLA on the CPU flushes subnormal inputs and
  outputs to zero across all backends (including FFI custom calls and host
  callbacks), so CPU losses and gradients vanish below the smallest normal number
  (float32 `exp(-87.3)`, float64 `exp(-708.4)`). The top-k target mass is formed
  independently of competitor entries, so competitor flushing does not zero a
  normal target entry. On the GPU, default FP32 work can lose individually
  underflowed exponentials before their sum is formed (`precision="float64"`
  retains them; see [aggregate-tail limits](../docs/numerics.md#aggregate-tail-limitations)).
- **Higher derivatives:** The first-order losses in the table above, and all losses
  with `backend="native"`, raise under `jvp`, `jacfwd` and `hessian`. Distillation
  losses raise under `jvp` and `jacfwd`, while `jax.hessian` and reverse-over-reverse
  work through their `jax.numpy` gradient.

The jax backend is `jax.numpy` rather than a Triton or XLA FFI kernel: it
needs no dependency or build beyond `jax`, runs on every jaxlib, and
computes what the PyTorch tensor backend computes, which matches the native
losses to the rounding of the scores (see the tests below).

JAX computes in float32 unless `jax_enable_x64` is on. `precision` on
`sdca_loss` and `distillation_loss` works as in
[libsdca.torch](TORCH.md#precision); float64 arithmetic in the jax backend
or distillation needs `jax_enable_x64`. Reductions sum the per-example
losses in their working dtype (float64 where JAX has it) and round once.

## Targets and validation

Targets are integer class indices of the logits shape without `axis`, or a
boolean mask of the logits shape, as in `libsdca.torch.sdca_loss`. `axis`
takes the place of `dim`.

Shapes and options are always validated statically. Concrete (untraced) logits
and targets are also checked before the call, raising `ValueError` on invalid
inputs. Under `jit` or `vmap`:
- `"native"` validates values at execution time and raises `JaxRuntimeError`
  with the native error message.
- `"jax"` cannot raise on traced values and instead returns `nan` loss and
  gradients for invalid targets, nonfinite logits, or non-convergent
  `proximal_cross_entropy` rows.
- `"auto"` follows the selected backend (on CPU with float64 or
  `precision="float32"`, calls of `<= 16,384` scores per example use `"jax"` and
  return `nan`, while larger calls use `"native"` and raise `JaxRuntimeError`).
- `proximal_cross_entropy` with a traced mask solves for every class as a
  candidate label and is slower than with integer targets or a concrete mask.

For `proximal_cross_entropy`, finite float64 inputs whose within-row spread
overflows float64 are outside the supported numerical range. The JAX backend
returns NaN values and gradients for those rows, while valid neighboring rows
remain usable. This detects an unsupported centered solve; it does not extend
the solver to arbitrary finite float64 spreads. A representable multilabel
mean remains supported when its unscaled sum would overflow: the jax loss
scales the label contributions before summing them.

## Distillation

`distillation_loss(student, teacher, temperature=..., reduction=..., axis=...)`
is the cross-entropy to `softmax(teacher / T)` of `libsdca.torch`. It runs in
`jax.numpy` on any device. The teacher gets no gradient, and the
first-order gradient is exactly zero at equal logits.
Under `jit`, a row with nonfinite student or teacher logits returns `nan`
loss and student gradients; eager calls reject those inputs with `ValueError`.

`proximal_distillation_loss(student, teacher, temperature=1.0, alpha=1.0, ...)`
is the Moreau envelope of `distillation_loss`, as in
[libsdca.torch](TORCH.md#teacher-logits-and-distillation): the residual solve of
`proximal_cross_entropy` in `jax.numpy`, with an exactly zero gradient at equal logits. An `alpha` outside the float32 range needs
`jax_enable_x64`. `temperature` and `alpha` are static under `jit`: a new value
compiles again. Every loss value has dtype `promote(logits.dtype, float32)` (float32
for 16-bit and float32 logits, float64 for float64), as in PyTorch and under
`torch.autocast`; gradients keep the logits dtype.

## Tests

`tests/python/test_jax.py` compares every loss with `loss_and_grad` under `jit`,
in float32 and float64. It also checks:

- `vmap` with batched and shared targets, nested `vmap`, and the class axis;
- the jax entropy against the native one, and its Hessian
  `diag(p) - p p^T`;
- validation;
- distillation against a NumPy reference, proximal distillation's KKT
  conditions and limits, and the float32 values of 16-bit logits.

`tests/python/test_jax_ffi.py` compares the FFI calls with the host callback bit for bit
(every loss, integer labels of several widths and masks, float32 and float64, x64 on and
off, forward alone and with the gradient, `jit` and `vmap`), their messages for invalid
traced inputs and thread counts, and the build rules of `SDCA_JAX_FFI`.

`tests/python/test_jax_backend.py` compares the jax backend of every other loss
with the native one, as `test_tensor_losses.py` and `test_tensor_prox.py`
do for PyTorch: random scores in float32 and float64, ties, far offsets,
cancelling margins, closed forms, `jit`, `vmap` and traced targets,
`precision="float32"`, half precision and the backend `"auto"` selects.

Both pass on the CPU with JAX 0.6.2 (Python 3.10) and 0.12 (Python 3.12),
on an H100 with JAX 0.12 and the pip CUDA 12 libraries, and on an A100 with JAX 0.12.0.1
with the FFI handlers on and off; an earlier
minimum-version run passed with JAX 0.4.35 ([tested configurations](../DEPENDENCIES.md#tested-configurations)).
`tests/python/test_jax_half_precision.py` holds float16 and bfloat16 logits, both backends,
every loss (top-k at k = 3), the three reductions and batches whose mean gradients reach
float16 subnormals to the 16-bit envelope of
[numerics.md](../docs/numerics.md#further-precision-limits), against float64 native losses and
80- and 120-digit distillation references of the represented inputs.

[Performance](../docs/performance.md#jax) has the timings.
