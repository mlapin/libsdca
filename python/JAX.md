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

- `"jax"` computes every loss with `jax.numpy` on any device: XLA
  operations only, so no host round trip, and `jax.export` works. Entropy
  of integer labels makes the passes of a composed log-softmax; it keeps
  tiny losses and the target gradient of confident rows (minus the mass of
  the other classes) to rounding. On the CPU, the entropy and distillation
  losses sum a row and count its tied maxima (and labels) in one variadic
  reduction, which XLA:CPU fuses: a plain reduction of a computed or boolean
  row first writes it to memory and reduces it in a slower reduce-window.
  The variadic reduction adds runs of at most 256 entries in order and the
  runs' sums pairwise, which rounds as a pairwise sum. Entropy
  of a mask keeps small logit differences exact in its gradient, as the
  PyTorch tensor entropy. Top-k entropy takes its k largest competitors from
  `lax.top_k`. The other losses are the algorithms of the PyTorch tensor
  backend (`include/sdca/loss.h`) in float64, rounded to the dtype of the
  scores; with `precision="float32"`, in float32 where float32 holds
  `gamma`, `alpha` and the spread of every row. float64 needs
  `jax_enable_x64`: without it, `backend="jax"` raises for these losses
  unless `precision="float32"`, and `"auto"` takes the native backend.
  The smooth and multilabel hinges solve from the 16 (then 64) largest
  margins, chosen by `lax.top_k` on the scores, and sort whole rows only
  when some row's solution reaches past them. Top-k entropy sums the
  competitors below its k largest in one pass. On the CPU, nonsmooth
  top-k hinges form margins only at the k + 1 largest scores (margins
  grow with the scores), and every margin only for the subgradient of a
  batch where rounding makes a smaller score's margin equal to the k-th
  largest one; `lax.top_k` takes float32 keys there: jaxlib 0.6's XLA:CPU
  sorts whole rows for any other dtype, 50 to 100 times slower; float64
  entries are ordered exactly among k + 1 candidates, and rows whose
  float32 keys tie unequal entries at the k-th take the exact `lax.top_k`.
  On a GPU or TPU, where each of these conditionals would wait for its
  predicate on the host, the hinges form every margin and `lax.top_k`
  takes the entries themselves. Under `vmap` every stage runs.
- `"native"` runs the C++ loss with a custom VJP, on the available CPUs
  (automatic threads) unless `SDCA_NUM_THREADS` is set. On the CPU it is an
  XLA FFI custom call: a small handler, compiled on the first native call
  against the running jaxlib's headers (a few seconds, then cached; see
  `SDCA_JAX_FFI` in [API.md](../docs/API.md#environment-variables)), runs the
  native rows on XLA's buffers inside the compiled computation, with no host
  round trip or copy. Elsewhere, and where the handler cannot be built (no C++
  compiler, a jax without `jax.ffi`, `SDCA_JAX_FFI=off`), it runs through
  `jax.pure_callback` with the same results; on a GPU or TPU the scores are
  copied to the host and back. Under `vmap` the whole batch goes to one call,
  and a forward call returns only the losses. The FFI call costs about 0.03 ms
  beyond the loss; through the callback a 256×10 call took 0.45–0.48 ms, and
  its copies, label mask and checks ran on one thread (Linux, jaxlib 0.12, 32
  threads: 4096×32000 multiclass hinge forward and backward in 55 ms against
  1067 ms; [performance](../docs/performance.md#jax)). float32 scores are
  solved in double, as by `loss_and_grad`. float64 scores stay in float64,
  and float16/bfloat16 scores are computed in float32. `jax.export` cannot
  serialize a host callback, and rejects the FFI call as a custom call without
  a compatibility guarantee.
- XLA on the CPU flushes subnormal results to zero and reads subnormal
  operands as zero, also in its custom calls and on the host callback's thread. The native loss
  computes with IEEE subnormals (every C ABI call runs with them, and the
  wrapper checks `gamma` and `alpha` by their bits), but its subnormal values
  and gradient entries reach XLA as zero: on the CPU every backend's losses
  and entries are zero below the smallest normal number (float32 `exp(-87.3)`,
  float64 `exp(-708.4)`), where the PyTorch frontend keeps them to the
  [subnormal limit](../docs/numerics.md#representation-limits). The top-k target mass is computed independently of the competitor entries, so
  their flushing does not erase a normal target entry. The jax backend
  gives `nan` there for a subnormal `gamma` or `alpha` (it computes them
  on a GPU); distillation handles any temperature on every device.
- On the GPU, default FP32 work can lose individually underflowed exponentials
  before their representable aggregate is formed. Masked CE and distillation
  probes expose this at losses around `6.8e-43`; explicit `precision="float64"`
  passes those probes. See the [aggregate-tail limits](../docs/numerics.md#aggregate-tail-limitations)
  for the inputs and distinction from CPU output flushing.
- On CPU, `"auto"` uses JAX for entropy (also `topk_cross_entropy` with
  `k=1`, the entropy of its one target). Where the jax arithmetic of the
  other losses is available (`jax_enable_x64`, or `precision="float32"`),
  it also uses JAX for calls of at most 16,384 scores except
  `proximal_cross_entropy` (a 256×10 top-k entropy takes 0.14 ms in jax
  and 0.25 ms native), and where the FFI handlers are unavailable also for
  the nonsmooth `multiclass_hinge`, top-k hinges at `k=1` and
  `multilabel_hinge`. With the FFI handlers, all larger non-entropy calls
  (the nonsmooth hinges included: at 32 threads the FFI call of
  `multiclass_hinge` takes 0.44–0.68 of the jax time from 4096×1000) stay
  native. Without float64 (JAX's default) these losses stay native at
  `precision=None`.
- The first-order losses of the table, and every loss with `backend="native"`, raise
  under `jvp`, `jacfwd` and `hessian`. The distillation
  losses raise under `jvp` and `jacfwd`, but their gradient is `jax.numpy` code:
  `jax.hessian` and reverse over reverse differentiate it again (tested against
  `jax.nn` for `distillation_loss` on the CPU).
- jaxlib 0.6's XLA:CPU miscompiled the smooth hinges given a constant boolean
  mask whose class axis was moved (a NumPy mask under `jit`, `axis=0`): before
  jaxlib 0.12 the jax backend keeps every mask an operand
  (`lax.optimization_barrier`); jaxlib 0.12 computes them right without it.

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

Shapes and options are always checked. Concrete (untraced) logits and
targets are also checked before the call, and errors raise `ValueError`.
Inside `jit` or `vmap`, the native loss checks them when it runs (the FFI
handler and the host callback alike, with the same messages), and JAX
raises a `JaxRuntimeError` carrying its message. The jax backend cannot
raise for traced values: an example with an invalid target or nonfinite logits
gives `nan` loss and gradients, as
does a `proximal_cross_entropy` solve that does not converge. With
`backend="auto"` the outcome follows the backend it picks: on the CPU with
float64 (or `precision="float32"`) that depends on the objective and, for
most losses, on the size, so an invalid traced target of such a loss raises
a `JaxRuntimeError` above 16,384 scores and gives `nan` at or below it.
Under `vmap` the size of one example decides, not that of the batch.
`proximal_cross_entropy` with a traced mask target solves for every class as a possible label,
which is slower than integer targets or a concrete mask.

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
