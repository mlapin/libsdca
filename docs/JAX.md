# JAX losses

[All guides](https://github.com/mlapin/libsdca/blob/master/docs/README.md) ·
[Shared API reference](https://github.com/mlapin/libsdca/blob/master/docs/API.md)

`libsdca.jax` provides classification and distillation losses as JAX functions that
compose with `jit`, `grad` and `vmap`, and `libsdca.optax` provides drop-ins for optax's
losses. The [technical report](https://mlapin.github.io/libsdca/report/) compares their
speed with JAX and optax.

## Installation

```bash
pip install 'libsdca[jax]'       # CPU
pip install 'libsdca[jax-cuda]'  # Linux NVIDIA GPUs
```

- **CUDA:** the extra includes CUDA 13 and Triton and needs Python 3.11+ and NVIDIA driver
  580+. The kernels launch through jax-triton (installed on Python 3.12+) or, where it is
  unavailable, a C++17 XLA FFI handler built on first use.
- **`ptxas`:** the kernels need the CUDA 13 `ptxas` of `jax[cuda13]`; if an older CUDA
  toolkit's `ptxas` comes first on `PATH`, put `site-packages/nvidia/cu13/bin` before it.

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

| Function | Loss | Targets | Parameters |
| --- | --- | --- | --- |
| `cross_entropy` | Softmax cross-entropy | labels or class probabilities | `weight`, `ignore_index=None`, `label_smoothing=0.0` |
| `topk_cross_entropy` | Top-*k* entropy | labels | `k` (required), `weight`, `ignore_index` |
| `multiclass_hinge_loss` | Crammer–Singer multiclass hinge | labels | `gamma=0.0`, `weight`, `ignore_index` |
| `topk_hinge_loss` | Top-*k* hinge, $\beta$ (default) or $\alpha$ | labels | `k` (required), `variant="beta"`, `gamma=0.0`, `weight`, `ignore_index` |
| `proximal_cross_entropy` | Moreau envelope of cross-entropy | labels | `gamma=1.0`, `weight`, `ignore_index` |
| `multilabel_cross_entropy` | Cross-entropy to the uniform target on the mask | boolean mask | — |
| `multilabel_hinge_loss` | Multilabel hinge | boolean mask | `gamma=0.0` |
| `distillation_loss` | Logit distillation | teacher logits | `temperature=1.0` |
| `proximal_distillation_loss` | Its Moreau envelope | teacher logits | `temperature=1.0`, `gamma=1.0` |

Signatures, for example:

```text
cross_entropy(logits, target, *, weight=None, ignore_index=None, label_smoothing=0.0,
              reduction="mean", axis=-1, backend="auto", precision=None)
topk_hinge_loss(logits, target, *, k, variant="beta", gamma=0.0, weight=None,
                ignore_index=None, reduction="mean", axis=-1, backend="auto", precision=None)
distillation_loss(student, teacher, *, temperature=1.0, reduction="mean", axis=-1,
                  backend="auto", precision=None)
```

Labels are integers of any dtype with the shape of `logits` without `axis`; masks are
booleans with the shape of `logits`. A target of the other kind raises `TypeError`. The
definitions are in [API.md](API.md#loss-definitions) and
[parameter domains](API.md#parameter-domains-and-numerical-limits).

### Common options

All options are keyword-only.

| Option | Default | Behavior |
| --- | --- | --- |
| `axis` | `-1` | Class axis; labels have the logits' shape with this axis removed |
| `reduction` | `"mean"` | `"none"` returns per-example values; `"sum"` sums them |
| `backend` | `"auto"` | `"auto"`, `"kernel"` or `"composed"` ([backends](#backends)) |
| `precision` | `None` | `None`, `"float32"` or `"float64"` (or `jnp.float32`, `jnp.float64`); `float32` work for 16- and 32-bit logits ([API.md](API.md#precision)); `"float64"` with `backend="composed"` requires `jax_enable_x64` |
| `weight` | `None` | Class weights of the single-label losses; the mean divides by the sum of unignored label weights |
| `ignore_index` | `None` | Matching labels contribute no loss or gradient; with probability targets it must be `None` or negative |
| `label_smoothing` | `0.0` | Cross-entropy only; mixes the target with a uniform distribution |

`k`, `gamma` and `temperature` must be Python numbers or concrete scalar arrays: close over
them or mark them static under `jit`. `weight` and `label_smoothing` may be traced.

### Weighted cross-entropy

For a label target, smoothing mixes the weighted label loss with a uniform-target term:

$$ (1-\varepsilon) w_y \ell + \frac{\varepsilon}{C} \sum_j w_j (\mathrm{lse}(z)-z_j), $$

with $\mathrm{lse}(z)=\log\sum_j\exp(z_j)$. Probability targets (finite and nonnegative)
give $-\sum_j w_j t_j \log\mathrm{softmax}(z)_j$, with the mean divided by the number of
examples.

### Transformations and sharding

- **`jax.checkpoint`** (`jax.remat`) can recompute the loss's row state in the backward
  pass.
- **`jax.shard_map`:** shards must hold complete class rows. With the legacy
  `jax.experimental.shard_map` and the CPU native-library route, pass `check_rep=False`.

### optax losses

`libsdca.optax` has optax 0.2.9's signatures and per-example outputs (no reduction), on the
same kernels as `libsdca.jax` (`jit`, `grad`, `vmap`):

| Function | Loss |
| --- | --- |
| `softmax_cross_entropy_with_integer_labels(logits, labels, axis=-1, where=None)` | Cross-entropy of integer labels; `axis` may be a tuple (a label is a flat index into those axes) |
| `softmax_cross_entropy(logits, labels, axis=-1, where=None)` | Cross-entropy of target probabilities of the logits' shape (not normalized; differentiable); `axis=None` takes every axis |
| `safe_softmax_cross_entropy(logits, labels)` | The same over the last axis |
| `multiclass_hinge_loss(scores, labels)` | Crammer–Singer hinge of integer labels over the last axis |

```python
from libsdca import optax as sdca_optax

loss = sdca_optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
```

`where` has optax's two meanings: of the logits' shape it is a class mask, of the labels'
shape it excludes whole examples. A row that `where` leaves no class gives optax's value
(`0.0` for labels, `-0.0` for probabilities). Results have optax's dtype, computed in
`float32` work for 16-bit logits and rounded once. Where optax and libsdca differ:

- **Masked classes.** A logit of `-inf`, or a class excluded by a class-shaped `where`,
  masks its class ([API.md](API.md#masked-classes)): without target mass it is removed
  (gradient `+0.0`), so `softmax_cross_entropy` and `multiclass_hinge_loss` are finite where
  optax gives `nan`; a label or target mass there gives `+inf` (optax gives a finite value).
- **Targets.** Targets must be finite and nonnegative and labels in `[0, classes)`; other
  rows give `nan`.

## Backends

| `backend` | Path | Derivatives |
| --- | --- | --- |
| `"auto"` (default), `"kernel"` | libsdca's kernels as one XLA call: on the CPU, kernels compiled on first use ([build tools](#build-tools)); on CUDA, the fused Triton kernels of `libsdca.torch` | First order in reverse mode (`grad`, `vjp`, `jacrev`, under `vmap` too); a second derivative raises PyTorch's `RuntimeError`, forward mode (`jvp`, `jacfwd`) JAX's `TypeError` |
| `"composed"` | `jax.numpy` on any device (`float64` work needs `jax_enable_x64`) | `cross_entropy`, `multilabel_cross_entropy`, `topk_cross_entropy` with $k = 1$, `distillation_loss` and the nonsmooth hinges at any order and in forward mode; the others first order |

Per-row results match PyTorch. Where no kernel computes a call (another platform, no build
tools for the CUDA launcher, a CUDA shape beyond the kernels), `"auto"` and `"kernel"` raise
`NotImplementedError` naming `backend="composed"`. CPU-committed arguments stay on the CPU.

**Subnormals:** the kernels' losses and gradients reach the subnormal range, as PyTorch's.
XLA's own operations flush them: an XLA reduction of subnormal row losses outside the
kernel, and `backend="composed"` results below `exp(-87.3)` (`float32`) or `exp(-708.4)`
(`float64`), are `0`.

```text
explain_backend(loss, /, *args, **kwargs) -> (backend, reason)
```

`explain_backend` reports the backend of the call `loss(*args, **kwargs)` and why; it runs
the call as given (eagerly, or traced under `jit`):

```python
from libsdca.jax import explain_backend, topk_hinge_loss

explain_backend(topk_hinge_loss, logits, labels, k=2)
# ('kernel', 'the CPU kernels through the FFI target (float32 work)')
```

### Build tools

The CPU handler is compiled on first use, which needs a C++17 compiler and a JAX that
exposes `jax.ffi` (as for [PyTorch](TORCH.md#build-tools)); no PyTorch is needed.

Without them, CPU calls run the same kernels in the prebuilt native library (`libsdca_c`)
through `jax.experimental.buffer_callback`, under `jit`, `grad` and `vmap`, after one
`RuntimeWarning` that names the failed build step. Results are the handler's, except that
the native library keeps subnormals; each call adds a host callback. JAX releases without
`buffer_callback` use `jax.pure_callback`, which copies arguments and results.

`SDCA_JAX_FFI=off` never builds the handler and takes the native library without the
warning ([environment variables](API.md#environment-variables)).

## Memory

- **Value only:** `jax.jit(loss)` evaluates the value alone; `jax.value_and_grad` and
  `jax.vjp` also compute what backward needs.
- **Buffer reuse:** the kernels can write the gradient into temporary logits after their
  last use, or into logits donated with `jax.jit(step, donate_argnums=...)`. A non-donated
  argument of the outermost `jit` keeps its buffer and gets a separate gradient
  (`pure_callback` always allocates one).
- **CPU sum or mean:** a differentiated sum or mean computes values and the gradient in one
  forward pass, which `jax.grad` uses directly; other cotangents can evaluate the gradient
  again.

## Errors

- **Arguments:** wrong types (non-floating `logits`, a target of the wrong kind, a
  non-integer `k`, a traced `gamma`, ...) raise `TypeError`; bad values (shapes, `k` outside
  $[1, C-1]$, a negative `gamma`, an unknown `reduction` or `backend`) raise `ValueError`,
  before any computation.
- **Labels:** a label outside the classes (other than the ignored label) raises `IndexError`
  in synchronous CPU calls; under `jit`, XLA reports it at execution. CUDA and traced
  composed calls give `nan` rows.
- **Data values:** `nan` or `+inf` logits, invalid rows with
  [masked classes](#masked-classes) and masks without a positive (or, for
  `multilabel_hinge_loss`, a negative) label give `nan` losses and gradients for those rows,
  on every backend.
- **No kernel:** `backend="auto"` and `"kernel"` raise `NotImplementedError` naming
  `backend="composed"` ([backends](#backends)).

## Masked classes

A logit of `-inf` masks its class by the rule of [API.md](API.md#masked-classes), on every
backend, in concrete calls and under `jit`, `grad` and `vmap`:

- A masked class that is not a label is removed, with a gradient of exactly `+0.0`.
- A masked label, a masked positive class, or a class the student masks where the teacher
  has probability gives the loss `+inf` with the limit of the gradient (for
  `cross_entropy`, the gradient of `torch.nn.functional.cross_entropy`). The CUDA kernels
  give this for the cross-entropy losses only, and `nan` for the others.
- **Distillation:** a class the teacher masks has teacher probability zero; a class both
  mask is removed.
- **Invalid rows** (a student or teacher row of `-inf` only) give `nan`.
- **Options:** class weights and ignored labels apply as for finite rows; a zero weight
  times an infinite loss is invalid. Label smoothing gives `+inf`. With probability targets,
  a masked class of target `0` is removed and one of a positive target gives `+inf`.

## Distillation

`distillation_loss` and `proximal_distillation_loss` take student and teacher logits:

- No $T^2$ factor is applied; a different teacher dtype is promoted with the student's.
- Equal logits give the student an **exact zero gradient**
  ([API.md](API.md#distillation-losses)).
- A differentiated teacher gets PyTorch's gradient.
