# PyTorch loss layers

[All guides](https://github.com/mlapin/libsdca/blob/master/docs/README.md) ·
[Shared API reference](https://github.com/mlapin/libsdca/blob/master/docs/API.md)

`libsdca.torch` provides the classification losses as differentiable PyTorch functions and
modules, on CPU and CUDA tensors in `float16`, `bfloat16`, `float32` and `float64`. The
[technical report](https://mlapin.github.io/libsdca/report/) compares their speed and
memory with eager and compiled PyTorch.

## Installation

```bash
pip install 'libsdca[torch]'
```

## Quick start

```python
import torch
from libsdca.torch import CrossEntropyLoss, TopKHingeLoss

logits = torch.randn(16, 10, requires_grad=True)
labels = torch.randint(0, 10, (16,))
CrossEntropyLoss()(logits, labels).backward()
TopKHingeLoss(k=2)(logits, labels).backward()
```

## Losses

| Loss | Module | Function | Own options | Target |
| --- | --- | --- | --- | --- |
| Cross-entropy | `CrossEntropyLoss` | `cross_entropy` | `weight`, `ignore_index=-100`, `label_smoothing=0.0` | labels or probabilities, classes on `dim=1` |
| Linear + cross-entropy | `LinearCrossEntropyLoss` | `linear_cross_entropy` | `class_weight`, `ignore_index=-100`, `label_smoothing=0.0`, `chunk_bytes`, `recompute` | labels or probabilities |
| Top-*k* cross-entropy | `TopKCrossEntropyLoss` | `topk_cross_entropy` | `k` (required, `1 <= k < C`) | labels |
| Multiclass hinge | `MulticlassHingeLoss` | `multiclass_hinge_loss` | `gamma=0.0` (`> 0` smooths) | labels |
| Top-*k* hinge | `TopKHingeLoss` | `topk_hinge_loss` | `k` (required), `variant="beta"` (or `"alpha"`), `gamma=0.0` | labels |
| Proximal cross-entropy | `ProximalCrossEntropyLoss` | `proximal_cross_entropy` | `gamma=1.0` (Moreau envelope) | labels |
| Multilabel cross-entropy | `MultilabelCrossEntropyLoss` | `multilabel_cross_entropy` | | boolean mask |
| Multilabel hinge | `MultilabelHingeLoss` | `multilabel_hinge_loss` | `gamma=0.0` | boolean mask |
| Logit distillation | `DistillationLoss` | `distillation_loss` | `temperature=1.0` | teacher logits |
| Proximal distillation | `ProximalDistillationLoss` | `proximal_distillation_loss` | `temperature=1.0`, `gamma=1.0` | teacher logits |

Signatures, for example:

```text
cross_entropy(input, target, weight=None, *, ignore_index=-100, reduction="mean",
              label_smoothing=0.0, backend="auto", precision=None, reuse_logits=True)
topk_hinge_loss(logits, target, *, k, variant="beta", gamma=0.0, weight=None,
                ignore_index=None, reduction="mean", dim=None, backend="auto",
                precision=None, reuse_logits=True)
distillation_loss(student, teacher, *, temperature=1.0, reduction="mean", dim=None,
                  backend="auto", precision=None, reuse_logits=True)
```

A module takes its function's options after the two tensors (`CrossEntropyLoss(weight)`,
`TopKHingeLoss(k=2)`), and class weights become a buffer moved by `.to()`. All options but
cross-entropy's `weight` are keyword-only.

The common options:

| Option | Default | Taken by |
| --- | --- | --- |
| `reduction` | `"mean"` (`"none"`, `"sum"`) | Every loss |
| `backend` ([backends](#devices-and-backends)) | `"auto"` | Every loss |
| `precision` ([precision](#precision)) | `None` | Every loss |
| `dim` (the class dimension) | `1` (`0` for 1-D logits) | All but `cross_entropy` (always `dim=1`) and `linear_cross_entropy` |
| `reuse_logits` ([memory](#memory)) | `True` | All but `linear_cross_entropy` |
| `weight`, `ignore_index` ([class weights](#class-weights-and-ignored-labels)) | `None` | The single-label losses besides cross-entropy |

Labels are class indices of any integer dtype; the multilabel losses take boolean masks of
the logits' shape, and a target of the other kind raises `TypeError`. The loss definitions
are in [API.md](API.md#loss-definitions).

## Build tools

On first use, `libsdca` compiles its CPU kernels against the installed PyTorch and caches
them in `SDCA_CACHE_DIR`. This needs a C++17 compiler (`CXX`, else `c++`;
`build-essential` on Debian/Ubuntu, Xcode Command Line Tools on macOS) and Python headers
(`python3-dev`; included in conda and uv-managed Pythons).

CUDA tensors use fused Triton kernels, launched by the same C++ extension; without it they
launch through Python, with much more overhead per call. On Linux, `libsdca[torch]`
installs a Zig C compiler that Triton can use when the host has none.

Without build tools, CPU calls run the same kernels in the prebuilt native library
(`libsdca_c`) with every option and dtype, after one `RuntimeWarning` per process that
names the failed build step; `explain_backend` reports the route:

```python
explain_backend(cross_entropy, logits, labels)
# ('kernel', 'the CPU kernels of the native library (float32 work; the CPU kernels of
#  libsdca.torch cannot be built here: no C++ compiler (c++ or CXX) on PATH (...))')
```

The native route gives the extension's results, except that `float32` and `float64`
gradients at an upstream derivative other than 1 can round twice (as in compiled graphs).
It allocates its own gradient (no [reuse](#memory)) and adds Python overhead per call.

A failed build is skipped for 24 hours. `SDCA_CPU_EXTENSION=on` retries it; `off` never
builds and takes the native library (CPU) or the Python launcher (CUDA) without the
warning ([environment variables](API.md#environment-variables)).

## Replace PyTorch cross-entropy

`CrossEntropyLoss` and `cross_entropy` accept class weights, ignored labels, reductions
and label smoothing. Pass the options after `weight` by keyword; instead of the deprecated
`size_average` and `reduce`, pass `reduction`.

```python
from libsdca.torch import CrossEntropyLoss

criterion = CrossEntropyLoss()           # drop-in for torch.nn.CrossEntropyLoss()
loss = criterion(net(features), target)  # raw logits, class labels
```

Confident predictions keep their loss and gradient down to the output dtype's subnormal
range (margins up to about 104 in `float32` and 745 in `float64`), where the standard
$\log(1 + \tau)$ rounds both to zero (from about 16.6 and 36.7 for two classes):

```python
logits = torch.tensor([[0., -20.]], requires_grad=True)
loss = CrossEntropyLoss()(logits, torch.tensor([0]))
loss.backward()
print(loss.item(), logits.grad)  # 2.061e-9, [[-2.061e-9, 2.061e-9]]
```

### Argument parity

| `nn.CrossEntropyLoss` feature | `libsdca.torch.CrossEntropyLoss` |
| --- | --- |
| `weight`, `ignore_index=-100`, `reduction` | As PyTorch, including the weighted mean (`float32` weights accepted for 16-bit autocast). |
| `label_smoothing` | As PyTorch. |
| Probability targets | A floating-point target of the logits' shape, finite and nonnegative, with weights and smoothing as PyTorch; `ignore_index` must stay negative. Targets that require gradients get PyTorch's, $-(1-\varepsilon)\,w_c \log p_c$ per unit of loss. |
| Shapes and dtypes | `(C,)`, `(N, C)`, `(N, C, d1, ...)`; non-contiguous inputs; labels of any integer dtype; 16-bit logits return a `float32` loss and 16-bit gradients. |
| Masked classes (logits of `-inf`) | As PyTorch ([masked classes](#masked-classes)). |
| Labels outside the class range | Synchronous CPU calls raise `IndexError`; CUDA and traced composed calls give `NaN` rows. |
| `NaN` or `+inf` logits; a row of `-inf` only | `NaN` loss and gradient for each affected row. |
| `torch.compile`, CUDA graphs | Supported ([compilation](#compilation-and-transformations)). |
| Double backward, forward mode, `hessian` | With `backend="composed"` ([compilation](#compilation-and-transformations)). |

## Class weights and ignored labels

`topk_cross_entropy`, the multiclass and top-*k* hinges and `proximal_cross_entropy` take
`weight` (`(C,)`, finite and nonnegative) and `ignore_index` (default `None`) with
PyTorch's cross-entropy rule:

- each example's loss is multiplied by the weight of its label;
- ignored examples contribute nothing;
- the mean divides by the sum of the label weights.

## Linear layer and cross-entropy without the logits

`linear_cross_entropy` and `LinearCrossEntropyLoss` evaluate
`cross_entropy(F.linear(hidden, weight, bias), target)` in row chunks without materializing
the `(N, V)` logits:

```text
linear_cross_entropy(hidden, weight, target, bias=None, *, class_weight=None,
                     ignore_index=-100, reduction="mean", label_smoothing=0.0,
                     backend="auto", precision=None, chunk_bytes=None, recompute=False)
LinearCrossEntropyLoss(class_weight=None, *, ...)  # the same options
LinearCrossEntropyLoss(...)(hidden, weight, target, bias=None)
```

```python
from libsdca.torch import linear_cross_entropy

hidden = model.body(tokens)                                          # (B, T, H)
loss = linear_cross_entropy(hidden, model.lm_head.weight, labels)    # labels (B, T); -100 ignored
loss.backward()
```

| Option | Meaning |
| --- | --- |
| `chunk_bytes` | Logits per chunk (default: `256 MiB` or `2048` rows, whichever is more; at least `8192` rows on CUDA `bfloat16`). |
| `recompute` | `False`: for `"mean"` and `"sum"` (not `float16`), forward also accumulates the gradients at upstream derivative 1 (3 GEMMs; peak extra memory is `grad_hidden`, the parameter gradients and one chunk). `True`, `reduction="none"` or `float16`: backward recomputes each chunk's logits (4 GEMMs; nothing saved). |
| `class_weight`, `ignore_index`, `label_smoothing` | As in `cross_entropy`; label smoothing still avoids the logits. |

- **Probability targets** have the logits' size, so the logits are computed in full.
- **`GradScaler`:** pass `recompute=True`, so that the scaled backward keeps gradient tails
  that would underflow `float32` at unit scale.
- **Invalid rows:** on CUDA, an unignored row with an out-of-range target or nonfinite
  logits makes a `"mean"` or `"sum"` loss and all parameter gradients `NaN`.

## Devices and backends

```text
explain_backend(loss, /, *args, **kwargs) -> (backend, reason)
```

`explain_backend` reports the path of the call `loss(*args, **kwargs)`, for a function or
a module:

```python
from libsdca.torch import explain_backend, topk_hinge_loss

explain_backend(topk_hinge_loss, logits, labels, k=2)
# ('kernel', 'the CPU kernels (float32 work)')
```

| `backend` | Path |
| --- | --- |
| `"auto"` (default), `"kernel"` | libsdca's kernels: the CPU kernels ([build tools](#build-tools)) or the fused Triton kernels on CUDA ($C \le 2^{25}$, $N \le 2^{28}$; the smooth top-*k* hinges $k \le 32$, the other top-*k* losses $k \le 256$ or $k=C-1$ for top-*k* cross-entropy) |
| `"composed"` | Composed tensor operations, on any device |

Where no kernel computes a call (another device, no CPU extension or native library, no
Triton, a CUDA shape beyond the kernels), `"auto"` and `"kernel"` raise
`NotImplementedError` naming `backend="composed"`; `explain_backend` raises it too.

## Precision

Loss values have dtype `promote(logits.dtype, float32)`; gradients keep `logits.dtype`.
`precision` takes `None`, `"float32"` or `"float64"` (or `torch.float32`, `torch.float64`).

| `precision` | Working arithmetic |
| --- | --- |
| `None` (default), `"float32"` | `float32` work for 16- and 32-bit logits, with `float64` row statistics and promotion of rows `float32` cannot hold ([API.md](API.md#precision)); `float64` for `float64` logits (`"float32"` raises) |
| `"float64"` | `float64` throughout, rounded once to the output dtype |

See also the [known limits](API.md#known-limits).

## Errors

- **Arguments:** wrong types raise `TypeError` (a non-tensor, integer logits, a boolean
  target for a single-label loss, `k=2.0`); bad values, shapes and options raise
  `ValueError` (`k=0`, `precision="fp32"` or `torch.float16`), in constructors as in calls.
- **Labels:** the CPU kernels and eager composed CPU calls raise `IndexError` for a label
  outside the classes (other than the ignored label); CUDA and compiled composed calls give
  `NaN` rows.
- **Data values:** `NaN` and `+inf` logits and invalid class weights make the affected
  examples (or a reduction over them) `NaN`, on every device and backend, without a host
  synchronization.
- **No kernel:** `backend="auto"` and `"kernel"` raise `NotImplementedError` naming
  `backend="composed"` ([backends](#devices-and-backends)).

### Masked classes

A logit of `-inf` masks its class by the rule of [API.md](API.md#masked-classes), on every
backend, in eager mode, compiled graphs and `torch.func` transforms. Finite logits give the
same results as without masking.

- A masked class that is not a label is removed, with a gradient of exactly `+0.0`.
- A masked label, a masked positive class, or a class the student masks where the teacher
  has probability gives the loss `+inf` with the limit of the gradient
  ([API.md](API.md#limits-of-the-gradient)). The CUDA kernels give this limit for the
  cross-entropy losses only, and `NaN` for the others.
- Invalid rows give `NaN`.

| Loss | A masked class | Invalid (`NaN`) |
| --- | --- | --- |
| `cross_entropy`, `topk_cross_entropy(k=1)`, `linear_cross_entropy` (a bias of `-inf`) | As `F.cross_entropy`: a masked label has loss `+inf` (gradient `-1` there and the softmax of the others); a row whose other classes are all masked has loss zero. | A row of `-inf` only; a zero class weight times `+inf`. |
| `cross_entropy` with label smoothing | As PyTorch: the loss is infinite. | |
| `cross_entropy` with probabilities | Target `0` on the class (after weights and smoothing): the loss of the row without it (PyTorch gives `NaN`). A positive target: loss `+inf`, gradient `-t` there. | Targets that are not finite and nonnegative; a row of `-inf` only. |
| `topk_cross_entropy`, `proximal_cross_entropy`, `multilabel_cross_entropy`, `multiclass_hinge_loss`, `topk_hinge_loss`, `multilabel_hinge_loss` | The loss of the row without the class. A row left with its label alone has loss zero; a top-*k* row with fewer than *k* other classes takes the limit of [API.md](API.md#masked-classes). A masked label or positive class: loss `+inf` and the limit of the gradient. | A row of `-inf` only; a zero class weight times `+inf`. |
| `distillation_loss`, `proximal_distillation_loss` | Masked by both: the loss of the row without the class. By the teacher only: teacher probability zero. By the student only: loss `+inf`, gradient `-p / T` there. | A teacher or student row of masked classes only. |

## Memory

For evaluation, use `torch.no_grad()` or `torch.inference_mode()`: the loss prepares
gradient state whenever autograd is recording and the logits require gradients
(`model.eval()` does not turn recording off).

**CPU sum or mean.** The eager CPU forward of `reduction="sum"` or `"mean"` computes the
gradient for an upstream derivative of 1 in the same pass, and `loss.backward()` returns it.
Peak memory is the logits plus their gradient, and the logits are left alone. Other
upstream gradients (a scaled loss, `reduction="none"`) compute the gradient in backward.

**Gradient over the logits (`reuse_logits`).** Every other eager backward (CUDA, and the
other CPU cases) writes the gradient over logits computed by the autograd graph when
nothing else can read them afterwards (PyTorch 2.10 or later). That holds for
`loss = criterion(net(x), y)`, and after `del logits`:

```python
logits = net(x)
loss = criterion(logits, y)
del logits       # backward writes the gradient into the logits' memory
loss.backward()
```

- **Still referenced** (by another tensor, view, Python reference, autograd node, or
  `retain_graph=True`): backward allocates a new gradient and warns once per process. Delete
  the references, pass `reuse_logits=False`, or filter the warning:
  `warnings.filterwarnings("ignore", message="libsdca: the logits are still referenced")`.
- **`reuse_logits=False`:** backward never overwrites the logits and never warns.
- **Leaf logits** (created with `requires_grad=True`, or a view of one) are never
  overwritten; their `.grad` is a new tensor.
- **No reuse and no warning:** under `torch.func` or `create_graph=True`, when the target or
  teacher requires a gradient, before PyTorch 2.10, and without the CPU extension.
- **Copied inputs** (classes not on the last dimension, as `(N, C, H, W)`; a student
  promoted to its teacher's dtype): backward reuses the copy.

`linear_cross_entropy` manages its own chunks
([above](#linear-layer-and-cross-entropy-without-the-logits)); `libsdca.jax` reuses
buffers where XLA can ([JAX.md](JAX.md#memory)).

## Compilation and transformations

**`torch.compile`.** Every loss is a `torch.library` custom op and traces through
`torch.compile(..., fullgraph=True)` without graph breaks (PyTorch 2.4 and later; tested
from 2.5), as does `backend="composed"`. Compiled graphs allocate gradients by Inductor's
buffer reuse rather than `reuse_logits`; to keep eager reuse inside a compiled model, wrap
the loss with `torch.compiler.disable`.

**Composed proximal losses.** With `backend="composed"`, `proximal_cross_entropy` and
`proximal_distillation_loss` solve for a root in a traced `torch.while_loop` (unrolled
where PyTorch lacks it), whose first compilation can take minutes. On the PyTorch 2.5.1
and 2.6.0 CPU wheels, compiling their training call can stall; use
`torch.compile(fn, backend="aot_eager", fullgraph=True)` there.

**`torch.func`.** `grad`, `vjp`, `vmap` and `jacrev` support every loss. The kernels are
first order: differentiating their gradient raises `RuntimeError`, and forward mode (`jvp`,
`jacfwd`, `hessian`) raises. With `backend="composed"`, `cross_entropy`,
`multilabel_cross_entropy`, `topk_cross_entropy` with `k = 1`, `distillation_loss` and the
nonsmooth hinges differentiate at any order and in forward mode; the others to first order.

## Distributed training

The kernels need complete class rows: local logits of DDP and FSDP2 heads work directly,
and a `DTensor` input is gathered in full on each rank (a class-sharded input allocates the
full logits per rank). A mean divides by the local count: for a global mean with different
token counts or weights across ranks, aggregate the numerators and divisors yourself.
Activation checkpointing and saved-tensor hooks (including CPU offload) are supported.

## Teacher logits and distillation

```python
from libsdca.torch import DistillationLoss

criterion = DistillationLoss(temperature=2.0)
criterion(student_logits, teacher_logits).backward()
```

- Both inputs are raw logits; no $T^2$ factor is applied.
- Equal student and teacher logits give the student an **exact zero gradient**
  ([API.md](API.md#distillation-losses)).
- A teacher that requires gradients gets PyTorch's gradient; a teacher of another
  floating-point dtype is promoted with the student's.
