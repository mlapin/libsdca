# PyTorch loss layers

`libsdca.torch` provides the classification losses as differentiable PyTorch functions and
modules (`pip install 'libsdca[torch]'`), on CPU and CUDA tensors in `float16`, `bfloat16`,
`float32` and `float64`:

| Loss | Module | Function | Own options | Target |
| --- | --- | --- | --- | --- |
| Cross-entropy | `CrossEntropyLoss` | `cross_entropy` | `weight`, `ignore_index=-100`, `label_smoothing=0.0` | labels or probabilities, classes on `dim=1` |
| Linear + cross-entropy | `LinearCrossEntropyLoss` | `linear_cross_entropy` | `class_weight`, `ignore_index=-100`, `label_smoothing=0.0`, `chunk_bytes`, `recompute` | labels or probabilities |
| Top-*k* cross-entropy | `TopKCrossEntropyLoss` | `topk_cross_entropy` | `k` (required, `1 <= k < C`) | labels |
| Multiclass hinge | `MulticlassHingeLoss` | `multiclass_hinge_loss` | `gamma=0.0` (`> 0` smooths) | labels |
| Top-*k* hinge | `TopKHingeLoss` | `topk_hinge_loss` | `k` (required), `variant="beta"` (or `"alpha"`), `gamma=0.0` | labels |
| Proximal cross-entropy | `ProximalCrossEntropyLoss` | `proximal_cross_entropy` | `alpha=1.0` (Moreau envelope) | labels |
| Multilabel cross-entropy | `MultilabelCrossEntropyLoss` | `multilabel_cross_entropy` | | boolean mask |
| Multilabel hinge | `MultilabelHingeLoss` | `multilabel_hinge_loss` | `gamma=0.0` | boolean mask |
| Logit distillation | `DistillationLoss` | `distillation_loss` | `temperature=1.0` | teacher logits |
| Proximal distillation | `ProximalDistillationLoss` | `proximal_distillation_loss` | `temperature=1.0`, `alpha=1.0` | teacher logits |

Every loss also takes `reduction="mean"` (`"none"`, `"sum"`), `backend="auto"` and
`precision=None`, and all but the cross-entropies the class dimension `dim=-1`. The
single-label losses besides cross-entropy take `weight=None` and `ignore_index=None` too.
Hyperparameters are keyword-only (modules mirror `torch.nn`: `CrossEntropyLoss(weight)`).
Labels are class indices of any integer dtype; the multilabel losses take boolean masks of
the logits' shape, and a target of the other kind raises `TypeError`.

```python
import torch
from libsdca.torch import CrossEntropyLoss, TopKHingeLoss

logits = torch.randn(16, 10, requires_grad=True)
labels = torch.randint(0, 10, (16,))
CrossEntropyLoss()(logits, labels).backward()
TopKHingeLoss(k=2)(logits, labels).backward()
```

## Build tools

On first use, `libsdca` compiles small C++ extensions against the installed PyTorch for its
fastest CPU and CUDA paths. This needs a C++17 compiler with OpenMP (`build-essential` on
Debian/Ubuntu, Xcode Command Line Tools on macOS), Python headers (`python3-dev`; included
in conda and uv-managed Pythons) and, on GPU, the CUDA headers. `libsdca[torch]` installs
`ninja` and, on Linux, a standalone Zig C compiler that Triton 3.4+ uses when the host has
none. Without build tools every loss still runs, on the prebuilt native library (CPU) or
Triton (CUDA), after a one-time warning; `SDCA_CPU_EXTENSION=off` and
`SDCA_CUDA_LAUNCHER=off` silence it ([environment variables](../docs/API.md#environment-variables)).

## Replace PyTorch cross-entropy

`CrossEntropyLoss` and `cross_entropy` take the arguments of `torch.nn.CrossEntropyLoss` and
`torch.nn.functional.cross_entropy`:

```python
from libsdca.torch import CrossEntropyLoss

criterion = CrossEntropyLoss()           # drop-in for torch.nn.CrossEntropyLoss()
loss = criterion(net(features), target)  # raw logits, class labels
```

When a prediction is confident, naive $\log(1 + \tau)$ evaluation rounds $1 + \tau$ to $1$
and collapses both the loss and the target gradient to zero (at margin $\sim 16.6$ in
`float32` and $\sim 36.7$ in `float64` for two classes). `CrossEntropyLoss` keeps both down
to the output dtype's subnormal range ($\sim 104$ in `float32`, $\sim 745$ in `float64`):

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
| `label_smoothing` | As PyTorch: `(1 - ε)` times the loss plus `ε / C` times the smoothing term, which a `logsumexp` in tensor operations adds to the fused loss. |
| Probability targets | A floating-point target of the logits' shape, as PyTorch (log-softmax in tensor operations; `ignore_index` must stay negative). |
| Shapes and dtypes | `(C,)`, `(N, C)`, `(N, C, d1, ...)`; non-contiguous inputs; labels of any integer dtype; 16-bit logits return a `float32` loss and 16-bit gradients. |
| Invalid labels / nonfinite logits | `ValueError` on the CPU; `NaN` on CUDA (no host synchronization) and inside `torch.func` transforms. |
| `torch.compile`, CUDA graphs, double backward | Supported ([compilation](#compilation-and-transformations)). |

Integer labels without smoothing take the fused kernels; smoothing and probability targets
work in `float32` for 16- and 32-bit logits (`float64` for `float64` logits or
`precision="float64"`). The other losses place classes on `dim=-1` (pass `dim=1` for
`(N, C, H, W)`).

## Class weights and ignored labels

`topk_cross_entropy`, the multiclass and top-*k* hinges and `proximal_cross_entropy` take
`weight` (`(C,)`, finite and nonnegative) and `ignore_index` (default `None`: no label is
ignored) with PyTorch's cross-entropy rule: each example's loss is multiplied by the weight
of its label, ignored examples contribute nothing (their logits are not read), and the mean
divides by the sum of the label weights. Without either option the call is the fused
kernel's.

## Linear layer and cross-entropy without the logits

`linear_cross_entropy(hidden, weight, target, bias=None, *, class_weight=None,
ignore_index=-100, reduction="mean", label_smoothing=0.0, backend="auto", precision=None,
chunk_bytes=None, recompute=False)` and `LinearCrossEntropyLoss` evaluate
`cross_entropy(F.linear(hidden, weight, bias), target)` in row chunks without materializing
the `(N, V)` logits:

```python
from libsdca.torch import linear_cross_entropy

hidden = model.body(tokens)                                          # (B, T, H)
loss = linear_cross_entropy(hidden, model.lm_head.weight, labels)    # labels (B, T); -100 ignored
loss.backward()
```

- **Chunking and memory:** Rows stream in chunks of `chunk_bytes` (default: `256 MiB` of logits
  or `2048` rows, whichever is more; at least `8192` rows on CUDA BF16). For `"mean"` / `"sum"`
  on BF16/FP32/FP64, forward accumulates `grad_hidden`, `grad_weight` and `grad_bias` in
  `float32` (`float64` for `float64`) at upstream derivative `1` (3 GEMMs total; peak extra
  memory is `grad_hidden`, the parameter gradients and one chunk). With `recompute=True`,
  `reduction="none"`, or `float16`, backward recomputes each chunk's logits (4 GEMMs total;
  nothing saved between forward and backward).
- **Label smoothing** composes the per-example losses with the target logits and one
  matrix-vector product, still without the logits. **Probability targets** have the logits'
  size, and compute the logits in full.
- **Precision and `GradScaler`:** Pass `recompute=True` when using a `GradScaler` that must
  recover subnormal gradient tails that would underflow `float32` at unit scale during
  forward.
- **Invalid rows:** On CUDA, any unignored row with an out-of-range target or nonfinite
  logits makes a `"mean"` or `"sum"` loss and all reduced parameter gradients `NaN`.

## Devices and backends

`backend="auto"` (default) selects the fastest path for each call; `explain_backend(loss,
*args, **kwargs)` reports the path of a call, a function or module with its arguments:

```python
from libsdca.torch import explain_backend, topk_hinge_loss

explain_backend(topk_hinge_loss, logits, labels, k=2)
# ('kernel', "the CPU extension's top-k kernels (float64)")
```

| Path (`backend` that selects it) | Where `backend="auto"` takes it |
| --- | --- |
| `"kernel"`: fused Triton kernels (launched from C++ autograd or by Triton) or the CPU extension | CUDA ($C \le 2^{25}$, $N \le 2^{28}$); CPU with build tools |
| `"native"`: the prebuilt native library on PyTorch's OpenMP pool (CPU tensors) | CPU without build tools; `float64` cross-entropy rows of more than 32 classes |
| `"torch"`: composed tensor operations (CPU or CUDA) | Other CUDA shapes |

`backend="kernel"` behaves as `"auto"` but raises `NotImplementedError` where no fused
kernel computes the call.

## Precision

Loss values have dtype `promote(logits.dtype, float32)`; gradients keep `logits.dtype`.
`precision` takes `None`, `"float32"` or `"float64"` (or `torch.float32`, `torch.float64`).

| `precision` | Working arithmetic |
| --- | --- |
| `None` (default), `"float32"` | `float32` work for 16- and 32-bit logits, with `float64` row statistics and promotion of rows `float32` cannot hold ([API.md](../docs/API.md#precision)); `float64` for `float64` logits (`"float32"` raises) |
| `"float64"` | `float64` throughout, rounded once to the output dtype |

16-bit logits halve memory traffic; arithmetic runs in `float32` or `float64`
([known limits](../docs/API.md#known-limits)).

## Errors

Wrong types raise `TypeError` (a non-tensor, integer logits, a boolean target for a
single-label loss, `k=2.0`); bad values, shapes and options raise `ValueError` (`k=0`,
`precision="fp32"` or `torch.float16`), in constructors as in calls. Data values are checked where it costs no synchronization: on the CPU nonfinite
logits, labels out of range and invalid class weights raise `ValueError`; on CUDA and inside
`torch.func` transforms the affected examples (or a reduction) are `NaN`.

## Memory

On CUDA (the C++ launcher), backward for `cross_entropy` and the hinge, top-*k* and
multilabel losses (not distillation or the proximal losses) overwrites the logits tensor in
place when no other reference holds it (`loss = criterion(net(x), y)` or `del logits` before
`loss.backward()`; PyTorch 2.10 or later). On CPU, peak extra memory is one gradient-sized
tensor.

## Compilation and transformations

All CPU and CUDA losses are registered as `torch.library` custom ops and trace through
`torch.compile(..., fullgraph=True)` without graph breaks (from PyTorch 2.4; tested from 2.5).
Inside a compiled graph, Inductor does not overwrite the logits in place; to keep in-place
backward memory savings while compiling the model, wrap the loss with `torch.compiler.disable`.

`torch.func` transforms (`grad`, `vjp`, `vmap`, `jacrev`) support every loss: each runs its
eager path on the rows (a `vmap` adds the mapped dimension to the rows) and saves the
gradient. `cross_entropy` and `CrossEntropyLoss` also support forward mode and `hessian`;
the other losses are first order, and differentiating their gradient raises `RuntimeError`.

## Distributed training

Fused kernels operate on full class rows (compatible with DDP, FSDP and pipeline
parallelism). Class-sharded `DTensor` inputs are supported via an all-gather fallback.

## Teacher logits and distillation

```python
from libsdca.torch import DistillationLoss

criterion = DistillationLoss()
criterion(student_logits, teacher_logits).backward()
```

Both inputs are raw logits; the teacher receives no gradient, no $T^2$ factor is applied,
and equal student and teacher logits produce an **exact zero gradient**
([API.md](../docs/API.md#distillation-losses)). A teacher of another floating-point dtype
is promoted with the student's.

## Benchmarks

Forward and backward on an NVIDIA A100 with PyTorch 2.15, against PyTorch implementations of
the same losses (`F.cross_entropy` for cross-entropy), eager and compiled alone with
`torch.compile`; median times in ms:

| Loss | Logits | dtype | `libsdca` | `torch.compile` | Eager PyTorch |
| --- | --- | --- | ---: | ---: | ---: |
| Cross-entropy | 8192 × 128256 | `bfloat16` | 4.73 | 4.97 | 13.55 |
| Cross-entropy | 8192 × 128256 | `float32` | 9.09 | 9.33 | 27.24 |
| Multiclass hinge | 4096 × 32000 | `bfloat16` | 0.53 | 1.99 | 6.52 |
| Multiclass hinge | 4096 × 32000 | `float32` | 0.78 | 3.66 | 9.33 |
| Smoothed multiclass hinge | 4096 × 32000 | `float32` | 2.88 | 24.34 | 41.92 |

The `bfloat16` multiclass hinge needs 250 MiB of extra memory at its peak, against 750 MiB
compiled and 1,375 MiB eager.

## Tests

```bash
python -m pytest -n auto tests/python --group torch   # from a source checkout
```
