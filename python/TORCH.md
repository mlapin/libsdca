# PyTorch loss layers

`libsdca` provides its classification losses as differentiable PyTorch modules and
functions (`pip install 'libsdca[torch]'`), running on CPU and CUDA tensors in
`float16`, `bfloat16`, `float32` and `float64`:

| Loss | Module | Function | Options / notes |
| --- | --- | --- | --- |
| Cross-entropy | `CrossEntropyLoss` | `cross_entropy` | `weight`, `ignore_index`, `(N, C, ...)` (integer labels, no smoothing) |
| Linear + cross-entropy | `LinearCrossEntropyLoss` | `linear_cross_entropy` | `class_weight`, `ignore_index`, `chunk_bytes`, `recompute` (no `(N, V)` logits) |
| Top-*k* cross-entropy | `TopKCrossEntropyLoss` | `topk_cross_entropy` | `k` (`1 <= k < C`) |
| Multiclass hinge | `MulticlassHingeLoss` | `multiclass_hinge_loss` | Crammer–Singer; `gamma=0.0` (`> 0` smooths) |
| Top-*k* hinge | `TopKHingeLoss` (`*Beta*`, `*Alpha*`) | `topk_hinge_loss` (`*_beta_loss`, `*_alpha_loss`) | `k`, `variant="beta"` (or `"alpha"`), `gamma=0.0` |
| Multilabel cross-entropy | `MultilabelCrossEntropyLoss` | `multilabel_cross_entropy` | boolean mask target (uniform over positives) |
| Multilabel hinge | `MultilabelHingeLoss` | `multilabel_hinge_loss` | boolean mask target, `gamma=0.0` |
| Proximal cross-entropy | `ProximalCrossEntropyLoss` | `proximal_cross_entropy` | Moreau envelope (`alpha=1.0`) |
| Logit distillation | `DistillationLoss` | `distillation_loss` | `temperature=1.0` |
| Proximal distillation | `ProximalDistillationLoss` | `proximal_distillation_loss` | `temperature=1.0`, `alpha=1.0` |
| Generic loss | `SDCALoss` | `sdca_loss` | `loss`, `k`, `gamma`, `alpha`, `dim=-1` |

```python
import torch
from libsdca import TopKCrossEntropyLoss

net = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 10))
criterion = TopKCrossEntropyLoss(k=3)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
features, target = torch.randn(16, 8), torch.randint(0, 10, (16,))
optimizer.zero_grad()
criterion(net(features), target).backward()
optimizer.step()
```

All names are exported by `libsdca` and `libsdca.torch` (`import libsdca` does not import
`torch` until a PyTorch symbol is accessed).

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

```python
from libsdca import CrossEntropyLoss, cross_entropy

criterion = CrossEntropyLoss()           # drop-in for torch.nn.CrossEntropyLoss(), except
                                         # integer class labels only and no label smoothing
loss = criterion(net(features), target)  # raw logits, int64 class indices
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

| `nn.CrossEntropyLoss` feature | `libsdca.CrossEntropyLoss` |
| --- | --- |
| `weight`, `ignore_index=-100`, `reduction` | Supported (`"none"`, `"sum"`, `"mean"`; `float32` weights accepted for 16-bit autocast). |
| Shapes and dtypes | `(C,)`, `(N, C)`, `(N, C, d1, ...)`; non-contiguous inputs supported; `int64` targets; 16-bit logits return `float32` loss and 16-bit gradients. |
| Invalid targets / nonfinite logits | Raises `ValueError` on CPU; returns `NaN` on CUDA without host synchronization. |
| `torch.compile`, CUDA graphs, double backward | Supported ([compilation](#compilation-and-transformations)). |
| `label_smoothing`, probability targets | Only `label_smoothing=0.0` in `CrossEntropyLoss`; use `distillation_loss` ([below](#label-smoothing-and-probability-targets)). |

All other `libsdca` losses place classes on `dim=-1` by default (pass `dim=1` for `(N, C, H, W)`).

### Label smoothing and probability targets

Pass log-probabilities as the teacher in `distillation_loss` to match `label_smoothing` (up
to rounding). Probability targets must be strictly positive, since teacher logits must be
finite:

```python
import torch.nn.functional as F
from libsdca import distillation_loss

eps = 0.1
q = (1 - eps) * F.one_hot(y, C).to(z.dtype) + eps / C
loss = distillation_loss(z, q.log(), temperature=1.0)
```

## Linear layer and cross-entropy without the logits

`linear_cross_entropy(hidden, weight, target, bias=None, *, class_weight=None,
ignore_index=-100, reduction="mean", label_smoothing=0.0, backend="auto", precision=None,
chunk_bytes=None, recompute=False)` and `LinearCrossEntropyLoss` evaluate
`cross_entropy(F.linear(hidden, weight, bias), target)` in row chunks without materializing
the `(N, V)` logits:

```python
from libsdca import linear_cross_entropy

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
- **Precision and `GradScaler`:** Pass `recompute=True` when using a `GradScaler` that must
  recover subnormal gradient tails that would underflow `float32` at unit scale during
  forward.
- **Invalid rows:** On CUDA, any unignored row with an out-of-range target or nonfinite
  logits makes a `"mean"` or `"sum"` loss and all reduced parameter gradients `NaN`.

## Devices and backends

`backend="auto"` (default) selects the fastest path for each call
(`libsdca.torch.explain_backend` reports the choice):

| Inputs | `backend="auto"` path |
| --- | --- |
| CUDA ($C \le 2^{25}$, $N \le 2^{28}$) | Fused Triton kernels via C++ autograd (`cuda-launcher`) or Triton (`triton`) |
| CUDA (other shapes) | Composed tensor operations (`torch`) |
| CPU (build tools available) | ATen/OpenMP C++ extension (`cpu-kernel`), or `native` for `float64` CE ($C > 32$) |
| CPU (without build tools) | Prebuilt native library (`native`) on PyTorch's OpenMP pool |

Pass `backend="torch"` to force PyTorch tensor operations or `backend="native"` to force
the C ABI library on CPU.

## Precision

Loss values have dtype `promote(logits.dtype, float32)`; gradients keep `logits.dtype`.

| `precision` | Working arithmetic |
| --- | --- |
| `None` (default), `"float32"` | `float32` work for 16- and 32-bit logits, with `float64` row statistics and promotion of rows `float32` cannot hold ([API.md](../docs/API.md#precision)); `float64` for `float64` logits (`"float32"` raises) |
| `"float64"` | `float64` throughout, rounded once to the output dtype |

16-bit logits halve memory traffic; arithmetic runs in `float32` or `float64`
([known limits](../docs/API.md#known-limits)).

## Memory

On CUDA (`cuda-launcher`), backward for `cross_entropy` and the hinge, top-*k* and
multilabel losses (not distillation or the proximal losses) overwrites the logits tensor in
place when no other reference holds it (`loss = criterion(net(x), y)` or `del logits` before
`loss.backward()`; PyTorch 2.10 or later). On CPU, peak extra memory is one gradient-sized
tensor.

## Compilation and transformations

All CPU and CUDA losses are registered as `torch.library` custom ops and trace through
`torch.compile(..., fullgraph=True)` without graph breaks (from PyTorch 2.4; tested from 2.5).
Inside a compiled graph, Inductor does not overwrite the logits in place; to keep in-place
backward memory savings while compiling the model, wrap the loss with `torch.compiler.disable`.

`torch.func` transforms (`grad`, `vmap`, `jacrev`, `hessian`, ...) support `cross_entropy`
and `CrossEntropyLoss` only; the other losses are not supported inside a transform (all but
`linear_cross_entropy` raise `ValueError`).

## Distributed training

Fused kernels operate on full class rows (compatible with DDP, FSDP and pipeline
parallelism). Class-sharded `DTensor` inputs are supported via an all-gather fallback.

## Teacher logits and distillation

```python
from libsdca import DistillationLoss

criterion = DistillationLoss(temperature=4.0)
criterion(student_logits, teacher_logits).backward()
```

Both inputs are raw logits; the teacher receives no gradient, no $T^2$ factor is applied,
and equal student and teacher logits produce an **exact zero gradient**
([API.md](../docs/API.md#distillation-losses)).

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
