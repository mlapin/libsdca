# PyTorch loss layers

`libsdca` provides its classification losses as differentiable PyTorch modules and
functions (`pip install 'libsdca[torch]'`), running on CPU and CUDA tensors in
`float16`, `bfloat16`, `float32` and `float64`:

| Loss | Module | Function | Options / notes |
| --- | --- | --- | --- |
| Cross-entropy (drop-in, except integer class labels only and no label smoothing) | `CrossEntropyLoss` | `cross_entropy` | `weight`, `ignore_index`, `(N, C, ...)` shapes |
| Linear + cross-entropy (unmaterialized logits) | `LinearCrossEntropyLoss` | `linear_cross_entropy` | `class_weight`, `ignore_index`, `chunk_bytes`, `recompute` |
| Top-$k$ cross-entropy | `TopKCrossEntropyLoss` | `topk_cross_entropy` | `k` (`1 <= k < C`) |
| Multiclass hinge (Crammer–Singer) | `MulticlassHingeLoss` | `multiclass_hinge_loss` | `gamma=0.0` (`> 0` smooths) |
| Top-$k$ hinge ($\beta$ default, or $\alpha$) | `TopKHingeLoss` (`TopKHingeBetaLoss`, `TopKHingeAlphaLoss`) | `topk_hinge_loss` (`topk_hinge_beta_loss`, `topk_hinge_alpha_loss`) | `k`, `variant="beta"`, `gamma=0.0` |
| Multilabel cross-entropy | `MultilabelCrossEntropyLoss` | `multilabel_cross_entropy` | boolean mask target (uniform over positives) |
| Multilabel hinge | `MultilabelHingeLoss` | `multilabel_hinge_loss` | boolean mask target, `gamma=0.0` |
| Proximal cross-entropy | `ProximalCrossEntropyLoss` | `proximal_cross_entropy` | Moreau envelope with step `alpha=1.0` ($\nabla_z \phi_\alpha = -a$) |
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

All names are exported by both `libsdca` and `libsdca.torch` (`import libsdca` does not
import `torch` until a PyTorch symbol is accessed).

## Build tools

On first use, `libsdca` compiles small C++ extensions against the installed PyTorch for
its fastest CPU and CUDA paths. This requires `ninja` (installed by `libsdca[torch]`), a
C++17/C++20 compiler with OpenMP (`build-essential` on Debian/Ubuntu, Xcode Command Line
Tools on macOS), Python headers (`python3-dev`), and CUDA headers on GPU. Without them,
every loss still runs on the prebuilt native library (CPU) or Triton (CUDA) with a
one-time warning (`SDCA_CPU_EXTENSION=off` and `SDCA_CUDA_LAUNCHER=off` silence it;
[environment variables](../docs/API.md#environment-variables)).

## Replace PyTorch cross-entropy

```python
from libsdca import CrossEntropyLoss, cross_entropy

criterion = CrossEntropyLoss()           # drop-in for torch.nn.CrossEntropyLoss(), except
                                         # integer class labels only and no label smoothing
loss = criterion(net(features), target)  # raw logits, int64 class indices
```

When a prediction is confident, naive $\log(1 + \tau)$ evaluation rounds $1 + \tau$
to $1$ and collapses both the loss and target gradient to zero (at margin $\sim 16.6$ in
`float32` and $\sim 36.7$ in `float64` for two classes, as in PyTorch $\le 2.15$).
`CrossEntropyLoss` preserves both down to the output dtype's subnormal limit
($\sim 104$ in `float32`, $\sim 745$ in `float64`;
[numerics](../docs/numerics.md#representation-limits)):

```python
logits = torch.tensor([[0., -20.]], requires_grad=True)
loss = CrossEntropyLoss()(logits, torch.tensor([0]))
loss.backward()
print(loss.item(), logits.grad)  # 2.061e-9, [[-2.061e-9, 2.061e-9]]
```

On an A100 at `8192 × 32000` and `8192 × 128256` (`bfloat16` / `float32`), forward +
backward takes `0.33–0.41×` eager `F.cross_entropy` time (`0.93–0.98×` `torch.compile`)
and saves a few scalars per row instead of an `(N, C)` tensor
([performance](../docs/performance.md#pytorch-on-cuda)).

### Argument parity

| `nn.CrossEntropyLoss` feature | `libsdca.CrossEntropyLoss` |
| --- | --- |
| `weight`, `ignore_index=-100`, `reduction` | Supported (`"none"`, `"sum"`, `"mean"`; `float32` weights accepted for 16-bit logits under autocast). |
| Shapes and dtypes | `(C,)`, `(N, C)`, `(N, C, d1, ...)`; non-contiguous inputs supported; `int64` targets; 16-bit logits return `float32` loss and 16-bit gradients. |
| Invalid targets / nonfinite logits | Raises `ValueError` on CPU; returns `NaN` on CUDA without host synchronization. |
| `torch.compile`, CUDA graphs, double backward | Supported ([compilation](#compilation-and-transformations)). |
| `label_smoothing`, probability targets | Only `label_smoothing=0.0` in `CrossEntropyLoss`; use `distillation_loss` ([below](#label-smoothing-and-probability-targets)). |

All other `libsdca` losses place classes on `dim=-1` by default (pass `dim=1` for `(N, C, H, W)`).

### Label smoothing and probability targets

Pass log-probabilities as the teacher in `distillation_loss` to match `label_smoothing`
exactly:

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

- **Chunking and memory:** Rows stream in chunks of `chunk_bytes` (default `256 MiB` or
  `2048` rows on CPU / CUDA FP32, and $\ge 8192$ rows on CUDA BF16). For `"mean"` / `"sum"`
  on BF16/FP32/FP64, forward accumulates `grad_hidden`, `grad_weight` and `grad_bias` in
  `float32` (`float64` for `float64`) at upstream derivative `1` (3 GEMMs total; peak extra
  memory is the parameter gradients plus one chunk). With `recompute=True`,
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
| CUDA (other shapes, or `precision="float32"` on hinges / masks / $k > 1$ top-$k$) | Composed tensor operations (`torch`) |
| CPU (build tools available) | ATen/OpenMP C++ extension (`cpu-kernel`), or `native` for `float64` CE ($C > 32$) and `precision="float32"` hinges/masks |
| CPU (without build tools) | Prebuilt native library (`native`) on PyTorch's OpenMP pool |

Pass `backend="torch"` to force PyTorch tensor operations or `backend="native"` to force
the C ABI library on CPU.

## Precision

Loss values have dtype `promote(logits.dtype, float32)` and gradients keep `logits.dtype`:

| `precision` | Working arithmetic |
| --- | --- |
| `None` (default) | `float64` for `float64` logits and fused hinges; `float32` exponentials with `float64` row statistics for 16-/32-bit cross-entropy and top-$k$ CE ($k \le 64$ CPU, $k \le 32$ CUDA) |
| `"float32"` | `float32` (promoted to `float64` when `gamma`, `alpha` or row spread exceeds `float32`) |
| `"float64"` | `float64` throughout, rounded once to the output dtype |

16-bit logits halve memory traffic while computing in `float32` or `float64` registers
(values within 1 output ulp + 100 `float32` eps; gradients within 1 output ulp + 2000
`float32` eps of the row maximum; [numerics.md](../docs/numerics.md)).

## Memory

On CUDA (`cuda-launcher`), backward for `cross_entropy`, $k=1$ hinges and multilabel losses
overwrites the logits tensor in place when no other reference holds it (`loss = criterion(net(x), y)`
or `del logits` before ` loss.backward()`). On CPU, peak extra memory is one gradient-sized tensor.

## Compilation and transformations

All CPU and CUDA losses are registered as `torch.library` custom ops and trace through
`torch.compile(..., fullgraph=True)` without graph breaks (from PyTorch 2.5). Inside a
compiled graph, Inductor does not overwrite the logits in place; to keep in-place backward
memory savings while compiling the model, wrap the loss with `torch.compiler.disable`.

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
and equal student and teacher logits give an **exact zero gradient**
([API.md](../docs/API.md#distillation-losses)).

## Tests

```bash
python -m pytest -n auto tests/python --group torch
```
