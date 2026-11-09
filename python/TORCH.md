# PyTorch loss layers

`libsdca` provides its classification losses as differentiable PyTorch modules
and functions, following `torch.nn` / `torch.nn.functional` conventions:

| Loss | Module | Function |
| --- | --- | --- |
| Cross-entropy (drop-in) | `CrossEntropyLoss` | `cross_entropy` |
| Linear layer + cross-entropy, without the logits | `LinearCrossEntropyLoss` | `linear_cross_entropy` |
| Top-k cross-entropy | `TopKCrossEntropyLoss` | `topk_cross_entropy` |
| Multiclass hinge (Crammer–Singer) | `MulticlassHingeLoss` | `multiclass_hinge_loss` |
| Top-k hinge, β (default) or α | `TopKHingeLoss` (`variant="beta"` or `"alpha"`) | `topk_hinge_loss` |
| Top-k hinge β | `TopKHingeBetaLoss` | `topk_hinge_beta_loss` |
| Top-k hinge α | `TopKHingeAlphaLoss` | `topk_hinge_alpha_loss` |
| Multilabel cross-entropy | `MultilabelCrossEntropyLoss` | `multilabel_cross_entropy` |
| Multilabel hinge | `MultilabelHingeLoss` | `multilabel_hinge_loss` |
| Proximal cross-entropy (Moreau envelope) | `ProximalCrossEntropyLoss` | `proximal_cross_entropy` |
| Logit distillation | `DistillationLoss` | `distillation_loss` |
| Proximal distillation | `ProximalDistillationLoss` | `proximal_distillation_loss` |

Every hinge takes `gamma`: 0 (the default) is the nonsmooth hinge, a positive
value smooths it. The losses run on CPU and CUDA tensors, in float16, bfloat16,
float32 and float64, through ordinary autograd. They need PyTorch 2.1 or newer and
are tested with 2.1, 2.5, 2.6, 2.8, 2.14 and 2.15
([tested configurations](../DEPENDENCIES.md#tested-configurations)).

```bash
pip install 'libsdca[torch]'
```

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

Every name is in `libsdca` and `libsdca.torch`; importing `libsdca` alone does
not import `torch`. The named losses call the generic
`sdca_loss(logits, target, loss="cross_entropy", *, backend="auto", k=1, gamma=0.0,
alpha=1.0, reduction="mean", dim=-1, precision=None)` and `SDCALoss(loss, **options)`.
The losses include no weight regularization; use the optimizer's weight decay.

## Build tools

The fastest CPU and CUDA paths are small C++ extensions that libsdca compiles
once, on first use, against the installed PyTorch. They need ninja (installed
with `libsdca[torch]`), a C++17 compiler with OpenMP (C++20 for PyTorch 2.12
and later; checked with GCC 11, Clang 14 and 18, Apple clang 21;
`build-essential` on Debian/Ubuntu, the Xcode Command Line Tools on macOS), the
Python headers, and on CUDA the CUDA toolkit headers. Without them every loss still runs, on the
prebuilt native library (CPU) or through Triton (CUDA), more slowly, with one
warning; `SDCA_CPU_EXTENSION=off` and `SDCA_CUDA_LAUNCHER=off` silence it
([environment variables](../docs/API.md#environment-variables)). Older PyTorch
releases build extensions through setuptools (2.1 also through its `pkg_resources`):
where those do not load, the warning names them, and `pip install 'setuptools<81'`
restores the extension.

## Replace PyTorch cross-entropy

```python
from libsdca import CrossEntropyLoss, cross_entropy

criterion = CrossEntropyLoss()           # was torch.nn.CrossEntropyLoss()
loss = criterion(net(features), target)  # raw logits, int64 class indices
loss = cross_entropy(net(features), target)
```

The class axis, `reduction`, class `weight`, `ignore_index` and `(N, C, ...)`
inputs behave as in PyTorch, and 16-bit logits return float32 losses with
gradients in the logits' dtype. Probability targets and `label_smoothing` are
not supported. Invalid inputs raise `ValueError` on the CPU; on CUDA they give
NaN without synchronizing with the host.

The objective is the same; the difference is accuracy when the correct class is
far ahead:

```python
logits = torch.tensor([[0., -20.]], requires_grad=True)
loss = CrossEntropyLoss()(logits, torch.tensor([0]))
loss.backward()
print(loss.item())  # about 2.061e-9; PyTorch returns 0 in float32
print(logits.grad)  # about [[-2.061e-9, 2.061e-9]]; PyTorch loses the target entry
```

For this two-class cross-entropy case, libsdca keeps the loss and gradient into
the output dtype's subnormal range: the first zero occurs near margin 104 in
float32, while float64 remains nonzero at 745, against PyTorch's collapse near
16.6 and 36.7. Default CUDA distillation has a separate extreme aggregate-tail
limitation ([numerics](../docs/numerics.md#aggregate-tail-limitations)).
On an A100, at 8192×32000 and 8192×128256 bfloat16 or float32 logits, its
forward+backward takes 0.33–0.41× the time of eager `F.cross_entropy` and 0.93–0.98×
`torch.compile`'s, and it keeps a few scalars per row instead of an `(N, C)` tensor between
forward and backward. Small calls, up to about 10⁶ logits, are bound by host overhead and
take longer than PyTorch's ([performance](../docs/performance.md#where-libsdca-is-slower)).

### Argument parity

| `nn.CrossEntropyLoss` feature | libsdca |
| --- | --- |
| `weight` | Supported: finite, nonnegative, shape `(C,)`, same device and dtype as the logits (float32 weights are accepted with 16-bit logits under autocast). Invalid weights raise `ValueError` on the CPU and give NaN on CUDA. |
| `ignore_index` (default `-100`) | Supported, any int64; keyword-only. |
| `reduction` | `"none"`, `"sum"`, `"mean"`. A weighted mean divides by the sum of the weights of the targets that are not ignored. Fully ignored input gives `sum = 0`, `mean = NaN` and zero gradients, as in PyTorch. |
| `label_smoothing` | Only `0.0`; see [below](#label-smoothing-and-probability-targets). |
| Probability targets | Not supported; use `distillation_loss`. |
| `size_average`, `reduce` | Not supported (deprecated in PyTorch); use `reduction`. |
| Positional arguments | Only `weight`; every later argument is keyword-only. |
| Input shapes | `(C,)`, `(N, C)`, `(N, C, d1, ...)`, with PyTorch's class axis; non-contiguous inputs are fine. |
| Target dtype | int64 only. |
| 16-bit logits | The gradient keeps the logits' dtype; the loss is float32, as under autocast. |
| Invalid targets, nonfinite logits | `ValueError` on the CPU; NaN on CUDA, without a host synchronization. |
| Double backward | Supported. |
| `torch.compile`, CUDA graphs | Supported ([compilation](#compilation-and-transformations)). Fused CUDA reductions are deterministic for a fixed batch size. |

The other libsdca losses put classes on `dim=-1` and take no `weight` or `ignore_index`;
pass `dim=1` for `(N, C, H, W)` logits.

### Label smoothing and probability targets

Smoothed targets are strictly positive, so `distillation_loss` with their logarithm as the
teacher reproduces `label_smoothing` exactly (without `weight` or `ignore_index`):

```python
import torch.nn.functional as F
from libsdca import distillation_loss

eps = 0.1
q = (1 - eps) * F.one_hot(y, C).to(z.dtype) + eps / C
loss = distillation_loss(z, q.log(), 1.0)  # == F.cross_entropy(z, y, label_smoothing=eps)
```

## Linear layer and cross-entropy without the logits

`linear_cross_entropy(hidden, weight, target, bias=None, *, class_weight=None,
ignore_index=-100, reduction="mean", label_smoothing=0.0, backend="auto", precision=None,
chunk_bytes=None, recompute=False)` and its module `LinearCrossEntropyLoss` compute
`cross_entropy(F.linear(hidden, weight, bias), target)`, the loss of a language-model head,
without storing the `(N, V)` logits:

```python
from libsdca import linear_cross_entropy

hidden = model.body(tokens)                                          # (B, T, H)
loss = linear_cross_entropy(hidden, model.lm_head.weight, labels)    # labels (B, T); -100 ignored
loss.backward()
```

**Chunking and memory.** Rows are processed in chunks of at most `chunk_bytes` of logits (at
least one row; default `256 MiB` or `2048` rows on CPU / CUDA FP32, and at least `8192` rows
on CUDA BF16). For each chunk, a GEMM computes the logits, `cross_entropy`'s fused kernels
turn them into their gradient `G` in place, and three GEMMs accumulate `grad_hidden = G W`,
`grad_weight += Gᵀ h` and `grad_bias += Σ G` before the chunk is discarded:
- **Saved gradients (default for `"mean"` / `"sum"` on BF16/FP32/FP64):** Forward computes and
  stores the float32 parameter gradients at upstream derivative `1`, and backward scales them by
  the upstream derivative (3 GEMMs total, peak memory = parameter gradients + one chunk).
- **Recomputed gradients (`recompute=True`, `reduction="none"`, or `float16`):** Forward keeps
  only the loss and one scalar; backward recomputes each chunk's logits and gradients (4 GEMMs
  total, nothing held between forward and backward).
- On the CPU, `ignore_index` rows are skipped before their GEMMs; on CUDA they produce zero
  gradients without host synchronization. Invalid targets or nonfinite logits on CUDA make a
  `"mean"` or `"sum"` loss and all its gradients `NaN`.

**Precision and `GradScaler`.** The loss and logit gradients keep `cross_entropy`'s guarantees
relative to the chunk GEMM's logits (no softmax collapse, float64 row-loss accumulation, and
single rounding after class weights and upstream scaling). When forward saves the gradients at
upstream derivative `1`, entries that already underflow float32 at scale `1` cannot be recovered
by a later `GradScaler` factor (for example, bfloat16 gap `94` or float32 gap `105` with scale
`65536` gives `0` when saved, vs. `±9.8e-37` / `±1.6e-41` with `recompute=True`). Pass
`recompute=True` when using a `GradScaler` that must preserve subnormal tails:

| Quantity | dtype |
| --- | --- |
| Compute dtype | that of `hidden`, `weight` and `bias` (must match); under autocast, the autocast dtype (`float64` stays `float64`), matching `F.linear` |
| Logits | GEMM of compute-dtype operands with float32 accumulation, rounded to compute dtype |
| Loss | `promote(compute dtype, float32)`, accumulated from float64 row losses |
| Logits' gradient | compute dtype; working arithmetic as in `cross_entropy` (`precision`) |
| `grad_hidden`, `grad_weight`, `grad_bias` | each input's dtype, accumulated in float32 (`float64` for `float64`) and rounded once |

**Backends and 16-bit CPU note.** On CUDA, `linear_cross_entropy` uses cuBLAS and the weighted
fused Triton kernels of `cross_entropy` ([performance](../docs/performance.md#pytorch-on-cuda)).
On the CPU, it uses the CPU extension (or the native library for `float64` / without the
extension), converting 16-bit operands to float32 in 64 MiB weight tiles to avoid slow 16-bit
CPU GEMMs (which uses 6 bytes per weight during the call, about `1.7×` eager PyTorch's memory at
`1024×4096×128256` bfloat16; [performance](../docs/performance.md#pytorch-on-the-cpu)).
`backend="torch"` forces tensor operations. Registered via `torch.library`, it traces under
`torch.compile` without graph breaks (first-order gradients only; `label_smoothing=0.0`;
`DTensor` inputs fall back to `cross_entropy(F.linear(...))`).

## Devices and backends

`backend="auto"`, the default, keeps logits on their device and takes the
fastest path that applies:

| Inputs | `auto` runs |
| --- | --- |
| CUDA, up to 2²⁵ classes and 2²⁸ rows | fused Triton kernels, launched from C++ autograd (`cuda-launcher`) or by Triton (`triton`) |
| CUDA otherwise | tensor operations (`torch`): beyond those limits, before PyTorch 2.7 / Triton 3.3, for the hinge, top-k (`k > 1`) and boolean-mask losses with `precision="float32"`, and for top-k losses with `k > 32` (top-k cross-entropy: `k ≥ classes − 1`) or 2³¹ logits or more |
| CPU, build tools available | the CPU extension (`cpu-kernel`) on PyTorch's thread pool; uses the native library instead for float64 integer cross-entropy with `classes > 32` and for `precision="float32"` on hinge, multilabel, and `k > 64` top-k losses |
| CPU otherwise | the prebuilt native library (`native`), on PyTorch's OpenMP threads (rows split as its static loops) |

`backend="torch"` forces tensor operations on any device and `backend="native"`
the native library on the CPU. `explain_backend` names the path a call takes and why:

```python
>>> from libsdca.torch import explain_backend
>>> logits = torch.randn(8, 1000, device="cuda", dtype=torch.bfloat16)
>>> explain_backend(logits, torch.randint(1000, (8,), device="cuda"), "cross_entropy")
('cuda-launcher', 'fused Triton kernels launched from C++ autograd (float32 exponentials, rows in registers)')
```

The CUDA kernels never synchronize with the host, so calls can be captured in
CUDA graphs (run each loss once on the capture stream first). Wide rows, such as
language-model vocabularies, stream through column chunks. Losses are reduced
deterministically for a given batch size. Cross-entropy has second derivatives;
the other losses are first order.

## Precision

Loss values have dtype `promote(logits.dtype, float32)` and gradients keep the
logits' dtype, on every backend. `precision` selects only the working arithmetic:

| `precision` | Working arithmetic |
| --- | --- |
| `None` (default) | float64 for float64 logits; for narrower logits, float32 exponentials with float64 row statistics in cross-entropy (integer targets and boolean masks) and top-k cross-entropy (`k <= 64` on CPU, `k <= 32` on CUDA), float64 in the fused hinge (top-k and multilabel hinges included) kernels |
| `"float32"` | float32, promoted to float64 where float32 cannot hold `gamma`, `alpha` or a row's spread; on CUDA the hinge, top-k (`k > 1`) and boolean-mask losses then take tensor operations |
| `"float64"` | float64 throughout, rounded once to the output dtypes |

Fused cross-entropy losses are within 8 float32 eps of 60-digit references, and
every gradient entry is scaled by the upstream factor and rounded once. CPU
boolean-mask cross-entropy of float32 logits (`precision=None`) takes float32
exponentials with float64 row statistics and the positives in float64: each
gradient entry is within one rounding plus 2 float32 eps of the row's largest
entry; with `precision="float64"` (and for float64 logits) every entry is the
float64 gradient rounded once. Accuracy of every loss and its tested limits:
[numerics.md](../docs/numerics.md).

### float16 and bfloat16 logits

16-bit logits keep 16-bit storage for the input and the gradient, which halves
memory traffic; the arithmetic is float32 or float64 in registers. Against
float64 results, values are within one output unit plus 100 float32 eps and
gradient entries within one unit of the logits' dtype plus 2000 float32 eps of
the row's largest entry.

### FAQ: Why is there no `bfloat16` or `float16` compute `precision`?

16-bit arithmetic is neither faster nor accurate enough here. GPUs evaluate `exp`
and `log` only in float32, memory traffic is already 16-bit, and a bfloat16 sum of
`exp` terms stops growing at 256 equal terms (float16 overflows above 65,504). On
an H100 a float32 row sum ran as fast as a bfloat16 one, with a 54,000–74,000×
smaller error.

## Memory

On CUDA (`cuda-launcher`), the backward of cross-entropy, the `k = 1` hinges and the
multilabel losses writes the gradient over the logits when nothing else holds them. In an
eager training step of an LLM head (8192 rows, 4096 features, 128256 classes, bfloat16
autocast, fused AdamW), the memory allocated above the step's start peaked at 1066 MiB for
eager libsdca, compared with 2068 MiB for libsdca inside `torch.compile` (which does not
write the gradient in place) and 8080 MiB for eager PyTorch. Keep
no other reference to the logits during `backward()` to get it
(`loss = criterion(net(x), y)`, or `del logits` first); the result is the same
either way.

### Memory on the CPU

A forward and backward on the CPU peaks at the gradient's size. Losses that
compute their gradient in the forward (the native library's top-k, hinge,
multilabel and proximal losses, the extension's proximal losses, and the
extension's `distillation_loss` of float32 and float64 logits) keep it until
backward, in float32 for 16-bit logits. Eager calls without gradient compute
the native and proximal ones on row blocks instead of the whole gradient.

## Compilation and transformations

The CPU and CUDA losses are `torch.library` operators: `torch.compile` traces them
without graph breaks, also with `fullgraph=True` (tested from PyTorch 2.5; releases
before 2.4 lack `torch.library.custom_op`: there, compile the model and leave the loss
outside, as below). They are opaque to Inductor, so
they are not fused with the surrounding graph, and inside a compiled graph the
gradient is not written over the logits. To keep that saving, compile the model
and leave the loss outside:

```python
loss_fn = torch.compiler.disable(cross_entropy)
step = torch.compile(lambda x, y: loss_fn(net(x), y))
```

Double backward needs eager mode. `vmap` and forward-mode AD are not tested.
On PyTorch 2.5 the first CPU `torch.compile` turns flush-to-zero on for the whole
process (2.6 does not), so subnormal values computed afterwards become zero;
`torch.set_flush_denormal(False)` restores gradual underflow.

## Distributed training

The fused kernels need every class of a row on one device, which DDP, FSDP and
pipeline parallelism satisfy. Class-sharded `DTensor` logits are accepted through
a fallback that gathers the full inputs on every rank and returns a replicated
result: correct, but not memory-efficient (tested from PyTorch 2.2, whose
`torch.distributed.device_mesh` builds the meshes; their higher derivatives need
PyTorch 2.12). For vocabulary-parallel cross-entropy
use PyTorch's [`loss_parallel()`](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel)
or Megatron's. With local tensors, normalizing the loss across ranks and
accumulation steps stays the caller's job.

## Other losses

| Loss | Meaning |
| --- | --- |
| `topk_cross_entropy` | Top-k entropy, `1 ≤ k ≤ classes − 1` |
| `multiclass_hinge` | Crammer–Singer hinge: either top-k hinge at `k = 1` |
| `topk_hinge_alpha` | `max(0, mean of the k largest margins 1 + z_j − z_y)` |
| `topk_hinge_beta` | Mean of the k largest positive parts of the margins |
| `multilabel_cross_entropy` | Cross-entropy with uniform probability on the positives of a boolean mask |
| `multilabel_hinge` | `max(0, 1 + max negative score − min positive score)` |
| `proximal_cross_entropy` | The Moreau envelope of cross-entropy with step `alpha` |

```python
from libsdca import SDCALoss

criterion = SDCALoss("topk_cross_entropy", k=2)
criterion = SDCALoss("multilabel_hinge", gamma=0.5)
```

Integer targets are `int64` with the class dimension removed; boolean masks have
the logits' shape. Class weights and `ignore_index` are `CrossEntropyLoss` only.

`proximal_cross_entropy` is the loss behind one SDCA update:

$$
\phi_\alpha(z,y)=\min_u\left\{\operatorname{CE}(u,y)+\frac{\|u-z\|^2}{2\alpha}\right\},
\qquad \nabla_z\phi_\alpha = -a,
$$

where `a` is the dual block of the native solver's step. It is a smoothed loss
with its exact gradient, not cross-entropy with a substituted backward.

## Teacher logits and distillation

```python
from libsdca import DistillationLoss

criterion = DistillationLoss(temperature=4.0)
criterion(student_logits, teacher_logits).backward()
```

Both arguments are logits and the teacher is detached. Identical logits give an
exactly zero gradient, and small temperatures do not overflow. There is no
automatic `temperature²` factor. `ProximalDistillationLoss(temperature, alpha)`
is the Moreau envelope of `distillation_loss`, which it approaches as
`alpha → 0`. Both have first derivatives only
([API.md](../docs/API.md#distillation-losses)).

## Tests

```bash
python -m pytest -n auto tests/python --group torch
```

The autograd functions are checked with `gradcheck`, the entropy losses also with
`gradgradcheck`, and the CUDA operators with `torch.library.opcheck`.
