# Performance

Runtime and memory of `libsdca` 0.3.0 compared with PyTorch, JAX and scikit-learn.
Ratios are `libsdca` time (or memory) divided by the reference's: below `1.0` means
`libsdca` is faster or uses less memory. Full benchmark tables and methodology are in the
[technical report](https://mlapin.github.io/libsdca/).

## Test systems

| | GPU | CPU (Linux) | CPU (macOS) |
| --- | --- | --- | --- |
| Hardware | NVIDIA A100-SXM4-80GB | AMD EPYC 7J13 (32 cores) | Apple M5 Pro |
| Software | PyTorch 2.15, Triton 3.9, Liger 0.8.5, JAX 0.12 | PyTorch 2.15, JAX 0.12, sklearn 1.9 | PyTorch 2.15, JAX 0.12 |
| Threads | — | 32 and 1 | 18 and 1 |

CUDA timings use CUDA events after warm-up (excluding compilation); CPU timings measure
wall time per call (including Python and autograd) as the median of 7–15 interleaved
rounds on `3·N(0, 1)` logits with `reduction="mean"`. Distillation uses `T = 2` and
masked CE marks ~1% of classes positive. "Compiled" is `torch.compile(dynamic=False)` on
the same objective in PyTorch.

## PyTorch on CUDA

Forward + backward on the A100 (ms):

| Loss, shape | dtype | libsdca | Eager | Compiled | Liger |
| --- | --- | ---: | ---: | ---: | ---: |
| CE, 8192 × 128256 | bfloat16 | 3.842 | 11.052 | 4.134 | 3.773 |
| CE, 8192 × 128256 | float32 | 7.329 | 22.268 | 7.581 | 7.656 |
| CE, 8192 × 32000 | bfloat16 | 0.999 | 2.438 | 1.030 | 1.121 |
| CE, 8192 × 32000 | float32 | 1.853 | 5.477 | 1.895 | 1.764 |
| Top-5 CE, 1024 × 128256 | bfloat16 | 1.110 | 10.165 | 4.255 | — |
| Top-5 CE, 1024 × 128256 | float32 | 1.407 | 10.813 | 6.035 | — |
| Masked CE, 4096 × 32000 | bfloat16 | 0.926 | 3.705 | 0.875 | — |
| Masked CE, 4096 × 32000 | float32 | 1.214 | 6.087 | 1.357 | — |
| Distillation, 4096 × 32000 | bfloat16 | 2.276 | 3.633 | 2.821 | — |
| Distillation, 4096 × 32000 | float32 | 2.920 | 7.259 | 4.680 | — |

- `libsdca` beats eager PyTorch on every row (`1.6–9.2×` faster) and beats `torch.compile`
  on 9 of 10 rows (`3.8–4.3×` faster on top-5 CE, `1.2–1.6×` faster on distillation;
  `bfloat16` masked CE is `1.06×`).
- Liger is `1.8%` faster at `bfloat16` `8192 × 128256` and `5.0%` faster at `float32`
  `8192 × 32000` because it overwrites the logits with the gradient during forward
  (rounding before the upstream scale); `libsdca` rounds each gradient entry once in backward.

### Tall logits

Short rows are packed multiple per thread block. Forward + backward in `bfloat16` (ms),
with ratios to compiled PyTorch in `bfloat16` (`float32` in parentheses):

| Loss, shape | libsdca | Compiled | Eager | / compiled, bf16 (fp32) |
| --- | ---: | ---: | ---: | ---: |
| CE, 262144 × 10 | 0.212 | 0.408 | 0.417 | 0.52 (0.56) |
| CE, 65536 × 100 | 0.182 | 0.295 | 0.231 | 0.61 (0.60) |
| CE, 16384 × 1000 | 0.206 | 0.438 | 0.239 | 0.47 (0.62) |
| Top-5 CE, 262144 × 10 | 0.624 | 1.201 | 23.687 | 0.52 (0.48) |
| Top-5 CE, 65536 × 100 | 0.487 | 1.036 | 7.179 | 0.47 (0.45) |
| Top-5 CE, 16384 × 1000 | 0.398 | 0.938 | 3.366 | 0.42 (0.31) |
| Masked CE, 262144 × 10 | 0.234 | 0.277 | 0.279 | 0.84 (0.79) |
| Masked CE, 65536 × 100 | 0.226 | 0.289 | 0.286 | 0.78 (0.80) |
| Masked CE, 16384 × 1000 | 0.244 | 0.304 | 0.525 | 0.80 (1.02) |
| Distillation, 262144 × 10 | 0.202 | 0.280 | 0.307 | 0.72 (0.73) |
| Distillation, 65536 × 100 | 0.207 | 0.294 | 0.309 | 0.70 (0.65) |
| Distillation, 16384 × 1000 | 0.354 | 0.301 | 0.501 | 1.18 (1.21) |

- `libsdca` beats eager PyTorch in all **24 / 24** cells (up to `38×` faster on top-5 CE)
  and beats `torch.compile` in **21 / 24** (`1.2–3.2×` faster).
- Only `16384 × 1000` `float32` masked CE (`1.02×`) and `16384 × 1000` distillation
  (`1.18–1.21×`) trail `torch.compile`.

### Memory

- Fused CUDA kernels save 4–13 `float64` scalars per row between forward and backward
  instead of an `(N, C)` tensor, and overwrite the logits in place during backward when no
  other reference holds them (`1066 MiB` peak extra allocation on an `8192 × 128256`
  `bfloat16` LLM head step, vs. `2068 MiB` inside `torch.compile` and `8080 MiB` for eager PyTorch).
- `linear_cross_entropy` never materializes `(N, V)` logits, running in `1.00–1.06×`
  compiled PyTorch time on CUDA with `20–24%` less `float32` memory.

## PyTorch on the CPU

Forward + backward time ratios (`libsdca` / PyTorch) at `4096 × 1000` and `4096 × 32000`
across all four dtypes with the CPU extension:

| Loss | Linux / eager | Linux / compiled | macOS / eager | macOS / compiled |
| --- | --- | --- | --- | --- |
| `cross_entropy` | 0.13–0.61 | 0.28–0.87 | 0.53–0.82 | 0.41–0.84 |
| `multiclass_hinge`, top-5 `topk_hinge_alpha` | 0.02–0.18 | 0.04–0.20 | 0.05–0.33 | 0.14–0.42 |
| `multilabel_cross_entropy`, `distillation_loss` | 0.08–1.15 | 0.32–0.93 | 0.35–1.09 | 0.40–0.91 |

- With the ATen/OpenMP C++ extension, `libsdca` beats `torch.compile` across all families:
  `1.15–3.6×` faster on `cross_entropy`, `2.4–25×` faster on multiclass and top-*k* hinges,
  and `1.08–3.1×` faster on `multilabel_cross_entropy` and `distillation_loss`.
- Peak extra memory for a CPU forward + backward pass is a single gradient-sized tensor.
  On 32 threads (`float32`), `linear_cross_entropy` takes `0.86–0.97×` eager time
  (`0.93–1.07×` compiled) and `0.34–0.56×` eager memory (`0.68–0.79×` compiled) from 4096 rows.

## JAX

On the CPU (`4096 × 1000` to `1024 × 128256`, 32 threads), `cross_entropy` under `jit`
takes `0.32–1.29×` the time of `jax.numpy` and `0.67–0.96×` optax's. On the A100 at
`4096 × 32000`, forward + backward (ms):

| Loss | dtype | libsdca | `jax.numpy` |
| --- | --- | ---: | ---: |
| Masked CE | float32 | 1.894 | 1.771 |
| Masked CE | bfloat16 | 1.395 | 1.289 |
| Masked CE | float64 | 3.902 | 4.595 |
| Distillation | float32 | 4.311 | 2.945 |
| Distillation | bfloat16 | 3.590 | 1.772 |
| Distillation | float64 | 11.325 | 6.085 |

- On `float64` masked CE, `libsdca.jax`'s 3-pass layout is `1.18×` faster (`0.85×` time)
  than `jax.numpy`, and within `6–8%` in `float32` and `bfloat16` (`1.07–1.08×`).
- On `distillation_loss`, `libsdca.jax` takes `1.46–2.03×` `jax.numpy` time because it
  performs an additional pass with `expm1` log-ratio centering to guarantee exact zero
  gradients at equal logits (whereas naive `jax.numpy` suffers from softmax cancellation).

## Solver

Comparing `fit` against scikit-learn to the same relative suboptimality (`1e-2` to `1e-5`,
EPYC 7J13, 1 and 32 threads):

- **`multiclass_hinge` vs. `LinearSVC` (Crammer–Singer):** `libsdca` takes `0.14–0.59×`
  `LinearSVC`'s time. On tall problems (`d < n`), primal SDCA is `2.0–4.4×` faster; on wide
  problems (`d >= n`), kernel SDCA is up to `44×` faster (`17×` with Gram construction).
- **`cross_entropy` vs. `LogisticRegression` (L-BFGS) and PyTorch L-BFGS:** On tall data,
  scikit-learn's L-BFGS is faster (`libsdca` takes `1.3–14×` its time), while `libsdca`
  primal SDCA is `1.2–2.2×` faster than `torch.optim.LBFGS` for $d \ge 500$. On wide data
  ($d = 16n$), kernel SDCA with a reused Gram matrix is `6.7×` faster than scikit-learn
  L-BFGS and `30×` faster than PyTorch L-BFGS (when counting fresh Gram construction
  $K = X X^\top$, total time is `1.1–1.9×` scikit-learn L-BFGS).

The two kernel SDCA curves below compare solve time alone (`Gram reused`, when $K = X X^\top$
is precomputed once and reused across a `C` or `gamma` sweep) against end-to-end time
including $K = X X^\top$ (`+ Gram build`, a single cold fit from features `X`):

```python
# 1. Build the (n, n) Gram matrix from features X (n, d):
t0 = time.perf_counter()
K = X @ X.T
gram_build_s = time.perf_counter() - t0

# 2. Certified SDCA solve on K:
t0 = time.perf_counter()
model = libsdca.fit(K, y, objective, C=0.1, tol=1e-3, kernel=True)
gram_reused_s = time.perf_counter() - t0          # "libsdca Kernel SDCA (Gram reused)"
gram_plus_build_s = gram_build_s + gram_reused_s  # "libsdca Kernel SDCA (+ Gram build)"
```

![Time to a relative suboptimality against d and n](figures/aspect-frontier.png)

*Why the curves bend near* $d = n = 2{,}000$: Each feature column is standardized to unit
variance without $1/\sqrt{d}$ row normalization, so $\Vert x_i \Vert_2^2 \approx d$ and rows become
nearly orthogonal for $d \ge n$, cutting the required SDCA epochs by `3–10×`.
In addition, `cross_entropy` converges in 3–6 epochs for $d \ge 2{,}000$, so the default
`eval_epoch = 10` check bounds its reported time at 10 epochs. We did not tune `eval_epoch`
during benchmarking, which gives a small advantage to the baselines on wide problems.

In `float32`, setting `precision="float32"` makes `cross_entropy` training `1.6–1.9×`
faster than the default `"float64"` ([numerics](numerics.md#what-each-precision-guarantees)).

## How the losses run

`backend="auto"` selects the fastest available path (`libsdca.torch.explain_backend`
reports the active choice):

- **CUDA:** Fused Triton kernels with `float64` row statistics, deterministic reductions,
  and no host synchronization, launched via a C++ `cuLaunchKernel` autograd extension (or
  Triton directly).
- **CPU (PyTorch):** An ATen/OpenMP C++ extension compiled on first use for the host ISA,
  falling back to the prebuilt native library.
- **Any device (`backend="torch"`):** Composed PyTorch tensor operations.
- **JAX:** `jax.numpy` compositions, plus an XLA FFI custom call into the native library on
  CPU ([TORCH.md](../python/TORCH.md#devices-and-backends), [JAX.md](../python/JAX.md#backends)).

## Where libsdca is slower

- **Tall CPU logits (`C = 10`):** `float32` CE is `1.30–2.78×` PyTorch, `multilabel_cross_entropy`
  is `4.03–5.43×`, 16-bit distillation (`C <= 100`) is `1.4–1.9×`, and JAX `float32` CE is
  `1.28–1.78×` because the CPU kernels reduce one row at a time.
- **Distillation (`16384 × 1000` CUDA, `4096 × 1000` 16-bit CPU, JAX GPU):** `1.18–1.21×`
  compiled PyTorch on CUDA, `1.14–1.51×` eager PyTorch in 16-bit on CPU, and `1.46–2.03×`
  `jax.numpy` on GPU due to extra passes for exact cancellation.
- **Masked CE on CUDA:** `1.06×` compiled PyTorch at `4096 × 32000` (`bfloat16`) and
  `1.02×` at `16384 × 1000` (`float32`).
- **Small host-bound calls:** Up to ~`10^6` logits on CUDA (`1.2–1.3×` PyTorch) and
  `256 × 10` forward-only CPU calls (a few microseconds).
- **16-bit CPU `linear_cross_entropy` memory:** Uses ~`1.7×` eager PyTorch memory at
  `1024 × 4096 × 128256` (`bfloat16`) to accumulate `grad_weight` in `float32`.
