# Numerical notes

What `libsdca` 0.3.0 guarantees in floating point, how it is verified, and its known limits
(measured on NVIDIA A100 / H100 GPUs and Linux x86-64 / macOS arm64 CPUs with PyTorch 2.15).

- **No softmax collapse:** Standard cross-entropy rounds $1 + \tau$ ($\tau = \sum_{j \neq y} e^{z_j - z_y}$)
  to $1$ once $\tau < \frac{1}{2}\operatorname{ulp}(1)$, collapsing both the loss and target
  gradient to zero (near margin `16.6` in `float32` and `36.7` in `float64` for two classes;
  [softmax collapse](softmax-collapse.md)). `libsdca` evaluates $\operatorname{log1p}(\tau)$
  and forms the target gradient directly from $-\tau / (1 + \tau)$, staying nonzero down to the
  output dtype's subnormal limit (`104` in `float32`, `745` in `float64`; see
  [aggregate-tail limitations](#aggregate-tail-limitations) for exceptions).
- **Single rounding per gradient entry:** Each gradient entry is scaled by the upstream
  cotangent (e.g. a `GradScaler` factor) and reduction divisor in working precision before a
  single rounding to the output dtype. At `float16` gap `20` with scale `65504`, `libsdca`
  returns `-1.35064e-4` (exact `-1.35014e-4`) where PyTorch returns `0`. (The exception is
  `linear_cross_entropy`'s saved gradients at unit scale; pass `recompute=True` for single
  rounding; [TORCH.md](../python/TORCH.md#linear-layer-and-cross-entropy-without-the-logits).)
- **Zero failures against 100-digit references ([verification](#how-it-is-verified)):**
  `43,865` loss and every-entry gradient checks (`30,145` CPU across extension modes,
  `13,720` CUDA) and `17,618` proximal oracle cases (`10,884` CPU, `6,734` CUDA) pass with
  `0` failures.

## What each precision guarantees

| | `"float32"`, `float32` data | `"float64"`, `float32` data (default) | `float64` data |
| --- | --- | --- | --- |
| Coordinate steps | `float` (hinge projections in `double`) | `double` | `double` |
| `fit` iterate (`W`, duals) | `float` | `double`, rounded once into the returned `float32` arrays at each evaluation | `double` |
| Losses | `float` (`double` when parameters or spread exceed `float`) | `double` | `double` |
| Evaluation and `solved` | `double` | `double` | `double` |
| Single-step entropy block error vs. `double` (`tests/cpp/regression.cpp`) | $\le (64n + 16\max|z|)\cdot 2^{-24}$ | $\le 8\cdot 2^{-24}$ (measured `1.86` float ulps) | reference |
| `multiclass_hinge` on standardized `digits`, lowest certified gap (5000 epochs) | `4.7e-6` | `3.5e-7` (`tol=1e-6` in 2132 epochs; `float64` data: 2100) | `< 1e-7` (2610 epochs) |

Keeping `W` and the duals in `float32` during `fit` loses coordinate updates smaller than
$\frac{1}{2}\operatorname{ulp}$ near the optimum (`89%` of weight updates and `75%` of dual
updates on `digits`). The default `"float64"` mode maintains `W` and the duals in `double`
and certifies their rounded `float32` copies at each evaluation, reaching the `float32`
representation floor (`3.5e-7`).

### The `solved` certificate

`solved` requires $P(W) - D(A) + e \le \text{tol} \cdot \max(|P(W)|, |D(A)|)$
([API.md](API.md#results)), where $P(W)$ and $D(A)$ are evaluated in `double` at the
returned arrays and $e$ bounds the evaluation's own rounding (`test_certificate.py` checks
every `solved` fit against a 50-digit primal). With a regularizer center $W_0$, the
evaluator forms $W - W_0$ directly so $\|W_0\|_2^2$ never cancels (`test_centered_certificate.py`).

### Projection precision

`projection_precision="float64"` (default) solves the `multilabel_hinge` two-simplex
projection in `double` and accepts the result only if every entry lies clear of its
threshold error bound; otherwise it re-solves in `"extended"` (`long double`). Top-$k$ hinge
projections always use the extended solver. Both modes agree within `16 eps` on offsets up
to `1e30` (`tests/cpp/regression.cpp`; [API.md](API.md#projection-precision)).

## Representation limits

First represented two-class margin $g$ (`logits = [0, -g]`, target `0`) at which the target
gradient underflows to exact zero (`test_half_zero_gaps.py`, `test_float32_zero_gaps.py`,
`test_softmax_limits.py`):

| Logits / gradient dtype | `libsdca` target gradient: first zero | PyTorch CPU loss / target: first zero | PyTorch CUDA loss / target: first zero |
| --- | ---: | ---: | ---: |
| `float16` | 17.34375 | 7.625 | 16.640625 |
| `bfloat16` | 93 | 5.5625 | 16.75 |
| `float32` | 103.97208 | 16.63553 | 16.63553 |
| `float64` | 745.13322 | 36.73680 | 36.73680 |

16-bit logits return `float32` loss values (first zero at margin `104` for two classes). With
$C - 1$ equal competitors, the boundary shifts by $\approx \log(C - 1)$.

## How it is verified

Every loss value and gradient entry is compared against a 100-digit `mpmath` oracle evaluated
on the exact represented input values (`2` to `32,769` classes, gaps up to `110` in `float32`
and `750` in `float64`, large offsets, ties, and $\pm\text{finfo.max}$). A check passes when
the loss and every gradient entry lie within `64 eps` of the output dtype (plus one subnormal
quantum near zero; `float32` work adds $C$ working eps):

| Verification suite | Result |
| --- | --- |
| Entropy & distillation stress (`2` to `32,769` classes, all dtypes and CPU/native/CUDA paths vs. 100-digit `mpmath`) | `0` failures in `30,145` CPU and `13,720` CUDA checks (max budget fraction `0.678` CPU, `0.622` CUDA) |
| Proximal oracle (`alpha` from `1e-300` to `1e300`, up to `32,768` classes vs. `mpmath`) | `10,884` CPU and `6,734` CUDA cases within 1 rounding + `64` (`float64`) / `60` (`float32`) working eps |
| Coordinate steps & projections (`tests/cpp/regression.cpp`, `stress.cpp`, `stress_projections.cpp`) | Randomized entropy KKT, warm hinge steps up to $2^{100}$ ($\le 4\text{ eps}$), projections at $10^{-30}\dots 10^{30}$ ($\le 64\text{ eps}$) |
| 16-bit & ordinary-input sweeps (`test_half_gradient_rounding.py`, `test_torch_common_inputs.py`, `test_ordinary_inputs.py`) | Every PyTorch CPU/CUDA path, JAX path, NumPy loss, and `fit` verified against `float64` / `mpmath` |

## Numerical contracts

- Coordinate updates and loss gradients preserve the zero-sum invariant across classes.
- Each gradient entry is scaled by the upstream cotangent and reduction factor before a
  single rounding to the output dtype.
- Python and Rust validate inputs before calling the C ABI; low-level C++ templates assume
  valid buffers.
- PyTorch and JAX tensor backends are tested against both the native C++ library and
  independent high-precision references.

## Known limits

### Certificate and solver

- Tolerances below ~`1e-12` in `float64` exceed evaluation rounding resolution and run to
  `max_epoch`.
- Large regularizer centers $W_0$ limit representable weight updates (`float32` spacing is
  `8` near `1e8`; `float64` spacing is `2` near `1e16`; `test_centered_certificate.py`).
- On platforms where `long double` is 64-bit (`macOS arm64`, MSVC), extended-precision
  retries match `double` (top-$k$ entropy blocks solve up to $C\|x\|_2^2 = 10^{20}$).

### Further precision limits

- **16-bit logits with `"float32"` work (`test_half_precision.py`):** Values within `100`
  `float32` eps of $\max(|\text{loss}|, 1)$; gradients within `2,000` `float32` eps of the
  row's largest gradient entry.
- **Cross-entropy with `float32` exponentials (default for 16-/32-bit logits):** Uses
  `float32` exponentials with `float64` row statistics, accurate within `8` `float32` eps
  of 60-digit references (`test_fused_entropy.py`, `test_topk_kernels.py`,
  `test_multilabel_kernels.py`, `test_cpu_extension.py`). Pass `precision="float64"` for
  full `float64` exponentials.
- **Distillation (`test_distillation_references.py`, `test_distillation.py`):** Within `64`
  output eps at $T \in [0.25, 4]$. With `float32` work, entries where student and teacher
  probabilities $q$ and $p$ nearly agree are within `64` `float32` eps of $(p + q)/T$ (CPU
  kernels promote rows with mean log-ratio $> 32$ to `float64`).
- **Top-$k$ cross-entropy at extreme score spreads (`StressLimit` in `tests/cpp/stress.cpp`):**
  - *Partially capped rows ($k < C - 1$):* Log-odds combine score differences with order-one
    logarithms in working precision, so error grows with competitor spread (for `[0, S, -S, -S]`,
    $k=2$: `float64` relative error is `5e-13` at $S=10^4$, `5e-10` at $10^8$, `9e-6` at
    $10^{12}$, `log 3` at $10^{16}$, and `0` at `finfo.max`; `prox_topk_entropy` behaves
    identically).
  - *All-capped rows ($k = C - 1$):* Uses a scaled double-double mean of score differences,
    resolving ~106 bits in `float64`; spreads spanning $> 2^{100}$ lose their smallest term
    (for `[M/4, H, M, 1, -H]` with $H=2^{200}, M=2^{100}, k=4$, all routes return
    `log 5 = 1.6094` instead of `1.8142`).

### Aggregate-tail limitations

- **CUDA distillation, default `float32` work:** When individual competitor gradients
  underflow `float32` while their sum over thousands of classes is a representable
  `float32` subnormal (e.g. `8,192` classes, student `[0, -110, ...]`, teacher
  `[0, -220, ...]`, $T=1$), the target gradient rounds to `0` instead of `-1.38e-44` while
  the loss (`1.40e-44`) remains accurate. Pass `precision="float64"` to retain it.
- **JAX GPU, default `float32` work:** Individually underflowed `float32` exponentials
  ($\text{gap} = 104$ across `1,000` classes) vanish before summation, returning `0` instead
  of `6.81e-43`; pass `precision="float64"` (with `jax_enable_x64`) to retain them.
