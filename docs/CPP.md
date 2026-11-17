# C++ quick start and contracts

[All guides](https://github.com/mlapin/libsdca/blob/master/docs/README.md) ·
[Shared API reference](https://github.com/mlapin/libsdca/blob/master/docs/API.md)

## Installation

A C++11 compiler is sufficient with the built-in BLAS (`BUILTIN`). With CMake 3.12+,
`AUTO` uses an external CBLAS when available and falls back to `BUILTIN`; `EXTERNAL`
requires an external CBLAS ([DEPENDENCIES.md](../DEPENDENCIES.md)).

```bash
cmake -S . -B build/release -DSDCA_BLAS_BACKEND=BUILTIN \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/install"
cmake --build build/release --parallel
cmake --install build/release
cmake -S examples/cpp -B build/consumer -DCMAKE_PREFIX_PATH="$PWD/build/install"
cmake --build build/consumer --parallel && build/consumer/sdca_example
```

In a downstream CMake project ([examples/cpp/train.cpp](../examples/cpp/train.cpp)):

```cmake
find_package(libsdca CONFIG REQUIRED)
target_link_libraries(sdca_example PRIVATE libsdca::libsdca)
```

The headers are self-contained (no source file to compile). Keep fast math disabled
(`-fno-fast-math -ffp-contract=off`). Public headers: `sdca/solver.h`, `sdca/loss.h`,
`sdca/prox.h` (`sdca::detail` is internal).

## Quick start

The commands above run a complete classifier. The following sections explain how to
configure that [example](../examples/cpp/train.cpp) for your own data.

## Training

Construct an input view, an output and an objective:

| Part | Call |
| --- | --- |
| Input view | `make_input_feature(n, d, X)` or `make_input_kernel(n, K)` |
| Output | `make_output_multiclass(first, last, num_classes)` |
| Output of label sets | `make_output_multilabel(label_first, label_last, offset_first, offset_last, num_classes)` or `make_output_multilabel(vector_of_label_vectors, num_classes)` |

Pass them with caller-owned buffers to `make_context`, configure `context.criteria`, and
call `make_solver(context).solve()`:

```cpp
auto context = sdca::make_context(sdca::make_input_feature(n, d, X), std::move(output),
                                  std::move(objective), W, A);       // W0 = center: optional 6th
auto kernel = sdca::make_context(sdca::make_input_kernel(n, K), std::move(output),
                                 std::move(objective), A);
```

### Choosing an objective

These factories return an `objective_spec` (validated: `std::invalid_argument`):

| Factory | Arguments |
| --- | --- |
| `cross_entropy_objective` | `c = 1` |
| `topk_cross_entropy_objective` | `c`, `k` |
| `multiclass_hinge_objective` | `c = 1`, `gamma = 0` |
| `topk_hinge_alpha_objective`, `topk_hinge_beta_objective` | `c`, `k`, `gamma = 0` |
| `multilabel_hinge_objective` | `c = 1`, `gamma = 0` |
| `multilabel_cross_entropy_objective` | `c = 1` |

A visitor receives the objective of a spec:

- `visit_multiclass_objective<Data, Result>(spec, visitor)` and
  `visit_multilabel_objective<Data, Result>(spec, visitor)` call `visitor(objective)` with
  its type.
- `gamma > 0` selects the smoothed hinge; `multiclass_hinge` is the top-1
  `topk_hinge_alpha`.
- The visitor is a functor with a templated `operator()` (C++11), as in the README and the
  example.

### Typed objectives

The objective types are all `<Data, Result>`, with `make_objective_l2_*` factories (`k` and
`gamma` required where the type takes them):

- `l2_cross_entropy`, `l2_topk_cross_entropy`, `l2_multilabel_cross_entropy`;
- `l2_topk_hinge_alpha`, `l2_topk_hinge_beta`, `l2_multilabel_hinge`, and the `_smooth`
  hinges, which need `gamma > 0`;
- `l2_distillation` (with `make_output_distribution`).

### Inputs and outputs

Every array is row-major and used in place, as in the other interfaces
([API.md](API.md#array-layouts)):

| Array | Shape | Notes |
| --- | --- | --- |
| Features `X` | `(n, d)` | `make_input_feature(n, d, X)` |
| Weights | `(m, d)` | One weight vector per class |
| Duals `A` | `(n, m)` | One block of `m` values per example |
| Center `W0` | `(m, d)` | Optional; regularizes $\frac12\Vert W - W_0 \Vert^2$ |

Labels are 0-based and must lie in `[0, num_classes)`. At the start the weights must equal
`W0 + A' X` (zero duals: `W0`). `sdca::batch_scores(kernel, n, d, m, X, weights, scores)`
computes the scores `(n, m)` of a batch with one `gemm` of the BLAS backend.

### Solver options

`context.criteria` holds `tol`, `max_epoch`, `eval_epoch`, `eval_on_start`,
`recompute_primal`, `shrinking`, `projection`, `num_threads` and `seed`:

- **`tol`:** Relative duality gap, `1e-3`.
- **`eval_epoch`:** `0` evaluates only at the end.
- **`seed`:** The order of the examples; `0` by default, seeds equal modulo $2^{31} - 2$
  give the same order.

### Precision

- **`Data`** sets buffer storage (`float` or `double`).
- **`Result`** sets working arithmetic and the solver iterate; it defaults to `Data` for
  loss functions and objective factories ([API.md](API.md#precision)).
- **`<float, double>`** keeps internal `double` copies in `context.primal_variables` and
  `context.dual_variables`, rounding into your `float` buffers at each evaluation.
- **Deduction:** Factory arguments deduce `Result` when supplied: use `<float, float>`
  explicitly if a double literal parameter should be converted to float work.

### Errors and convergence

Inspect `context.status == sdca::solver_status::solved`; reaching
`max_epoch` does not guarantee convergence. `context.is_kernel()` tells a Gram-matrix
context (no `W`).

### Logging

Off by default. `sdca::logging::set_level(level::warning)` (or `info`,
`verbose`, `debug`) enables the solver's messages, `set_stream(os)` redirects them from
`std::cout`, and `set_format` sets their number format.

## Losses and projections

### Loss values and gradients

`sdca::loss_and_gradient<Data, Result>` computes the loss of `m` scores in class order and
writes `m` gradient entries unless `gradient` is null:

| Arguments | Target | Losses |
| --- | --- | --- |
| `(kind, m, scores, label, gradient, params)` | `label` | The single-label losses: `cross_entropy`, `topk_cross_entropy`, `multiclass_hinge`, `topk_hinge_alpha`, `topk_hinge_beta`, `proximal_cross_entropy` |
| `(kind, m, scores, labels, num_labels, gradient, params)` | A sorted label set | `multilabel_hinge`, which needs a class outside the set, and `multilabel_cross_entropy`, up to every class |

- **Parameters:** `sdca::loss_parameters` holds `k` (top-k losses, required), `gamma`
  (hinges), `alpha` (`proximal_cross_entropy`) and `projection` (`multilabel_hinge`); a
  loss takes only its own, the others keep their defaults.
- **Errors:** Invalid arguments throw `std::invalid_argument`; numerical failures throw
  `std::runtime_error`.
- **`float` work** forms the smoothed hinges' margins and gradient entries in `double`.

### Distillation, proximal operators and BLAS

- **Distillation:** `distillation_loss(m, student, teacher, temperature, gradient)` and
  `proximal_distillation_loss(m, student, teacher, temperature, alpha, gradient)`.
- **Proximal operators:** see [below](#projections-and-proximal-operators).
- **BLAS:** The wrappers' transposition argument is `sdca::CBLAS_TRANSPOSE`
  (`sdca::CblasNoTrans`, `sdca::CblasTrans`): CBLAS's own enum with an external CBLAS, else
  one in namespace `sdca`, so that a CBLAS header (such as Accelerate's) included alongside
  the built-in backend does not clash.

## Projections and proximal operators

`sdca/prox.h` declares `sdca::prox_<name>` for every operator of
[API.md](API.md#projections-and-proximal-operators). Each replaces a range of scores by the
solution of its problem:

```cpp
#include "sdca/prox.h"

std::vector<double> a = {0.9, 0.4, -0.2};
sdca::prox_knapsack_eq(a.begin(), a.end());       // onto the simplex: {0.75, 0.25, 0}

std::vector<double> b = {0.9, 0.4, -0.2};
sdca::prox_topk_simplex(b.begin(), b.end(), 2);   // top-2 simplex: {0.5, 0.5, 0}

// Two rows of three entries; aux is scratch for one row.
std::vector<float> rows = {0.9f, 0.4f, -0.2f, 2.0f, 0.0f, 0.0f}, aux(3);
sdca::prox_knapsack_eq(3, rows.begin(), rows.end(), aux.begin());
```

Every operator has three overloads; the parameters follow in the order of the API table,
with its defaults:

| Call | Operates on |
| --- | --- |
| `prox_<name>(first, last, params...)` | `[first, last)` in place, through a temporary copy |
| `prox_<name>(first, last, aux, params...)` | `[first, last)` in place; `aux` (as long) is scratch |
| `prox_<name>(dim, first, last, aux, params...)` | Each block of `dim` consecutive entries; `aux` holds `dim` entries |

The two-block operators (`prox_two_simplex`, `prox_two_simplex_sort`, `prox_two_entropy`)
take their blocks in one of these forms:

| Call | Blocks |
| --- | --- |
| `(a_first, a_last, b_first, b_last, params...)` | Two ranges, through temporary copies |
| `(a_first, a_last, b_first, b_last, a_aux, b_aux, params...)` | Two ranges with scratch of their lengths |
| `(first, middle, last, params...)` | `[first, middle)` and `[middle, last)` |
| `(p, first, last, aux, params...)` | The first `p` entries and the rest |
| `(dim, p, first, last, aux, params...)` | Each block of `dim` entries, split at `p` (not `prox_two_entropy`) |

- **Parameters:** `k` is the iterator's `difference_type`; `lo`, `hi`, `rhs`, `rho` and
  `alpha` have the type `Result`. `prox_two_simplex` takes a last argument
  `sdca::projection_precision` (`float64`, the default, or `extended`;
  [API.md](API.md#projection-precision)).
- **Working precision:** the first template parameter `Result` (default `double`) sets the
  arithmetic, whatever the element type: `sdca::prox_knapsack_eq<long double>(first, last)`.
- **Thresholds:** `thresholds_<name>(first, last, params...)` returns the solution in
  closed form without writing it: the fields `t`, `lo` and `hi` of
  `x[i] = max(lo, min(a[i] - t, hi))` (with `exp` or the Lambert map applied to `a[i] - t`
  for the entropic operators). It may reorder the range.
- **Errors:** `std::invalid_argument` for an empty range, nonfinite scores, an infeasible
  constraint (such as `rhs` outside `[n lo, n hi]` in `knapsack_eq`), a negative `rho`, or
  `k` or block sizes that do not fit the range; `std::runtime_error` if a solve does not
  converge.
- **Lambert function:** `sdca::lambert_w_exp(x)` returns $W(e^x)$ for `float`, `double`
  and `long double`.

## Benchmarks

The [solver benchmarks](https://github.com/mlapin/libsdca/blob/master/docs/PYTHON.md#benchmarks)
compare the shared solver through Python with scikit-learn. They help choose between
feature and kernel inputs; they do not measure this binding's call overhead.
No separate end-to-end timings for this interface are published here.
