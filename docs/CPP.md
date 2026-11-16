# C++ quick start and contracts

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

## Training

Construct an input view (`make_input_feature(d, n, X)` or `make_input_kernel(n, K)`), an
output (`make_output_multiclass(first, last, num_classes)`, or label sets with
`make_output_multilabel(label_first, label_last, offset_first, offset_last, num_classes)` /
`make_output_multilabel(vector_of_label_vectors, num_classes)`) and an objective, pass them
with caller-owned buffers to `make_context`, configure `context.criteria`, and call
`make_solver(context).solve()`:

```cpp
auto context = sdca::make_context(sdca::make_input_feature(d, n, X), std::move(output),
                                  std::move(objective), W, A);       // W0 = center: optional 6th
auto kernel = sdca::make_context(sdca::make_input_kernel(n, K), std::move(output),
                                 std::move(objective), A);
```

- **Objectives by the shared names:** `cross_entropy_objective(c = 1)`,
  `topk_cross_entropy_objective(c, k)`, `multiclass_hinge_objective(c = 1, gamma = 0)`,
  `topk_hinge_alpha_objective(c, k, gamma = 0)`, `topk_hinge_beta_objective(c, k, gamma =
  0)`, `multilabel_hinge_objective(c = 1, gamma = 0)` and
  `multilabel_cross_entropy_objective(c = 1)` return an `objective_spec` (validated:
  `std::invalid_argument`). `visit_multiclass_objective<Data, Result>(spec, visitor)` and
  `visit_multilabel_objective<Data, Result>(spec, visitor)` call `visitor(objective)` with
  its type: `gamma > 0` selects the smoothed hinge, `multiclass_hinge` is the top-1
  `topk_hinge_alpha`. The visitor is a functor with a templated `operator()` (C++11), as in
  the README and the example.
- **Typed objectives:** `l2_cross_entropy`, `l2_topk_cross_entropy`, `l2_topk_hinge_alpha`,
  `l2_topk_hinge_beta`, `l2_multilabel_hinge`, `l2_multilabel_cross_entropy` (and the
  `_smooth` hinges, which need `gamma > 0`), `l2_distillation` (with
  `make_output_distribution`), all `<Data, Result>`, with `make_objective_l2_*` factories
  (`k` and `gamma` required where the type takes them).
- **Layouts:** Features `X` are column-major `(d, n)` (identical byte order to row-major
  `(n, d)` in NumPy/C); weights `W` are column-major `(d, m)`; duals `A` store each
  example's `m`-class block contiguously (`m`-by-`n` column-major). Labels are 0-based and
  must lie in `[0, num_classes)`. The optional center `W0` (`(d, m)`) regularizes
  $\frac12\Vert W - W_0 \Vert^2$; `W` must equal `W0 + X A'` at the start (zeros: `W0`).
- **Criteria (`context.criteria`):** `tol` (relative duality gap, `1e-3`), `max_epoch`,
  `eval_epoch` (`0`: only at the end), `eval_on_start`, `recompute_primal`, `shrinking`,
  `projection`, `num_threads`, and `seed` (the order of the examples; `0` by default, seeds
  equal modulo $2^{31} - 2$ give the same order).
- **Precision (`<Data, Result>`):** `Data` sets buffer storage (`float` or `double`);
  `Result` defaults to `Data` for loss functions and objective factories and sets working
  arithmetic and the solver iterate (`<float, double>` keeps internal
  `double` copies in `context.primal_variables` and `context.dual_variables`, rounding into
  your `float` buffers at each evaluation; [API.md](API.md#precision)). Factory arguments
  deduce `Result` when supplied: use `<float, float>` explicitly if a double literal
  parameter should be converted to float work.
- **Convergence:** Inspect `context.status == sdca::solver_status::solved`; reaching
  `max_epoch` does not guarantee convergence. `context.is_kernel()` tells a Gram-matrix
  context (no `W`).
- **Logging:** Off by default. `sdca::logging::set_level(level::warning)` (or `info`,
  `verbose`, `debug`) enables the solver's messages, `set_stream(os)` redirects them from
  `std::cout`, and `set_format` sets their number format.

## Losses and projections

- **`sdca::loss_and_gradient<Data, Result>(kind, m, scores, label, gradient, params)`:**
  The loss of a single-label loss (`cross_entropy`, `topk_cross_entropy`,
  `multiclass_hinge`, `topk_hinge_alpha`, `topk_hinge_beta`, `proximal_cross_entropy`) of
  `m` scores in class order at `label`; writes `m` gradient entries unless `gradient` is
  null. The overload `(kind, m, scores, labels, num_labels, gradient, params)` takes a sorted
  label set (`multilabel_hinge`, which needs a class outside the set, and
  `multilabel_cross_entropy`, up to every class). `sdca::loss_parameters` holds `k` (top-k
  losses, required), `gamma` (hinges), `alpha` (`proximal_cross_entropy`) and `projection`
  (`multilabel_hinge`); a loss takes only its own, the others keep their defaults. Invalid
  arguments throw `std::invalid_argument`; numerical failures throw `std::runtime_error`.
  `float` work forms the smoothed hinges' margins and gradient entries in `double`.
- **Distillation:** `distillation_loss(m, student, teacher, temperature, gradient)` and
  `proximal_distillation_loss(m, student, teacher, temperature, alpha, gradient)`.
- **Proximal operators (`include/sdca/prox.h`):** Operate in place on mutable iterator
  ranges (`prox_knapsack_eq`, `prox_knapsack_le`, `prox_two_simplex`, `prox_topk_simplex`,
  `prox_topk_cone`, `prox_entropy`, `prox_topk_entropy`).
- **BLAS:** The wrappers' transposition argument is `sdca::CBLAS_TRANSPOSE`
  (`sdca::CblasNoTrans`, `sdca::CblasTrans`): CBLAS's own enum with an external CBLAS, else
  one in namespace `sdca`, so that a CBLAS header (such as Accelerate's) included alongside
  the built-in backend does not clash.
