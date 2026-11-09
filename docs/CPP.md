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

Include `sdca/utility/logging.cpp` in **exactly one** translation unit per binary (or
compile it once alongside your sources). Keep fast math disabled (`-fno-fast-math
-ffp-contract=off`).

## Training

Construct an input view (`make_input_feature(d, n, X)` or `make_input_kernel(n, K)`),
output specification (`multiclass_output(m, labels)`), and objective (such as
`l2_entropy<Data, Result>(C)`), pass them with caller-owned weight and dual buffers to
`make_context`, configure `context.criteria`, and call `make_solver(context).solve()`:

- **Layouts:** Features `X` are column-major `(d, n)` (identical byte order to row-major
  `(n, d)` in NumPy/C); weights `W` are column-major `(d, m)`; duals `A` store each
  example's `m`-class block contiguously (`m`-by-`n` column-major). Labels are 0-based.
- **Precision (`<Data, Result>`):** `Data` sets buffer storage (`float` or `double`);
  `Result` sets working arithmetic and the solver iterate (`<float, double>` keeps internal
  `double` copies in `context.primal_variables` and `context.dual_variables`, rounding into
  your `float` buffers at each evaluation; [API.md](API.md#precision)).
- **Convergence:** Inspect `context.status == sdca::solver_status::solved`; reaching
  `max_epoch` is not a convergence certificate.

## Losses and projections

- **`sdca::loss_and_gradient(kind, m, p, scores, gradient, k, gamma, alpha)`:** Returns the
  `double` loss and writes `m` gradient entries for a single row where the first `p`
  entries of `scores` are the positive classes (`p = 1` for single-label objectives).
  Invalid parameters throw `std::invalid_argument`; numerical failures throw
  `std::runtime_error`.
- **Proximal operators (`include/sdca/prox.h`):** Operate in place on mutable iterator
  ranges (`prox_knapsack`, `prox_two_simplex`, `prox_topk_simplex`, `prox_topk_cone`,
  `prox_entropy`, `prox_topk_entropy`, `lambert_w_exp`).
