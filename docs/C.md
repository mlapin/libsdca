# C interface

`include/sdca/c_api.h` declares the exception-safe C ABI used by the Python and Rust
bindings and available to any language with a C foreign function interface. Its comments
are the reference; this page summarizes them.

## Building

```bash
cmake -S . -B build -DSDCA_BUILD_C_API=ON -DSDCA_BLAS_BACKEND=BUILTIN \
  -DCMAKE_INSTALL_PREFIX=$PWD/build/install
cmake --build build --target install
```

This installs `lib/libsdca_c.so` (macOS: `lib/libsdca_c.dylib`) and the headers. CMake
consumers link `libsdca::sdca_c`:

```cmake
find_package(libsdca CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE libsdca::sdca_c)
```

Without CMake, compile `src/c_api.cpp` directly with any C++11 compiler:

```bash
c++ -std=c++11 -O2 -shared -fPIC -pthread -fno-fast-math -ffp-contract=off \
  -fvisibility=hidden -fvisibility-inlines-hidden -DBLAS_BUILTIN \
  "-DSDCA_VERSION=\"$(cat VERSION)\"" -Iinclude src/c_api.cpp -o libsdca_c.so
```

## Entry points

| Function | Does |
| --- | --- |
| `sdca_loss_evaluate` | Loss values and (optionally) gradients of any loss at a batch of scores |
| `sdca_dual_update` | One coordinate step of a training objective per row (new dual blocks) |
| `sdca_model_fit` | Trains a linear or kernel model; returns `W`, the duals, a status and a history |
| `sdca_history_rows` | The history rows a fit with these options may write |
| `sdca_max_extent` | The largest size a fit takes (`n`, `d`, `num_classes`, weight and dual counts) |
| `sdca_abi_version`, `sdca_version`, `sdca_last_error` | The ABI, the library version, the last failure |

## Arguments

Every call separates what is computed from how:

- **`struct sdca_loss`:** the loss (`enum sdca_loss_kind`, one per shared name of
  [API.md](API.md#loss-definitions)) and its own parameters: `k` (top-*k* losses,
  required), `gamma` (hinges; `0` is nonsmooth), `alpha` (proximal losses), `temperature`
  (distillation losses), `projection_precision` (`multilabel_hinge`). Every other field
  keeps its default from `SDCA_LOSS_INIT` (`k = 0`, `gamma = 0`, `alpha = 1`,
  `temperature = 1`, `SDCA_PROJECTION_FLOAT64`); a parameter the loss does not take is
  `SDCA_INVALID_ARGUMENT`.
- **`struct sdca_execution`:** the working precision (`SDCA_PRECISION_DEFAULT` works in the
  storage type, `_FLOAT32` needs `float` storage, `_FLOAT64` works in `double`) and
  `num_threads` (`0`: `SDCA_NUM_THREADS`, else the available CPUs; `n`: up to `n`).
  Storage types are `SDCA_DTYPE_FLOAT32` and `SDCA_DTYPE_FLOAT64`.
- **`struct sdca_targets`:** a kind and the fields of that kind (the others `NULL`):

  | Kind | Fields | Losses |
  | --- | --- | --- |
  | `SDCA_TARGET_LABELS` | `labels` (`n`, 0-based) | single-label losses, `proximal_cross_entropy` |
  | `SDCA_TARGET_LABEL_SETS` | `labels`, `offsets` (`n + 1`) | `multilabel_hinge`, `multilabel_cross_entropy` |
  | `SDCA_TARGET_MASK` | `mask` (`n, num_classes` of `0/1`) | the same |
  | `SDCA_TARGET_PROBABILITIES` | `probabilities` (`n, num_classes`) | `cross_entropy` in `sdca_dual_update` and `sdca_model_fit` (distillation objective) |
  | `SDCA_TARGET_LOGITS` | `logits` (teacher, in the scores' dtype) | `distillation`, `proximal_distillation` |

  Label sets are sorted, duplicate-free, nonempty and leave a class out (the steps of the
  multilabel objectives need one).
- **`struct sdca_fit_options`:** `c`, `tol`, `max_epoch`, `eval_epoch` (`0`: evaluate only
  at the end), `eval_on_start`, `recompute_primal`, `shrinking` (hinge objectives only;
  the others take `SDCA_SHRINKING_AUTO`) and `seed` (the order of the examples).
- **`struct sdca_input`:** features `(n, d)` or a Gram matrix `(n, n)`
  (`SDCA_INPUT_KERNEL`), with its dtype.

Training takes every loss but the proximal and distillation ones, and `cross_entropy`
against target probabilities.

```c
#include "sdca/c_api.h"

if (sdca_abi_version() != SDCA_ABI_VERSION) { /* mismatched library */ }
struct sdca_loss loss = SDCA_LOSS_INIT;           /* cross_entropy */
struct sdca_fit_options options = SDCA_FIT_OPTIONS_INIT;
struct sdca_execution execution = SDCA_EXECUTION_INIT;
struct sdca_input input = SDCA_INPUT_INIT;        /* float64 features */
struct sdca_targets targets = SDCA_TARGETS_INIT;  /* labels */
struct sdca_fit_result result = SDCA_FIT_RESULT_INIT;
input.n = n, input.d = d, input.x = x;
targets.labels = labels;
size_t rows = sdca_history_rows(&options);        /* or history NULL, rows 0 */
if (sdca_model_fit(&loss, &options, &execution, &input, num_classes, &targets, NULL, NULL,
                   w, a, history, rows, &result) != SDCA_OK)
  fprintf(stderr, "%s\n", sdca_last_error());
```

See [examples/c/train.c](../examples/c/train.c) for a complete program.

## Conventions

- **Errors:** Functions return `SDCA_OK` (`0`), `SDCA_INVALID_ARGUMENT` or `SDCA_FAILURE`
  and set a thread-local message read by `sdca_last_error()` (`""` after a success). Every
  call validates all its arguments before writing any output, so `SDCA_INVALID_ARGUMENT`
  leaves the outputs as given; after `SDCA_FAILURE` (a failed computation or allocation)
  they are unspecified. A fit whose coordinate steps fail numerically returns `SDCA_OK`
  with `SDCA_STATUS_FAILED` and the last consistent iterate.
- **Status:** `SDCA_STATUS_SOLVED` (only it guarantees the duality gap), `_NO_PROGRESS`,
  `_MAX_EPOCH`, `_FAILED`.
- **Warm starts:** `initial_dual` (may be the `dual` output itself) must be feasible for the
  objective; the library checks it, gives single-label rows the label entry that makes them
  sum to zero, and builds `W = center + x' A` in `double`.
- **History:** `SDCA_HISTORY_FIELDS` columns named by `enum sdca_history_field`; a fit writes
  at most `sdca_history_rows(&options)` rows (`max_epoch / eval_epoch + 2`, `2` for
  `eval_epoch = 0`) and reports the rows written in `result.history_rows`.
- **Layouts:** features `(n, d)`, Gram matrices `(n, n)`, scores and duals
  `(n, num_classes)` are row-major; weights and regularizer centers `(d, num_classes)` are
  column-major.
- **Versioning:** each struct starts with `size`, set by its `SDCA_*_INIT`; check
  `sdca_abi_version() == SDCA_ABI_VERSION` once. Within an ABI version fields are only
  appended, and the library accepts every size of that version.

`src/c_api_internal.h` declares the `sdca_internal_*` entry points of the Python package's
PyTorch and JAX CPU paths (16-bit storage, saved row statistics, class weights and
`ignore_index`). They are not part of the public ABI and change without notice.

See [API.md](API.md) for the shared names, defaults and definitions.
