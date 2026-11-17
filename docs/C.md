# C interface

[All guides](https://github.com/mlapin/libsdca/blob/master/docs/README.md) ·
[Shared API reference](https://github.com/mlapin/libsdca/blob/master/docs/API.md)

`include/sdca/c_api.h` declares the exception-safe C ABI used by the Python and Rust
bindings and available to any language with a C foreign function interface. Its comments
are the reference; this page summarizes them.

## Installation

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

## Quick start

Build and run the complete [training example](../examples/c/train.c) from the repository root:

```bash
cmake -S examples/c -B build/c-consumer -DCMAKE_PREFIX_PATH="$PWD/build/install"
cmake --build build/c-consumer
LD_LIBRARY_PATH="$PWD/build/install/lib" build/c-consumer/sdca_c_example
```

On macOS use `DYLD_LIBRARY_PATH` in place of `LD_LIBRARY_PATH`.

## Entry points

| Function | Does |
| --- | --- |
| `sdca_loss_evaluate` | Loss values and (optionally) gradients of any loss at a batch of scores |
| `sdca_dual_update` | One coordinate step of a training objective per row (new dual blocks) |
| `sdca_model_fit` | Trains a linear or kernel model; returns the weights, the duals, a status and a history |
| `sdca_model_scores` | Scores of a batch of examples under a fitted model: one BLAS matrix product |
| `sdca_history_rows` | The history rows a fit with these options may write |
| `sdca_max_extent` | The largest size a fit takes (`n`, `d`, `num_classes`, weight and dual counts) |
| `sdca_abi_version`, `sdca_version`, `sdca_last_error` | The ABI, the library version, the last failure |

## Arguments

Every call separates what is computed from how:

### Loss

`struct sdca_loss` holds the loss (`enum sdca_loss_kind`, one per shared name of
[API.md](API.md#loss-definitions)) and its own parameters:

| Field | Losses | `SDCA_LOSS_INIT` default |
| --- | --- | --- |
| `k` | Top-*k* losses, required | `0` |
| `gamma` | Hinges; `0` is nonsmooth | `0` |
| `alpha` | Proximal losses | `1` |
| `temperature` | Distillation losses | `1` |
| `projection_precision` | `multilabel_hinge` | `SDCA_PROJECTION_FLOAT64` |

Every other field keeps its default from `SDCA_LOSS_INIT`; a parameter the loss does not
take is `SDCA_INVALID_ARGUMENT`.

### Precision and threads

`struct sdca_execution` selects the working precision and the threads:

- **Precision:** `SDCA_PRECISION_DEFAULT` works in the storage type, `_FLOAT32` needs
  `float` storage, `_FLOAT64` works in `double`.
- **`num_threads`:** `0` uses `SDCA_NUM_THREADS`, else the available CPUs; `n` uses up to
  `n`.

Storage types are `SDCA_DTYPE_FLOAT32` and `SDCA_DTYPE_FLOAT64`.

### Targets

`struct sdca_targets` holds a kind and the fields of that kind (the others `NULL`):

| Kind | Fields | Losses |
| --- | --- | --- |
| `SDCA_TARGET_LABELS` | `labels` (`n`, 0-based) | single-label losses, `proximal_cross_entropy` |
| `SDCA_TARGET_LABEL_SETS` | `labels`, `offsets` (`n + 1`) | `multilabel_hinge`, `multilabel_cross_entropy` |
| `SDCA_TARGET_MASK` | `mask` (`n, num_classes` of `0/1`) | the same |
| `SDCA_TARGET_PROBABILITIES` | `probabilities` (`n, num_classes`) | `cross_entropy` in `sdca_dual_update` and `sdca_model_fit` (distillation objective) |
| `SDCA_TARGET_LOGITS` | `logits` (teacher, in the scores' dtype) | `distillation`, `proximal_distillation` |

Label sets are sorted, duplicate-free, nonempty and leave a class out (the steps of the
multilabel objectives need one).

### Training options

Training takes every loss but the proximal and distillation ones, and `cross_entropy`
against target probabilities.

`struct sdca_fit_options` holds `c`, `tol`, `max_epoch`, `eval_epoch`, `eval_on_start`,
`recompute_primal`, `shrinking` and `seed`:

- **`eval_epoch`:** `0` evaluates only at the end.
- **`shrinking`:** Hinge objectives only; the others take `SDCA_SHRINKING_AUTO`.
- **`seed`:** The order of the examples.

### Input

`struct sdca_input` describes features `(n, d)` or a Gram matrix `(n, n)`
(`SDCA_INPUT_KERNEL`), with its dtype.

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
  and set a thread-local message read by `sdca_last_error()` (`""` after a success).
  - Every call validates all its arguments before writing any output, so
    `SDCA_INVALID_ARGUMENT` leaves the outputs as given.
  - After `SDCA_FAILURE` (a failed computation or allocation) the outputs are unspecified.
  - A fit whose coordinate steps fail numerically returns `SDCA_OK` with
    `SDCA_STATUS_FAILED` and the last consistent iterate.
- **Status:** `SDCA_STATUS_SOLVED` (only it guarantees the duality gap), `_NO_PROGRESS`,
  `_MAX_EPOCH`, `_FAILED`.
- **Warm starts:** `initial_dual` (may be the `dual` output itself) must be feasible for the
  objective; the library checks it, gives single-label rows the label entry that makes them
  sum to zero, and builds `weights = center + A' x` in `double`.
- **History:** `SDCA_HISTORY_FIELDS` columns named by `enum sdca_history_field`; a fit writes
  at most `sdca_history_rows(&options)` rows (`max_epoch / eval_epoch + 2`, `2` for
  `eval_epoch = 0`) and reports the rows written in `result.history_rows`.
- **Layouts:** every array is row-major: features `(n, d)`, Gram matrices `(n, n)`, scores
  and duals `(n, num_classes)`, weights and regularizer centers `(num_classes, d)` (one
  weight vector per class). The library uses the caller's buffers in place
  ([API.md](API.md#array-layouts)).
- **Scores:** `sdca_model_scores(&input, num_classes, weights, scores)` writes
  `x weights'` for features; with `input.kind = SDCA_INPUT_KERNEL`, `x` is the kernel
  `(n, d)` to the `d` training examples and the coefficients are the duals.
- **Versioning:** each struct starts with `size`, set by its `SDCA_*_INIT`; check
  `sdca_abi_version() == SDCA_ABI_VERSION` once. Within an ABI version fields are only
  appended, and the library accepts every size of that version.

`src/c_api_internal.h` declares the `sdca_internal_*` entry points of the Python package's
PyTorch and JAX CPU paths (16-bit storage, saved row statistics, class weights and
`ignore_index`). They are not part of the public ABI and change without notice.

See [API.md](API.md) for the shared names, defaults and definitions.

## Benchmarks

The [solver benchmarks](https://github.com/mlapin/libsdca/blob/master/docs/PYTHON.md#benchmarks)
compare the shared solver through Python with scikit-learn. They help choose between
feature and kernel inputs; they do not measure this binding's call overhead.
No separate end-to-end timings for this interface are published here.
