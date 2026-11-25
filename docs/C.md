# libsdca for C

[All guides](https://github.com/mlapin/libsdca/blob/master/docs/README.md) ·
[Shared API reference](https://github.com/mlapin/libsdca/blob/master/docs/API.md)

`include/sdca/c_api.h` declares the exception-safe C ABI of `libsdca_c`, which the Python and
Rust bindings use and any language with a C foreign function interface can call. The
header's comments document every field; this page summarizes them.

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

Without CMake, compile `src/c_api.cpp` with the flags of `src/native_build.txt` (every
instruction set of the CPU is in the one translation unit, chosen at run time); add
`-DBLAS_DEFAULT_LOCAL_HEADER` and a CBLAS (such as `-lopenblas`) in place of `-DBLAS_BUILTIN`
for an external BLAS:

```bash
c++ $(sed -n 's/^flags //p' src/native_build.txt) -shared -fPIC -DBLAS_BUILTIN \
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
| `sdca_blas_backend` | The BLAS of the library: `"external"` (or `"external: <library>"`), `"builtin"` or `"fallback"` ([API.md](API.md#blas)) |
| `sdca_history_rows` | The history rows a fit with these options may write |
| `sdca_max_extent` | The largest size a fit takes (`n`, `d`, `num_classes`, weight and dual counts) |
| `sdca_abi_version`, `sdca_version`, `sdca_last_error` | The ABI, the library version, the last failure |

The PyTorch and JAX bindings use three more entry points as their CPU route when their own
extensions are not built. They are part of the ABI, but most programs need only the ones
above:

| Function | Does |
| --- | --- |
| `sdca_loss_rows` | `sdca_loss_evaluate` with the options of the PyTorch and JAX losses (`struct sdca_row_options`: probability storage dtype and gradient factors) and 16-bit storage |
| `sdca_reduce_rows` | Writes double row values to float32 or float64 output. Reduction 0 copies each value, 1 sums them, and 2 divides their sum by the sum of the optional denominator terms (row count when `terms == NULL`). |
| `sdca_reduce_row_groups` | Applies reduction 1 or 2 to each consecutive group of `rows_per_group` values and writes one result per group. |

## Arguments

Every call separates what is computed from how:

### Loss

`struct sdca_loss` holds the loss (`enum sdca_loss_kind`, one per shared name of
[API.md](API.md#loss-definitions), and `SDCA_LOSS_TOPK_HINGE_ALPHA` and
`SDCA_LOSS_TOPK_HINGE_BETA` for the two variants of `topk_hinge`) and its own parameters.
`SDCA_LOSS_INIT(kind)` gives the defaults of a kind, as a constant expression (C89
initializers too):

| Field | Losses | `SDCA_LOSS_INIT(kind)` default |
| --- | --- | --- |
| `k` | Top-*k* losses, required | `0` |
| `gamma` | Hinges: the smoothing (`0`: nonsmooth); proximal losses: the step, `> 0` | `0`; `1` for the proximal losses |
| `temperature` | Distillation losses | `1` |
| `weight` | Single-label losses: class weights `(num_classes)`, finite and `>= 0` | `NULL` (every weight `1`) |
| `label_smoothing` | `cross_entropy`: $\epsilon \in [0, 1]$ | `0` |
| `ignore`, `ignore_index` | Single-label losses: `ignore = 1` gives the rows of label `ignore_index` the loss `0` and a gradient of `+0.0` | `0`, `-100` |

Every other field keeps its default; a parameter the loss does not take is
`SDCA_INVALID_ARGUMENT` ("multilabel_hinge takes no parameter weight"). `weight`,
`label_smoothing` and `ignore_index` are the options of `sdca_loss_evaluate`, with the
definitions of PyTorch's `cross_entropy`
([API.md](API.md#loss-evaluation-options)); `sdca_dual_update` and `sdca_model_fit` take
their defaults. A label out of range is an error unless it is the ignored one.

### Precision and threads

`struct sdca_execution` selects the working precision and the threads:

- **Precision:** `SDCA_PRECISION_DEFAULT` works in the storage type, `_FLOAT32` needs
  `float` storage, `_FLOAT64` works in `double`.
- **`num_threads`:** `0` uses `SDCA_NUM_THREADS`, else the available CPUs; `n` uses up to
  `n`.
- **`check_rows`** (`sdca_loss_evaluate`): `0`, the default, gives an invalid row `NaN`
  outputs and returns `SDCA_INVALID_ROWS`; `1` checks every row before writing any output
  and returns `SDCA_INVALID_ARGUMENT` for an invalid row, leaving the outputs as given, at
  the cost of one more pass over the scores.
- **`report`** (optional): a `struct sdca_loss_report` that receives the number of invalid
  rows and the first of them.

Storage types are `SDCA_DTYPE_FLOAT32` and `SDCA_DTYPE_FLOAT64`.

### Targets

`struct sdca_targets` holds a kind and the fields of that kind (the others `NULL`):

| Kind | Fields | Losses |
| --- | --- | --- |
| `SDCA_TARGET_LABELS` | `labels` (`n`, 0-based) | single-label losses, `proximal_cross_entropy` |
| `SDCA_TARGET_LABEL_SETS` | `labels`, `offsets` (`n + 1`) | `multilabel_hinge`, `multilabel_cross_entropy` |
| `SDCA_TARGET_MASK` | `mask` (`n, num_classes` of `0/1`) | the same |
| `SDCA_TARGET_PROBABILITIES` | `probabilities` (`n, num_classes`, double) | `cross_entropy`: as given in `sdca_loss_evaluate` (as PyTorch's `cross_entropy`); finite, nonnegative rows that sum to one within the rounding of a row normalized in the storage dtype ($\gamma_{m+1}$, [API.md](API.md#targets)) in `sdca_dual_update` and `sdca_model_fit` (distillation objective), which renormalize a copy |
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
struct sdca_loss loss = SDCA_LOSS_INIT(SDCA_LOSS_CROSS_ENTROPY);
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
  (errors are negative; `sdca_loss_evaluate` also returns `SDCA_INVALID_ROWS`, `1`) and set
  a thread-local message read by `sdca_last_error()` (`""` after `SDCA_OK`).
  - Every call validates all its arguments before writing any output, so
    `SDCA_INVALID_ARGUMENT` leaves the outputs as given.
  - `sdca_loss_evaluate` takes scores of `-inf` as
    [masked classes](API.md#masked-classes) for every loss. A label out of range is
    `SDCA_INVALID_ARGUMENT` (`Target 7 is out of bounds.`). A row it cannot compute (`NaN`
    or `+inf` scores, an invalid row of masked classes) gets a `NaN`
    value and `NaN` gradients; the call writes every output, the valid rows hold their
    results, and it returns `SDCA_INVALID_ROWS` with the count and the first invalid row in
    `sdca_last_error()` (and in `execution.report` if given). With
    `execution.check_rows = 1` an invalid row is `SDCA_INVALID_ARGUMENT` before any output
    is written.
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

See [API.md](API.md) for the shared names, defaults and definitions.
