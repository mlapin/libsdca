# C interface

`include/sdca/c_api.h` declares a plain C ABI over the C++ headers. The Python
and Rust packages are built on it, and any language with a C FFI can use it.

## Building

```bash
cmake -S . -B build -DSDCA_BUILD_C_API=ON -DSDCA_BLAS_BACKEND=BUILTIN \
  -DCMAKE_INSTALL_PREFIX=$PWD/build/install
cmake --build build --target install
```

This installs `lib/libsdca_c.so` (macOS: `lib/libsdca_c.dylib`) and the headers. CMake consumers link the
exported target:

```cmake
find_package(libsdca CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE libsdca::sdca_c)
```

Without CMake, compile `src/c_api.cpp` with any C++11 compiler, e.g.
`c++ -std=c++11 -O2 -shared -fPIC -pthread -DBLAS_BUILTIN '-DSDCA_VERSION="0.3.0"' -Iinclude src/c_api.cpp -o libsdca_c.so`
(without `SDCA_VERSION`, `sdca_version()` returns `"unknown"`).

## Usage

[examples/c/train.c](../examples/c/train.c) trains a model and evaluates a
loss gradient:

```c
#include "sdca/c_api.h"

if (sdca_abi_version() != SDCA_ABI_VERSION) { /* stale library */ }
struct sdca_options options = SDCA_OPTIONS_DEFAULT;
struct sdca_fit_result result = SDCA_FIT_RESULT_INIT;  /* status, epochs, history rows */
options.c = 10.0;
options.max_epoch = 100;
if (sdca_fit(&options, n, d, num_classes, x, labels, /* offsets */ NULL, /* center */ NULL, w, a,
             history, history_rows, &result) != SDCA_OK)
  fprintf(stderr, "%s\n", sdca_last_error());
```

```bash
cmake -S examples/c -B build/c-example -DCMAKE_PREFIX_PATH=$PWD/build/install
cmake --build build/c-example && build/c-example/sdca_c_example
```

## Conventions

- Every computing function returns `SDCA_OK` (0), `SDCA_INVALID_ARGUMENT`
  for invalid inputs or options, or `SDCA_FAILURE` for a failed computation;
  `sdca_last_error()` then describes the failure (thread-local). Outputs are
  left unchanged when `sdca_loss_gradient`, `sdca_entropy_*` or
  `sdca_update_objective` fails, and a failed fit leaves consistent weights
  and duals. The [tensor entry points](#tensor-entry-points) state their own
  rule.
- Arrays are row-major: features `(n, d)`, Gram matrix `(n, n)`, scores and
  dual variables `(n, num_classes)`. Weights and the center are `(d, num_classes)`
  column-major.
- Options go in a `struct sdca_options`, started from `SDCA_OPTIONS_DEFAULT`
  (the defaults of every interface); each function reads the fields that
  name it in the header. `sdca_fit` writes its status, epochs and history
  rows to a `struct sdca_fit_result` started from `SDCA_FIT_RESULT_INIT`.
  Both structs begin with their `size`, set by these initializers: later
  libraries accept the sizes of earlier headers, so fields can be appended
  without breaking callers.
- Labels are zero-based `int64_t`. Multilabel targets are sorted label sets
  without duplicates, delimited by `n + 1` offsets; pass `NULL` offsets for
  single-label objectives. `sdca_loss_gradient` takes an `(n, num_classes)`
  0/1 mask; `options.loss` is an id of `enum sdca_loss` (`SDCA_CROSS_ENTROPY`,
  ..., `SDCA_PROXIMAL_CROSS_ENTROPY`), and `sdca_fit` and
  `sdca_update_objective` take the same ids except the last. `gamma >= 0` smooths the hinges (0: nonsmooth).
- `options.precision` selects storage and arithmetic: `SDCA_FLOAT32` (float data,
  float arithmetic), `SDCA_FLOAT32_FLOAT64` (float data, double arithmetic) or
  `SDCA_FLOAT64`. Data pointers are `float*` for the first two and `double*`
  for the last; loss values and history are always `double`.
- The history buffer holds `history_rows * sdca_history_fields()` doubles,
  with `history_rows >= max_epoch / eval_epoch + 2`. Its columns are listed
  in the header and in [API.md](API.md#results).
- Compare `sdca_abi_version()` with `SDCA_ABI_VERSION` before any other call.
  ABI 5 takes the option and result structs; development libraries reporting
  an earlier ABI must be rebuilt, even if their package version is `0.3.0`.
  `sdca_version()` returns the library version string (e.g. `"0.3.0"`).
- `options.projection_precision` is `SDCA_PROJECTION_FLOAT64` (double with
  a checked result, the default of every interface) or
  `SDCA_PROJECTION_EXTENDED` (always the extended-precision solver); it acts
  on the `multilabel_hinge` projections, and other values are invalid
  arguments ([API.md](API.md#projection-precision)).
- `options.num_threads` of `sdca_loss_gradient` and `sdca_update_objective`
  defaults to `SDCA_THREADS_DEFAULT` (`SDCA_NUM_THREADS`, else automatic) and
  0 is automatic. The results do not depend on it ([API.md](API.md#threads)).
  The library uses `std::thread`: link it with `-pthread` when building it by
  hand.

- The functions check that numeric inputs are finite, options are in range
  and buffer extents fit in `size_t`, and reject `NULL` required pointers,
  returning `SDCA_INVALID_ARGUMENT`. Buffer capacities and non-overlapping
  outputs remain the caller's responsibility. A coordinate step that fails
  during `sdca_fit` ends training with `SDCA_STATUS_FAILED` in `result.status`.

## Tensor entry points

The PyTorch CPU losses call these functions on the tensors' memory. They take
scores in any `enum sdca_dtype` (`SDCA_DTYPE_FLOAT32`, `_FLOAT64`, `_FLOAT16`,
`_BFLOAT16`; 16-bit values as their raw bits) and read `size`, `precision` (the
working arithmetic only) and `num_threads` of the options.

| Function | Computes |
| --- | --- |
| `sdca_cross_entropy_forward`, `sdca_cross_entropy_backward` | cross-entropy of integer labels as `torch.nn.functional.cross_entropy`, with class weights and `ignore_index` |
| `sdca_loss_gradient_rows` | `sdca_loss_gradient` of labels or a 0/1 mask |
| `sdca_distillation_forward`, `sdca_distillation_backward` | `distillation_loss` of student and teacher logits at a temperature |
| `sdca_proximal_distillation_rows` | `proximal_distillation_loss`: values and gradients |

The header states each function's arguments, outputs and checks. A failed
`sdca_cross_entropy_*` call writes nothing; a failed `sdca_loss_gradient_rows`
or distillation call may have written some rows. The losses are defined in
[API.md](API.md#distillation-losses).

The names, options and defaults are shared with every other interface; see
[API.md](API.md).
