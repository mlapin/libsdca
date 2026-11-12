# C interface

`include/sdca/c_api.h` declares the exception-safe C ABI used by the Python and Rust
bindings and available to any language with a C foreign function interface.

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
  '-DSDCA_VERSION="0.3.1"' -Iinclude src/c_api.cpp -o libsdca_c.so
```

## Usage

See [examples/c/train.c](../examples/c/train.c) for a complete program:

```c
#include "sdca/c_api.h"

if (sdca_abi_version() != SDCA_ABI_VERSION) { /* mismatched library */ }
struct sdca_options options = SDCA_OPTIONS_DEFAULT;
struct sdca_fit_result result = SDCA_FIT_RESULT_INIT;
options.c = 10.0;
options.max_epoch = 100;
if (sdca_fit(&options, n, d, num_classes, x, labels, NULL, NULL, w, a,
             history, history_rows, &result) != SDCA_OK)
  fprintf(stderr, "%s\n", sdca_last_error());
```

## Conventions

- **Status and errors:** Functions return `SDCA_OK` (`0`), `SDCA_INVALID_ARGUMENT`, or
  `SDCA_FAILURE`, setting a thread-local message read by `sdca_last_error()`. A failed
  coordinate step during `sdca_fit` stops cleanly with `SDCA_STATUS_FAILED` in
  `result.status` and leaves the last consistent iterate.
- **Layouts:** Features `(n, d)`, Gram matrices `(n, n)`, scores `(n, num_classes)` and
  duals `(n, num_classes)` are row-major; weights and regularizer centers `(d, num_classes)`
  are column-major.
- **Versioned structs:** Initialize `struct sdca_options` with `SDCA_OPTIONS_DEFAULT` and
  `struct sdca_fit_result` with `SDCA_FIT_RESULT_INIT` (which set `size = sizeof(...)`),
  and check `sdca_abi_version() == SDCA_ABI_VERSION`.
- **Targets:** Single-label targets are 0-based `int64_t` indices with `offsets = NULL`.
  Multilabel targets pass sorted, duplicate-free class indices delimited by `n + 1` offsets
  (or an `(n, num_classes)` `0/1` mask for `sdca_loss_gradient`).
- **Precision (`options.precision`):** `SDCA_FLOAT32` (`float` data and arithmetic),
  `SDCA_FLOAT32_FLOAT64` (`float` data, `double` arithmetic and iterate), or `SDCA_FLOAT64`
  (`double` data and arithmetic). Loss values and `history` (`history_rows >= max_epoch / eval_epoch + 2`,
  `sdca_history_fields()` columns) are always `double`.

## Tensor entry points

Used by the Python and JAX bindings to evaluate losses directly on tensor buffers across
`SDCA_DTYPE_FLOAT32`, `_FLOAT64`, `_FLOAT16`, and `_BFLOAT16` (16-bit values passed as raw
`uint16_t` bits):

| Function | Computes |
| --- | --- |
| `sdca_cross_entropy_forward`, `sdca_cross_entropy_backward` | Integer-target cross-entropy with class weights and `ignore_index` |
| `sdca_loss_gradient_rows` | Any `enum sdca_loss` on integer labels or a `0/1` mask |
| `sdca_distillation_forward`, `sdca_distillation_backward` | `distillation_loss` on student and teacher logits |
| `sdca_proximal_distillation_rows` | `proximal_distillation_loss` values and gradients |

See [API.md](API.md) for shared option defaults and definitions.
