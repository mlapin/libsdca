# C++ quick start and contracts

A C++11 compiler and CMake 3.12+ are sufficient with `BUILTIN`. `AUTO` uses
external CBLAS when available and otherwise falls back to built-in routines.
Choose `EXTERNAL` for a tuned BLAS deployment. See [dependencies](../DEPENDENCIES.md).
The commands below use CMake 3.20+ conveniences; the project configuration
itself retains its 3.12 minimum. CI builds and tests the C++ library on Linux and
macOS arm64; the installed-consumer tests run on Linux.

```bash
cmake -S . -B build/release -DSDCA_BLAS_BACKEND=BUILTIN \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/install"
cmake --build build/release --parallel
cmake --install build/release
cmake -S examples/cpp -B build/consumer \
  -DCMAKE_PREFIX_PATH="$PWD/build/install"
cmake --build build/consumer --parallel
build/consumer/sdca_example
```

The [example](../examples/cpp/train.cpp) trains a three-class model and evaluates a
loss gradient. Its independent CMake project uses:

```cmake
find_package(libsdca CONFIG REQUIRED)
target_link_libraries(sdca_example PRIVATE libsdca::libsdca)
```

Linking the target carries include paths, C++11, backend definitions/libraries,
and strict numerical compiler options. The logger is the one implementation
file: include `sdca/utility/logging.cpp` in exactly one translation unit, as in
the example, or compile it separately. Do not include it in every source file.

## Training

Construct a `make_input_feature(features, examples, data)` or
`make_input_kernel(examples, gram)` view, output labels, and an objective such
as `l2_entropy<double, double>(C)`. Pass them and caller-owned dual/weight
buffers to `make_context`, set `context.criteria`, then call
`make_solver(context).solve()`. Buffers must have the sizes and layouts below,
remain alive until solving ends, and not alias one another. Feature inputs and kernel
matrices must be finite; a kernel must be positive semidefinite. C++ low-level APIs do
not validate every element or pointer.

Features are column-major `(features, examples)`, so each example's feature vector is
contiguous; NumPy's row-major `(examples, features)` array has the same byte order.
Weights are column-major `(features, classes)`. The duals store each example's class
block contiguously. The logger implementation (`sdca/utility/logging.cpp`) must be
compiled into exactly one translation unit per executable or library.

The default regularizer is `0.5*||W||²` and losses are summed with multiplier
`C`, not averaged. There is no implicit intercept. Objective template `Data`
controls storage; `Result` controls result arithmetic and, with the wider of the
two (`context_type::variable_type`), the solver's iterate: `<float, double>` keeps
`W` and the duals in double copies (`context.primal_variables`,
`context.dual_variables`) of the caller's float buffers (`context.primal_buffer`,
`context.dual_buffer`), loads them from the buffers when a solve starts, and at
each evaluation stores them there rounded and evaluates the stored values; read
the results from your buffers ([API.md](API.md#precision)). Positive class labels
are zero-based. A solver context retains references to buffers, evaluation
history, status, and stopping criteria. Check `context.status` and the final
gap; an epoch limit is not a convergence certificate.

## Losses and projections

`sdca::loss_and_gradient(kind, n, p, scores, gradient, k, gamma, alpha)` returns
a double loss and writes `n` gradient entries in the input storage type.
Scores and gradients are separate contiguous buffers. The first `p` scores
are positive labels; arrange classes before the call and restore their order
afterward. Entropy uses a uniform target over those labels. Single-target
top-k objectives require `p=1`. All score entries must be finite; `k` is in
`[1,n-1]`, `gamma` is finite and nonnegative (0: nonsmooth) and `alpha` finite
and positive.

Hinge gradients are deterministic selected subgradients at ties.
`proximal_cross_entropy` is a Moreau envelope, not ordinary cross-entropy.
Invalid dimensions or parameters throw `std::invalid_argument`; failed numerical solves can
throw `std::runtime_error` or `std::overflow_error`. No general strong exception
guarantee is provided for caller-owned output buffers.

Proximal operators in `sdca/prox.h` work on mutable iterator ranges. They may
partition/reorder scratch values while computing thresholds; consult the
individual operator's signature and tests before reusing buffers. Internal
`detail` functions are implementation details. Keep fast math disabled for
the documented stability guarantees.
