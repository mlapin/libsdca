# Dependencies and optional builds

`libsdca` requires a C++11 compiler. An external BLAS library is optional: a built-in
backend lets the solver and loss functions run with no third-party runtime dependencies.
CMake 3.12+ is required only for the CMake packages (`libsdca::libsdca` and
`libsdca::sdca_c`).

## Tested configurations

| Component | Validated environments |
| --- | --- |
| C++ & C ABI | Ubuntu 22.04 / 24.04 (`x86_64`) with GCC 11.4 / 13 and Clang 18 (`BUILTIN` and `EXTERNAL` BLAS, C and C++ consumers, GCC ASan/UBSan, Clang TSan); macOS (`arm64`) with Apple Clang 21 (`BUILTIN` and Accelerate) |
| Python (CPU) | Python 3.10–3.14, NumPy 1.23–2.5, SciPy 1.10–1.18, scikit-learn 1.6–1.9, PyTorch 2.1–2.15 (with CPU extension syntax and runtime checks across AVX-512, AVX2, and generic/arm64), JAX 0.4.35–0.12 (with XLA FFI handlers on and off) |
| CUDA | NVIDIA A100-SXM4-80GB and H100 (PyTorch 2.14–2.15, Triton 3.8–3.9, both C++ launcher and Triton launch paths; JAX 0.12 with CUDA plugin) |
| Packaging | Portable `py3-none` wheels (`SDCA_BLAS_BACKEND=BUILTIN`) via `cibuildwheel` on `manylinux_2_28` `x86_64` and macOS 11+ `arm64`, sdist builds, and standalone `.crate` archives |
| Rust | Rust 1.74 (MSRV) and stable on Linux `x86_64`, linked against `BUILTIN`, reference BLAS, OpenBLAS and BLIS (shared and static) |
| Octave | GNU Octave 10.3 (`mkoctfile` MEX gateways) |

- **PyTorch & JAX version floors:** Requires `python >= 3.10`, `numpy >= 1.23`,
  `torch >= 2.1` (use `numpy < 2` with PyTorch 2.1 wheels), `jax >= 0.4.35`, and
  `scikit-learn >= 1.6`.
- **CUDA fused kernels:** Require `torch >= 2.7` and `triton >= 3.3` (older versions fall
  back to composed PyTorch tensor ops). The C++ `cuLaunchKernel` launcher supports Triton
  3.3–3.9 (`SDCA_CUDA_LAUNCHER` overrides selection; `libsdca.torch.explain_backend`
  reports the active backend).

## C++ configuration

| CMake option | Default | Description |
| --- | --- | --- |
| `SDCA_BLAS_BACKEND` | `AUTO` | `AUTO` (prefer external CBLAS, else built-in), `BUILTIN` (no external BLAS/OpenMP), or `EXTERNAL` (require external CBLAS) |
| `SDCA_BUILD_C_API` | `OFF` | Build and install the C ABI library `libsdca::sdca_c` (links `Threads`) |
| `SDCA_WARNINGS_AS_ERRORS` | `OFF` | Treat compiler warnings as errors (`-Werror`) |
| `BUILD_TESTING` | `OFF` | Build bundled GoogleTest and C++ test suites |
| `SDCA_BUILD_MATLAB` | `OFF` | Build MATLAB MEX gateways |
| `USE_CHRONO` | `ON` | Use `<chrono>` wall clock (`OFF` selects POSIX `gettimeofday`) |
| `USE_SEQUENTIAL`, `USE_ILP64`, `USE_THREADING_SUFFIX` | `ON`, `ON`, `OFF` | Intel MKL threading layer, 64-bit integer ABI, and MEX binary suffix |

Minimal installation and CMake usage:

```bash
cmake -S . -B build/minimal -DSDCA_BLAS_BACKEND=BUILTIN -DCMAKE_INSTALL_PREFIX=/your/prefix
cmake --build build/minimal --target install
```

```cmake
find_package(libsdca CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE libsdca::libsdca)
```

Include `sdca/utility/logging.cpp` in exactly one translation unit. Direct header users
without CMake default to `BLAS_BUILTIN` and `<chrono>` (or define `BLAS_DEFAULT`,
`BLAS_DEFAULT_LOCAL_HEADER`, `BLAS_MKL`, `BLAS_ACCELERATE`, or `BLAS_MATLAB` and link the
corresponding CBLAS library).

## Python configuration

The PyTorch CPU extension and CUDA launcher compile on first use via
`torch.utils.cpp_extension` (requiring `ninja`, a C++17/C++20 compiler with OpenMP, Python
headers, and CUDA headers for the launcher); native JAX CPU losses likewise compile a small
XLA FFI handler against `jax.ffi.include_dir()` ([environment variables](docs/API.md#environment-variables)).
Without build tools, both fall back automatically with a one-time warning.

```bash
# Default source install (uses system CBLAS if found, else built-in):
python -m pip install .

# Force built-in BLAS (no external BLAS dependency):
SDCA_BLAS_BACKEND=BUILTIN python -m pip install .

# Require external BLAS:
SDCA_BLAS_BACKEND=EXTERNAL python -m pip install .

# Pure-Python wheel for PyTorch/JAX tensor losses only (no C++ compiler needed):
SDCA_BUILD_NATIVE=OFF python -m pip install '.[torch]'
```

| Optional extra | Packages | Purpose |
| --- | --- | --- |
| *(default)* | `numpy>=1.23` | NumPy solver and loss interface |
| `torch` | `torch>=2.1`, `ninja` | PyTorch loss layers and JIT CPU/CUDA extension builds |
| `jax` | `jax>=0.4.35` | JAX loss functions (`libsdca.jax`) |
| `sklearn` | `scikit-learn>=1.6` | `SDCAClassifier` estimator |
| `test`, `test-torch`, `test-jax` | `pytest>=9`, `pytest-xdist>=3.6`, `scipy`, `mpmath`, ... | Test suites (`--group core`, `torch`, `jax`, `all`) |
| `coverage` | `coverage[toml]>=7` | Python line-coverage instrumentation |

## Rust configuration

The `libsdca` crate has zero Rust dependencies and requires Rust 1.74+. Its `build.rs`
compiles `src/c_api.cpp` with `CXX` (default `c++`) and the built-in BLAS, or links an
external CBLAS via `SDCA_BLAS` ([rust/README.md](rust/README.md#building)).
