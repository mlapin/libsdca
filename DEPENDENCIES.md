# Dependencies and builds

`libsdca` needs only a C++11 compiler. An external CBLAS is optional (a built-in backend is
included), and CMake 3.12+ is needed only for the CMake packages `libsdca::libsdca` and
`libsdca::sdca_c`.

## Tested configurations

| Component | Environments |
| --- | --- |
| C++ and C ABI | Ubuntu 22.04 / 24.04 `x86_64` with GCC 11 and 13 and Clang 18 (built-in and external BLAS, ASan/UBSan, TSan); macOS `arm64` with Apple Clang (built-in BLAS and Accelerate) |
| Python | Python 3.10–3.13, NumPy 1.26–2.5, SciPy 1.15–1.18, scikit-learn 1.7–1.9, PyTorch 2.1–2.15, JAX 0.4.35–0.12 |
| CUDA | NVIDIA A100 with PyTorch 2.14–2.15, Triton 3.8–3.9 and JAX 0.12 (hosted CI has no GPU) |
| Packaging | `py3-none` wheels via `cibuildwheel` on `manylinux_2_28` `x86_64` and macOS 11+ `arm64`, sdist, `.crate` |
| Rust | Rust 1.74 (MSRV) and stable on Linux `x86_64` (built-in BLAS, reference BLAS, OpenBLAS, BLIS) |
| Octave | GNU Octave 10 (`mkoctfile`) |

## Python

| Extra | Packages | Purpose |
| --- | --- | --- |
| *(default)* | `numpy>=1.23` | NumPy solver and losses |
| `sklearn` | `scikit-learn>=1.6` | `SDCAClassifier` |
| `torch` | `torch>=2.1`, `ninja`, `setuptools>=77,<81`, `ziglang>=0.13,<0.18` (Linux) | PyTorch losses and their C++ extensions |
| `jax` | `jax>=0.4.35` | JAX losses (`libsdca.jax`) |
| `jax-cuda` | `jax[cuda13]>=0.7.2`, `triton>=3.3` (Linux) | JAX losses on CUDA |
| `test`, `test-torch`, `test-jax` | `pytest`, `pytest-xdist`, `scipy`, `mpmath`, ... | Test suites (`--group core\|torch\|jax\|all`) |

- **Versions:** PyTorch 2.1 wheels need `numpy<2`. The fused CUDA kernels need `triton>=3.3`,
  which PyTorch 2.7 and later install (older versions use composed PyTorch operations); the C++
  launcher supports Triton 3.3–3.9. `libsdca.torch.explain_backend` reports the active path.
- **Runtime extensions:** The PyTorch CPU extension and CUDA launcher compile on first use
  with `torch.utils.cpp_extension` (a C++17 compiler with OpenMP, Python headers, and CUDA
  headers for the launcher). JAX compiles a small XLA FFI handler against
  `jax.ffi.include_dir()`. Without build tools both fall back with a one-time warning.

```bash
python -m pip install .                                      # system CBLAS if found, else built-in
SDCA_BLAS_BACKEND=BUILTIN python -m pip install .            # no external BLAS
SDCA_BLAS_BACKEND=EXTERNAL python -m pip install .           # require an external CBLAS
SDCA_BLAS="-L/opt/blas/lib -lopenblas" python -m pip install .   # explicit linker flags
SDCA_BUILD_NATIVE=OFF python -m pip install '.[torch]'       # tensor losses only, no compiler
```

## C++ and CMake

| CMake option | Default | Description |
| --- | --- | --- |
| `SDCA_BLAS_BACKEND` | `AUTO` | `AUTO` (external CBLAS if found, else built-in), `BUILTIN`, or `EXTERNAL` |
| `SDCA_BUILD_C_API` | `OFF` | Build and install the C ABI library `libsdca::sdca_c` |
| `SDCA_BUILD_MATLAB` | `OFF` | Build the MATLAB MEX gateways |
| `BUILD_TESTING` | `OFF` | Build the bundled GoogleTest suites |
| `SDCA_WARNINGS_AS_ERRORS` | `OFF` | Add `-Werror` |
| `USE_CHRONO` | `ON` | `<chrono>` wall clock (`OFF`: POSIX `gettimeofday`) |
| `USE_SEQUENTIAL`, `USE_ILP64`, `USE_THREADING_SUFFIX` | `ON`, `ON`, `OFF` | Intel MKL threading layer, 64-bit integers, MEX suffix |

```bash
cmake -S . -B build -DSDCA_BLAS_BACKEND=BUILTIN -DCMAKE_INSTALL_PREFIX="$PWD/build/install"
cmake --build build --target install
```

```cmake
find_package(libsdca CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE libsdca::libsdca)
```

Without CMake, define one of `BLAS_BUILTIN` (default), `BLAS_DEFAULT`,
`BLAS_DEFAULT_LOCAL_HEADER`, `BLAS_MKL`, `BLAS_ACCELERATE` or `BLAS_MATLAB`, link the matching
CBLAS, and compile `sdca/utility/logging.cpp` in exactly one translation unit.

## Rust

The crate has no Rust dependencies and needs Rust 1.74+. `build.rs` compiles `src/c_api.cpp`
with `CXX` (default `c++`) and the built-in BLAS, or links an external CBLAS given by
`SDCA_BLAS` ([rust/README.md](rust/README.md#building)).
