# Dependencies and builds

`libsdca` needs only a C++11 compiler (the first-use PyTorch and JAX CPU kernels, C++17). An external
CBLAS is recommended: every build links a system one when it finds one, and otherwise uses
the built-in products and warns ([API.md](docs/API.md#blas)). CMake 3.12+ is needed only for
the CMake packages `libsdca::libsdca` and `libsdca::sdca_c`.

## Tested configurations

| Component | Environments |
| --- | --- |
| C++ and C ABI | Ubuntu 22.04 / 24.04 `x86_64` with GCC 11 and 13 and Clang 18 (built-in and external BLAS, ASan/UBSan, TSan); macOS `arm64` with Apple Clang (built-in BLAS and Accelerate) |
| Python | Python 3.10–3.14, NumPy 1.26–2.5, SciPy 1.15–1.18, scikit-learn 1.7–1.9, PyTorch 2.1–2.15, JAX 0.4.35–0.12 |
| CUDA | NVIDIA A100 with PyTorch 2.14–2.15, Triton 3.8–3.9 and JAX 0.12 (hosted CI has no GPU) |
| Packaging | `py3-none` wheels via `cibuildwheel` on `manylinux_2_28` `x86_64` (OpenBLAS of `scipy-openblas32`) and macOS 11+ `arm64` (Accelerate), sdist, `.crate` |
| Rust | Rust 1.74 (MSRV) and stable on Linux `x86_64` (built-in BLAS, reference BLAS, OpenBLAS, BLIS) |

## Python

| Extra | Packages | Purpose |
| --- | --- | --- |
| *(default)* | `numpy>=1.23`; `scipy-openblas32` (Linux wheels) | NumPy solver and losses; the BLAS of the Linux wheels |
| `sklearn` | `scikit-learn>=1.6` | `SDCAClassifier` |
| `torch` | `torch>=2.1`, `setuptools>=77,<81`, `ziglang>=0.13,<0.18` (Linux) | PyTorch losses |
| `jax` | `jax>=0.4.35` | JAX losses (`libsdca.jax`) |
| `jax-cuda` | `jax[cuda13]>=0.7.2`; `triton>=3.7`, `jax-triton>=0.5` (Linux, Python 3.12+) | JAX losses on CUDA |
| `test`, `test-torch`, `test-jax` | `pytest`, `pytest-xdist`, `scipy`, `mpmath`; `hypothesis>=6.136` (`test-torch`, `test-jax`), ... | Test suites (`--group core\|torch\|jax\|all`); property tests (`test_properties_*.py`) |

- **Versions:** PyTorch 2.1 wheels need `numpy<2`. The fused CUDA kernels need `triton>=3.3`,
  which PyTorch 2.7 and later install; without it CUDA calls raise `NotImplementedError`
  naming `backend="composed"`. `libsdca.torch.explain_backend` reports the active path.
- **First-use builds:** The PyTorch and JAX CPU kernels compile on first use against the
  installed framework (a C++17 compiler and Python headers; JAX against
  `jax.ffi.include_dir()`). Without build tools, CPU calls with `backend="auto"` raise
  `NotImplementedError` naming `backend="composed"`.
- **Licences:** `libsdca` is MIT. The Linux wheels require, and load at run time, the
  `scipy-openblas32` package (BSD-2-Clause), which ships OpenBLAS and LAPACK (BSD-3-Clause)
  and the GCC runtime libraries `libgfortran` and `libquadmath` (GPL-3.0 with the GCC Runtime
  Library Exception); its wheel carries their texts. The `libsdca` wheels include none of them.

```bash
python -m pip install .                                      # a system CBLAS, else built-in (warns)
SDCA_BLAS_BACKEND=BUILTIN python -m pip install .            # no external BLAS
SDCA_BLAS_BACKEND=EXTERNAL python -m pip install .           # require an external CBLAS
SDCA_BLAS_BACKEND=PACKAGE python -m pip install --no-build-isolation .  # scipy-openblas32 (installed)
SDCA_BLAS="-L/opt/blas/lib -lopenblas" python -m pip install .   # explicit linker flags
SDCA_BUILD_NATIVE=OFF python -m pip install '.[torch]'       # tensor losses only, no compiler
```

## C++ and CMake

| CMake option | Default | Description |
| --- | --- | --- |
| `SDCA_BLAS_BACKEND` | `AUTO` | `AUTO` (external CBLAS if found, else built-in with a warning), `BUILTIN`, or `EXTERNAL` ([API.md](docs/API.md#blas)) |
| `SDCA_BLAS` | `$ENV{SDCA_BLAS}` | Linker flags of an external CBLAS, in place of the candidates of `src/native_build.txt` |
| `SDCA_BUILD_C_API` | `OFF` | Build and install the C ABI library `libsdca::sdca_c` |
| `BUILD_TESTING` | `OFF` | Build the bundled GoogleTest suites |
| `SDCA_WARNINGS_AS_ERRORS` | `OFF` | Add `-Werror` |
| `USE_CHRONO` | `ON` | `<chrono>` wall clock (`OFF`: POSIX `gettimeofday`) |

```bash
cmake -S . -B build -DSDCA_BLAS_BACKEND=BUILTIN -DCMAKE_INSTALL_PREFIX="$PWD/build/install"
cmake --build build --target install
```

```cmake
find_package(libsdca CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE libsdca::libsdca)
```

Without CMake, define `BLAS_DEFAULT_LOCAL_HEADER` and link a CBLAS (such as `-lopenblas` or
`-framework Accelerate`), or define `BLAS_BUILTIN` for the built-in products; with neither, the
headers take the built-in products and the solver logs a warning once. The headers need no
separate source file. Every build reads its BLAS candidates and compile flags from
`src/native_build.txt` ([API.md](docs/API.md#blas)); Intel MKL links through `SDCA_BLAS`
(for example `-lmkl_rt`, its 32-bit integer interface).

## Rust

The crate has no Rust dependencies and needs Rust 1.74+. `build.rs` compiles `src/c_api.cpp`
with `CXX` (default `c++`) and the BLAS of the shared rule: a system CBLAS if one links, else
the built-in products with a Cargo warning ([docs/RUST.md](docs/RUST.md#building)).
