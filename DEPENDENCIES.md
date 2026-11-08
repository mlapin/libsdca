# Dependencies and optional builds

libsdca needs a C++11 compiler. A BLAS library is optional: a built-in
standard-library backend lets the full solver run without a third-party runtime
library. CMake 3.12 or newer is needed for the CMake package; direct header use
does not require CMake.

## Tested configurations

0.3.0 is validated in full on Linux x86-64; macOS arm64 is validated for the
C++ tests and the core and PyTorch CPU suites (row below). 0.3.0 was validated
on Ubuntu 22.04.5 with GCC 11.4 and Clang 18.1.8, using both
built-in and external BLAS. The Clang environment uses the conda-forge GCC 13
C++ runtime. These runs reproduce the checks in the GitHub workflows; they do not establish
compatibility with every toolchain on GitHub's Ubuntu 24.04 images.

| Component | Validated versions and configurations |
| --- | --- |
| C++ | GCC 11.4 and Clang 18.1.8, both BLAS backends, installed C and C++ consumers; GCC ASan/UBSan, Clang TSan |
| Python | 3.10.12 and 3.12.7 |
| CPU, Python 3.10 | NumPy 2.2.6, PyTorch 2.14.1+cpu, JAX 0.6.2; scikit-learn and SciPy are also exercised |
| CPU, Python 3.12 | NumPy 2.5.4, PyTorch 2.15.0+cpu, JAX 0.12.0.1, scikit-learn 1.9.1, SciPy 1.18.1 |
| Older PyTorch | 2.1.0+cpu with Python 3.10 and NumPy 1.26.4: named losses, tensor/native API compatibility and higher derivatives |
| Older PyTorch, CPU extension | 2.1.2, 2.5.1, 2.6.0 and 2.8.0 (+cpu, Python 3.10): the extension builds (GCC 11 and Clang 18 syntax checks of its AVX-512, AVX2 and generic code) and the torch suite passes with the extension on and off (461/323 tests on 2.1.2, 609/471 on the others); `DTensor` higher derivatives need 2.12 (checked on 2.10, 2.11 and 2.12) |
| CUDA | PyTorch 2.14.1+cu126 and Triton 3.8 on an NVIDIA H100, both launcher settings; JAX 0.12.0.1 on the same GPU. Release CUDA numerics, the CUDA and JAX GPU suites (JAX FFI handlers on and off) with PyTorch 2.15.0+cu132, Triton 3.9.0 and JAX 0.12.0.1 on an NVIDIA A100-SXM4-80GB |
| Packaging | Built-in BLAS, one `py3-none` wheel per platform: cibuildwheel 4.3.5 in the manylinux_2_28 x86-64 image (GCC 14.2, auditwheel tags `manylinux_2_27`/`manylinux_2_28`); macOS 11+ arm64 built with `python -m build` and delocate 0.13.0 as cibuildwheel does; installed core tests and strict Twine metadata checks |
| Rust | 1.74 (MSRV) and 1.99.1; GCC and Clang; downstream built-in, reference BLAS, OpenBLAS and BLIS link configurations |
| Octave | 10.3 |
| macOS arm64 | macOS 26.6, Apple clang 21, Python 3.12.13, NumPy 2.5.4, PyTorch 2.15.0, SciPy, scikit-learn: CMake C++ tests (built-in BLAS and Accelerate, `-Werror`, also with Ninja); core and torch suites against Accelerate's CBLAS (`python/build.py`, with the PyTorch CPU extension) and against the installed built-in-BLAS wheel; the wheel's core suite also on Python 3.14.9 |

The optional PyTorch CPU extension is checked with PyTorch 2.1.2, 2.5.1, 2.6.0,
2.8.0, 2.14.1 and 2.15.0.
The manylinux_2_28 test environment (CPython 3.12.16) uses NumPy 2.5.4,
SciPy 1.18.1 and scikit-learn 1.9.1. The wheels have no NumPy or CPython C
ABI dependency; the macOS wheel links only libc++ and libSystem.

Earlier minimum-version validation covered NumPy 1.23.5, scikit-learn 1.6.0, SciPy 1.10.1, and JAX 0.4.35 with NumPy 1.24.4. Those results are historical evidence rather
than a rerun of the complete final matrix at every minimum version.

Use `numpy<2` with the older PyTorch 2.1 wheels when using the native backend:
their NumPy bridge is incompatible with NumPy 2 (confirmed with 2.2.6).
This is an older PyTorch wheel constraint; the NumPy interface itself and
current PyTorch do not require that upper bound.

PyTorch 2.0 lacks `torch.backends.cpu` (used by `SDCA_CPU_EXTENSION=on`) and
JAX 0.4.34 the `vmap_method` of `pure_callback`. PyTorch 2.1 itself needs
`packaging` or a setuptools that still ships `pkg_resources`.

On CUDA, the fused kernels need PyTorch >=2.7 and Triton >=3.3. PyTorch 2.7.0
requires `triton==3.3.0` on Linux x86-64. Older releases (PyTorch 2.6 with
Triton 3.2) take the composed tensor path. Triton newer than 3.9, the last
release verified with the C++ launcher, launches the same kernels through Triton
instead.
`SDCA_CUDA_LAUNCHER` overrides that choice
([environment variables](docs/API.md#environment-variables)), and
`libsdca.torch.explain_backend` reports the backend in use and why.

Not yet built or tested: Windows, Intel macOS, Linux aarch64 (these install
from the sdist), other GPUs, and compiler/runtime combinations outside the
configurations above. On macOS the CMake `EXTERNAL` backend uses Accelerate
(`BLAS_ACCELERATE`); the Python builds link Accelerate through the bundled CBLAS
declarations. The Rust crate is untested on macOS.

## C++ configuration

| Setting | Default | Meaning |
| --- | --- | --- |
| `SDCA_BLAS_BACKEND` | `AUTO` | Prefer external BLAS, otherwise use the built-in backend. |
| `SDCA_BLAS_BACKEND=BUILTIN` | | Do not discover or link BLAS, MKL, or OpenMP. |
| `SDCA_BLAS_BACKEND=EXTERNAL` | | Require a usable external provider; fail instead of silently falling back. |
| `SDCA_BUILD_C_API` | `OFF` | Build and install the C ABI library `libsdca::sdca_c`; find Threads. |
| `SDCA_WARNINGS_AS_ERRORS` | `OFF` | Treat the project's compiler warnings as errors. |
| `BUILD_TESTING` | `OFF` | Build bundled GoogleTest and C++ test executables; find Threads for these tests. |
| `SDCA_BUILD_MATLAB` | `OFF` | Discover MATLAB and build MEX interfaces; MATLAB is required when enabled. |
| `USE_CHRONO` | `ON` | Standard C++ wall clock; `OFF` explicitly selects POSIX `gettimeofday`. |
| `USE_SEQUENTIAL` | `ON` | Select sequential MKL; `OFF` selects its threaded layer and required OpenMP runtime. |
| `USE_ILP64` | `ON` | MKL integer ABI; use `OFF` for LP64. Other providers use their existing ABI. |
| `USE_THREADING_SUFFIX` | `OFF` | Append the existing threading-mode suffix to MEX names. |

Minimal installation:

```bash
cmake -S . -B build/minimal -DSDCA_BLAS_BACKEND=BUILTIN \
  -DBUILD_TESTING=OFF -DSDCA_BUILD_MATLAB=OFF -DCMAKE_INSTALL_PREFIX=/your/prefix
cmake --build build/minimal --target install
```

CMake consumers can use the installed target:

```cmake
find_package(libsdca CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE libsdca::libsdca)
```

The target propagates C++11, include paths, the selected backend, and numerical
settings. Include `sdca/utility/logging.cpp` once in the program, as required
by the existing logger, or compile that file once alongside your sources.
Direct header users default to the built-in backend and `<chrono>`. To use an
external provider directly, define one of `BLAS_DEFAULT` (the system `<cblas.h>`),
`BLAS_DEFAULT_LOCAL_HEADER` (the bundled declarations), `BLAS_MKL`,
`BLAS_ACCELERATE`, or `BLAS_MATLAB`, and supply its link libraries.
Use consistent backend definitions across translation units. The bundled
`cblas.h` contains declarations, not an external BLAS implementation.

The minimal configuration does not require Threads or GoogleTest; the C ABI
(`SDCA_BUILD_C_API`) and the MATLAB gateways find Threads, since their loss
rows run on `std::thread` workers. Ordinary
compiler/runtime libraries such as the C++ runtime and system math library
remain normal toolchain dependencies. External providers may themselves depend
on threading/Fortran/OpenMP runtimes. Generic CMake selection verifies CBLAS
symbols, including versioned runtime-only Linux installations; a Fortran-only
BLAS is not sufficient. `BLAS_LIBRARIES` can specify external linker inputs.

## Python configuration

The PyTorch CPU extension builds on the first CPU loss call and is cached against
the installed PyTorch using `torch.utils.cpp_extension`, so it additionally needs
ninja (in `libsdca[torch]`), a C++17 compiler with OpenMP (C++20 for PyTorch
2.12 and later; checked with GCC 11, Clang 14 and 18, Apple clang 21; Linux:
GCC, e.g. the `build-essential` or `gcc-c++` package; macOS: the Xcode Command Line Tools) and
the Python headers (Debian/Ubuntu: `python3-dev`; included with the python.org,
conda and Homebrew Pythons); the CUDA launcher also needs the CUDA headers and
`libcuda.so.1`. Without them the prebuilt native library serves the CPU losses (and
Triton launches the CUDA kernels), with one warning naming what is missing
([environment variables](docs/API.md#environment-variables)). The native JAX
losses build their XLA FFI handlers on the first native call on the CPU (about 2 s),
against the installed jaxlib's headers (`jax.ffi.include_dir()`), with a C++17
compiler only; without one they run through `jax.pure_callback`, with one
warning (`SDCA_JAX_FFI`). The extension uses
ATen SIMD and intra-op threading, with strict floating-point flags, and has been
tested with GCC 11.4 on Linux x86-64 (AVX2, AVX-512) and Apple clang on arm64. The
default native build remains independent of PyTorch and OpenMP. The optional
kernel's source is shipped in the Python package.

Automatic native-library discovery probes `blas`, then `openblas`, and checks
the required CBLAS symbols before accepting either. A missing or incompatible
first candidate does not hide a usable second candidate. `SDCA_BLAS` remains
the explicit linker override.

```bash
# Native solver without external BLAS:
SDCA_BLAS_BACKEND=BUILTIN python -m pip install .

# Native solver requiring external BLAS:
SDCA_BLAS_BACKEND=EXTERNAL python -m pip install .

# Tensor losses only: no native compilation or BLAS discovery:
SDCA_BUILD_NATIVE=OFF python -m pip install '.[torch]'
```

The last command produces a `py3-none-any` wheel. It still requires NumPy and,
for tensor losses, PyTorch. It cannot train an SDCA model or use
`backend="native"`. Every `sdca_loss` objective, including
`proximal_cross_entropy`, and `cross_entropy` run with `backend="torch"`. On the
CPU, `auto` sends logits of every dtype to the CPU extension or the native
library, so without them CPU calls of every dtype need the explicit
`backend="torch"`. `distillation_loss` and `proximal_distillation_loss` take no
`backend` and on the CPU always run in the extension or the native library:
without them, such an install runs these two losses only on CUDA tensors.
NumPy and PyTorch retain their own transitive numerical/runtime dependencies;
this flag removes libsdca's native build, not those packages' dependencies.
The native builder does not use pybind11 or the Python C API. Native wheels
and CBLAS discovery are validated on Linux x86-64 and macOS arm64
([tested configurations](#tested-configurations)); pure Python packaging has
no native platform tag.

`SDCA_BUILD_NATIVE=ON` is the default. The two Python build paths share backend
selection. `SDCA_BLAS` supplies explicit external linker flags, while
`SDCA_BLAS_BACKEND=BUILTIN` bypasses them. Backend-specific build directories and
forced native rebuilds prevent stale objects or a previous native wheel from
leaking into a different configuration. `python/build.py` also accepts
`--blas-backend` and `--output`.

| Dependency group | Packages | Used for |
| --- | --- | --- |
| Runtime | NumPy >=1.23 | Array interface; no effect on C++ users. |
| `torch` | PyTorch >=2.1, ninja | Optional autograd and tensor CPU/CUDA losses; ninja builds the CPU extension. |
| `jax` | JAX >=0.4.35 | Optional JAX losses (`libsdca.jax`). |
| `sklearn` | scikit-learn >=1.6 | Optional `SDCAClassifier`. |
| `test` | pytest, pytest-xdist, SciPy, mpmath | Independent optimization and high-precision reference tests. |
| `test-torch` | pytest, pytest-xdist, SciPy, mpmath, PyTorch, ninja, scikit-learn | Full Python, PyTorch and scikit-learn tests. |
| `test-jax` | pytest, pytest-xdist, SciPy, mpmath, JAX | Python and JAX tests. |
| `coverage` | coverage.py | Python instrumentation; GCC/gcov/gcov-tool are separate developer tools. |
| Build | setuptools >=77 | Python package creation, including pure wheels. |

`python -m pytest tests/python --group core` skips importing PyTorch- and
JAX-dependent test modules; `--group torch`, `--group jax` and `--group all`
require the optional dependencies of their tests. Without `--group`, the core
tests run with the groups whose packages are installed.
CUDA is optional and used through PyTorch, not by the C++ solver. On NVIDIA
GPUs, the CUDA `proximal_cross_entropy` uses Triton where PyTorch provides it (its
Linux CUDA builds do) and the tensor solve otherwise; libsdca does not
require it. With Triton installed, `test_tensor_prox.py` also runs the
kernel under Triton's CPU interpreter.

## Rust configuration

The crate has no Rust dependencies. Its build script compiles
`src/c_api.cpp` with `CXX` (default `c++`) and the built-in BLAS; set
`SDCA_BLAS` to `-L<dir>`, `-l<name>`, `-l:<file>` or absolute library paths
(and optionally `SDCA_BLAS_BACKEND=EXTERNAL`) to link an external CBLAS
instead ([rust/README.md](rust/README.md#building)). It needs Rust 1.74 or
newer. CI links it into a downstream crate with reference BLAS, OpenBLAS and BLIS
in each of those forms, shared and static.

## Cost of the built-in backend

The built-in backend accumulates float products in double, so results are
not bitwise identical to an external BLAS. Keep `AUTO`, choose `EXTERNAL`
when a tuned provider matters and `BUILTIN` for minimal deployments.
