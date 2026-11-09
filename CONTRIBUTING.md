# Contributing

`libsdca` is a header-only C++11 library with a C ABI and Python (NumPy,
scikit-learn, PyTorch, JAX), Rust and MATLAB/Octave interfaces. This guide lists
the development workflow, repository layout, quality gates and coding rules.

## Development setup

```bash
python3 -m venv build/dev && source build/dev/bin/activate
pip install -r requirements-dev.txt numpy scipy mpmath scikit-learn jax ninja
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .                        # rerun after changing include/ or src/
python -m pytest -n auto --dist worksteal tests/python --group all   # or core, torch, jax
python -m pytest tests/python --group all -m serial
cmake -S . -B build/cpp -DBUILD_TESTING=ON -DSDCA_BUILD_C_API=ON -DSDCA_WARNINGS_AS_ERRORS=ON
cmake --build build/cpp -j4
ctest --test-dir build/cpp --output-on-failure --parallel 4
python tools/quality.py
python tools/coverage.py --cpp-only
cargo fmt --check --manifest-path rust/Cargo.toml
cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path rust/Cargo.toml
```

- `tools/quality.py --fix` formats C++ and Python sources before running the checks.
- If you run `python/build.py`, delete `build/python` (or set `SDCA_LIBRARY_DIR`) when
  switching back to `pip install -e .`, as `build/python/libsdca_c.so` takes precedence.
- Pytest markers (`tests/python/conftest.py`): `slow` and `long` tests are scheduled first
  under `-n auto --dist worksteal` (CI deselects `long` with `-m "not long"`); `serial`
  tests require an unloaded machine and run via `-m serial` without `-n`.
- For repeated local JAX test runs, set `JAX_COMPILATION_CACHE_DIR=<dir>` and
  `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`.

## Layout

| Location | Contents |
| --- | --- |
| `include/sdca/math/`, `include/sdca/prox/` | Compensated math, entropy roots, BLAS, projections and proximal operators |
| `include/sdca/solver/`, `include/sdca/loss.h` | Data views, contexts, objectives, dual updates, stopping criteria, evaluation and loss dispatch |
| `include/sdca/c_api.h`, `src/c_api.cpp` | Exception-safe C ABI used by Python and Rust |
| `python/libsdca/_core.py`, `sklearn.py` | NumPy API and `SDCAClassifier` |
| `python/libsdca/torch.py`, `_cpu.py`, `_torch_*` | PyTorch autograd layers, tensor losses, and ATen/OpenMP CPU extension |
| `python/libsdca/_triton_*.py`, `_cuda_launch.py` | Fused Triton CUDA kernels and C++ `cuLaunchKernel` launcher |
| `python/libsdca/jax.py`, `_jax_*` | JAX losses and XLA FFI custom-call handler |
| `rust/`, `matlab/`, `src/matlab/` | Rust crate (`rust/consumer` tests BLAS links) and MATLAB/Octave gateways |
| `tests/`, `examples/`, `docs/`, `tools/` | C++/Python/MATLAB test suites, runnable examples, docs, and quality/coverage scripts |

## Checks

| Tool | Scope | Rule |
| --- | --- | --- |
| `clang-format` | `include/sdca`, `src`, `tests`, `tools`, `python/libsdca` | `.clang-format` (v18): Google style, C++11, 100 columns |
| `clang-tidy` | `src/c_api.cpp`, `tests/cpp/regression.cpp` (both BLAS backends) | `.clang-tidy` (v18): correctness and performance checks, warnings as errors |
| `lizard` | `include/sdca`, `src`, `python/libsdca` | Cyclomatic complexity $\le 20$, length $\le 100$ lines (`tools/complexity_baseline.txt` may only shrink) |
| `ruff` | `python`, `tests/python`, `examples/python`, `tools`, build scripts | `pyproject.toml`: `E`, `F`, `I`, `B`, `UP`, `C90` (complexity $\le 15$) |
| `basedpyright` | Python package, tests, examples, build scripts, tools | `pyproject.toml`: standard mode, warnings as errors, rule-scoped `# pyright: ignore[rule]` |
| `gcovr`, `coverage.py` | `include/sdca`, `src/c_api.cpp`, `python/libsdca/` | $\ge 90\%$ line coverage across C++ and Python host code (only `@triton.jit` device bodies excluded) |
| `rustfmt`, `clippy` | `rust/` | Standard style, `-D warnings`, all public items documented |
| C++ compilers | All C++ targets | `-pedantic -Werror -W -Wall -Wextra` under GCC and Clang |

To combine GPU and CPU Python host coverage from the same source checkout:

```bash
python tools/coverage.py --write-python-manifest build/gpu.coverage
PYTHONPATH=python COVERAGE_FILE=build/gpu.coverage SDCA_CUDA_LAUNCHER=on \
  python -m coverage run --source=libsdca -m pytest tests/python --group torch
PYTHONPATH=python COVERAGE_FILE=build/gpu.coverage SDCA_CUDA_LAUNCHER=off \
  python -m coverage run --append --source=libsdca -m pytest tests/python --group torch
python tools/coverage.py --python-data build/gpu.coverage
```

## Coding guidelines

- **C++:** C++11, header-only, strict IEEE semantics (`-fno-fast-math -ffp-contract=off`),
  no `new` (use `std::vector` or `detail::entropy_buffer`). Public functions validate inputs
  and throw `std::invalid_argument` or `std::runtime_error`; `detail` helpers document their
  preconditions.
- **Python:** Annotate all functions; keep `import libsdca` free of `torch`, `jax`,
  `sklearn` and native-library loads; raise `ValueError` or `TypeError` at the API boundary
  and never copy tensors across devices implicitly.
- **Rust:** No `unsafe` outside FFI calls, each with a `// SAFETY:` comment.
- **Cross-language vocabulary:** Every new objective, option or status uses the same name
  across C++, the C ABI, Python, Rust and MATLAB ([docs/API.md](docs/API.md)), pinned by
  `tests/python/test_api_contracts.py` and `rust/tests/api.rs`.
- **Numerical changes:** Every fix includes a regression test against an independent
  analytic, `mpmath`, or KKT reference. Refactorings of `include/sdca` must leave results
  bitwise unchanged, and performance-sensitive changes must report before/after timings
  ([docs/performance.md](docs/performance.md#where-libsdca-is-slower)).

## Releases

The version string is checked across 10 files by `tools/quality.py`: `CMakeLists.txt`,
`pyproject.toml`, `setup.py`, `python/build.py`, `python/libsdca/__init__.py`,
`matlab/make.m`, `rust/Cargo.toml`, `include/sdca/c_api.h`, `docs/C.md`, and `CHANGELOG.md`.

`.github/workflows/packages.yml` builds and tests the sdist, `manylinux_2_28` `x86_64` and
macOS 11+ `arm64` wheels (`cibuildwheel`), and the `.crate` archive (`rust/package.sh`),
publishing to TestPyPI (manual dispatch) or, on `v*` tags, to PyPI and then crates.io via OIDC
Trusted Publishing:

1. In GitHub *Settings → Environments*, create `testpypi`, `pypi` and `crates-io` (restrict
   `pypi` and `crates-io` to `v*` tags, and `pypi` to required reviewers: approving it releases
   both, since the crates.io job waits for the PyPI upload).
2. On [TestPyPI](https://test.pypi.org/manage/account/publishing/) and
   [PyPI](https://pypi.org/manage/account/publishing/), register `mlapin/libsdca`, workflow
   `packages.yml`, and environment `testpypi` / `pypi` as a trusted publisher.
3. In the [crates.io](https://crates.io/crates/libsdca) crate settings, add a GitHub trusted
   publisher for `mlapin/libsdca`, workflow `packages.yml`, and environment `crates-io`.
