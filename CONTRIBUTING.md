# Contributing

`libsdca` is a header-only C++11 library with a C ABI and Python (NumPy, scikit-learn,
PyTorch, JAX), Rust and MATLAB/Octave interfaces.

## Development setup

On Debian/Ubuntu:

```bash
sudo apt-get install -y build-essential python3-dev python3-venv cmake ninja-build clang \
  libomp-dev libblas-dev pkg-config
python3 -m venv build/dev && source build/dev/bin/activate
pip install -r requirements-dev.txt numpy scipy mpmath scikit-learn jax ninja
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .                        # rerun after changing include/ or src/
```

## Tests and checks

```bash
python -m pytest -n auto --dist worksteal tests/python --group all   # or core, torch, jax
python -m pytest tests/python --group all -m serial                  # timing-sensitive, unloaded host
cmake -S . -B build/cpp -DBUILD_TESTING=ON -DSDCA_BUILD_C_API=ON -DSDCA_WARNINGS_AS_ERRORS=ON
cmake --build build/cpp -j4 && ctest --test-dir build/cpp --output-on-failure --parallel 4
python tools/quality.py                 # --fix formats first; --skip-tidy skips clang-tidy
python tools/coverage.py --cpp-only
cargo fmt --check --manifest-path rust/Cargo.toml
cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path rust/Cargo.toml
OCTAVE_HOME=<prefix> tests/matlab/run_octave_tests.sh
```

- **Markers:** `slow` and `long` tests are scheduled first under `--dist worksteal` (CI
  deselects `long`); `serial` tests run without `-n` on an unloaded machine.
- **CUDA:** On a machine with an NVIDIA GPU, install a CUDA build of PyTorch (and
  of JAX); the same pytest command then runs the CUDA tests, which skip otherwise.
  `python tests/python/release_numerics.py --device cuda --output <dir>` also checks the
  CUDA entropy and distillation losses against `mpmath` references and exits 1 on a failure.
- **Runtime:** `tests/python/test_work_counts.py` bounds the solver's epochs on seeded
  problems, so convergence regressions fail without a timer. On a CUDA machine,
  `python tools/perf_gate.py --output <file>` times the losses against `torch.compile` on
  32,000- and 128,256-class rows and exits 1 when a speedup falls below its floor or libsdca
  needs more peak memory.
- **Installed package:** `python -m pytest tests/python --installed` tests the installed
  `libsdca` instead of the source tree.
- **`python/build.py`** builds `build/python/libsdca_c.so`, which takes precedence over an
  editable install; delete it (or set `SDCA_LIBRARY_DIR`) when switching back.
- **JAX:** For repeated runs, set `JAX_COMPILATION_CACHE_DIR=<dir>` and
  `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`.

| Tool | Rule |
| --- | --- |
| `clang-format`, `clang-tidy` (v18) | `.clang-format` (Google style, C++11, 100 columns) and `.clang-tidy`, warnings as errors |
| `ruff`, `basedpyright` | `pyproject.toml`: `E`, `F`, `I`, `B`, `UP`, `C90`; standard mode, warnings as errors |
| `lizard` | Cyclomatic complexity $\le 20$, length $\le 100$ lines (`tools/complexity_baseline.txt` may only shrink) |
| `gcovr`, `coverage.py` | $\ge 90\%$ line coverage of `include/sdca`, `src/c_api.cpp` and `python/libsdca` |
| `rustfmt`, `clippy` | `-D warnings`, all public items documented |
| C++ compilers | `-pedantic -Werror -W -Wall -Wextra` under GCC and Clang |

## Layout

| Location | Contents |
| --- | --- |
| `include/sdca/math/`, `include/sdca/prox/` | Compensated math, BLAS, projections and proximal operators |
| `include/sdca/solver/`, `include/sdca/loss.h` | Data views, objectives, dual updates, stopping criteria and losses |
| `include/sdca/c_api.h`, `src/c_api.cpp` | Exception-safe C ABI used by Python and Rust |
| `python/libsdca/_core.py`, `sklearn.py` | NumPy API and `SDCAClassifier` |
| `python/libsdca/torch.py`, `_cpu.py`, `_torch_*` | PyTorch layers, tensor losses and the CPU extension |
| `python/libsdca/_triton_*.py`, `_cuda_launch.py` | Triton CUDA kernels and the C++ launcher |
| `python/libsdca/jax.py`, `_jax_*` | JAX losses and XLA FFI handlers |
| `rust/`, `matlab/`, `src/matlab/` | Rust crate and MATLAB/Octave gateways |
| `tests/`, `examples/`, `docs/`, `tools/` | Test suites, examples, guides and developer scripts |

## Guidelines

- **C++:** C++11, header-only, IEEE semantics (`-fno-fast-math -ffp-contract=off`), no `new`.
  Public functions validate inputs and throw `std::invalid_argument` or `std::runtime_error`.
- **Python:** Annotate all functions; keep `import libsdca` free of `torch`, `jax`, `sklearn`
  and native-library loads; raise `ValueError` or `TypeError` at the API boundary; never copy
  tensors across devices implicitly.
- **Rust:** No `unsafe` outside FFI calls, each with a `// SAFETY:` comment.
- **Names:** A new objective, option or status uses the same name in every interface
  ([docs/API.md](docs/API.md)), pinned by `tests/python/test_api_contracts.py` and
  `rust/tests/api.rs`.
- **Numerical changes:** Every fix comes with a regression test against an independent
  analytic, `mpmath` or KKT reference. Do not loosen a tolerance to make a test pass.

## Releases

```bash
python tools/bump_version.py 0.3.2      # VERSION and its copies; "## Unreleased" gets the date
python tools/quality.py --skip-tidy     # checks that every copy agrees
git tag v0.3.2 && git push origin v0.3.2
```

`VERSION` is read by CMake, `setup.py`, `pyproject.toml`, `python/build.py` and
`matlab/make.m`. On a `v*` tag, `.github/workflows/packages.yml` builds the sdist, the wheels
and the crate and publishes them to PyPI and crates.io with trusted publishing.
