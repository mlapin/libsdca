# Contributing

libsdca is a header-only C++11 library with a C ABI and Python (NumPy,
scikit-learn, PyTorch, JAX), Rust and MATLAB interfaces. This page lists the
rules a change has to follow and the checks that enforce them.

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
python tools/coverage.py
cargo fmt --check --manifest-path rust/Cargo.toml
cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path rust/Cargo.toml
```

`tools/quality.py --fix` applies the formatters first. Its basedpyright check
resolves the optional packages installed above. The CI workflow
(`.github/workflows/ci.yml`) runs the same commands, testing the sources with
`python/build.py` and `PYTHONPATH=python` instead of an install. A
`build/python/libsdca_c.so` left by `python/build.py` takes precedence over the
library of `pip install -e .`: delete `build/python` (or set `SDCA_LIBRARY_DIR`)
when switching to the editable install.

The suites run on pytest (`tests/python/conftest.py` defines `--group` and
`--installed`); the header names the `libsdca` under test, and any working
directory works. `--installed` also rejects the source tree and skips the
tests of source-tree helpers. New tests use pytest's plain `assert`,
`pytest.raises` and fixtures; `subtests` labels the cases of a loop. Tests that start Python
subprocesses use `tests/python/child_process.py`, so a child imports the same
package as the test, whatever the working directory; Triton interpreter checks
run through `tests/python/interpreter.py`.

`-n auto` (pytest-xdist) runs the tests in one worker per CPU, each with an
equal share of the CPUs as `OMP_NUM_THREADS` unless it is set; the CPU
extension is built once before the workers start. Tests marked `slow` (about a
minute on a loaded machine) start first, one per worker under
`--dist worksteal`. A test that needs the machine to itself (thread timing) is
marked `serial`: `-n` deselects it, even when named, and the `-m serial` run,
without `-n`, runs it (only `core` and `all` have such tests; for `torch` or
`jax` it selects none and pytest exits with 5). Keep tests safe to run in
parallel: temporary files in `tmp_path` or `tempfile` directories, environment
changes through `mock.patch.dict`, and no fixed ports. A loop of many slow cases is split into parametrized parts
(`part`) so the workers share it: every part runs the whole loop, drawing the
same random inputs, and checks only its own cases.

Most of the JAX tests' time is XLA compilation. For repeated local runs,
`JAX_COMPILATION_CACHE_DIR=<dir>` with `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`
reuses the previous run's compilations.

## Layout

| Directory | Contents |
| --- | --- |
| `include/sdca/` | The header-only C++11 library and the C ABI header |
| `src/` | The C ABI (`c_api.cpp`) and the MATLAB gateways (`src/matlab/`) |
| `python/` | The Python package `libsdca` and `build.py` |
| `rust/`, `matlab/` | The Rust crate (with its `tests/`) and the MATLAB/Octave package |
| `tests/` | `cpp/` (GoogleTest and C ABI), `python/`, `matlab/` |
| `examples/`, `docs/` | Usage examples, documentation |
| `tools/` | Quality, coverage and C++ test scripts |
| `cmake/`, `vendor/` | CMake modules, vendored GoogleTest |

The C++ headers own the numerical algorithms; the bindings adapt layouts, validate
inputs and expose them under one vocabulary. The main source files:

| Location | Contents |
| --- | --- |
| `include/sdca/math/`, `include/sdca/prox/` | Stable scalar arithmetic, entropy roots, BLAS abstraction; projections and proximal operators |
| `include/sdca/solver/` | Data views, contexts, objectives (values, duals, coordinate updates), iteration, stopping, evaluation |
| `include/sdca/loss.h` | Loss and gradient dispatch |
| `include/sdca/c_api.h`, `src/c_api.cpp` | The exception-safe C ABI used by Python and Rust |
| `python/libsdca/_core.py`, `sklearn.py` | NumPy API and `SDCAClassifier` |
| `python/libsdca/torch.py`, `_cpu.py`, `_torch_*.{h,cpp}` | PyTorch autograd boundary, tensor losses, CPU extension |
| `python/libsdca/_triton_*.py`, `_cuda_launch.py` | Fused CUDA kernels and their C++ launcher |
| `python/libsdca/jax.py`, `_jax_losses.py`, `_jax_ffi.*` | JAX losses and the XLA FFI handler |

## Checks

| Tool | Scope | Rule |
| --- | --- | --- |
| clang-format 18 | `include/sdca`, `src`, `tests`, `tools`, `python/libsdca` | `.clang-format`: Google style, C++11, 100 columns, include order kept |
| clang-tidy 18 | the C ABI and the regression tests, both BLAS backends | `.clang-tidy`: correctness and performance checks, warnings are errors |
| lizard | library functions in `include/sdca`, `src` and `python/libsdca`; tests are exempt | cyclomatic complexity at most 20, at most 100 lines; `tools/complexity_baseline.txt` lists the existing exceptions, which may only shrink |
| ruff | `python`, `tests/python`, `examples/python`, `tools`, build scripts | `pyproject.toml`: pycodestyle errors, pyflakes, isort, bugbear, pyupgrade, complexity at most 15 |
| basedpyright | the Python package, tests, examples, build scripts and tools | `pyproject.toml`: standard mode, warnings are errors, ignores name their rule and must be needed, parameters annotated outside Triton kernels |
| gcovr, coverage.py | `include/sdca`, `src/c_api.cpp`, `python/libsdca/` | at least 90% of the lines run by the C++ and Python suites |
| rustfmt, clippy | `rust/` | default style, all clippy warnings are errors, documented public items |
| compilers | all C++ | `-pedantic -Werror -W -Wall -Wextra` under GCC and Clang |

Python's 90% line-coverage floor measures host code, including CUDA launch and
dispatch wrappers. Only functions decorated with `@triton.jit` are excluded:
their compiled device bodies cannot be traced by coverage.py. GPU oracle,
gradient, dispatch and launcher-on/off tests validate those kernels separately;
host coverage is not a device-code coverage claim.
The narrow exclusion uses coverage.py's
[decorator matching](https://coverage.readthedocs.io/en/latest/excluding.html#advanced-exclusion).

Collect GPU host coverage from the same source revision, then combine it with
CPU coverage (including the optional CPU extension and scikit-learn):

```bash
python tools/coverage.py --write-python-manifest build/gpu.coverage
PYTHONPATH=python COVERAGE_FILE=build/gpu.coverage SDCA_CUDA_LAUNCHER=on \
  python -m coverage run --source=libsdca -m pytest tests/python --group torch
PYTHONPATH=python COVERAGE_FILE=build/gpu.coverage SDCA_CUDA_LAUNCHER=off \
  python -m coverage run --append --source=libsdca -m pytest tests/python --group torch
python tools/coverage.py --python-data build/gpu.coverage
```

`--python-data` is repeatable.
Input data files are preserved; combined data and a machine-readable report
are written to `.coverage-host` and `python.json` in the coverage build directory.
Keep package sources unchanged throughout collection. Each input needs its
`<data-path>.sources.json` sidecar containing SHA256 hashes of every Python
package file; combination rejects missing or incompatible manifests before
mapping checkout paths. CPU collection writes and verifies its own manifest.
The CPU interpreter needs the development tools, NumPy, SciPy, mpmath, PyTorch,
JAX, scikit-learn and Ninja. A CPU-only run still reports missing GPU host code;
it does not weaken the threshold or omit host modules.

GitHub's CPU runner uses `tools/coverage.py --collect-python`: it runs the CPU
and optional CPU-extension suites, enforces the C++ 90% floor, and uploads raw
Python coverage, its source manifest, and `python-provisional.json`. The output
is explicitly provisional because that runner cannot exercise GPU host paths.
This collection mode does not certify Python's release floor: release
verification combines compatible CPU/GPU data and still requires 90% Python
host coverage.

`vendor/` and the MATLAB scripts are outside the format and Python lint/type
gates; tests are exempt from complexity limits. MATLAB gateways have separate
compile and runtime checks.

The complexity baseline may only shrink by project policy. The current check
rejects unlisted offenders and stale entries, but records function names rather
than per-function complexity or line ceilings. Review must therefore enforce
the policy against new baseline entries and growth within existing exceptions;
the gate does not enforce those two restrictions automatically.

## C++ guidelines

- C++11, header-only, no global mutable state beyond the existing logger
  level and format. Owned buffers are `std::vector` or the stack-backed
  `detail::entropy_buffer`; no `new`.
- Public functions validate their arguments and throw
  `std::invalid_argument` for invalid input and `std::runtime_error` for a
  numerical failure, with a message that names the function or quantity.
  Internal helpers in `namespace detail` document their preconditions
  instead of checking them again.
- One function, one job. Split a long function into named steps; keep a
  formula together when splitting it would hide the mathematics.
- Names describe roles (`threshold`, `free_mass`, `cap`); short names are
  fine where they are the mathematics (`x`, `a`, `s`, `t`, `k`, `n`).
- Comments explain why: the invariant, the failure a guard prevents, the
  regime a branch is for. A public function's comment states the problem
  it solves, its arguments and outputs, and what it throws.
- Numerical code keeps IEEE semantics (`-fno-fast-math
  -ffp-contract=off`). Do not reassociate sums of large and small terms;
  keep offsets apart from the differences they shift.

## Python guidelines

- Annotate every function; public functions have a docstring stating
  shapes, dtypes, defaults, outputs and exceptions.
- `import libsdca` must not import torch or load the native library.
- Raise `ValueError` or `TypeError` with an actionable message at the API
  boundary; never copy tensors between devices implicitly.

## Interface guidelines

- One vocabulary ([docs/API.md](docs/API.md)): a new objective, option or
  status gets the same name in C++, the C ABI, Python, Rust and MATLAB.
- Python and Rust call the C ABI (`include/sdca/c_api.h`); validation shared
  by several bindings belongs there, and its ids are tied to the C++ enums by
  `static_assert`. MATLAB calls the headers directly, and the PyTorch tensor
  backend is a separate implementation of the losses (see the
  [numerical contracts](docs/numerics.md#numerical-contracts)).
- `tests/python/test_api_contracts.py` and `rust/tests/api.rs` pin the public
  names, signatures and error behavior; change them only with the API.

## Rust guidelines

- No `unsafe` outside the FFI calls, each with a `SAFETY` comment.
- Public items are documented, errors are `Error::InvalidArgument` or
  `Error::Native`, and options are structs with `Default`.

## Numerical changes

- Every fix comes with a test that fails before it, against an
  independent reference (an analytic solution, a high-precision oracle, or
  a KKT check), not against the code's own previous output.
- A refactoring of `include/sdca` must leave the results bitwise unchanged;
  an intended change states which results change and by how much.
- Report the cost of a change to the coordinate updates, projections or
  training with before and after timings: libsdca and the reference on the same
  threads, several rounds, medians with their spread.
- Every case where libsdca is slower than its reference is listed in
  [docs/performance.md](docs/performance.md#where-libsdca-is-slower) with its
  measured cause, or as open work to close without loosening the precision
  guarantees.

## Releases

A release passes the checks above, the PyTorch CPU extension's syntax under GCC and
Clang for each instruction set it builds for (`tools/check_extension_builds.py`:
AVX-512, AVX2 and the generic build of arm64 and older x86), both C++ BLAS backends
under GCC and Clang, the sanitizers, the MEX build, line coverage, the Python, Rust
and Octave suites, and the sdist, wheel and crate tested outside the checkout.

The PyTorch tests also build the optional CPU extension, so Ninja and a C++
compiler must be on `PATH`. `tools/run_cpp_tests.sh --sanitize` runs
ThreadSanitizer with Clang (`SDCA_TSAN_CXX`; validated with Clang 18): GCC 11's
runtime does not intercept `pthread_cond_clockwait` and
reports false races. Older TSan runtimes fail at start on kernels whose
`vm.mmap_rnd_bits` exceeds 28 (`sysctl vm.mmap_rnd_bits=28`). With OpenMP, a run of
`tests/cpp/c_api_threads.cpp` on an OpenMP team needs LLVM's libomp with its Archer
tool (`OMP_TOOL_LIBRARIES`); TSan does not see libgomp's synchronization. The downstream Rust BLAS matrix and
platform-specific wheel builds run separately in GitHub Actions.

The version lives in `CMakeLists.txt`, `pyproject.toml`, `setup.py`,
`python/build.py`, `python/libsdca/__init__.py`, `matlab/make.m`,
`rust/Cargo.toml`, the `sdca_version()` example in `include/sdca/c_api.h`,
`docs/C.md` and the first `## <version> (YYYY-MM-DD)` heading of
`CHANGELOG.md`; `tools/quality.py` fails when they differ.

`.github/workflows/packages.yml` builds and checks the sdist, the wheels
(cibuildwheel, built-in BLAS: manylinux_2_28 x86-64 and macOS 11+ arm64,
each tested with `pytest tests/python --group core --installed`) and the
crate (`rust/package.sh`, `cargo publish --dry-run`), and keeps them as
artifacts. Uploads use PyPI Trusted Publishing (no API tokens):

- A manual run (*Actions → Packages → Run workflow*) with *Upload to
  TestPyPI* checked uploads the sdist and wheels to TestPyPI. A version can
  be uploaded only once; bump it (e.g. `0.3.0rc1`) to upload again.
- Pushing a tag `v<version>` checks that the tag matches `pyproject.toml`
  and uploads to PyPI after the `pypi` environment's approval.
- The crate is not published by the workflow; `cargo publish` from
  `rust/` after `rust/package.sh` remains a manual step.

One-time setup:

1. GitHub, *Settings → Environments*: create `testpypi` and `pypi`. Give
   `pypi` a required reviewer and limit its deployments to tags `v*`.
2. TestPyPI (<https://test.pypi.org/manage/account/publishing/>) and PyPI
   (<https://pypi.org/manage/account/publishing/>): add a pending trusted
   publisher for the project `libsdca`: the owner and repository of
   <https://github.com/mlapin/libsdca>, workflow `packages.yml` and
   environment `testpypi` or `pypi` respectively. The first upload creates
   the project.

## Commits

Small commits with a one-line summary in the imperative and short bullet
points, each passing the tests on its own.
