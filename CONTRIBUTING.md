# Contributing

`libsdca` is a header-only C++11 library with a C ABI and Python (NumPy, scikit-learn,
PyTorch, JAX) and Rust interfaces.

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
python tests/python/release_numerics.py --device cpu --jobs 4 --output numerics-cpu  # mpmath corpus
cmake -S . -B build/cpp -DBUILD_TESTING=ON -DSDCA_BUILD_C_API=ON -DSDCA_WARNINGS_AS_ERRORS=ON
cmake --build build/cpp -j4 && ctest --test-dir build/cpp --output-on-failure --parallel 4
python tools/quality.py                 # --fix formats first; --skip-tidy skips clang-tidy
python tools/coverage.py --cpp-only
cargo fmt --check --manifest-path rust/Cargo.toml
cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path rust/Cargo.toml
```

- **Markers:** `slow` and `long` tests are scheduled first under `--dist worksteal` (CI
  deselects `long`); `serial` tests run without `-n` on an unloaded machine.
- **Conformance matrix:** The default runs the full matrix.
  - `--conformance ci` runs about a tenth of the conformance cells (every loss on the
    default backends, cross-entropy and the multiclass hinge on every backend, dtype and
    precision), as CI does.
  - A manual CI run (*Run workflow*, `conformance: full`) runs the full matrix on the hosted
    runners and, when cells fail, uploads their records (`SDCA_CONFORMANCE_RECORD`) as an
    artifact.
  - A reviewed failure set of other hardware is added to `known_failures/` as a variant.
- **Property tests:** `tests/python/test_properties_{torch,jax}.py` (Hypothesis) check that
  gradient rows sum to zero, that no NaN appears, that no loss is negative, that distillation
  of identical logits has a zero gradient and that cross-entropy at its own softmax keeps
  only the rounding of the targets, on every interface and dtype
  (Optax: its four functions). Plain pytest loads the `ci` profile (the explicit examples
  and 200 derandomized examples per test), as CI does, together with the CPU `mpmath` corpus
  (`release_numerics.py --device cpu`, with `SDCA_CPU_EXTENSION` unset and `on`).
- **Failures and limits:** a failing input becomes a documented limit or an exemption only
  after it is shrunk to a minimal reproducer and its mechanism is found in the code (vary one
  parameter at a time until the boundary is located); the limit then names that condition,
  and the reproducer becomes a test. Checkers reject `NaN` explicitly (`excess > 0` passes a
  `NaN`).
- **Release-time checks:** `--hypothesis-profile=search --hypothesis-seed=<seed>` (10,000
  examples per test; record the seed), `-m sweep` (the batched sweep of the property
  tests), and on a GPU the CUDA corpus and the CUDA cells through `tools/gpu_session.py`
  (below).
- **CUDA:** On a machine with an NVIDIA GPU, install a CUDA build of PyTorch (and
  of JAX); the same pytest command then runs the CUDA tests, which skip otherwise.
  `python tests/python/release_numerics.py --device cuda --output <dir>` also checks the
  CUDA entropy and distillation losses against `mpmath` references and exits 1 on a failure.
  `python3 tools/gpu_session.py ARCHIVE.tgz` runs all of it from a `git archive` tarball (environment, build, pytest,
  `perf_gate`, `ctest`, `cargo test`) and packs JUnit, logs and a summary into one archive.
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
| `include/sdca/core/` | Lanes, math, reductions, the batch driver, projections and proximal maps |
| `include/sdca/loss/`, `include/sdca/kernel.h` | Loss rows and the kernel entry (`kernel::evaluate`) |
| `include/sdca/solver/`, `include/sdca/math/blas*.h` | Data views, objectives, dual updates, stopping criteria, BLAS |
| `include/sdca/loss.h`, `prox.h`, `solver.h` | Public C++ API |
| `include/sdca/c_api.h`, `src/c_api.cpp` | Exception-safe C ABI used by Python and Rust |
| `python/libsdca/_native.py`, `sklearn.py` | NumPy API and `SDCAClassifier` |
| `python/libsdca/_frontend.py`, `_frontend_cpu.h` | Argument handling and CPU entry shared by PyTorch and JAX |
| `python/libsdca/torch.py`, `_torch_*` | PyTorch losses and the CPU extension |
| `python/libsdca/jax.py`, `_jax_*` | JAX losses and the XLA FFI handler |
| `python/libsdca/triton/` | Triton GPU kernels |
| `python/libsdca/_composed/` | Losses composed from framework operations (`backend="composed"`) |
| `rust/` | Rust crate |
| `tests/`, `examples/`, `docs/`, `tools/` | Test suites, examples, guides and developer scripts |

## Guidelines

- **C++:** C++11, header-only, IEEE semantics (`-fno-fast-math -ffp-contract=off`), no
  `new`.
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

## Documentation

Keep API guides in `docs/`.

- **Structure:** The guides are a reference manual: installation, a runnable quick start,
  signatures, option tables, task examples, precision and errors. Keep language-specific
  sections such as C layouts and PyTorch transforms. Benchmarks belong to the technical
  report and `benchmarks/README.md`; test commands belong here.
- **Shared contract:** Link to `docs/API.md` for shared definitions and defaults instead of
  repeating the contract.
- **Style:** Use short paragraphs and tables for options and results; put longer derivations
  after the usage examples.
- **Benchmarks:** Benchmark tables must identify hardware and timing boundaries and include
  slower cases. Do not attribute one binding's timings to another.
- **Checks:** `python tools/check_docs.py` checks every link in the tracked Markdown: the
  file (with its exact case), the heading anchor and links to this repository on GitHub. It
  also rejects unsupported math commands; `tools/quality.py` runs it in CI. Use
  `\mathrm{...}` for named math functions. Add `--external` to request every external URL.
- **Absolute links:** The Python and Rust guides also render on PyPI and crates.io, so use
  absolute links there.
- **Rust guide:** Cargo packages `docs/RUST.md` as `RUST.md`; the build script feeds the
  same guide to doctests in both a checkout and an extracted crate.

## Releases

```bash
python tools/bump_version.py 0.3.2      # VERSION and its copies; "## 0.3.2 (unreleased)" gets the date
python tools/quality.py --skip-tidy     # checks that every copy agrees
python tools/check_docs.py --external   # every external link answers
git tag v0.3.2 && git push origin v0.3.2
```

`VERSION` is read by CMake, `setup.py`, `pyproject.toml` and `python/build.py`. On a `v*`
tag, `.github/workflows/packages.yml` builds the sdist, the wheels and the crate and publishes
them to PyPI and crates.io with trusted publishing.
