# Developer tools

Scripts used in development and CI ([CONTRIBUTING.md](../CONTRIBUTING.md)):

| File | Purpose |
| --- | --- |
| `quality.py` | `ruff`, `clang-format`, `basedpyright`, version checks, `lizard` and `clang-tidy` (`--fix`, `--skip-tidy`) |
| `coverage.py` | C++ and Python line coverage with the $\ge 90\%$ floor |
| `bump_version.py` | Sets `VERSION` and the copies that cannot read it |
| `run_cpp_tests.sh` | Builds and runs the C++ and C ABI suites without CMake (`--sanitize` for ASan, UBSan, TSan) |
| `check_extension_builds.py` | Syntax-compiles the PyTorch CPU extension (every instruction set in one unit) with each compiler |
| `check_install_archives.py` | Compares packaged Python modules and shared kernel headers against the source tree and checks the native library in wheels |
| `check_docs.py` | Checks the links and heading anchors of every tracked Markdown file and GitHub math compatibility; included in `quality.py`; `--external` also requests every external URL |
| `perf_gate.py` | CUDA speed and memory against `torch.compile`, with per-loss speedup floors |
| `gpu_session.py` | One GPU test session from a `git archive` tarball: dependencies, build, the CUDA tests, `release_numerics`, `perf_gate`, `ctest` and `cargo test`, with JUnit and JSON results (`--dry-run` collects only) |
| `apt_mirror.sh` | Adds retries and a fallback Ubuntu mirror before `apt-get` in CI |
| `complexity_baseline.txt` | Functions above the `lizard` limits (may only shrink) |
| `typings/triton/` | Type stubs for checking the Triton kernels without a GPU |

```bash
python tools/quality.py --fix
python tools/coverage.py --cpp-only
tools/run_cpp_tests.sh --sanitize
```
