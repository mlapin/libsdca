# Developer tools

Scripts used in development and CI ([CONTRIBUTING.md](../CONTRIBUTING.md)):

| File | Purpose |
| --- | --- |
| `quality.py` | `ruff`, `clang-format`, `basedpyright`, version checks, `lizard` and `clang-tidy` (`--fix`, `--skip-tidy`) |
| `coverage.py` | C++ and Python line coverage with the $\ge 90\%$ floor |
| `bump_version.py` | Sets `VERSION` and the copies that cannot read it |
| `run_cpp_tests.sh` | Builds and runs the C++ and C ABI suites without CMake (`--sanitize` for ASan, UBSan, TSan) |
| `check_extension_builds.py` | Compiles the PyTorch CPU extension for AVX-512, AVX2 and generic targets |
| `check_install_archives.py` | Checks that an sdist or wheel contains the shared CPU sources and native libraries |
| `readme_smoke.py` | Installs a wheel into fresh uv or conda environments and runs the README examples verbatim |
| `check_docs.py` | Checks local Markdown links, heading anchors and GitHub math compatibility; included in `quality.py` |
| `readme_*_smoke.py` | Installed-package probes used by `readme_smoke.py` (precision, CPU and CUDA dispatch, contracts) |
| `perf_gate.py` | CUDA speed and memory against `torch.compile`, with per-loss speedup floors |
| `apt_mirror.sh` | Adds retries and a fallback Ubuntu mirror before `apt-get` in CI |
| `complexity_baseline.txt` | Functions above the `lizard` limits (may only shrink) |
| `typings/triton/` | Type stubs for checking the Triton kernels without a GPU |

```bash
python tools/quality.py --fix
python tools/coverage.py --cpp-only
tools/run_cpp_tests.sh --sanitize
python -m build --wheel && python tools/readme_smoke.py --wheel dist/libsdca-*.whl --output build/readme
python tools/readme_smoke.py --device cuda --profiles torch jax-cuda --wheel dist/libsdca-*.whl --output build/readme-cuda
```
