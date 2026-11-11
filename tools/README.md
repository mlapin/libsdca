# Developer tools

Quality, coverage and build-verification scripts used in local development and CI
(see [CONTRIBUTING.md](../CONTRIBUTING.md) for the full setup):

| File | Purpose |
| --- | --- |
| `quality.py` | Runs `ruff`, `clang-format`, `basedpyright`, version-string checks, `lizard` complexity checks and `clang-tidy` (`--fix` formats first; `--skip-tidy` skips `clang-tidy`) |
| `coverage.py` | Builds C++ and the C ABI with `--coverage`, runs C++ and Python test suites, and enforces the $\ge 90\%$ line-coverage floor in `gcovr` and `coverage.py` |
| `check_extension_builds.py` | Syntax-checks the PyTorch CPU extension (`_torch_entropy.cpp`) across target ISAs (`AVX-512`, `AVX2`, generic) and compilers |
| `run_cpp_tests.sh` | Builds and runs the C++ and C ABI GoogleTest suites directly (`--sanitize` enables ASan, UBSan and Clang TSan) |
| `apt_mirror.sh` | Used by CI before `apt-get`: adds retries and a fallback Ubuntu mirror (`mirrors.edge.kernel.org`), placed first when the runner cannot resolve Ubuntu's package hosts |
| `complexity_baseline.txt` | Grandfathered functions above the `lizard` complexity ($\le 20$) or line-count ($\le 100$) limits (may only shrink) |
| `typings/triton/` | Minimal type stubs used by `basedpyright` to type-check the Triton kernel modules without a GPU |

```bash
python tools/quality.py --fix
python tools/coverage.py --cpp-only
python tools/check_extension_builds.py
tools/run_cpp_tests.sh --sanitize
```
