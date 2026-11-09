# Examples

Runnable examples for each language interface:

| Directory | Contents | Guide |
| --- | --- | --- |
| [python/](python/README.md) | PyTorch (`CrossEntropyLoss` drop-in, loss tour, MNIST LeNet-5, distillation), NumPy (`fit`, `loss_and_grad`), scikit-learn (`SDCAClassifier`) and JAX (`sdca_loss`) scripts | [python/README.md](python/README.md) |
| [cpp/](cpp) | [`train.cpp`](cpp/train.cpp) and [`CMakeLists.txt`](cpp/CMakeLists.txt) linking `libsdca::libsdca` | [docs/CPP.md](../docs/CPP.md) |
| [c/](c) | [`train.c`](c/train.c) and [`CMakeLists.txt`](c/CMakeLists.txt) linking `libsdca::sdca_c` | [docs/C.md](../docs/C.md) |

Rust and MATLAB examples live with their packages:
[`rust/examples/train.rs`](../rust/examples/train.rs) and
[`matlab/demo.m`](../matlab/demo.m).

## Running

```bash
# Python
pip install 'libsdca[torch,jax,sklearn]'
python examples/python/torch_drop_in_cross_entropy.py

# C++ and C (from a local install prefix)
cmake -S . -B build/release -DSDCA_BUILD_C_API=ON -DSDCA_BLAS_BACKEND=BUILTIN \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/install"
cmake --build build/release --target install
cmake -S examples/cpp -B build/example-cpp -DCMAKE_PREFIX_PATH="$PWD/build/install"
cmake --build build/example-cpp && build/example-cpp/sdca_example
cmake -S examples/c -B build/example-c -DCMAKE_PREFIX_PATH="$PWD/build/install"
cmake --build build/example-c && build/example-c/sdca_c_example
```
