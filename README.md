# libsdca

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

libsdca is a library for multiclass classification based on stochastic dual coordinate ascent (SDCA).

Features

- Matlab interface
- Top-k Multiclass SVM
- Multiclass SVM of Crammer and Singer
- Proximal operators including efficient projections onto the unit simplex and the top-k simplex

The library is currently in active development and more features are planned.

## Installation instructions

libsdca can be installed on Linux and OS X by an automatic CMake build.

```
mkdir build
cd build
cmake ..
make install -j2
```

libsdca requires [CMake](https://cmake.org) version >= 2.8.9.

## Quick start

The Matlab interface is installed to `build/matlab`, which should contain two mex files:

- `libsdca_prox` provides proximal operators;
- `libsdca_solve` provides solvers for multiclass classification.

To train the Multiclass SVM of Crammer and Singer on some random data, run
```
model = libsdca_solve(randn(2,15),randi(3,15,1))
```
(note that `libsdca_solve` should be in a folder on the Matlab search path).

To check the top-k training accuracies, see
```
model.evals(end).accuracy
```

To train the Top-k Multiclass SVM, specify the corresponding objective and the k
```
model = libsdca_solve(randn(2,15),randi(3,15,1),struct('objective','topk_svm','k',2))
```

Type `libsdca_prox('help')` and `libsdca_solve('help')` for further information.

## License and Citation

libsdca is released under the [MIT license](https://github.com/mlapin/libsdca/blob/master/LICENSE).

Please cite libsdca in your publications if it helps your research:
```
@inproceedings{lapin2015topk,
  title = {Top-k Multiclass {SVM}},
  author = {Lapin, Maksim and Hein, Matthias and Schiele, Bernt},
  booktitle = {NIPS},
  year = {2015}
}
```
