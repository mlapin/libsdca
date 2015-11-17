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

## License and Citation

libsdca is released under the [MIT license](https://github.com/mlapin/libsdca/blob/master/LICENSE).

Please cite libsdca in your publications if it helps your research:

    @inproceedings{lapin2015topk,
      title = {Top-k Multiclass {SVM}},
      author = {Lapin, Maksim and Hein, Matthias and Schiele, Bernt},
      booktitle = {NIPS},
      year = {2015}
    }
