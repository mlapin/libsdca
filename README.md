# libsdca

libsdca is a library for multiclass classification based on stochastic dual coordinate ascent (SDCA).

Below is a brief overview of supported training objectives, inputs, proximal operators, and interfaces.

**Multiclass objectives**:
- Multiclass SVM of Crammer and Singer [1]
- Multiclass SVM with smoothed hinge loss
- Top-k Multiclass SVM [4]
  - top-k hinge loss alpha (non-smooth and smooth)
  - top-k hinge loss beta (non-smooth and smooth)
- Multiclass cross-entropy loss (softmax loss)
- Top-k entropy loss [5]

**Multilabel objectives**:
- Multilabel SVM of Crammer and Singer [2]
- Multilabel SVM with smoothed hinge loss
- Multilabel cross-entropy loss

**Input data**:
- features
- kernels
- float or double precision
- multiple datasets at once (e.g. to monitor performance on a validation set)

**Proximal operators** and more (e.g. compute projections onto various sets):
- simplex (implements the algorithm of [3])
- top-k simplex
- top-k cone
- entropic projections
- Lambert W function of the exponent (computes W(exp(x))
- Further details: `matsdca_prox('help','prox')`

**Interfaces**:
- C++11 headers (simply include and use; no additional libraries to compile and link)
- Matlab mex files
- Python support is only partial at the moment, see PR #2

For the *truncated top-k entropy loss* [5], see `libsdca_gd` in a previous release [v0.2.0](https://github.com/mlapin/libsdca/releases/tag/v0.2.0).


## Installation instructions

libsdca can be installed on Linux and OS X in two ways.

- Using a make script in Matlab:
```
run matlab/make.m
```

- Using CMake (requires [CMake](https://cmake.org) version >= 2.8.9):
```
mkdir build && cd build && cmake .. && make install -j4
```

The Matlab interface is installed in the directory `matlab` and provides two mex files:
- `matsdca_fit` - solvers;
- `matsdca_prox` - proximal operators.

The directory should be added to the Matlab search path with `addpath matlab`.


## Quick start

Using the library is as easy as
```
model = matsdca_fit(X, Y, opts);
```

A quick demo script is available at `matlab/demo.m`.

Type `matsdca_fit('help')` and `matsdca_prox('help')` for help.


## References and citation

Please cite libsdca in your publications if it helps your research.
- Top-k Multiclass SVM (top-k hinge loss alpha and beta):
```
@inproceedings{lapin2015topk,
  title = {Top-k Multiclass {SVM}},
  author = {Lapin, Maksim and Hein, Matthias and Schiele, Bernt},
  booktitle = {NIPS},
  year = {2015}
}
```
- Smooth top-k losses and cross-entropy losses:
```
@inproceedings{lapin2016loss,
  title = {Loss Functions for Top-k Error: Analysis and Insights},
  author = {Lapin, Maksim and Hein, Matthias and Schiele, Bernt},
  booktitle = {CVPR},
  year = {2016}
}
```

**References**

<sup>[1] K. Crammer and Y. Singer. On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines. In JMLR, 2001.</sup>  
<sup>[2] K. Crammer and Y. Singer. A family of additive online algorithms for category ranking. In JMLR, 2003.</sup>  
<sup>[3] K.C. Kiwiel. Variable fixing algorithms for the continuous quadratic knapsack problem. In JOTA, 2008.</sup>  
<sup>[4] M. Lapin, M. Hein, and B. Schiele. Top-k multiclass SVM. In NIPS, 2015.</sup>  
<sup>[5] M. Lapin, M. Hein, and B. Schiele. Loss Functions for Top-k Error: Analysis and Insights. In CVPR, 2016.</sup>  
