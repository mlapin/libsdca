# libsdca

libsdca is a library for multiclass classification based on stochastic dual coordinate ascent (SDCA).

Objectives:
- Multiclass softmax loss
- Multiclass SVM of Crammer and Singer [1]
- Top-k Multiclass SVM
  - hinge-of-top-k and top-k-of-hinge (the latter is an instance of the OWPC loss of [2])
  - non-smooth and smoothed losses
- Top-k Entropy loss

Inputs:
- float or double precision
- features directly (primal)
- kernels (dual)
- multiple datasets at once (e.g. to monitor performance on a validation set)

Proximal operators (compute projections onto various sets):
- simplex (implements the algorithm of [3])
- top-k simplex, top-k cone
- entropic projections
- Lambert W function of the exponent (computes W(exp(x)), not a prox operator)

Interfaces:
- C++ headers (no additional libraries to link)
- Matlab

[1] K. Crammer and Y. Singer. On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines. In JMLR, 2001.  
[2] N. Usunier, D. Buffoni, and P. Gallinari. Ranking with ordered weighted pairwise classification. In ICML, 2009.  
[3] K.C. Kiwiel. Variable fixing algorithms for the continuous quadratic knapsack problem. In JOTA, 2008.

## Installation instructions

libsdca can be installed on Linux and OS X in two ways.

- Using a make script in Matlab:
```
run matlab/make.m
```

- Using CMake (requires [CMake](https://cmake.org) version >= 2.8.9):
```
mkdir build && cd build && cmake .. && make install -j2
```

Once libsdca is compiled, add the corresponding directory to the Matlab search path:
```
addpath matlab
```

## Quick start

The Matlab interface is installed to the directory `matlab`, which should contain two mex files:

- `libsdca_prox` provides proximal operators;
- `libsdca_solve` provides solvers for multiclass classification.

There is also `libsdca_gd` which is not officially a part of libsdca and implements a simple batch gradient descent for the (nonconvex) truncated top-k softmax loss.

#### Examples

To train the Multiclass SVM of Crammer and Singer on some random data, run
```
model = libsdca_solve(randn(2,15),randi(3,15,1))
```

To check the top-k training accuracies, see
```
model.evals(end).accuracy
```

To train the Top-k Multiclass SVM, specify the corresponding objective and the k
```
model = libsdca_solve(randn(2,15),randi(3,15,1),struct('objective','topk_svm','k',2))
```

Type `libsdca_prox('help')` and `libsdca_solve('help')` for further information.

## Citation

Please cite libsdca in your publications if it helps your research:
```
@inproceedings{lapin2015topk,
  title = {Top-k Multiclass {SVM}},
  author = {Lapin, Maksim and Hein, Matthias and Schiele, Bernt},
  booktitle = {NIPS},
  year = {2015}
}
```
