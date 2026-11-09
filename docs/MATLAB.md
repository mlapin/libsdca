# MATLAB and Octave interface

Three MEX gateways wrap the C++ headers:

| Function | Purpose |
| --- | --- |
| `matsdca_fit` | Train a linear or kernel classifier with any objective. |
| `matsdca_loss` | Evaluate loss values and score gradients. |
| `matsdca_prox` | Apply proximal operators and projections (simplex, top-*k* cone, entropic). |

## Building

From MATLAB in the repository root:

```matlab
run matlab/make.m
addpath matlab
```

Or with CMake (installs MEX binaries into `<prefix>/matlab`):

```bash
cmake -S . -B build -DSDCA_BUILD_MATLAB=ON -DCMAKE_INSTALL_PREFIX=$PWD/build/install
cmake --build build --target install
```

With GNU Octave (compiles with `mkoctfile` and runs `test_numerics.m` and `demo.m`):

```bash
OCTAVE_HOME=<prefix> tests/matlab/run_octave_tests.sh
```

## Training

Features `X` are column-major `d`-by-`n`; labels `Y` are `1:m` (or `0:m-1` when class `0`
is present). Multilabel objectives take a sparse `double` or `logical` `m`-by-`n` label
matrix.

```matlab
opts = struct('objective', 'cross_entropy', 'C', 1, 'tol', 1e-6);
model = matsdca_fit(X, Y, opts);
scores = model.W' * Xtest;            % m-by-n_test
[~, predicted] = max(scores);
```

`opts` accepts the shared options of [API.md](API.md#options) (`objective`, `C`, `k`,
`gamma`, `tol`, `max_epoch`, `eval_epoch`, `eval_on_start`, `recompute_primal`,
`shrinking`, `projection_precision`, `kernel`, `center`, `initial_dual`, `precision`), plus
`max_cpu_time` and `max_wall_time` (seconds, `0` = unlimited), `log_level` (`'none'`,
`'warning'`, `'info'`, `'verbose'`, `'debug'`), and `log_format` (`'short_f'`,
`'short_e'`, `'long_f'`, `'long_e'`). Setting `eval_epoch = 0` evaluates only at the end.

Passing cell arrays `{Xtrn, Xval}` and `{Ytrn, Yval}` trains on the first dataset and
evaluates all datasets at each check (`model.test`). Passing `opts.initial_dual = model.A`
warm-starts from a previous dual solution.

The returned `model` struct contains `W` (`d`-by-`m`, feature input only), `A` (`m`-by-`n`),
`status` (only `'solved'` certifies the gap), `epoch`, `relative_gap`, `cpu_time`,
`wall_time`, `train` (evaluation history), `test`, and the options used
(`matsdca_fit('help')` prints full usage).

## Losses and gradients

```matlab
% Scores: classes-by-examples. Values: 1-by-examples.
[values, gradients] = matsdca_loss(scores, Y, struct('loss', 'cross_entropy'));
[values, gradients] = matsdca_loss(scores, Y, ...
    struct('loss', 'topk_hinge_alpha', 'k', 2, 'gamma', 0.7));
```

`matsdca_loss` accepts the loss names and options of [API.md](API.md#objectives-and-losses)
(`loss`, `k`, `gamma`, `alpha`, `precision`, `projection_precision`, `num_threads`).
Targets `Y` are 1-based labels (`double`) or a `logical` `m`-by-`n` mask. Returned
`gradients` match the `single` or `double` class of `scores`.

## Proximal operators

```matlab
projected = matsdca_prox(A, struct('prox', 'knapsack'));         % columns onto the simplex
projected = matsdca_prox(A, struct('prox', 'topk_entropy_biased', 'k', 2));
```

`matsdca_prox('help')` lists all supported operators (`knapsack`, `two_simplex`,
`topk_simplex`, `topk_cone`, `entropy`, `topk_entropy`, `lambert_w_exp`, ...). `opts.precision`
selects working arithmetic (`'float32'` / `'single'`, `'float64'` / `'double'`, or
`'long_double'`) independently of the input array class.
