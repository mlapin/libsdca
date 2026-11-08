# MATLAB and Octave interface

Three MEX gateways wrap the C++ headers:

| Function | Purpose |
| --- | --- |
| `matsdca_fit` | Train a model with any objective on features or a Gram matrix. |
| `matsdca_loss` | Loss values and gradients with respect to the scores. |
| `matsdca_prox` | Proximal operators and projections (simplex, top-k cone, entropic, ...). |

## Building

From MATLAB, in the repository root:

```matlab
run matlab/make.m
addpath matlab
```

or with CMake, which installs the MEX files into `<prefix>/matlab`:

```bash
cmake -S . -B build -DSDCA_BUILD_MATLAB=ON -DCMAKE_INSTALL_PREFIX=$PWD/build/install
cmake --build build --target install
```

GNU Octave builds and tests the same gateways without MATLAB:

```bash
OCTAVE_HOME=<prefix> tests/matlab/run_octave_tests.sh
```

This compiles the MEX files with `mkoctfile` (with the `-fno-fast-math
-ffp-contract=off` of `make.m` and CMake) and runs `tests/matlab/test_numerics.m`
and `matlab/demo.m` under `octave-cli`. It links a system CBLAS when one is
found and the built-in BLAS otherwise.

## Training

Data are column-major: features are `d`-by-`n`, labels are `1:m` (or `0:m-1`).
The base is detected from each label vector: labels whose smallest value is 1
are one-based, so zero-based labels must include class 0 (otherwise every
label shifts down by one and the classes lose one). Test labels of a cell
array are detected separately; give them in the training base, with every
class present, or shift them to one-based labels first.

```matlab
opts = struct('objective', 'cross_entropy', 'C', 1, 'tol', 1e-6);
model = matsdca_fit(X, Y, opts);
scores = model.W' * Xtest;            % m-by-n_test
[~, predicted] = max(scores);
```

`opts` takes the names of the other interfaces (`objective`, `C`, `k`, `gamma`,
`tol`, `max_epoch`, `eval_epoch`, `eval_on_start`, `kernel`, `center`,
`initial_dual`, `recompute_primal`, `projection_precision`, `precision`) as well
as the historical ones (`c`, `epsilon`, `is_dual`, `W0`, `A`). MATLAB adds
`max_cpu_time` and `max_wall_time` (seconds, 0: no limit), `log_level`
(`'none'`, `'warning'`, `'info'` (default), `'verbose'`, `'debug'`) and
`log_format` (`'short_f'`, `'short_e'` (default), `'long_f'`, `'long_e'`). For feature input with `W0`, history and
stopping use the full centered objective `0.5*||W-W0||^2 + C*sum(loss_i)`;
no center-dependent constant is omitted. The objective names are those of [API.md](API.md#objectives-and-losses),
plus the historical v0.2.0 names `softmax`, `msvm`, `msvm_smooth`,
`l2_hinge_topk`, ... (`matsdca_fit('help', 'objective')` maps them). The shared
names smooth a hinge only with `gamma > 0` (default 0); the historical names
keep their default `gamma = 1`. `model.objective` is the name as given. Unlike Python and Rust, `eval_epoch = 0` is accepted and evaluates
only at the end. A cell array of data matrices and labels trains on the first
and reports the others at every evaluation (e.g. a validation set):

```matlab
model = matsdca_fit({Xtrn, Xval}, {Ytrn, Yval}, opts);
fprintf('validation accuracy %.2f\n', 100 * model.test(end, 1).accuracy);
```

Multilabel objectives take a sparse double or logical `m`-by-`n` label
matrix; its row count is the number of classes, so classes without positives
keep their rows. For a Gram matrix set `opts.kernel = true` and predict with
`model.A * Ktrn_test`.

`initial_dual` (`m`-by-`n`, e.g. `model.A`) resumes a fit. For features the
weights are rebuilt as `W = center + X * initial_dual'`; a given `opts.W` is
only checked for its shape. The duals are checked for shape and finiteness,
not for feasibility.

```matlab
model = matsdca_fit(X, Y, opts);
opts.initial_dual = model.A;
model = matsdca_fit(X, Y, opts);      % continues from the previous solution
```
The returned `model` holds `W` (`d`-by-`m`, features only) and `A`
(`m`-by-`n`); `status` (only `'solved'` certifies the gap), `epoch`,
`cpu_time`, `wall_time` and `relative_gap`; the options used (`objective`,
`c`, `k`, `gamma`, `epsilon`, `max_epoch`, ..., `projection_precision`,
`precision`, `data_precision`); `train`, one entry per evaluation (`epoch`,
`accuracy` or `rank_loss`, `relative_gap`, `primal`, `dual`, `primal_loss`,
`dual_loss`, the regularizers and times); and `test`, one row per evaluation
and one column per further data set.

`matsdca_fit('help')`, `matsdca_fit('help', 'objective')` and
`matsdca_fit('help', 'data')` print the full reference.

## Losses and gradients

```matlab
% Scores: classes-by-examples. Values: 1-by-examples.
% Gradients have the same shape and single/double type as the scores.
[values, gradients] = matsdca_loss(scores, Y, struct('loss', 'cross_entropy'));
[values, gradients] = matsdca_loss(scores, Y, ...
    struct('loss', 'topk_hinge_alpha', 'k', 2, 'gamma', 0.7));
```

`matsdca_loss` accepts the loss names of [API.md](API.md#objectives-and-losses),
including `multilabel_cross_entropy` (entropy with a mask), with the options `loss`
(or `objective`), `k`, `gamma`, `alpha`, `precision` (the working arithmetic
of single scores; double scores compute in double), `projection_precision`
and `num_threads` (by default `SDCA_NUM_THREADS`, else 0: automatic;
[API.md](API.md#threads)); `matsdca_loss('help')` lists them.
`opts.projection_precision` is `'float64'` (default: double with a checked
result) or `'extended'` (always the extended-precision solver) and acts on the
`multilabel_hinge` projections; `matsdca_fit` reports it as
`model.projection_precision` ([API.md](API.md#projection-precision)). Targets are one-based labels
(double) or a full logical classes-by-examples mask; entropy with a mask puts
equal target probability on the positive classes. The gradients are with
respect to the scores, without weight regularization; nonsmooth hinges return
subgradients.

## Proximal operators

```matlab
projected = matsdca_prox(A, struct('prox', 'knapsack'));         % columns on the simplex
projected = matsdca_prox(A, struct('prox', 'topk_entropy_biased', 'k', 2));
```

`matsdca_prox('help')` and `matsdca_prox('help', 'prox')` list the operators
and their options, including `two_simplex` (and `two_simplex_sort`), which
projects each column `[a; b]` (`a` of length `opts.p`) onto
`<1, x> = <1, y> <= rhs`, `x, y >= 0`. The output argument is required; the input is never
modified. Scalar options must be finite real scalars and integer options
integral.
`opts.precision` selects working arithmetic independently of the input/output
class for `matsdca_prox`: `float32`, `fp32`, `single` select
float32 work (the ambiguous `float` is rejected); `float64`, `fp64`, `double` select float64 work (the default).
The prox gateway also retains its explicit `long_double` / `long double`
extension. Returned projections keep the input's `single` or `double` class.
These float32/float64 aliases also apply to the fit and loss gateways.

## Tests

`tests/matlab/test_numerics.m` checks the loss gradients against finite differences,
extreme scores, option aliases, objective names, warm starts, sparse targets
and input validation. `matlab/demo.m` trains on
synthetic data. `tests/matlab/check_mex_compiles.sh` compiles the gateways
against a stub `mex.h` when neither MATLAB nor Octave is available.
