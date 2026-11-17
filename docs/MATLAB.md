# MATLAB and Octave interface

[All guides](https://github.com/mlapin/libsdca/blob/master/docs/README.md) ·
[Shared API reference](https://github.com/mlapin/libsdca/blob/master/docs/API.md)

Three MEX gateways wrap the C++ headers:

| Function | Purpose |
| --- | --- |
| `matsdca_fit` | Train a linear or kernel classifier on a training objective. |
| `matsdca_loss` | Evaluate loss values and score gradients. |
| `matsdca_prox` | Apply proximal operators and projections (simplex, top-*k* cone, entropic). |

They take the loss and option names shared by every interface ([API.md](API.md)); an
unknown option field, or a parameter that the loss does not take, is an error. Labels are
1-based. `matsdca_fit('help')`, `matsdca_loss('help')` and `matsdca_prox('help')` print the
full reference.

## Installation

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

## Quick start

```matlab
X = [1 0 -1; 0 1 -1];       % features-by-examples
Y = [1 2 3];                % labels start at 1
model = matsdca_fit(X, Y);
[~, predicted] = max(model.W' * X);
disp(model.status);
disp(predicted);
```

## Training

- **Features:** `X` is column-major `d`-by-`n` (`single` or `double`); a Gram matrix is
  `n`-by-`n` with `opts.kernel = true`.
- **Single-label losses** take a `double` vector of labels in `1:m`.
- **Multilabel losses** take an `m`-by-`n` sparse (`double` or `logical`) or `logical` label
  matrix whose nonzero entries mark each example's labels.

```matlab
model = matsdca_fit(X, Y);                                        % cross_entropy, C = 1
model = matsdca_fit(X, Y, struct('loss', 'topk_hinge_beta', 'k', 2, 'C', 10));
scores = model.W' * Xtest;            % m-by-n_test
[~, predicted] = max(scores);
```

| Option | Default | Meaning |
| --- | --- | --- |
| `loss` | `'cross_entropy'` | `cross_entropy`, `topk_cross_entropy`, `multiclass_hinge`, `topk_hinge_alpha`, `topk_hinge_beta`, `multilabel_hinge`, `multilabel_cross_entropy` |
| `C` | `1` | Loss weight in $0.5\Vert W - W_0 \Vert_2^2 + C \sum_i \ell_i$ |
| `k` | required | Top-*k* losses only: an integer in `1:m-1` |
| `gamma` | `0` | Hinge losses only: `0` nonsmooth, `> 0` smoothed |
| `num_classes` | `max(Y)` | Number of classes `m` of single labels (label matrices have `m` rows) |
| `kernel` | `false` | `X` is an `n`-by-`n` Gram matrix (test sets: `Xtrain' * Xtest`) |
| `tol` | `1e-3` | Relative duality gap of `'solved'` (`0`: never solved) |
| `max_epoch` | `1000` | Epoch limit |
| `eval_epoch` | `10` | Epochs between evaluations (`0`: only at the end) |
| `eval_on_start` | `false` | Also evaluate before the first epoch |
| `recompute_primal` | `true` | Rebuild $W = W_0 + X A^\top$ at each evaluation |
| `shrinking` | `'auto'` | Hinge losses only: `'auto'`, `'off'`, `'heuristic'`, `'safe'`, `'both'` ([API.md](API.md#shrinking)) |
| `projection_precision` | `'float64'` | `multilabel_hinge` only: `'float64'` or `'extended'` ([API.md](API.md#projection-precision)) |
| `seed` | `0` | Order of the examples in the epochs |
| `num_threads` | `SDCA_NUM_THREADS`, else the CPUs | Threads of the evaluations (the epochs are sequential) |
| `precision` | class of `X` | `'float32'` (`single` data only) or `'float64'` (`single` data keeps `double` iterates) |
| `initial_dual` | zeros | `m`-by-`n` duals to start from, e.g. `model.A`; checked to be feasible |
| `center` | zeros | `d`-by-`m` center $W_0$ of the regularizer (features only) |
| `log_level` | `'none'` | `'none'`, `'warning'`, `'info'`, `'verbose'`, `'debug'` |
| `log_format` | `'short_e'` | `'short_f'`, `'short_e'`, `'long_f'`, `'long_e'` |

A warm start (`initial_dual`) must be feasible for the loss (each column sums to zero,
nonpositive entries off the labels, and the loss's box and top-*k* constraints, to rounding);
a single-label column gets the label entry that makes it sum to zero exactly, and $W$
starts at $W_0 + X A^\top$, computed in `double`.

Passing cell arrays `{Xtrn, Xval}` and `{Ytrn, Yval}` trains on the first data set and
evaluates every data set at each evaluation (`model.test`).

### Results

| Field | Meaning |
| --- | --- |
| `W`, `A` | `d`-by-`m` weights (features only), `m`-by-`n` duals; both keep the class of `X` |
| `loss`, `C`, and option fields | Configuration used by the fit, where applicable |
| `num_classes`, `num_examples`, `num_features` | Problem dimensions |
| `status`, `epochs` | Stop reason and completed epochs; only `'solved'` guarantees the gap |
| `relative_gap`, `cpu_time`, `wall_time` | Final gap and elapsed time |
| `history` | One entry per evaluation; shared [history fields](API.md#results) |
| `test` | With cell-array inputs, one row per evaluation and one column per further dataset, holding `accuracy` and `primal_loss` |

### Precision and errors

Storage follows `X` (`single` or `double`). `precision='float64'` keeps double working
iterates for single input; see [precision](API.md#precision). Invalid inputs raise
`LIBSDCA:*` errors. A fit can return a model with `status='failed'` after a numerical
failure; always inspect `status` before relying on convergence.

## Losses and gradients

```matlab
% Scores: classes-by-examples. Values: 1-by-examples.
[values, gradients] = matsdca_loss(scores, Y);                       % cross_entropy
[values, gradients] = matsdca_loss(scores, Y, ...
    struct('loss', 'topk_hinge_alpha', 'k', 2, 'gamma', 0.7));
[values, gradients] = matsdca_loss(student, teacher, ...
    struct('loss', 'distillation', 'temperature', 4));
```

`matsdca_loss` takes every loss of [API.md](API.md#loss-definitions) and the options `loss`,
`k`, `gamma`, `alpha`, `temperature`, `projection_precision`, `precision` and
`num_threads`, each only where the loss takes it.

| Losses | Targets |
| --- | --- |
| Single-label losses, `proximal_cross_entropy` | 1-based `double` labels |
| Multilabel losses | An `m`-by-`n` `logical` or sparse label matrix (`multilabel_hinge` needs a negative class in every example) |
| `distillation`, `proximal_distillation` | Teacher logits of the size and class of the scores |

`precision` defaults to the class of the scores (`'float32'` for `single`);
`proximal_distillation` computes in `double`. The gradients have the class of the scores.

## Proximal operators

```matlab
projected = matsdca_prox(A, struct('prox', 'knapsack'));         % columns onto the simplex
projected = matsdca_prox(A, struct('prox', 'topk_entropy_biased', 'k', 2));
```

`matsdca_prox(A, opts)` applies `opts.prox` to each column of `A` and returns the result;
`A` is not modified. It takes every operator of
[API.md](API.md#projections-and-proximal-operators) except `two_entropy`, and the
elementwise `lambert_w_exp`. `knapsack` (the default) is a synonym of `knapsack_eq`.
`matsdca_prox('help', 'prox')` prints the problems.

| Option | Default | Meaning |
| --- | --- | --- |
| `prox` | `'knapsack'` | The operator |
| `lo`, `hi` | `0`, `1` | Bounds of the knapsack and entropy operators (`lo <= hi`, finite) |
| `rhs` | `1` | Right-hand side of the sum constraint (nonnegative, except for the knapsack operators) |
| `rho` | `1` | Weight of the biased operators (nonnegative) |
| `alpha` | `1` | Weight of the quadratic term in `topk_entropy_biased` (positive) |
| `k` | `1` | The `k` of the top-*k* operators, in `1:m` for `m` rows |
| `p` | `1` | Length of the first block in `two_simplex` and `two_simplex_sort`, in `1:m-1` |
| `precision` | class of `A` | `'float32'`, `'float64'` or `'long_double'` working arithmetic; the output keeps the class of `A` |

`A` must be finite; an unknown option is an error.

## Benchmarks

The [solver benchmarks](https://github.com/mlapin/libsdca/blob/master/docs/PYTHON.md#benchmarks)
compare the shared solver through Python with scikit-learn. They help choose between
feature and kernel inputs; they do not measure this binding's call overhead.
No separate end-to-end timings for this interface are published here.
