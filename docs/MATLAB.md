# MATLAB and Octave interface

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

Features `X` are column-major `d`-by-`n` (`single` or `double`); a Gram matrix is `n`-by-`n`
with `opts.kernel = true`. The single-label losses take a `double` vector of labels in
`1:m`; the multilabel losses take an `m`-by-`n` sparse (`double` or `logical`) or `logical`
label matrix whose nonzero entries mark each example's labels.

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

The returned `model` struct holds `W` (`d`-by-`m`, features only) and `A` (`m`-by-`n`) in the
class of `X`; the loss and options used (`loss`, `C`, `k`, `gamma`, `kernel`, `tol`,
`max_epoch`, `eval_epoch`, `eval_on_start`, `recompute_primal`, `shrinking`,
`projection_precision`, `seed`, `num_threads`, `precision`, each where it applies);
`num_classes`, `num_examples` and `num_features`; `status` (`'solved'`, `'no_progress'`,
`'max_epoch'` or `'failed'`; only `'solved'` guarantees the gap), `epochs`,
`relative_gap`, `cpu_time` and `wall_time`; `history`, one entry per evaluation with the
fields of [API.md](API.md#results) (`epoch`, `primal`, `dual`, `primal_loss`, `dual_loss`,
`regularizer`, `relative_gap`, `accuracy` (the rank loss of the multilabel losses),
`cpu_time`, `wall_time`); and with cell arrays `test`, one row per evaluation and one column
per further data set (`accuracy`, `primal_loss`).

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
`num_threads`, each only where the loss takes it. Targets: 1-based `double` labels for the
single-label losses and `proximal_cross_entropy`; an `m`-by-`n` `logical` or sparse label
matrix for the multilabel losses (`multilabel_hinge` needs a negative class in every
example); teacher logits of the size and class of the scores for `distillation` and
`proximal_distillation`. `precision` defaults to the class of the scores (`'float32'` for
`single`); `proximal_distillation` computes in `double`. The gradients have the class of
the scores.

## Proximal operators

```matlab
projected = matsdca_prox(A, struct('prox', 'knapsack'));         % columns onto the simplex
projected = matsdca_prox(A, struct('prox', 'topk_entropy_biased', 'k', 2));
```

`matsdca_prox('help', 'prox')` lists all supported operators (`knapsack`, `two_simplex`,
`topk_simplex`, `topk_cone`, `entropy`, `topk_entropy`, `lambert_w_exp`, ...). Its options
are `prox`, `lo`, `hi`, `rhs`, `rho`, `alpha`, `k`, `p` and `precision`: `'float32'`,
`'float64'` or `'long_double'` working arithmetic, by default the class of `A`; the output
keeps the class of `A`.
