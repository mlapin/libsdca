#include <cstdint>
#include <memory>

#include "mex_util.h"
#include "sdca/solver.h"

#ifndef MEX_SOLVE
#define MEX_SOLVE "mex_solve"
#endif
#ifndef LIBSDCA_VERSION
#ifdef SDCA_VERSION
#define LIBSDCA_VERSION SDCA_VERSION
#else
#define LIBSDCA_VERSION "0.0.0"
#endif
#endif

using namespace sdca;

static constexpr char default_loss[] = "cross_entropy";

inline void printUsage() {
  mexPrintf(
      "Usage: model = %s(X, Y, opts);\n"
      "  See %s('help') and %s('version') for more information.\n",
      MEX_SOLVE, MEX_SOLVE, MEX_SOLVE);
}

inline void printVersion() { mexPrintf("%s version %s.\n", MEX_SOLVE, LIBSDCA_VERSION); }

inline void printHelp(const mxArray* opts) {
  if (opts == nullptr) {
    mexPrintf(
        "Usage: model = %s(X, Y, opts);\n"
        "  Minimizes 0.5 ||W - center||^2 + C sum_i loss(W' x_i, y_i) by SDCA.\n"
        "\n"
        "  opts is a struct with the following fields (defaults in [brackets]);\n"
        "  an unknown field, or a parameter the loss does not take, is an error:\n"
        "\n"
        "    loss       ['%s'] - the training objective (%s('help', 'loss'));\n"
        "    C          [1]       - the loss weight, finite and positive;\n"
        "    k          [none]    - the k of the top-k losses (required), in 1:m-1;\n"
        "    gamma      [0]       - the smoothing of the hinge losses (0: nonsmooth);\n"
        "    num_classes [max(Y)] - the number of classes m (single-label Y);\n"
        "    kernel     [false]   - whether X is an n-by-n Gram matrix;\n"
        "\n"
        "    tol        [1e-3]    - the relative duality gap of 'solved':\n"
        "      (primal - dual) <= tol * max(abs(primal), abs(dual)); 0: never solved;\n"
        "    max_epoch  [1000]    - the epoch limit;\n"
        "    eval_epoch [10]      - evaluate every eval_epoch epochs (0: only at the end);\n"
        "                           up to two more fall where the gap's trend reaches tol;\n"
        "    eval_on_start [false] - also evaluate before the first epoch;\n"
        "    recompute_primal [true] - rebuild W = center + X * A' at each evaluation\n"
        "                           (false keeps the incrementally updated W);\n"
        "    shrinking  ['auto']  - the hinge losses only: what the epochs leave out:\n"
        "                           'heuristic' (skip unchanged examples between full\n"
        "                           passes), 'safe' (gap-safe screening), 'both', 'off';\n"
        "                           'auto' is 'both' (see docs/API.md);\n"
        "    projection_precision ['float64'] - multilabel_hinge only: 'float64' solves\n"
        "                           its projections in double with a checked result,\n"
        "                           re-solved in extended precision if the check fails;\n"
        "                           'extended' always in extended precision;\n"
        "    seed       [0]       - the order of the examples in the epochs;\n"
        "    num_threads [SDCA_NUM_THREADS, else the CPUs] - threads of the evaluations\n"
        "                           (the epochs are sequential); results do not depend on it;\n"
        "    precision  [class of X] - working precision: 'float32' (single X only) or\n"
        "                           'float64' (single X keeps double iterates);\n"
        "    log_level  ['none']  - 'none', 'warning', 'info', 'verbose', 'debug';\n"
        "    log_format ['short_e'] - 'short_f', 'short_e', 'long_f', 'long_e';\n"
        "\n"
        "    initial_dual [zeros] - m-by-n dual variables to start from, e.g. model.A;\n"
        "                           they must be feasible for the loss; for features W\n"
        "                           starts at center + X * A' (computed in double);\n"
        "    center     [zeros]   - d-by-m center of the regularizer (features only).\n"
        "\n"
        "  The model struct holds W (d-by-m, features only) and A (m-by-n); the loss\n"
        "  and options used (loss, C, k, gamma, tol, ...); num_classes, num_examples\n"
        "  and num_features; status ('solved', 'no_progress', 'max_epoch' or 'failed';\n"
        "  only 'solved' guarantees the gap), epochs, relative_gap, cpu_time and\n"
        "  wall_time; history, one entry per evaluation (epoch, primal, dual,\n"
        "  primal_loss, dual_loss, regularizer, relative_gap, accuracy (the rank loss\n"
        "  of the multilabel losses), cpu_time, wall_time); and with cell array\n"
        "  data, test: one row per evaluation and one column per further data set\n"
        "  (accuracy, primal_loss).\n"
        "\n"
        "  Prediction scores: scores = model.W' * Xtest, or model.A * Ktest for a\n"
        "  Gram matrix Ktest = Xtrain' * Xtest.\n"
        "\n"
        "  See %s('help', 'data') for the data and label formats.\n",
        MEX_SOLVE, default_loss, MEX_SOLVE, MEX_SOLVE);
    return;
  }

  std::string arg = mxGetString(opts, "help argument");
  if (arg == "data" || arg == "labels" || arg == "input") {
    mexPrintf(
        "Input data is given in the first two arguments, X and Y.\n"
        "\n"
        "  X can be:\n"
        "    - a d-by-n feature matrix (d features, n examples);\n"
        "    - an n-by-n Gram matrix (opts.kernel = true);\n"
        "    - a cell array of feature matrices or of Gram matrices, but not a mix:\n"
        "      the first trains, the others are evaluated at each evaluation (e.g. a\n"
        "      validation set). Test Gram matrices are Xtrain' * Xtest\n"
        "      (n_train-by-n_test).\n"
        "\n"
        "  Y can be:\n"
        "    - a vector of n class labels in 1:m (double), for the single-label losses;\n"
        "    - an m-by-n sparse (double or logical) or logical matrix, for the\n"
        "      multilabel losses: nonzero entries mark the labels of each example;\n"
        "    - a cell array of the above, one per data set.\n"
        "\n"
        "  X must be dense, finite, single or double.\n");
  } else if (arg == "loss" || arg == "losses") {
    mexPrintf(
        "opts.loss - the training objective (L2-regularized); the hinges are\n"
        "  smoothed by gamma > 0 (gamma = 0: nonsmooth, the default).\n"
        "  Single-label (labels in 1:m):\n"
        "    cross_entropy      - multiclass cross-entropy (softmax) loss\n"
        "    topk_cross_entropy - top-k entropy loss (k)\n"
        "    multiclass_hinge   - multiclass SVM of Crammer and Singer (gamma)\n"
        "    topk_hinge_alpha   - top-k hinge alpha: hinge of the top-k average (k, gamma)\n"
        "    topk_hinge_beta    - top-k hinge beta: average of the top-k hinges (k, gamma)\n"
        "  Multilabel (label sets):\n"
        "    multilabel_hinge         - multilabel SVM of Crammer and Singer (gamma,\n"
        "                               projection_precision)\n"
        "    multilabel_cross_entropy - cross-entropy to the uniform target on the labels\n"
        "\n"
        "  Default value:\n"
        "    %s\n",
        default_loss);
  } else {
    mex_error(err_id[err_help_arg], err_msg[err_help_arg], arg.c_str());
  }
}

// The options of matsdca_fit, read and checked before anything is computed.
struct fit_options {
  std::string loss;
  int id = 0;
  double C = 1;
  size_type k = 0;
  double gamma = 0;
  size_type num_classes = 0;  // 0: not given
  bool kernel = false;
  bool num_threads_given = false;
  stopping_criteria criteria;
};

// opts.(name) as a logical or numeric 0/1 scalar.
inline bool flag_option(const mxArray* opts, const char* name, const bool value) {
  return mxGetFieldValueOrDefault<double>(opts, name, value ? 1 : 0) != 0;
}

inline fit_options read_fit_options(const mxArray* opts) {
  check_option_names(opts,
                     {"loss",
                      "C",
                      "k",
                      "gamma",
                      "num_classes",
                      "kernel",
                      "tol",
                      "max_epoch",
                      "eval_epoch",
                      "eval_on_start",
                      "recompute_primal",
                      "shrinking",
                      "projection_precision",
                      "seed",
                      "num_threads",
                      "precision",
                      "log_level",
                      "log_format",
                      "initial_dual",
                      "center"},
                     MEX_SOLVE);
  fit_options o;
  o.loss = mxGetFieldValueOrDefault(opts, "loss", std::string(default_loss));
  o.id = shared_loss_id(o.loss);
  if (o.id < 0 || o.id >= num_training_losses)
    throw std::invalid_argument("Unknown loss '" + o.loss + "'; see " + MEX_SOLVE +
                                "('help', 'loss')");
  o.C = mxGetFieldValueOrDefault<double>(opts, "C", 1);
  if (!(o.C > 0) || !std::isfinite(o.C))
    throw std::invalid_argument("C must be finite and positive");
  if (loss_takes_k(o.id)) {
    if (!mxIsField(opts, "k"))
      throw std::invalid_argument("The loss '" + o.loss + "' needs opts.k");
    o.k = mxGetFieldValueOrDefault<size_type>(opts, "k", 1);
  } else {
    check_not_given(opts, "k", o.loss);
  }
  if (loss_takes_gamma(o.id)) {
    o.gamma = mxGetFieldValueOrDefault<double>(opts, "gamma", 0);
    if (!(o.gamma >= 0) || !std::isfinite(o.gamma))
      throw std::invalid_argument("gamma must be finite and nonnegative");
  } else {
    check_not_given(opts, "gamma", o.loss);
    // The entropy objectives update every class of every example.
    check_not_given(opts, "shrinking", o.loss);
  }
  if (o.id != 5) check_not_given(opts, "projection_precision", o.loss);
  if (mxIsField(opts, "num_classes")) {
    o.num_classes = mxGetFieldValueOrDefault<size_type>(opts, "num_classes", 0);
    if (o.num_classes < 2) throw std::invalid_argument("num_classes must be an integer >= 2");
  }
  o.kernel = flag_option(opts, "kernel", false);

  stopping_criteria& c = o.criteria;
  c.tol = mxGetFieldValueOrDefault<double>(opts, "tol", c.tol);
  if (!(c.tol >= 0) || !std::isfinite(c.tol))
    throw std::invalid_argument("tol must be finite and nonnegative");
  c.max_epoch = mxGetFieldValueOrDefault<size_type>(opts, "max_epoch", c.max_epoch);
  c.eval_epoch = mxGetFieldValueOrDefault<size_type>(opts, "eval_epoch", c.eval_epoch);
  c.eval_on_start = flag_option(opts, "eval_on_start", c.eval_on_start);
  c.recompute_primal = flag_option(opts, "recompute_primal", c.recompute_primal);
  const std::string shrinking = mxGetFieldValueOrDefault(opts, "shrinking", std::string("auto"));
  if (!shrinking_mode_of(shrinking, c.shrinking))
    throw std::invalid_argument("Unknown shrinking '" + shrinking +
                                "' (use 'auto', 'off', 'heuristic', 'safe' or 'both')");
  const std::string projection =
      mxGetFieldValueOrDefault(opts, "projection_precision", std::string("float64"));
  if (!projection_precision_of(projection, c.projection))
    throw std::invalid_argument("Unknown projection_precision '" + projection +
                                "' (use 'float64' or 'extended')");
  c.seed = mxGetFieldValueOrDefault<std::uint64_t>(opts, "seed", 0);
  c.num_threads = num_threads_option(opts);
  o.num_threads_given = mxIsField(opts, "num_threads");
  return o;
}

//-----------------------------------------------------------------------------
// Labels
//-----------------------------------------------------------------------------

// 1-based class labels (a double vector of n entries) as 0-based ones; the
// largest label in labels_max.
inline std::vector<size_type> class_labels(const mxArray* labels, const size_type num_examples,
                                           size_type& labels_max) {
  if (mxIsSparse(labels) || mxIsComplex(labels) || !mxIsDouble(labels))
    throw std::invalid_argument(
        "Y must be a double vector of labels in 1:num_classes for this loss");
  mxCheckVector(labels, "labels", num_examples);
  const double* y = mxGetPr(labels);
  std::vector<size_type> out(num_examples);
  labels_max = 0;
  for (size_type i = 0; i < num_examples; ++i) {
    const double v = y[i];
    if (!std::isfinite(v) || std::floor(v) != v || v < 1 || v > 9.007199254740992e15)
      throw std::invalid_argument("Labels must be integers in 1:num_classes");
    out[i] = static_cast<size_type>(v) - 1;
    labels_max = std::max(labels_max, out[i] + 1);
  }
  return out;
}

inline multiclass_output make_multiclass(const mxArray* labels, const size_type num_examples,
                                         const size_type num_classes) {
  size_type largest = 0;
  std::vector<size_type> y = class_labels(labels, num_examples, largest);
  const size_type m = num_classes ? num_classes : largest;
  if (m < 2) throw std::invalid_argument("Training needs num_classes >= 2");
  if (largest > m) throw std::invalid_argument("Labels must be integers in 1:num_classes");
  return multiclass_output(m, std::move(y));
}

// Label sets as an m-by-n sparse (double or logical) or full logical matrix:
// the nonzero rows of column i are the labels of example i.
inline multilabel_output make_multilabel(const mxArray* labels, const size_type num_examples,
                                         const size_type num_classes) {
  const bool sparse = mxIsSparse(labels);
  if (mxIsComplex(labels) || (!mxIsLogical(labels) && !(sparse && mxIsDouble(labels))))
    throw std::invalid_argument(
        "Y must be an m-by-n sparse or logical label matrix for the multilabel losses");
  const size_type m = mxGetM(labels);
  if (mxGetN(labels) != num_examples)
    throw std::invalid_argument("Y must have one column per example");
  if (num_classes && m != num_classes) throw std::invalid_argument("Y must have num_classes rows");
  if (m < 2) throw std::invalid_argument("Training needs num_classes >= 2");
  std::vector<size_type> classes, offsets(1, 0);
  if (sparse) {
    // Row indices are zero-based classes; jc[n], not nzmax, is the entry count.
    const mwIndex* ir = mxGetIr(labels);
    const mwIndex* jc = mxGetJc(labels);
    const bool logical = mxIsLogical(labels);
    const double* values = logical ? nullptr : mxGetPr(labels);
    const mxLogical* flags = logical ? static_cast<const mxLogical*>(mxGetData(labels)) : nullptr;
    for (size_type i = 0; i < num_examples; ++i) {
      for (mwIndex j = jc[i]; j < jc[i + 1]; ++j) {
        // Stored zeros mark no label.
        if (logical ? flags[j] != 0 : values[j] != 0) classes.push_back(ir[j]);
      }
      offsets.push_back(classes.size());
    }
  } else {
    const mxLogical* flags = static_cast<const mxLogical*>(mxGetData(labels));
    for (size_type i = 0; i < num_examples; ++i) {
      for (size_type j = 0; j < m; ++j)
        if (flags[i * m + j]) classes.push_back(j);
      offsets.push_back(classes.size());
    }
  }
  return multilabel_output(m, std::move(classes), std::move(offsets));
}

//-----------------------------------------------------------------------------
// Warm start
//-----------------------------------------------------------------------------

// Throws unless the dual block a (m entries) is finite and sums to zero
// relative to its largest entry, to eps.
template <typename Data>
inline void check_dual_sum(const Data* a, const size_type m, const double eps) {
  double largest = 0, sum = 0, total = 0;
  for (size_type j = 0; j < m; ++j) {
    if (!std::isfinite(a[j])) throw std::invalid_argument("initial_dual must be finite");
    largest = std::max(largest, std::abs(double(a[j])));
  }
  const double scale = largest > 0 ? largest : 1;
  for (size_type j = 0; j < m; ++j) {
    sum += double(a[j]) / scale;
    total += std::abs(double(a[j])) / scale;
  }
  if (std::abs(sum) > eps * total)
    throw std::invalid_argument("initial_dual columns must sum to zero to storage precision");
}

// The hinge constraints of check_initial_dual on the dual block a (m entries)
// whose positive entries sum to pos_sum: pos_sum is at most C, and the top-k
// cone or cap holds.
template <typename Data>
inline void check_hinge_dual(const int id, const double C, const size_type k, const double slack,
                             const Data* a, const size_type m, const double pos_sum) {
  if (pos_sum > C + slack) throw std::invalid_argument("initial_dual is outside the feasible set");
  if (id == 1 || id == 2 || id == 3) {
    const size_type kk = id == 2 ? 1 : k;
    const double cone = double(std::max<size_type>(1, m / kk)) * 2 * slack;
    for (size_type j = 0; j < m; ++j)
      if (-double(a[j]) > pos_sum / double(kk) + cone)
        throw std::invalid_argument("initial_dual violates the top-k cone");
  } else if (id == 4) {
    for (size_type j = 0; j < m; ++j)
      if (-double(a[j]) > C / double(k) + slack)
        throw std::invalid_argument("initial_dual violates the top-k cap");
  }
}

// The sign, box and cone constraints of check_initial_dual on the dual block a
// with the p positive classes of positive.
template <typename Data>
inline void check_label_dual(const int id, const double C, const size_type k, const double slack,
                             const Data* a, const std::vector<char>& positive, const size_type p) {
  const size_type m = positive.size();
  double pos_sum = 0;
  for (size_type j = 0; j < m; ++j) {
    const double v = double(a[j]);
    if (!positive[j] && v > 0)
      throw std::invalid_argument("initial_dual: non-label entries must be nonpositive");
    if (positive[j]) pos_sum += v;
  }
  for (size_type j = 0; j < m; ++j) {
    if (!positive[j]) continue;
    const double v = double(a[j]);
    if (id == 0 || id == 6) {  // the entropy losses
      if (v > C / double(p) + slack)
        throw std::invalid_argument("initial_dual is outside the entropy feasible set");
    } else if (v < -slack) {
      throw std::invalid_argument("initial_dual is outside the feasible set");
    }
  }
  if (id != 0 && id != 6) check_hinge_dual(id, C, k, slack, a, m, pos_sum);
}

// Rejects dual blocks (m-by-n, column i the block of example i) outside the
// feasible set of the loss, to rounding; single-label blocks get the label
// entry that makes them sum to zero exactly. As the C ABI's check.
template <typename Data, typename Output>
inline void check_initial_dual(const int id, const double C, const size_type k, const Output& out,
                               Data* A) {
  const size_type m = out.num_classes, n = out.num_examples();
  const double eps = 32 * double(std::numeric_limits<Data>::epsilon()), slack = eps * C;
  std::vector<char> positive(m);
  for (size_type i = 0; i < n; ++i) {
    Data* a = A + i * m;
    check_dual_sum(a, m, eps);
    std::fill(positive.begin(), positive.end(), 0);
    const size_type p = out.num_labels(i);
    std::vector<size_type> labels(m);
    for (size_type j = 0; j < m; ++j) labels[j] = j;
    out.move_front(i, labels.data());
    for (size_type j = 0; j < p; ++j) positive[labels[j]] = 1;
    check_label_dual(id, C, k, slack, a, positive, p);
    if (!loss_is_multilabel(id)) {
      const size_type y = labels[0];
      double others = 0;
      for (size_type j = 0; j < m; ++j)
        if (j != y) others += double(a[j]);
      a[y] = static_cast<Data>(-others);
    }
  }
}

// W (d-by-m) = center + X * A', accumulated in double.
template <typename Data>
inline void warm_primal(const feature_input<Data>& in, const size_type m, const Data* A,
                        const Data* center, Data* W) {
  const size_type d = in.num_dimensions, n = in.num_examples;
  std::vector<double> w(d * m, 0.0);
  if (center)
    for (size_type j = 0; j < d * m; ++j) w[j] = double(center[j]);
  for (size_type i = 0; i < n; ++i) {
    const Data* x = in.features + i * d;
    for (size_type c = 0; c < m; ++c) {
      const double a = double(A[i * m + c]);
      if (a == 0) continue;
      double* col = &w[c * d];
      for (size_type l = 0; l < d; ++l) col[l] += a * double(x[l]);
    }
  }
  for (size_type j = 0; j < d * m; ++j) W[j] = static_cast<Data>(w[j]);
}

// The m-by-n duals: a checked copy of opts.initial_dual, or zeros.
template <typename Data, typename Output>
inline mxArray* initial_duals(const mxArray* opts, const fit_options& o, const Output& out) {
  const size_type m = out.num_classes, n = out.num_examples();
  const mxArray* given = opts != nullptr ? mxGetField(opts, 0, "initial_dual") : nullptr;
  mxArray* A = nullptr;
  if (given != nullptr) {
    mxCheckNotSparse(given, "initial_dual");
    mxCheckMatrix(given, "initial_dual", m, n, mex_class<Data>::id());
    A = mxDuplicateArray(given);
    mxCheckCreated(A, "A");
    check_initial_dual(o.id, o.C, o.k, out, static_cast<Data*>(mxGetData(A)));
  } else {
    A = mxCreateNumericMatrix(m, n, mex_class<Data>::id(), mxREAL);
    mxCheckCreated(A, "A");
  }
  return A;
}

// The d-by-m center, or null.
template <typename Data>
inline const Data* center_of(const mxArray* opts, const size_type d, const size_type m) {
  const mxArray* center = opts != nullptr ? mxGetField(opts, 0, "center") : nullptr;
  if (center == nullptr) return nullptr;
  mxCheckNotSparse(center, "center");
  mxCheckMatrix(center, "center", d, m, mex_class<Data>::id());
  mxCheckFinite<Data>(center, "center");
  return static_cast<const Data*>(mxGetData(center));
}

//-----------------------------------------------------------------------------
// The model
//-----------------------------------------------------------------------------

template <typename Result>
inline double history_metric(const eval_train<Result, multiclass_output>& e) {
  return e.topk_accuracy(0);
}
template <typename Result>
inline double history_metric(const eval_train<Result, multilabel_output>& e) {
  return e.rank_loss;
}
template <typename Result>
inline double test_metric(const eval_test<Result, multiclass_output>& e) {
  return e.topk_accuracy(0);
}
template <typename Result>
inline double test_metric(const eval_test<Result, multilabel_output>& e) {
  return e.rank_loss;
}

// The status names shared with the other interfaces.
inline const char* status_name(const solver_status status) {
  switch (status) {
    case solver_status::solved:
      return "solved";
    case solver_status::no_progress:
      return "no_progress";
    case solver_status::failed:
      return "failed";
    default:
      return "max_epoch";
  }
}

template <typename Context>
inline void add_history(const Context& ctx, model_info<mxArray*>& info) {
  const char* names[] = {"epoch",       "primal",       "dual",     "primal_loss", "dual_loss",
                         "regularizer", "relative_gap", "accuracy", "cpu_time",    "wall_time"};
  const auto& evals = ctx.train.evals;
  mxArray* pa = mxCreateStructMatrix(evals.size(), 1, 10, names);
  mxCheckCreated(pa, "history");
  size_type i = 0;
  for (const auto& e : evals) {
    const double row[] = {double(e.epoch),
                          double(e.primal),
                          double(e.dual),
                          double(e.primal_loss),
                          double(e.dual_loss),
                          double(e.primal_regularizer),
                          double(e.relative_gap()),
                          history_metric(e),
                          e.cpu_time(),
                          e.wall_time()};
    for (int f = 0; f < 10; ++f) mxSetFieldByNumber(pa, i, f, mxCreateScalar(row[f]));
    ++i;
  }
  info.add("history", pa);

  if (ctx.test.empty()) return;
  const char* test_names[] = {"accuracy", "primal_loss"};
  mxArray* pt = mxCreateStructMatrix(ctx.test[0].evals.size(), ctx.test.size(), 2, test_names);
  mxCheckCreated(pt, "test");
  i = 0;
  for (const auto& set : ctx.test)
    for (const auto& e : set.evals) {
      mxSetFieldByNumber(pt, i, 0, mxCreateScalar(test_metric(e)));
      mxSetFieldByNumber(pt, i, 1, mxCreateScalar(e.primal_loss));
      ++i;
    }
  info.add("test", pt);
}

template <typename Input>
inline void add_num_features(const Input&, model_info<mxArray*>&) {}
template <typename Data>
inline void add_num_features(const feature_input<Data>& in, model_info<mxArray*>& info) {
  info.add("num_features", mxCreateScalar(in.num_dimensions));
}

template <typename Context>
inline void add_model(const fit_options& o, const Context& ctx, model_info<mxArray*>& info) {
  const stopping_criteria& c = ctx.criteria;
  info.add("loss", mxCreateString(o.loss.c_str()));
  info.add("C", mxCreateScalar(o.C));
  if (loss_takes_k(o.id)) info.add("k", mxCreateScalar(o.k));
  if (loss_takes_gamma(o.id)) info.add("gamma", mxCreateScalar(o.gamma));
  info.add("num_classes", mxCreateScalar(ctx.train.num_classes()));
  info.add("num_examples", mxCreateScalar(ctx.train.num_examples()));
  add_num_features(ctx.train.in, info);
  info.add("kernel", mxCreateLogicalScalar(ctx.is_kernel()));
  info.add("tol", mxCreateScalar(c.tol));
  info.add("max_epoch", mxCreateScalar(c.max_epoch));
  info.add("eval_epoch", mxCreateScalar(c.eval_epoch));
  info.add("eval_on_start", mxCreateLogicalScalar(c.eval_on_start));
  info.add("recompute_primal", mxCreateLogicalScalar(c.recompute_primal));
  if (loss_takes_gamma(o.id))
    info.add("shrinking", mxCreateString(shrinking_mode_name(c.shrinking).c_str()));
  if (o.id == 5)
    info.add(
        "projection_precision",
        mxCreateString(c.projection == projection_precision::float64 ? "float64" : "extended"));
  info.add("seed", mxCreateScalar(c.seed));
  if (o.num_threads_given) info.add("num_threads", mxCreateScalar(c.num_threads));
  info.add("precision", mxCreateString(precision_name<typename Context::result_type>()));

  info.add("status", mxCreateString(status_name(ctx.status)));
  info.add("epochs", mxCreateScalar(ctx.epoch));
  const auto& evals = ctx.train.evals;
  info.add("relative_gap", mxCreateScalar(evals.empty() ? std::numeric_limits<double>::quiet_NaN()
                                                        : double(evals.back().relative_gap())));
  info.add("cpu_time", mxCreateScalar(ctx.cpu_time()));
  info.add("wall_time", mxCreateScalar(ctx.wall_time()));
  add_history(ctx, info);
}

//-----------------------------------------------------------------------------
// Data
//-----------------------------------------------------------------------------

template <typename Data>
inline void validate_matrix(const mxArray* data, const bool kernel = false) {
  mxCheckNotSparse(data, "X");
  mxCheckNotEmpty(data, "X");
  mxCheckReal(data, "X");
  mxCheckClass(data, "X", mex_class<Data>::id());
  if (kernel) mxCheckSquare(data, "X");
}

template <typename Data>
inline feature_input<Data> make_test_input(const feature_input<Data>& trn_in, const mxArray* data,
                                           const size_type id) {
  if (trn_in.num_dimensions != mxGetM(data)) {
    mex_error(err_id[err_num_dim], err_msg[err_num_dim], static_cast<unsigned long long>(id));
  }
  return sdca::make_input_feature(mxGetM(data), mxGetN(data), static_cast<Data*>(mxGetData(data)));
}

template <typename Data>
inline kernel_input<Data> make_test_input(const kernel_input<Data>& trn_in, const mxArray* data,
                                          const size_type id) {
  if (trn_in.num_examples != mxGetM(data)) {
    mex_error(err_id[err_num_examples], err_msg[err_num_examples],
              static_cast<unsigned long long>(id));
  }
  return sdca::make_input_kernel(mxGetM(data), mxGetN(data), static_cast<Data*>(mxGetData(data)));
}

inline multiclass_output make_output(const mxArray* labels, const size_type n,
                                     const size_type num_classes, const multiclass_output*) {
  return make_multiclass(labels, n, num_classes);
}
inline multilabel_output make_output(const mxArray* labels, const size_type n,
                                     const size_type num_classes, const multilabel_output*) {
  return make_multilabel(labels, n, num_classes);
}

template <typename Data, typename Result, template <typename> class Input, typename Output,
          template <typename, typename> class Objective>
inline void set_test_data(const mxArray* all_data, const mxArray* all_labels,
                          solver_context<Data, Result, Input, Output, Objective>& ctx) {
  // The first data set trains.
  const size_type num_datasets = mxGetNumberOfElements(all_data);
  for (size_type i = 1; i < num_datasets; ++i) {
    const mxArray* data = mxGetCell(all_data, i);
    const mxArray* labels = mxGetCell(all_labels, i);
    validate_matrix<Data>(data);
    mxCheckFinite<Data>(data, "X");
    Input<Data> in = make_test_input(ctx.train.in, data, i + 1);
    Output out = make_output(labels, in.num_examples, ctx.train.out.num_classes,
                             static_cast<const Output*>(nullptr));
    ctx.add_test(std::move(in), std::move(out));
  }
}

// The W of a feature fit, from the duals and the center; nothing for a kernel.
template <typename Data, typename Output>
inline Data* primal_variables(const mxArray* opts, const feature_input<Data>& in, const Output& out,
                              const Data* A, const Data*& center, model_info<mxArray*>& info) {
  const size_type m = out.num_classes;
  center = center_of<Data>(opts, in.num_dimensions, m);
  mxArray* W = mxCreateNumericMatrix(in.num_dimensions, m, mex_class<Data>::id(), mxREAL);
  mxCheckCreated(W, "W");
  info.add("W", W);
  Data* w = static_cast<Data*>(mxGetData(W));
  warm_primal(in, m, A, center, w);
  return w;
}

template <typename Data, typename Output>
inline Data* primal_variables(const mxArray* opts, const kernel_input<Data>&, const Output&,
                              const Data*, const Data*&, model_info<mxArray*>&) {
  if (mxIsField(opts, "center"))
    throw std::invalid_argument("center is only supported for feature input");
  return nullptr;
}

template <typename Data, typename Result, template <typename, typename> class Objective,
          typename Output>
inline auto make_fit_context(feature_input<Data>&& in, Output&& out, Objective<Data, Result>&& obj,
                             Data* W, Data* A, const Data* center)
    -> decltype(make_context(std::move(in), std::move(out), std::move(obj), W, A, center)) {
  return make_context(std::move(in), std::move(out), std::move(obj), W, A, center);
}

template <typename Data, typename Result, template <typename, typename> class Objective,
          typename Output>
inline auto make_fit_context(kernel_input<Data>&& in, Output&& out, Objective<Data, Result>&& obj,
                             Data*, Data* A,
                             const Data*) -> decltype(make_context(std::move(in), std::move(out),
                                                                   std::move(obj), A)) {
  return make_context(std::move(in), std::move(out), std::move(obj), A);
}

// Trains objective on the data and writes the model to plhs[0].
template <typename Data, typename Result, template <typename> class Input, typename Output,
          template <typename, typename> class Objective>
inline void train(mxArray* plhs[], const mxArray* prhs[], const mxArray* opts, const fit_options& o,
                  Input<Data>&& in, Output&& out, Objective<Data, Result>&& obj) {
  model_info<mxArray*> info;
  mxArray* mxA = initial_duals<Data>(opts, o, out);
  Data* A = static_cast<Data*>(mxGetData(mxA));
  const Data* center = nullptr;
  Data* W = primal_variables(opts, in, out, A, center, info);
  info.add("A", mxA);

  auto ctx = make_fit_context(std::move(in), std::move(out), std::move(obj), W, A, center);
  if (prhs != nullptr && mxIsCell(prhs[0]) && mxIsCell(prhs[1]))
    set_test_data(prhs[0], prhs[1], ctx);
  ctx.criteria = o.criteria;

  sdca::make_solver(ctx).solve();

  add_model(o, ctx, info);
  plhs[0] = mxCreateStruct(info.fields, "model");
}

// The typed objective of the single-label losses.
template <typename Data, typename Result, template <typename> class Input>
inline void dispatch(mxArray* plhs[], const mxArray* prhs[], const mxArray* opts,
                     const fit_options& o, Input<Data>&& in, multiclass_output&& out) {
  const Result c = static_cast<Result>(o.C), gamma = static_cast<Result>(o.gamma);
  const bool smooth = o.gamma > 0;
  if (loss_takes_k(o.id) && (o.k < 1 || o.k >= out.num_classes))
    throw std::invalid_argument("k must be an integer in 1:num_classes-1");
  switch (o.id) {
    case 0:
      train(plhs, prhs, opts, o, std::move(in), std::move(out),
            make_objective_l2_cross_entropy<Data, Result>(c));
      break;
    case 1:
      train(plhs, prhs, opts, o, std::move(in), std::move(out),
            make_objective_l2_topk_cross_entropy<Data, Result>(c, o.k));
      break;
    case 2:
    case 3: {
      const size_type k = o.id == 2 ? 1 : o.k;
      if (smooth)
        train(plhs, prhs, opts, o, std::move(in), std::move(out),
              make_objective_l2_topk_hinge_alpha_smooth<Data, Result>(c, k, gamma));
      else
        train(plhs, prhs, opts, o, std::move(in), std::move(out),
              make_objective_l2_topk_hinge_alpha<Data, Result>(c, k));
      break;
    }
    case 4:
      if (smooth)
        train(plhs, prhs, opts, o, std::move(in), std::move(out),
              make_objective_l2_topk_hinge_beta_smooth<Data, Result>(c, o.k, gamma));
      else
        train(plhs, prhs, opts, o, std::move(in), std::move(out),
              make_objective_l2_topk_hinge_beta<Data, Result>(c, o.k));
      break;
    default:
      throw std::invalid_argument("Unknown loss '" + o.loss + "'");
  }
}

// The typed objective of the multilabel losses.
template <typename Data, typename Result, template <typename> class Input>
inline void dispatch(mxArray* plhs[], const mxArray* prhs[], const mxArray* opts,
                     const fit_options& o, Input<Data>&& in, multilabel_output&& out) {
  const Result c = static_cast<Result>(o.C), gamma = static_cast<Result>(o.gamma);
  if (o.id == 5 && o.gamma > 0)
    train(plhs, prhs, opts, o, std::move(in), std::move(out),
          make_objective_l2_multilabel_hinge_smooth<Data, Result>(c, gamma));
  else if (o.id == 5)
    train(plhs, prhs, opts, o, std::move(in), std::move(out),
          make_objective_l2_multilabel_hinge<Data, Result>(c));
  else if (o.id == 6)
    train(plhs, prhs, opts, o, std::move(in), std::move(out),
          make_objective_l2_multilabel_cross_entropy<Data, Result>(c));
  else
    throw std::invalid_argument("Unknown loss '" + o.loss + "'");
}

template <typename Data, typename Result, template <typename> class Input>
inline void set_output(mxArray* plhs[], const mxArray* prhs[], const mxArray* opts,
                       const fit_options& o, const mxArray* labels, Input<Data>&& in) {
  mxCheckNotEmpty(labels, "Y");
  if (loss_is_multilabel(o.id)) {
    multilabel_output out = make_multilabel(labels, in.num_examples, o.num_classes);
    dispatch<Data, Result>(plhs, prhs, opts, o, std::move(in), std::move(out));
  } else {
    multiclass_output out = make_multiclass(labels, in.num_examples, o.num_classes);
    dispatch<Data, Result>(plhs, prhs, opts, o, std::move(in), std::move(out));
  }
}

template <typename Data, typename Result>
inline void set_input(mxArray* plhs[], const mxArray* prhs[], const mxArray* opts,
                      const fit_options& o) {
  const mxArray* data;
  const mxArray* labels;
  if (!mxIsCell(prhs[0]) && !mxIsCell(prhs[1])) {
    data = prhs[0];
    labels = prhs[1];
  } else {
    mxCheckCellArrays(prhs[0], prhs[1]);
    data = mxGetCell(prhs[0], 0);
    labels = mxGetCell(prhs[1], 0);
  }

  validate_matrix<Data>(data, o.kernel);
  mxCheckFinite<Data>(data, "X");
  Data* p_data = static_cast<Data*>(mxGetData(data));
  if (o.kernel) {
    set_output<Data, Result>(plhs, prhs, opts, o, labels,
                             sdca::make_input_kernel(mxGetN(data), p_data));
  } else {
    set_output<Data, Result>(plhs, prhs, opts, o, labels,
                             sdca::make_input_feature(mxGetM(data), mxGetN(data), p_data));
  }
}

template <typename Data>
inline void mex_main(mxArray* plhs[], const int nrhs, const mxArray* prhs[]) {
  const mxArray* opts = nrhs > 2 ? prhs[2] : nullptr;
  mxCheckStruct(opts, "opts");
  const fit_options o = read_fit_options(opts);
  set_logging_options(opts);
  if (float32_work(opts, std::is_same<Data, float>::value)) {
    set_input<Data, float>(plhs, prhs, opts, o);
  } else {
    set_input<Data, double>(plhs, prhs, opts, o);
  }
}

void mexFunction(const int nlhs, mxArray* plhs[], const int nrhs, const mxArray* prhs[]) {
  mxCheckArgNum(nrhs, 1, 3, printUsage);
  mxCheckArgNum(nlhs, 0, 1, printUsage);

  if (mxIsChar(prhs[0])) {
    std::string command = mxGetString(prhs[0], "command");
    const mxArray* opts = (nrhs > 1) ? prhs[1] : nullptr;
    if (command == "help") {
      printHelp(opts);
    } else if (command == "version") {
      printVersion();
    } else {
      mex_error(err_id[err_command], err_msg[err_command], command.c_str());
    }
  } else {
    mxCheckArgNum(nrhs, 2, 3, printUsage);

    mat_cout_hijack mat_cout;
    try {
      const mxArray* first =
          mxIsCell(prhs[0]) && !mxIsEmpty(prhs[0]) ? mxGetCell(prhs[0], 0) : prhs[0];
      if (mxIsDouble(first)) {
        mex_main<double>(plhs, nrhs, prhs);
      } else if (mxIsSingle(first)) {
        mex_main<float>(plhs, nrhs, prhs);
      } else {
        mex_error(err_id[err_arg], "X must be single or double.");
      }
    } catch (const std::exception& e) {
      mex_error(err_id[err_exception], err_msg[err_exception], e.what());
    }
  }
}
