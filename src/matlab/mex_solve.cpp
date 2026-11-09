#include <memory>

#include "mex_util.h"
#include "sdca/solver.h"

#include "sdca/utility/logging.cpp"

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

static constexpr char default_objective[] = "cross_entropy";

inline void printUsage() {
  mexPrintf(
      "Usage: model = %s(data, labels, opts);\n"
      "  See %s('help') and %s('version') for more information.\n",
      MEX_SOLVE, MEX_SOLVE, MEX_SOLVE);
}

inline void printVersion() { mexPrintf("%s version %s.\n", MEX_SOLVE, LIBSDCA_VERSION); }

inline void printHelp(const mxArray* opts) {
  if (opts == nullptr) {
    mexPrintf(
        "Usage: model = %s(data, labels, opts);\n"
        "  Optimizes an objective given in opts.objective using the data and labels.\n"
        "\n"
        "  opts is a struct with the following fields (defaults in [brackets]):\n"
        "\n"
        "    objective ['%s'] - the objective to optimize;\n"
        "    c         [1]           - the loss weight C (alias C);\n"
        "    k         [1]           - the k in top-k optimization;\n"
        "    gamma     [0]           - the smoothing of the hinge losses (0: nonsmooth;\n"
        "                              the historical names default to 1);\n"
        "    is_dual   [false]       - whether data is given as Gram matrix (alias kernel);\n"
        "\n"
        "    epsilon        [1e-3]   - relative duality gap bound (alias tol), stop if\n"
        "      (primal - dual) <= epsilon * max(abs(primal), abs(dual))\n"
        "    max_epoch      [1000]   - epochs limit;\n"
        "    max_cpu_time   [0]      - CPU time limit (0: no limit);\n"
        "    max_wall_time  [0]      - wall time limit (0: no limit);\n"
        "    eval_on_start  [false]  - whether to check the duality gap on start;\n"
        "    eval_epoch     [10]     - how often to check the gap\n"
        "                              (0: only at the end, unlike Python and Rust);\n"
        "    recompute_primal [true] - whether each check first rebuilds W = W0 + X * A'\n"
        "                              (false keeps the incrementally updated W);\n"
        "    projection_precision ['float64'] - how the multilabel hinge projections\n"
        "                              solve: 'float64' in double with a checked result,\n"
        "                              re-solved in extended precision if the check\n"
        "                              fails; 'extended' always in extended precision;\n"
        "    shrinking  ['auto']     - what the epochs of the hinge objectives leave out:\n"
        "                              'heuristic' (skip unchanged examples between\n"
        "                              full passes), 'safe' (gap-safe screening), 'both',\n"
        "                              'off'; 'auto' is 'both' (see docs/API.md);\n"
        "\n"
        "    log_level  ['info']     - logging verbosity:\n"
        "                              'none', 'warning', 'info', 'verbose', 'debug';\n"
        "    log_format ['short_e']  - numeric format:\n"
        "                              'short_f', 'short_e', 'long_f', 'long_e';\n"
        "    precision  ['double']   - floating-point precision for intermediate\n"
        "                              computations (e.g. proximal update steps):\n"
        "                              'double' (or 'float64', 'fp64'),\n"
        "                              'single' (or 'float32', 'fp32');\n"
        "\n"
        "  Warm restart:\n"
        "    initial_dual [none] - m-by-n dual variables, e.g. model.A (alias A);\n"
        "      for features the primal variables are rebuilt as W = W0 + X * A';\n"
        "      opts.W is only checked for its shape and otherwise ignored.\n"
        "\n"
        "  Proximal regularization:\n"
        "    center [none] - d-by-m primal variables for the regularization\n"
        "                    1/2 ||W - W0||^2 (alias W0);\n"
        "\n"
        "  The model struct holds W (d-by-m, features only) and A (m-by-n); status,\n"
        "  epoch, cpu_time, wall_time and relative_gap; the options used (objective,\n"
        "  c, k, gamma, epsilon, ..., precision, data_precision); train, one entry\n"
        "  per evaluation (epoch, accuracy or rank_loss, relative_gap, primal, dual,\n"
        "  primal_loss, dual_loss, the regularizers and times); and test, one row\n"
        "  per evaluation and one column per further data set.\n"
        "\n"
        "  Prediction scores can be computed as:\n"
        "    scores = model.W' * X;\n"
        "    scores = model.A * Xtrn' * Xtst;\n"
        "\n"
        "  See %s('help', 'data') for more information on\n"
        "  supported input data formats.\n"
        "  See %s('help', 'objective') for more information on\n"
        "  currently supported training objectives.\n",
        MEX_SOLVE, default_objective, MEX_SOLVE, MEX_SOLVE);
    return;
  }

  std::string arg = mxGetString(opts, "help argument");
  if (arg == "data" || arg == "labels" || arg == "input") {
    mexPrintf(
        "Input data is given in the first two arguments, data and labels.\n"
        "\n"
        "  data can be:\n"
        "    - a d-by-n feature matrix,\n"
        "      where d is the number of features and n is the number of examples.\n"
        "    - a n-by-n Gram matrix (requires opts.is_dual=true),\n"
        "      where n is the number of training examples.\n"
        "    - a cell array containing either the feature or the Gram matrices,\n"
        "      but not a mixture of both.\n"
        "      In this case, the first matrix is used for training,\n"
        "      while the rest is used for evaluation only\n"
        "      (e.g., can be used to monitor performance on a validation set).\n"
        "      The Gram matrices for testing should be computed as\n"
        "        Ktst = Xtrn' * Xtst\n"
        "      and should be num_train_examples-by-num_test_examples.\n"
        "\n"
        "  labels can be:\n"
        "    - a n-by-1 or a 1-by-n vector of class labels;\n"
        "      labels must be in the range 0:(m-1) or 1:m,\n"
        "      where m is the number of classes.\n"
        "    - a sparse double or logical m-by-n matrix (multilabel objectives\n"
        "      only), where m is the number of classes,\n"
        "      with nonzero entries indicating class membership.\n"
        "    - a cell array with the same number of elements as in data\n"
        "      containing labels as above.\n"
        "\n"
        "  data matrices must be non-sparse and of type single or double\n"
        "\n"
        "  label vectors must be of type double\n");
  } else if (arg == "obj" || arg == "objective") {
    mexPrintf(
        "opts.objective - the training objective to optimize (all l2 regularized).\n"
        "  The hinges are smoothed by gamma > 0 (gamma = 0: nonsmooth, the default).\n"
        "  Multiclass:\n"
        "    cross_entropy      - multiclass cross-entropy (softmax) loss\n"
        "    topk_cross_entropy - top-k entropy loss (cross_entropy for k=1)\n"
        "    multiclass_hinge   - multiclass SVM of Crammer and Singer (ignores k)\n"
        "    topk_hinge_alpha   - top-k hinge alpha loss (hinge of the top-k average)\n"
        "    topk_hinge_beta    - top-k hinge beta loss (average of the top-k hinges)\n"
        "\n"
        "  Multilabel (sparse labels):\n"
        "    multilabel_hinge         - multilabel SVM of Crammer and Singer\n"
        "    multilabel_cross_entropy - multilabel cross-entropy loss\n"
        "\n"
        "  Historical names of v0.2.0, with gamma = 1 by default:\n"
        "    softmax, l2_multiclass_entropy       = cross_entropy\n"
        "    l2_entropy_topk                      = topk_cross_entropy\n"
        "    msvm, l2_multiclass_hinge            = multiclass_hinge, gamma = 0\n"
        "    msvm_smooth, l2_multiclass_hinge_smooth\n"
        "                                         = multiclass_hinge\n"
        "    l2_hinge_topk                        = topk_hinge_alpha\n"
        "    l2_topk_hinge                        = topk_hinge_beta\n"
        "    mlsvm, l2_multilabel_hinge           = multilabel_hinge, gamma = 0\n"
        "    mlsvm_smooth, l2_multilabel_hinge_smooth\n"
        "                                         = multilabel_hinge\n"
        "    mlsoftmax, l2_multilabel_entropy     = multilabel_cross_entropy\n"
        "\n"
        "  Default value:\n"
        "    %s\n",
        default_objective);
  } else {
    mex_error(err_id[err_help_arg], err_msg[err_help_arg], arg.c_str());
  }
}

template <typename Result>
inline void add_train_evals(const std::vector<eval_train<Result, multiclass_output>>& evals,
                            model_info<mxArray*>& info) {
  const char* names[] = {"epoch",
                         "accuracy",
                         "relative_gap",
                         "primal",
                         "dual",
                         "primal_loss",
                         "dual_loss",
                         "primal_regularizer",
                         "dual_regularizer",
                         "solve_time_cpu",
                         "solve_time_wall",
                         "eval_time_cpu",
                         "eval_time_wall",
                         "accuracies"};
  mxArray* pa = mxCreateStructMatrix(evals.size(), 1, 14, names);
  mxCheckCreated(pa, "train");
  size_type i(0);
  for (auto& a : evals) {
    mxSetFieldByNumber(pa, i, 0, mxCreateScalar(a.epoch));
    mxSetFieldByNumber(pa, i, 1, mxCreateScalar(a.topk_accuracy(0)));
    mxSetFieldByNumber(pa, i, 2, mxCreateScalar(a.relative_gap()));
    mxSetFieldByNumber(pa, i, 3, mxCreateScalar(a.primal));
    mxSetFieldByNumber(pa, i, 4, mxCreateScalar(a.dual));
    mxSetFieldByNumber(pa, i, 5, mxCreateScalar(a.primal_loss));
    mxSetFieldByNumber(pa, i, 6, mxCreateScalar(a.dual_loss));
    mxSetFieldByNumber(pa, i, 7, mxCreateScalar(a.primal_regularizer));
    mxSetFieldByNumber(pa, i, 8, mxCreateScalar(a.dual_regularizer));
    mxSetFieldByNumber(pa, i, 9, mxCreateScalar(a.solve_time_cpu));
    mxSetFieldByNumber(pa, i, 10, mxCreateScalar(a.solve_time_wall));
    mxSetFieldByNumber(pa, i, 11, mxCreateScalar(a.eval_time_cpu));
    mxSetFieldByNumber(pa, i, 12, mxCreateScalar(a.eval_time_wall));
    mxSetFieldByNumber(pa, i, 13, mxCreateVector(a.accuracy, "accuracies"));
    ++i;
  }
  info.add("train", pa);
}

template <typename Result>
inline void add_train_evals(const std::vector<eval_train<Result, multilabel_output>>& evals,
                            model_info<mxArray*>& info) {
  const char* names[] = {"epoch",
                         "rank_loss",
                         "relative_gap",
                         "primal",
                         "dual",
                         "primal_loss",
                         "dual_loss",
                         "primal_regularizer",
                         "dual_regularizer",
                         "solve_time_cpu",
                         "solve_time_wall",
                         "eval_time_cpu",
                         "eval_time_wall"};
  mxArray* pa = mxCreateStructMatrix(evals.size(), 1, 13, names);
  mxCheckCreated(pa, "train");
  size_type i = 0;
  for (auto& a : evals) {
    mxSetFieldByNumber(pa, i, 0, mxCreateScalar(a.epoch));
    mxSetFieldByNumber(pa, i, 1, mxCreateScalar(a.rank_loss));
    mxSetFieldByNumber(pa, i, 2, mxCreateScalar(a.relative_gap()));
    mxSetFieldByNumber(pa, i, 3, mxCreateScalar(a.primal));
    mxSetFieldByNumber(pa, i, 4, mxCreateScalar(a.dual));
    mxSetFieldByNumber(pa, i, 5, mxCreateScalar(a.primal_loss));
    mxSetFieldByNumber(pa, i, 6, mxCreateScalar(a.dual_loss));
    mxSetFieldByNumber(pa, i, 7, mxCreateScalar(a.primal_regularizer));
    mxSetFieldByNumber(pa, i, 8, mxCreateScalar(a.dual_regularizer));
    mxSetFieldByNumber(pa, i, 9, mxCreateScalar(a.solve_time_cpu));
    mxSetFieldByNumber(pa, i, 10, mxCreateScalar(a.solve_time_wall));
    mxSetFieldByNumber(pa, i, 11, mxCreateScalar(a.eval_time_cpu));
    mxSetFieldByNumber(pa, i, 12, mxCreateScalar(a.eval_time_wall));
    ++i;
  }
  info.add("train", pa);
}

template <typename Result, typename Input>
inline void add_test_evals(
    const std::vector<dataset<Input, multiclass_output, eval_test<Result, multiclass_output>>>&
        sets,
    model_info<mxArray*>& info) {
  if (sets.size() == 0) return;

  const char* names[] = {"accuracy", "primal_loss", "accuracies"};
  mxArray* pa = mxCreateStructMatrix(sets[0].evals.size(), sets.size(), 3, names);
  mxCheckCreated(pa, "test");
  size_type i = 0;
  for (auto& testset : sets) {
    for (auto& a : testset.evals) {
      mxSetFieldByNumber(pa, i, 0, mxCreateScalar(a.topk_accuracy(0)));
      mxSetFieldByNumber(pa, i, 1, mxCreateScalar(a.primal_loss));
      mxSetFieldByNumber(pa, i, 2, mxCreateVector(a.accuracy, "accuracies"));
      ++i;
    }
  }
  info.add("test", pa);
}

template <typename Result, typename Input>
inline void add_test_evals(
    const std::vector<dataset<Input, multilabel_output, eval_test<Result, multilabel_output>>>&
        sets,
    model_info<mxArray*>& info) {
  if (sets.size() == 0) return;

  const char* names[] = {"rank_loss", "primal_loss"};
  mxArray* pa = mxCreateStructMatrix(sets[0].evals.size(), sets.size(), 2, names);
  mxCheckCreated(pa, "test");
  size_type i = 0;
  for (auto& testset : sets) {
    for (auto& a : testset.evals) {
      mxSetFieldByNumber(pa, i, 0, mxCreateScalar(a.rank_loss));
      mxSetFieldByNumber(pa, i, 1, mxCreateScalar(a.primal_loss));
      ++i;
    }
  }
  info.add("test", pa);
}

template <typename Context>
inline void add_info_solution(const Context& ctx, model_info<mxArray*>& info) {
  info.add("status", mxCreateString(ctx.status_name().c_str()));
  info.add("epoch", mxCreateScalar(ctx.epoch));
  info.add("cpu_time", mxCreateScalar(ctx.cpu_time()));
  info.add("wall_time", mxCreateScalar(ctx.wall_time()));

  const auto& evals = ctx.train.evals;
  if (evals.size() > 0) {
    info.add("relative_gap", mxCreateScalar(evals.back().relative_gap()));
  }

  add_train_evals(evals, info);
  add_test_evals(ctx.test, info);
}

template <typename Input>
inline void add_info_input(const Input&, model_info<mxArray*>&) {}

template <typename Data>
inline void add_info_input(const feature_input<Data>& in, model_info<mxArray*>& info) {
  info.add("num_dimensions", mxCreateScalar(in.num_dimensions));
}

template <typename Objective>
inline void add_info_k(const Objective&, model_info<mxArray*>&,
                       typename std::enable_if<!has_param_k<Objective>::value>::type* = 0) {}

template <typename Objective>
inline void add_info_k(const Objective& obj, model_info<mxArray*>& info,
                       typename std::enable_if<has_param_k<Objective>::value>::type* = 0) {
  info.add("k", mxCreateScalar(obj.k));
}

template <typename Objective>
inline void add_info_gamma(const Objective&, model_info<mxArray*>&,
                           typename std::enable_if<!has_param_gamma<Objective>::value>::type* = 0) {
}

template <typename Objective>
inline void add_info_gamma(const Objective& obj, model_info<mxArray*>& info,
                           typename std::enable_if<has_param_gamma<Objective>::value>::type* = 0) {
  info.add("gamma", mxCreateScalar(obj.gamma));
}

template <typename Context>
inline void add_info(const mxArray* opts, const Context& ctx, model_info<mxArray*>& info) {
  add_info_input(ctx.train.in, info);
  info.add("num_examples", mxCreateScalar(ctx.train.num_examples()));
  info.add("num_classes", mxCreateScalar(ctx.train.num_classes()));
  info.add("is_dual", mxCreateScalar(ctx.is_dual()));

  std::string obj = mxGetFieldValueOrDefault(opts, "objective", std::string(default_objective));
  info.add("objective", mxCreateString(obj.c_str()));
  info.add("c", mxCreateScalar(ctx.objective.c));
  add_info_k(ctx.objective, info);
  add_info_gamma(ctx.objective, info);

  info.add("epsilon", mxCreateScalar(ctx.criteria.epsilon));
  info.add("max_epoch", mxCreateScalar(ctx.criteria.max_epoch));
  info.add("max_cpu_time", mxCreateScalar(ctx.criteria.max_cpu_time));
  info.add("max_wall_time", mxCreateScalar(ctx.criteria.max_wall_time));
  info.add("eval_on_start", mxCreateScalar(ctx.criteria.eval_on_start));
  info.add("recompute_primal", mxCreateScalar(ctx.criteria.recompute_primal));
  info.add("projection_precision",
           mxCreateString(ctx.criteria.projection == projection_precision::float64 ? "float64"
                                                                                   : "extended"));
  info.add("eval_epoch", mxCreateScalar(ctx.criteria.eval_epoch));
  info.add("shrinking", mxCreateString(shrinking_mode_name(ctx.criteria.shrinking).c_str()));

  info.add("data_precision", mxCreateString(precision_name<typename Context::data_type>()));
  info.add("precision", mxCreateString(precision_name<typename Context::result_type>()));
  info_add_opts_field(opts, "log_level", info);
  info_add_opts_field(opts, "log_format", info);

  add_info_solution(ctx, info);
}

template <typename Data>
inline void validate_matrix(const mxArray* data, const bool is_dual = false) {
  mxCheckNotSparse(data, "data");
  mxCheckNotEmpty(data, "data");
  mxCheckReal(data, "data");
  mxCheckClass(data, "data", mex_class<Data>::id());
  if (is_dual) {
    mxCheckSquare(data, "data");
  }
}

inline void validate_labels(const mxArray* labels, const size_type num_examples) {
  mxCheckNotEmpty(labels, "labels");
  if (mxIsSparse(labels)) {
    if (!mxIsLogical(labels)) mxCheckDouble(labels, "labels");
    mxCheck(std::equal_to<size_type>(), mxGetN(labels), num_examples, "num_examples");
  } else {
    mxCheckVector(labels, "labels", num_examples);
  }
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

template <typename Output>
inline Output make_output(const mxArray*, const size_type) {
  return Output();
}

template <>
inline multiclass_output make_output<multiclass_output>(const mxArray* labels,
                                                        const size_type num_examples) {
  mxCheckNotSparse(labels, "labels");
  auto first = mxGetPr(labels);
  auto last = mxGetPr(labels) + num_examples;
  return make_output_multiclass(first, last);
}

template <>
inline multilabel_output make_output<multilabel_output>(const mxArray* labels,
                                                        const size_type num_examples) {
  if (mxIsSparse(labels)) {
    // Row indices are zero-based classes; jc[n], not nzmax, is the entry count.
    const mwIndex* ir = mxGetIr(labels);
    const mwIndex* jc = mxGetJc(labels);
    std::vector<size_type> classes(ir, ir + jc[num_examples]);
    std::vector<size_type> offsets(jc, jc + num_examples + 1);
    return multilabel_output(mxGetM(labels), classes, offsets);
  } else {
    auto first = mxGetPr(labels);
    auto last = mxGetPr(labels) + num_examples;
    return make_output_multilabel(first, last);
  }
}

template <typename Data, typename Result, template <typename> class Input, typename Output,
          template <typename, typename> class Objective>
inline void set_test_data(const mxArray* all_data, const mxArray* all_labels,
                          solver_context<Data, Result, Input, Output, Objective>& ctx) {
  // The first dataset is for training
  size_type num_datasets = mxGetNumberOfElements(all_data);
  for (size_type i = 1; i < num_datasets; ++i) {
    auto data = mxGetCell(all_data, i);
    auto labels = mxGetCell(all_labels, i);
    validate_matrix<Data>(data);
    mxCheckFinite<Data>(data, "data");
    validate_labels(labels, mxGetN(data));

    Input<Data> in = make_test_input(ctx.train.in, data, i + 1);
    Output out = make_output<Output>(labels, in.num_examples);
    if (out.num_classes != ctx.train.out.num_classes) {
      mex_error(err_id[err_num_classes], err_msg[err_num_classes],
                static_cast<unsigned long long>(i) + 1);
    }

    ctx.add_test(std::move(in), std::move(out));
  }
}

template <typename Context>
inline void set_stopping_criteria(const mxArray* opts, Context& ctx) {
  auto c = &ctx.criteria;
  mxSetFieldValue(opts, "epsilon", c->epsilon);
  mxSetFieldValue(opts, "eval_on_start", c->eval_on_start);
  mxSetFieldValue(opts, "recompute_primal", c->recompute_primal);
  const std::string projection =
      mxGetFieldValueOrDefault(opts, "projection_precision", std::string("float64"));
  if (!projection_precision_of(projection, c->projection))
    mex_error(err_id[err_arg], "Unknown projection_precision '%s' (use 'float64' or 'extended').",
              projection.c_str());
  const std::string shrinking = mxGetFieldValueOrDefault(opts, "shrinking", std::string("auto"));
  if (!shrinking_mode_of(shrinking, c->shrinking))
    mex_error(err_id[err_arg],
              "Unknown shrinking '%s' (use 'auto', 'off', 'heuristic', 'safe' or 'both').",
              shrinking.c_str());
  mxSetFieldValue(opts, "eval_epoch", c->eval_epoch);
  mxSetFieldValue(opts, "max_epoch", c->max_epoch);
  mxSetFieldValue(opts, "max_cpu_time", c->max_cpu_time);
  mxSetFieldValue(opts, "max_wall_time", c->max_wall_time);

  // Integer options are checked nonnegative on reading; eval_epoch = 0 evaluates only at the end.
  mxCheck<double>(std::greater_equal<double>(), c->epsilon, 0, "epsilon");
  mxCheck<double>(std::greater_equal<double>(), c->max_cpu_time, 0, "max_cpu_time");
  mxCheck<double>(std::greater_equal<double>(), c->max_wall_time, 0, "max_wall_time");
}

// The m-by-n dual variables: a copy of opts.A (alias initial_dual) or zeros.
template <typename Data>
inline Data* set_dual_variables(const mxArray* opts, const size_type num_examples,
                                const size_type num_classes, model_info<mxArray*>& info) {
  mxArray* mxA =
      mxDuplicateFieldOrCreateMatrix(opts, "A", num_classes, num_examples, mex_class<Data>::id());
  validate_matrix<Data>(mxA);
  mxCheckFinite<Data>(mxA, "initial_dual");
  info.add("A", mxA);
  return static_cast<Data*>(mxGetData(mxA));
}

// W = W0 + X * A' for the given duals; a given opts.W is checked but not used.
template <typename Data, typename Output>
inline void set_variables(const mxArray* opts, feature_input<Data>& in, Output& out,
                          model_info<mxArray*>& info, Data*& A, Data*& W, Data*& W0) {
  const auto D = static_cast<blas_int>(in.num_dimensions);
  const auto M = static_cast<blas_int>(out.num_classes);
  const auto N = static_cast<blas_int>(in.num_examples);
  A = set_dual_variables<Data>(opts, in.num_examples, out.num_classes, info);
  if (mxIsField(opts, "W")) {
    mxCheckMatrix(mxGetField(opts, 0, "W"), "W", in.num_dimensions, out.num_classes,
                  mex_class<Data>::id());
  }

  mxArray* mxW =
      mxCreateNumericMatrix(in.num_dimensions, out.num_classes, mex_class<Data>::id(), mxREAL);
  mxCheckCreated(mxW, "W");
  info.add("W", mxW);
  W = static_cast<Data*>(mxGetData(mxW));

  mxArray* mxW0 = mxGetField(opts, 0, "W0");
  if (mxW0 != nullptr) {
    mxCheckMatrix(mxW0, "W0", in.num_dimensions, out.num_classes, mex_class<Data>::id());
    validate_matrix<Data>(mxW0);
    mxCheckFinite<Data>(mxW0, "center");
    W0 = static_cast<Data*>(mxGetData(mxW0));
    sdca_blas_copy(D * M, W0, W);
  }
  if (mxIsField(opts, "A")) {
    sdca_blas_gemm(D, M, N, in.features, D, A, M, W, CblasNoTrans, CblasTrans, Data(1),
                   W0 != nullptr ? Data(1) : Data(0));
  }
}

template <typename Data, template <typename> class Input, typename Output>
inline void set_variables(const mxArray* opts, Input<Data>& in, Output& out,
                          model_info<mxArray*>& info, Data*& A, Data*&, Data*&) {
  A = set_dual_variables<Data>(opts, in.num_examples, out.num_classes, info);
}

template <typename Data, typename Result, template <typename> class Input, typename Output,
          template <typename, typename> class Objective>
inline void set_context(mxArray* plhs[], const mxArray* prhs[], const mxArray* opts,
                        Input<Data>&& in, Output&& out, Objective<Data, Result>&& obj) {
  Data *A(0), *W(0), *W0(0);
  model_info<mxArray*> info;
  set_variables(opts, in, out, info, A, W, W0);

  auto ctx = make_context(std::move(in), std::move(out), std::move(obj), A, W);
  ctx.primal_initial = W0;

  if (mxIsCell(prhs[0]) && mxIsCell(prhs[1])) {
    set_test_data(prhs[0], prhs[1], ctx);
  }

  set_stopping_criteria(opts, ctx);

  auto solver = sdca::make_solver(ctx);
  solver.solve();

  add_info(opts, ctx, info);
  plhs[0] = mxCreateStruct(info.fields, "model");
}

template <typename Data, typename Result, template <typename> class Input>
inline void set_objective(mxArray* plhs[], const mxArray* prhs[], const mxArray* opts,
                          const std::string& obj, Input<Data>&& in, multiclass_output&& out) {
  auto c = mxGetFieldValueOrDefault<Result>(opts, "c", 1);
  mxCheck<Result>(std::greater<Result>(), c, 0, "c");

  auto gamma = mxGetFieldValueOrDefault<Result>(opts, "gamma", 0);
  mxCheck<Result>(std::greater_equal<Result>(), gamma, 0, "gamma");

  auto k = mxGetFieldValueOrDefault<size_type>(opts, "k", 1);
  mxCheckRange<size_type>(k, 1, out.num_classes - 1, "k");

  if (obj == "multiclass_hinge") {
    k = 1;  // as cross_entropy, it has no top-k
    if (gamma > 0) {
      set_context(plhs, prhs, opts, std::move(in), std::move(out),
                  make_objective_l2_hinge_topk_smooth<Data>(c, k, gamma));
    } else {
      set_context(plhs, prhs, opts, std::move(in), std::move(out),
                  make_objective_l2_hinge_topk<Data>(c, k));
    }
  } else if (obj == "msvm" || obj == "l2_multiclass_hinge") {
    mxCheckRange<size_type>(k, 1, 1, "k");
    set_context(plhs, prhs, opts, std::move(in), std::move(out),
                make_objective_l2_topk_hinge<Data>(c, k));
  } else if (obj == "msvm_smooth" || obj == "l2_multiclass_hinge_smooth") {
    mxCheckRange<size_type>(k, 1, 1, "k");
    set_context(plhs, prhs, opts, std::move(in), std::move(out),
                make_objective_l2_topk_hinge_smooth<Data>(c, k, gamma));
  } else if (obj == "cross_entropy") {
    // As multiclass_hinge, it has no top-k: k is checked above, as in the other interfaces.
    set_context(plhs, prhs, opts, std::move(in), std::move(out),
                make_objective_l2_entropy<Data>(c));
  } else if (obj == "softmax" || obj == "l2_multiclass_entropy") {
    mxCheckRange<size_type>(k, 1, 1, "k");
    set_context(plhs, prhs, opts, std::move(in), std::move(out),
                make_objective_l2_entropy<Data>(c));
  } else if (obj == "l2_hinge_topk" || obj == "topk_hinge_alpha") {
    if (gamma > 0) {
      set_context(plhs, prhs, opts, std::move(in), std::move(out),
                  make_objective_l2_hinge_topk_smooth<Data>(c, k, gamma));
    } else {
      set_context(plhs, prhs, opts, std::move(in), std::move(out),
                  make_objective_l2_hinge_topk<Data>(c, k));
    }
  } else if (obj == "l2_topk_hinge" || obj == "topk_hinge_beta" || obj == "topk_hinge") {
    if (gamma > 0) {
      set_context(plhs, prhs, opts, std::move(in), std::move(out),
                  make_objective_l2_topk_hinge_smooth<Data>(c, k, gamma));
    } else {
      set_context(plhs, prhs, opts, std::move(in), std::move(out),
                  make_objective_l2_topk_hinge<Data>(c, k));
    }
  } else if (obj == "l2_entropy_topk") {
    set_context(plhs, prhs, opts, std::move(in), std::move(out),
                make_objective_l2_entropy_topk<Data>(c, k));
  } else {
    mex_error(err_id[err_objective], err_msg[err_objective], obj.c_str());
  }
}

template <typename Data, typename Result, template <typename> class Input>
inline void set_objective(mxArray* plhs[], const mxArray* prhs[], const mxArray* opts,
                          const std::string& obj, Input<Data>&& in, multilabel_output&& out) {
  auto c = mxGetFieldValueOrDefault<Result>(opts, "c", 1);
  mxCheck<Result>(std::greater<Result>(), c, 0, "c");

  auto gamma = mxGetFieldValueOrDefault<Result>(opts, "gamma", 0);
  mxCheck<Result>(std::greater_equal<Result>(), gamma, 0, "gamma");

  // Unused by the multilabel losses, but checked as in the other interfaces.
  auto k = mxGetFieldValueOrDefault<size_type>(opts, "k", 1);
  mxCheckRange<size_type>(k, 1, out.num_classes - 1, "k");

  if (obj == "multilabel_hinge") {
    if (gamma > 0) {
      set_context(plhs, prhs, opts, std::move(in), std::move(out),
                  make_objective_l2_multilabel_hinge_smooth<Data>(c, gamma));
    } else {
      set_context(plhs, prhs, opts, std::move(in), std::move(out),
                  make_objective_l2_multilabel_hinge<Data>(c));
    }
  } else if (obj == "mlsvm" || obj == "l2_multilabel_hinge") {
    set_context(plhs, prhs, opts, std::move(in), std::move(out),
                make_objective_l2_multilabel_hinge<Data>(c));
  } else if (obj == "mlsvm_smooth" || obj == "l2_multilabel_hinge_smooth") {
    set_context(plhs, prhs, opts, std::move(in), std::move(out),
                make_objective_l2_multilabel_hinge_smooth<Data>(c, gamma));
  } else if (obj == "mlsoftmax" || obj == "l2_multilabel_entropy") {
    set_context(plhs, prhs, opts, std::move(in), std::move(out),
                make_objective_l2_multilabel_entropy<Data>(c));
  } else {
    mex_error(err_id[err_objective], err_msg[err_objective], obj.c_str());
  }
}

// The shared objective names that set_objective takes by a historical name.
static const std::pair<const char*, const char*> shared_objectives[] = {
    {"topk_cross_entropy", "l2_entropy_topk"}, {"multilabel_cross_entropy", "mlsoftmax"}};

inline std::string historical_objective(const std::string& obj) {
  for (const auto& name : shared_objectives)
    if (obj == name.first) return name.second;
  return obj;
}

inline bool shared_objective(const std::string& obj) {
  for (int i = 0; i < num_shared_names - 1; ++i)
    if (obj == shared_names[i]) return true;
  return false;
}

template <typename Data, typename Result, template <typename> class Input>
inline void set_output(mxArray* plhs[], const mxArray* prhs[], const mxArray* opts,
                       const mxArray* labels, Input<Data>&& in) {
  validate_labels(labels, in.num_examples);
  std::string obj = historical_objective(
      mxGetFieldValueOrDefault(opts, "objective", std::string(default_objective)));
  if ((obj == "mlsoftmax") || (obj == "l2_multilabel_entropy") || (obj == "mlsvm") ||
      (obj == "l2_multilabel_hinge") || (obj == "mlsvm_smooth") ||
      (obj == "l2_multilabel_hinge_smooth") || (obj == "multilabel_hinge")) {
    // Multilabel
    set_objective<Data, Result>(plhs, prhs, opts, obj, std::move(in),
                                make_output<multilabel_output>(labels, in.num_examples));
  } else {
    // Multiclass
    set_objective<Data, Result>(plhs, prhs, opts, obj, std::move(in),
                                make_output<multiclass_output>(labels, in.num_examples));
  }
}

template <typename Data, typename Result>
inline void set_input(mxArray* plhs[], const mxArray* prhs[], const mxArray* opts) {
  bool is_dual = mxGetFieldValueOrDefault(opts, "is_dual", false);
  const mxArray* data;
  const mxArray* labels;
  if (mxIsNumeric(prhs[0]) && (mxIsNumeric(prhs[1]) || mxIsLogical(prhs[1]))) {
    data = prhs[0];
    labels = prhs[1];
  } else {
    mxCheckCellArrays(prhs[0], prhs[1]);
    data = mxGetCell(prhs[0], 0);
    labels = mxGetCell(prhs[1], 0);
  }

  validate_matrix<Data>(data, is_dual);
  mxCheckFinite<Data>(data, "data");
  Data* p_data = static_cast<Data*>(mxGetData(data));
  if (is_dual) {
    set_output<Data, Result>(plhs, prhs, opts, labels,
                             sdca::make_input_kernel(mxGetN(data), p_data));
  } else {
    set_output<Data, Result>(plhs, prhs, opts, labels,
                             sdca::make_input_feature(mxGetM(data), mxGetN(data), p_data));
  }
}

// mxSetField and mxRemoveField do not free the previous field value.
inline void set_field(mxArray* opts, const char* name, mxArray* value) {
  if (mxGetFieldNumber(opts, name) < 0) mxAddField(opts, name);
  mxArray* old = mxGetField(opts, 0, name);
  if (old != nullptr) mxDestroyArray(old);
  mxSetField(opts, 0, name, value);
}

// Moves the value of opts.(alias) to opts.(name).
inline void rename_field(mxArray* opts, const char* alias, const char* name) {
  mxArray* value = mxGetField(opts, 0, alias);
  if (value == nullptr) return;
  if (mxGetField(opts, 0, name) != nullptr)
    mex_error(err_id[err_arg], "Specify only opts.%s or its alias opts.%s.", name, alias);
  mxSetField(opts, 0, alias, nullptr);
  mxRemoveField(opts, mxGetFieldNumber(opts, alias));
  set_field(opts, name, value);
}

// Without opts.gamma, the historical names of v0.2.0 smooth with gamma = 1;
// the shared names default to 0 (nonsmooth). The model reports the given name.
inline void set_objective_smoothing(mxArray* opts) {
  if (mxGetField(opts, 0, "gamma") != nullptr) return;
  const std::string obj =
      mxGetFieldValueOrDefault(opts, "objective", std::string(default_objective));
  if (!shared_objective(obj)) set_field(opts, "gamma", mxCreateDoubleScalar(1));
}

// A copy of opts with the option names shared with the other interfaces
// (C, tol, kernel, center, initial_dual) renamed to the historical ones.
inline mxArray* shared_options(const mxArray* opts) {
  mxArray* copy = opts == nullptr ? mxCreateStructMatrix(1, 1, 0, nullptr) : mxDuplicateArray(opts);
  rename_field(copy, "C", "c");
  rename_field(copy, "tol", "epsilon");
  rename_field(copy, "kernel", "is_dual");
  rename_field(copy, "center", "W0");
  rename_field(copy, "initial_dual", "A");
  set_objective_smoothing(copy);
  return copy;
}

struct mx_array_deleter {
  void operator()(mxArray* pa) const { mxDestroyArray(pa); }
};

template <typename Data>
inline void mex_main(mxArray* plhs[], const int nrhs, const mxArray* prhs[]) {
  mxCheckStruct(nrhs > 2 ? prhs[2] : nullptr, "opts");
  std::unique_ptr<mxArray, mx_array_deleter> copy(shared_options(nrhs > 2 ? prhs[2] : nullptr));
  const mxArray* opts = copy.get();
  set_logging_options(opts);
  std::string precision = mxGetFieldValueOrDefault(opts, "precision", std::string("double"));
  if (precision == "double" || precision == "float64" || precision == "fp64") {
    set_input<Data, double>(plhs, prhs, opts);
  } else if (precision == "single" || precision == "float32" || precision == "fp32") {
    set_input<Data, float>(plhs, prhs, opts);
  } else {
    mex_error(err_id[err_precision], err_msg[err_precision], precision.c_str());
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
      if (mxIsDouble(prhs[0]) ||
          (mxIsCell(prhs[0]) && !mxIsEmpty(prhs[0]) && mxIsDouble(mxGetCell(prhs[0], 0)))) {
        mex_main<double>(plhs, nrhs, prhs);
      } else if (mxIsSingle(prhs[0]) ||
                 (mxIsCell(prhs[0]) && !mxIsEmpty(prhs[0]) && mxIsSingle(mxGetCell(prhs[0], 0)))) {
        mex_main<float>(plhs, nrhs, prhs);
      } else {
        mex_error(err_id[err_arg], err_msg[err_arg]);
      }
    } catch (const std::exception& e) {
      mex_error(err_id[err_exception], err_msg[err_exception], e.what());
    }
  }
}
