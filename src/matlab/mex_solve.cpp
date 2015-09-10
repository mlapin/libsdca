#include "mex_util.h"
#include "solve/solve.h"

#ifndef MEX_SOLVE
#define MEX_SOLVE "mex_solve"
#endif
#ifndef LIBSDCA_VERSION
#define LIBSDCA_VERSION "0.0.0"
#endif

using namespace sdca;

inline void
printUsage() {
  mexPrintf("Usage: MODEL = %s(X, Y);\n"
            "       MODEL = %s(DATA, LABELS, OPTS);\n"
            "  See %s('version') and %s('help') for more information.\n",
            MEX_SOLVE, MEX_SOLVE, MEX_SOLVE, MEX_SOLVE);
}

inline void
printVersion() {
  mexPrintf("%s version %s.\n", MEX_SOLVE, LIBSDCA_VERSION);
}

inline void
printHelp() {
  mexPrintf("Usage: MODEL = %s(X, Y);\n"
            "       MODEL = %s(DATA, LABELS, OPTS);\n"
            "  DATA is either a d-by-n feature matrix (default)\n"
            "              or a n-by-n Gram matrix (requires is_dual=true).\n"
            "  LABELS is a n-by-1 or a 1-by-n vector of class labels.\n"
            "    If m is the number of classes (tasks),\n"
            "    then LABELS must be in the range 0:(m-1) or 1:m.\n"
            "  DATA can be single or double; LABELS must be double.\n"
            "\n"
            "  OPTS is a struct with the following fields:\n"
            "  (default values in [brackets], synonyms in (parenthesis))\n"
            "    'objective' :\n"
            "      ['l2_topk_hinge'] : L2-regularized top-k-of-hinge loss;\n"
            "       'l2_hinge_topk'  : L2-regularized hinge-of-top-k loss.\n"
            "  Regularization parameter (standard SVM C parameter):\n"
            "    'c' : [1], divided by n in the objective;\n"
            "    'C' : [c/n], used directly in the objective.\n"
            "  Loss smoothing parameter:\n"
            "    'gamma' : [0], use gamma > 0 for a smooth version of\n"
            "                   l2_topk_hinge and l2_hinge_topk.\n"
            "  Top-k loss parameter:\n"
            "    'k' : [1], note that for k=1 and gamma=0 both\n"
            "               l2_topk_hinge and l2_hinge_topk coincide\n"
            "               with the multiclass SVM of Crammer and Singer.\n"
            "  Training in the dual:\n"
            "    'is_dual' : [false], if DATA is the Gram matrix,\n"
            "                         set is_dual=true.\n"
            "    No primal variables (W) are maintained if is_dual=true.\n"
            "    Moreover, training in the dual is faster for d>=n.\n"
            "  Stopping criteria:\n"
            "    'check_on_start' : [false], check duality gap on start;\n"
            "    'check_epoch'    : [1], how often to check the gap;\n"
            "    'max_epoch'      : [1000], epochs limit;\n"
            "    'max_cpu_time'   : [0], CPU time limit (0: no limit);\n"
            "    'max_wall_time'  : [0], wall time limit (0: no limit);\n"
            "    'epsilon'        : [1e-3], relative duality gap bound:\n"
            "      (primal - dual) <= epsilon * max(abs(primal), abs(dual)).\n"
            "  Warm restart (resume training):\n"
            "    'W' : initial primal variables (if is_dual=false);\n"
            "    'A' : initial dual variables such that W = X*A'.\n"
            "    If only A is available, use check_on_start=true.\n"
            "    Note that one can simply pass a computed MODEL as OPTS.\n"
            "  Logging options:\n"
            "    'log_level'  : 'none', ['info'], 'verbose', 'debug'.\n"
            "    'log_format' : ['short_f'], 'short_e', 'long_f', 'long_e'.\n"
            "                   short/long: 4/15 digits; f/e: float/exp fmt.\n"
            "  Options that influence the accuracy of computations:\n"
            "    'precision' : 'single' ('float'), ['double'],\n"
            "                  'long double' ('long_double');\n"
            "    'summation' : ['standard'] ('default'), 'kahan'.\n"
            "\n"
            "  Prediction scores can be computed as:\n"
            "    scores = model.W' * X;\n"
            "    scores = model.A * Xtrn' * Xtst.\n",
            MEX_SOLVE, MEX_SOLVE);
}

template <typename Result>
inline void
add_records(
    const std::vector<train_point<Result>>& records,
    model_info<mxArray*>& info
  ) {
  const char* names[] = {"epoch", "primal", "dual", "gap",
    "loss", "dual_loss", "regularizer", "wall_time", "cpu_time",
    "solve_wall_time", "solve_cpu_time", "eval_wall_time", "eval_cpu_time"};
  mxArray* pa = mxCreateStructMatrix(records.size(), 1, 13, names);
  mxCheckCreated(pa, "records");
  size_type i = 0;
  for (auto& a : records) {
    mxSetFieldByNumber(pa, i, 0, mxCreateScalar(a.epoch));
    mxSetFieldByNumber(pa, i, 1, mxCreateScalar(a.primal));
    mxSetFieldByNumber(pa, i, 2, mxCreateScalar(a.dual));
    mxSetFieldByNumber(pa, i, 3, mxCreateScalar(a.gap));
    mxSetFieldByNumber(pa, i, 4, mxCreateScalar(a.primal_loss));
    mxSetFieldByNumber(pa, i, 5, mxCreateScalar(a.dual_loss));
    mxSetFieldByNumber(pa, i, 6, mxCreateScalar(a.regularizer));
    mxSetFieldByNumber(pa, i, 7, mxCreateScalar(a.wall_time));
    mxSetFieldByNumber(pa, i, 8, mxCreateScalar(a.cpu_time));
    mxSetFieldByNumber(pa, i, 9, mxCreateScalar(a.solve_wall_time));
    mxSetFieldByNumber(pa, i, 10, mxCreateScalar(a.solve_cpu_time));
    mxSetFieldByNumber(pa, i, 11, mxCreateScalar(a.eval_wall_time));
    mxSetFieldByNumber(pa, i, 12, mxCreateScalar(a.eval_cpu_time));
    ++i;
  }
  info.add("records", pa);
}

template <typename Result>
inline void
add_evaluations(
    const std::vector<std::vector<test_point<Result>>>& evals,
    model_info<mxArray*>& info
  ) {
  const char* names[] = {"loss", "accuracy"};
  mxArray* pa = mxCreateStructMatrix(evals[0].size(), evals.size(), 2, names);
  mxCheckCreated(pa, "evals");
  size_type i = 0;
  for (auto& dataset_evals : evals) {
    for (auto& a : dataset_evals) {
      mxSetFieldByNumber(pa, i, 0, mxCreateScalar(a.loss));
      mxSetFieldByNumber(pa, i, 1, mxCreateVector(a.accuracy, "accuracy"));
      ++i;
    }
  }
  info.add("evals", pa);
}

template <typename Solver>
inline void
solve_objective_add_info(
    Solver solver,
    model_info<mxArray*>& info
  ) {
  solver.solve();
  info.add("status", mxCreateString(solver.status_name().c_str()));
  info.add("primal", mxCreateScalar(solver.primal()));
  info.add("dual", mxCreateScalar(solver.dual()));
  info.add("absolute_gap", mxCreateScalar(solver.absolute_gap()));
  info.add("relative_gap", mxCreateScalar(solver.relative_gap()));
  info.add("epoch", mxCreateScalar(solver.epoch()));
  info.add("wall_time", mxCreateScalar(solver.wall_time()));
  info.add("cpu_time", mxCreateScalar(solver.cpu_time()));
  info.add("solve_wall_time", mxCreateScalar(solver.solve_wall_time()));
  info.add("solve_cpu_time", mxCreateScalar(solver.solve_cpu_time()));
  info.add("eval_wall_time", mxCreateScalar(solver.eval_wall_time()));
  info.add("eval_cpu_time", mxCreateScalar(solver.eval_cpu_time()));
  add_records(solver.records(), info);
  add_evaluations(solver.evaluations(), info);
}

template <typename Objective,
          typename Data>
inline void
make_solver_solve(
    solver_context<Data>& context,
    model_info<mxArray*>& info,
    const Objective& objective
  ) {
  typedef typename Objective::result_type Result;
  if (context.is_dual) {
    solve_objective_add_info(
      dual_solver<Objective, Data, Result>(objective, context), info);
  } else {
    solve_objective_add_info(
      primal_solver<Objective, Data, Result>(objective, context), info);
  }
}

template <typename Data>
inline void
set_variables(
    const dataset<Data>& trn_data,
    const mxArray* opts,
    solver_context<Data>& context,
    model_info<mxArray*>& info
  ) {
  mxArray *mxA = mxDuplicateFieldOrCreateMatrix(opts, "A",
    trn_data.num_tasks, trn_data.num_examples, mex_class<Data>::id());
  context.dual_variables = static_cast<Data*>(mxGetData(mxA));
  info.add("A", mxA);

  if (!context.is_dual) {
    mxArray *mxW = mxDuplicateFieldOrCreateMatrix(opts, "W",
      trn_data.num_dimensions, trn_data.num_tasks, mex_class<Data>::id());
    context.primal_variables = static_cast<Data*>(mxGetData(mxW));
    info.add("W", mxW);
  }
}

template <typename Data>
inline void
set_stopping_criteria(
    const mxArray* opts,
    solver_context<Data>& context
  ) {
  auto c = &context.criteria;
  mxSetFieldValue(opts, "check_on_start", c->check_on_start);
  mxSetFieldValue(opts, "check_epoch", c->check_epoch);
  mxSetFieldValue(opts, "max_epoch", c->max_epoch);
  mxSetFieldValue(opts, "max_cpu_time", c->max_cpu_time);
  mxSetFieldValue(opts, "max_wall_time", c->max_wall_time);
  mxSetFieldValue(opts, "epsilon", c->epsilon);
  mxCheck<size_type>(std::greater_equal<size_type>(),
    c->check_epoch, 0, "check_epoch");
  mxCheck<size_type>(std::greater_equal<size_type>(),
    c->max_epoch, 0, "max_epoch");
  mxCheck<double>(std::greater_equal<double>(),
    c->max_cpu_time, 0, "max_cpu_time");
  mxCheck<double>(std::greater_equal<double>(),
    c->max_wall_time, 0, "max_wall_time");
  mxCheck<double>(std::greater_equal<double>(),
    c->epsilon, 0, "epsilon");
}

template <typename Data>
inline void
set_labels(
    const mxArray* labels,
    dataset<Data>& data_set
  ) {
  size_type n = data_set.num_examples;
  mxCheckVector(labels, "labels", n);

  std::vector<size_type> vec(mxGetPr(labels), mxGetPr(labels) + n);
  auto minmax = std::minmax_element(vec.begin(), vec.end());
  if (*minmax.first == 1) {
    std::for_each(vec.begin(), vec.end(), [](size_type &x){ x -= 1; });
  } else if (*minmax.first != 0) {
    mexErrMsgIdAndTxt(err_id[err_labels_range], err_msg[err_labels_range]);
  }

  data_set.labels = std::move(vec);
  data_set.num_tasks = static_cast<size_type>(*minmax.second) + 1;
}

template <typename Data>
inline void
set_dataset(
    const mxArray* data,
    const mxArray* labels,
    solver_context<Data>& context
  ) {
  mxCheckNotSparse(data, "data");
  mxCheckNotEmpty(data, "data");
  mxCheckReal(data, "data");
  mxCheckClass(data, "data", mex_class<Data>::id());

  mxCheckNotSparse(labels, "labels");
  mxCheckNotEmpty(labels, "labels");
  mxCheckDouble(labels, "labels");

  dataset<Data> data_set;
  data_set.data = static_cast<Data*>(mxGetData(data));
  data_set.num_dimensions = (context.is_dual) ? 0 : mxGetM(data);
  data_set.num_examples = mxGetN(data);
  set_labels(labels, data_set);

  context.datasets.emplace_back(data_set);
}

template <typename Data>
inline void
set_datasets(
    const mxArray* data,
    const mxArray* labels,
    solver_context<Data>& context
  ) {
  if (mxIsNumeric(data)) {
    if (context.is_dual) {
      mxCheckSquare(data, "data");
    }
    set_dataset(data, labels, context);
  } else {
    mxCheckCellArrays(data, labels);
    if (context.is_dual) {
      mxCheckSquare(mxGetCell(data, 0), "data");
    }

    // Training dataset
    set_dataset(mxGetCell(data, 0), mxGetCell(labels, 0), context);
    size_type num_dimensions = context.datasets[0].num_dimensions;
    size_type num_examples = context.datasets[0].num_examples;
    size_type num_tasks = context.datasets[0].num_tasks;

    // Testing datasets
    size_type num_datasets = mxGetNumberOfElements(data);
    for (size_type i = 1; i < num_datasets; ++i) {
      set_dataset(mxGetCell(data, i), mxGetCell(labels, i), context);
      if (num_dimensions != context.datasets[i].num_dimensions) {
        mexErrMsgIdAndTxt(err_id[err_num_dim], err_msg[err_num_dim], i + 1);
      }
      if (num_tasks != context.datasets[i].num_tasks) {
        mexErrMsgIdAndTxt(err_id[err_num_tasks], err_msg[err_num_tasks], i + 1);
      }
      if (context.is_dual && num_examples != mxGetM(mxGetCell(data, i))) {
        mexErrMsgIdAndTxt(err_id[err_num_ex], err_msg[err_num_ex], i + 1);
      }
    }
  }
}

template <typename Data,
          typename Result,
          typename Summation>
void
mex_main(
    mxArray* plhs[],
    const mxArray* prhs[],
    const mxArray* opts,
    Summation& sum
    ) {
  solver_context<Data> context;
  context.is_dual = mxGetFieldValueOrDefault(opts, "is_dual", false);
  set_datasets(prhs[0], prhs[1], context);
  set_stopping_criteria(opts, context);

  model_info<mxArray*> info;
  auto trn_data = context.datasets.front();
  set_variables(trn_data, opts, context, info);

  info.add("is_dual", mxCreateScalar(context.is_dual));
  if (!context.is_dual) {
    info.add("num_dimensions", mxCreateScalar(trn_data.num_dimensions));
  }
  info.add("num_examples", mxCreateScalar(trn_data.num_examples));
  info.add("num_tasks", mxCreateScalar(trn_data.num_tasks));

  std::string objective = mxGetFieldValueOrDefault(
    opts, "objective", std::string("l2_topk_hinge"));
  info.add("objective", mxCreateString(objective.c_str()));

  auto c = mxGetFieldValueOrDefault<Result>(opts, "c", 1);
  mxCheck<Result>(std::greater<Result>(), c, 0, "c");

  Result num_examples(static_cast<Result>(trn_data.num_examples));
  auto C = mxGetFieldValueOrDefault<Result>(opts, "C", c / num_examples);
  mxCheck<Result>(std::greater<Result>(), C, 0, "C");

  c = (C != c / num_examples) ? C * num_examples : c;
  info.add("c", mxCreateScalar(c));
  info.add("C", mxCreateScalar(C));

  auto k = mxGetFieldValueOrDefault<size_type>(opts, "k", 1);
  mxCheckRange<size_type>(k, 1, trn_data.num_tasks - 1, "k");

  auto gamma = mxGetFieldValueOrDefault<Result>(opts, "gamma", 0);
  mxCheck<Result>(std::greater_equal<Result>(), gamma, 0, "gamma");

  if (objective == "l2_entropy_topk") {
    info.add("k", mxCreateScalar(k));
    make_solver_solve(context, info,
      l2_entropy_topk<Data, Result, Summation>(k, C, sum));
  } else if (objective == "l2_topk_hinge") {
    info.add("k", mxCreateScalar(k));
    info.add("gamma", mxCreateScalar(gamma));
    if (gamma > 0) {
      make_solver_solve(context, info,
        l2_topk_hinge_smooth<Data, Result, Summation>(k, C, gamma, sum));
    } else {
      make_solver_solve(context, info,
        l2_topk_hinge<Data, Result, Summation>(k, C, sum));
    }
  } else if (objective == "l2_hinge_topk") {
    info.add("k", mxCreateScalar(k));
    info.add("gamma", mxCreateScalar(gamma));
    if (gamma > 0) {
      make_solver_solve(context, info,
        l2_hinge_topk_smooth<Data, Result, Summation>(k, C, gamma, sum));
    } else {
      make_solver_solve(context, info,
        l2_hinge_topk<Data, Result, Summation>(k, C, sum));
    }
  } else {
    mexErrMsgIdAndTxt(
      err_id[err_objective], err_msg[err_objective], objective.c_str());
  }

  info.add("check_on_start", mxCreateScalar(context.criteria.check_on_start));
  info.add("check_epoch", mxCreateScalar(context.criteria.check_epoch));
  info.add("max_epoch", mxCreateScalar(context.criteria.max_epoch));
  info.add("max_cpu_time", mxCreateScalar(context.criteria.max_cpu_time));
  info.add("max_wall_time", mxCreateScalar(context.criteria.max_wall_time));
  info.add("epsilon", mxCreateScalar(context.criteria.epsilon));
  info.add("log_level", mxCreateString(logging::get_level_name()));
  info.add("log_format", mxCreateString(logging::get_format_name()));
  info.add("summation", mxCreateString(sum.name()));
  info.add("precision", mxCreateString(type_traits<Result>::name()));
  info.add("data_precision", mxCreateString(type_traits<Data>::name()));

  plhs[0] = mxCreateStruct(info.fields, "model");
}

template <typename Data,
          typename Result>
inline void
mex_main(
    mxArray* plhs[],
    const mxArray* prhs[],
    const mxArray* opts
    ) {
  std::string summation = mxGetFieldValueOrDefault(
    opts, "summation", std::string("standard"));
  if (summation == "standard" || summation == "default") {
    std_sum<Data*, Result> sum;
    mex_main<Data, Result, std_sum<Data*, Result>>(plhs, prhs, opts, sum);
  } else if (summation == "kahan") {
    kahan_sum<Data*, Result> sum;
    mex_main<Data, Result, kahan_sum<Data*, Result>>(plhs, prhs, opts, sum);
  } else {
    mexErrMsgIdAndTxt(
      err_id[err_summation], err_msg[err_summation], summation.c_str());
  }
}

inline void
set_logging_options(
    const mxArray* opts
  ) {
  std::string log_level = mxGetFieldValueOrDefault(
    opts, "log_level", std::string("info"));
  if (log_level == "none") {
    logging::set_level(logging::none);
  } else if (log_level == "info") {
    logging::set_level(logging::info);
  } else if (log_level == "verbose") {
    logging::set_level(logging::verbose);
  } else if (log_level == "debug") {
    logging::set_level(logging::debug);
  } else {
    mexErrMsgIdAndTxt(
      err_id[err_log_level], err_msg[err_log_level], log_level.c_str());
  }

  std::string log_format = mxGetFieldValueOrDefault(
    opts, "log_format", std::string("short_f"));
  if (log_format == "short_f") {
    logging::set_format(logging::short_f);
  } else if (log_format == "short_e") {
    logging::set_format(logging::short_e);
  } else if (log_format == "long_f") {
    logging::set_format(logging::long_f);
  } else if (log_format == "long_e") {
    logging::set_format(logging::long_e);
  } else {
    mexErrMsgIdAndTxt(
      err_id[err_log_format], err_msg[err_log_format], log_format.c_str());
  }
}

template <typename Data>
inline void
mex_main(
    mxArray* plhs[],
    const int nrhs,
    const mxArray* prhs[]
    ) {
  const mxArray* opts = (nrhs > 2) ? prhs[2] : nullptr;
  mxCheckStruct(opts, "opts");
  set_logging_options(opts);
  std::string precision = mxGetFieldValueOrDefault(
    opts, "precision", std::string("double"));
  if (precision == "double") {
    mex_main<Data, double>(plhs, prhs, opts);
  } else if (precision == "single" || precision == "float") {
    mex_main<Data, float>(plhs, prhs, opts);
  } else if (precision == "long double" || precision == "long_double") {
    mex_main<Data, long double>(plhs, prhs, opts);
  } else {
    mexErrMsgIdAndTxt(
      err_id[err_precision], err_msg[err_precision], precision.c_str());
  }
}

void
mexFunction(
    const int nlhs,
    mxArray* plhs[],
    const int nrhs,
    const mxArray* prhs[]
    ) {
  mxCheckArgNum(nrhs, 1, 3, printUsage);
  mxCheckArgNum(nlhs, 0, 1, printUsage);

  if (mxIsChar(prhs[0])) {
    std::string command = mxGetString(prhs[0], "command");
    if (command == "help" || command == "--help" || command == "-h") {
      printHelp();
    } else if (command == "version" || command == "--version") {
      printVersion();
    } else {
      mexErrMsgIdAndTxt(
        err_id[err_command], err_msg[err_command], command.c_str());
    }
  } else {
    mxCheckArgNum(nrhs, 2, 3, printUsage);

    logging::format_push();
    mat_cout_hijack mat_cout;
    if (mxIsDouble(prhs[0]) || (mxIsCell(prhs[0]) && !mxIsEmpty(prhs[0])
        && mxIsDouble(mxGetCell(prhs[0], 0)))) {
       mex_main<double>(plhs, nrhs, prhs);
    } else if (mxIsSingle(prhs[0]) || (mxIsCell(prhs[0]) && !mxIsEmpty(prhs[0])
        && mxIsSingle(mxGetCell(prhs[0], 0)))) {
       mex_main<float>(plhs, nrhs, prhs);
    } else {
      mexErrMsgIdAndTxt(err_id[err_arg], err_msg[err_arg]);
    }
    mat_cout.release();
    logging::format_pop();
  }
}
