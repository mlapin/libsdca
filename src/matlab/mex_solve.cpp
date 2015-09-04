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


template <typename Data,
          typename Result>
struct solver_context {
  typedef Data data_type;
  typedef Result result_type;
  bool is_dual = false;
  stopping_criteria criteria;
  std::vector<dataset<Data>> datasets;
  std::vector<std::pair<const char*, const mxArray*>> fields;
};

template <typename Data,
          typename Result>
inline void
add_field(
    const char* name,
    const mxArray* value,
    solver_context<Data, Result>& context
  ) {
  context.fields.emplace_back(std::make_pair(name, value));
}

template <typename Data,
          typename Result,
          typename Type>
inline void
add_field_scalar(
    const char* name,
    const Type value,
    solver_context<Data, Result>& context
  ) {
  context.fields.emplace_back(std::make_pair(name, mxCreateScalar(value)));
}

template <typename Data,
          typename Result,
          typename Type>
inline void
add_field_opts_value(
    const mxArray* opts,
    const char* name,
    Type& value,
    solver_context<Data, Result>& context
  ) {
  mxSetFieldValue(opts, name, value);
  context.fields.emplace_back(std::make_pair(name, mxCreateScalar(value)));
}

template <typename Data,
          typename Result>
inline mxArray*
add_field_opts_matrix(
    const mxArray* opts,
    const char* name,
    const size_type m,
    const size_type n,
    const mxClassID class_id,
    solver_context<Data, Result>& context
  ) {
  mxArray *pa = mxGetField(opts, 0, name);
  if (pa != nullptr) {
    mxCheckMatrix(pa, name, m, n, class_id);
    pa = mxDuplicateArray(pa);
  } else {
    pa = mxCreateNumericMatrix(m, n, class_id, mxREAL);
  }
  mxCheckCreated(pa, name);
  add_field(name, pa, context);
  return pa;
}

template <typename Data,
          typename Result>
inline void
add_states(
    const std::vector<state<Result>>& states,
    solver_context<Data, Result>& context
  ) {
  const char* names[] =
    {"epoch", "cpu_time", "wall_time", "primal", "dual", "gap"};
  mxArray* pa = mxCreateStructMatrix(states.size(), 1, 6, names);
  mxCheckCreated(pa, "states");
  size_type i = 0;
  for (auto& state : states) {
    mxSetFieldByNumber(pa, i, 0, mxCreateScalar(state.epoch));
    mxSetFieldByNumber(pa, i, 1, mxCreateScalar(state.cpu_time));
    mxSetFieldByNumber(pa, i, 2, mxCreateScalar(state.wall_time));
    mxSetFieldByNumber(pa, i, 3, mxCreateScalar(state.primal));
    mxSetFieldByNumber(pa, i, 4, mxCreateScalar(state.dual));
    mxSetFieldByNumber(pa, i, 5, mxCreateScalar(state.gap));
    ++i;
  }
  context.fields.emplace_back(
    std::make_pair("states", const_cast<const mxArray*>(pa)));
}

template <typename Solver,
          typename Data,
          typename Result>
inline void
solve_model(
    Solver solver,
    solver_context<Data, Result>& context
  ) {
  solver.solve();
  add_field("status", mxCreateString(solver.status_name().c_str()), context);
  add_field_scalar("primal", solver.primal(), context);
  add_field_scalar("dual", solver.dual(), context);
  add_field_scalar("absolute_gap", solver.absolute_gap(), context);
  add_field_scalar("relative_gap", solver.relative_gap(), context);
  add_field_scalar("epoch", solver.epoch(), context);
  add_field_scalar("cpu_time", solver.cpu_time(), context);
  add_field_scalar("wall_time", solver.wall_time(), context);
  add_states(solver.states(), context);
}

template <typename Objective,
          typename Data,
          typename Result>
inline void
make_solver_solve(
    solver_context<Data, Result>& context,
    const Objective& objective
  ) {
  if (context.is_dual) {
    solve_model(dual_solver<Objective, Data, Result>(
      context.datasets.front(), context.criteria, objective), context);
  } else {
    solve_model(primal_solver<Objective, Data, Result>(
      context.datasets.front(), context.criteria, objective), context);
  }
}

template <typename Data,
          typename Result>
inline void
set_logging_options(
    const mxArray* opts,
    solver_context<Data, Result>& context
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

  add_field("log_level", mxCreateString(log_level.c_str()), context);
  add_field("log_format", mxCreateString(log_format.c_str()), context);
}

template <typename Data,
          typename Result>
inline void
set_stopping_criteria(
    const mxArray* opts,
    solver_context<Data, Result>& context
  ) {
  auto c = &context.criteria;
  add_field_opts_value(opts, "check_on_start", c->check_on_start, context);
  add_field_opts_value(opts, "check_epoch", c->check_epoch, context);
  add_field_opts_value(opts, "max_epoch", c->max_epoch, context);
  add_field_opts_value(opts, "max_cpu_time", c->max_cpu_time, context);
  add_field_opts_value(opts, "max_wall_time", c->max_wall_time, context);
  add_field_opts_value(opts, "epsilon", c->epsilon, context);
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

template <typename Data,
          typename Result,
          typename Summation>
inline void
set_precision_options(
    Summation& sum,
    solver_context<Data, Result>& context
  ) {
  add_field("summation", mxCreateString(sum.name()), context);
  add_field("precision", mxCreateString(type_traits<Result>::name()), context);
  add_field("data_precision",
    mxCreateString(type_traits<Data>::name()), context);
}

template <typename Data>
inline void
set_labels(
    const mxArray* labels,
    dataset<Data>& data_set
  ) {
  auto n = data_set.num_examples;
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

template <typename Data,
          typename Result>
inline void
set_datasets(
    const mxArray* data,
    const mxArray* labels,
    const mxArray* opts,
    solver_context<Data, Result>& context
  ) {
  dataset<Data> trn_data;
  trn_data.data = static_cast<Data*>(mxGetData(data));
  trn_data.num_examples = mxGetN(data);
  set_labels(labels, trn_data);

  context.is_dual = mxGetFieldValueOrDefault(opts, "is_dual", false);
  if (context.is_dual) {
    // Work in the dual
    trn_data.num_dimensions = 0;
    mxCheckSquare(data, "data");
  } else {
    // Work in the primal
    trn_data.num_dimensions = mxGetM(data);
    mxArray *mxW = add_field_opts_matrix(opts, "W",
      trn_data.num_dimensions, trn_data.num_tasks, mxGetClassID(data), context);
    trn_data.primal_variables = static_cast<Data*>(mxGetData(mxW));
  }

  mxArray *mxA = add_field_opts_matrix(opts, "A",
    trn_data.num_tasks, trn_data.num_examples, mxGetClassID(data), context);
  trn_data.dual_variables = static_cast<Data*>(mxGetData(mxA));

  add_field_scalar("num_dimensions", trn_data.num_dimensions, context);
  add_field_scalar("num_examples", trn_data.num_examples, context);
  add_field_scalar("num_tasks", trn_data.num_tasks, context);
  add_field_scalar("is_dual", context.is_dual, context);
  context.datasets.emplace_back(trn_data);
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
  solver_context<Data, Result> context;
  set_datasets(prhs[0], prhs[1], opts, context);
  set_logging_options(opts, context);
  set_precision_options(sum, context);
  set_stopping_criteria(opts, context);

  auto trn_data = context.datasets.front();

  std::string objective = mxGetFieldValueOrDefault(
    opts, "objective", std::string("l2_topk_hinge"));
  add_field("objective", mxCreateString(objective.c_str()), context);

  auto c = mxGetFieldValueOrDefault<Result>(opts, "c", 1);
  mxCheck<Result>(std::greater<Result>(), c, 0, "c");

  Result num_examples(static_cast<Result>(trn_data.num_examples));
  auto C = mxGetFieldValueOrDefault<Result>(opts, "C", c / num_examples);
  mxCheck<Result>(std::greater<Result>(), C, 0, "C");

  c = (C != c / num_examples) ? C * num_examples : c;
  add_field_scalar("c", c, context);
  add_field_scalar("C", C, context);

  auto k = mxGetFieldValueOrDefault<size_type>(opts, "k", 1);
  mxCheckRange<size_type>(k, 1, trn_data.num_tasks - 1, "k");

  auto gamma = mxGetFieldValueOrDefault<Result>(opts, "gamma", 0);
  mxCheck<Result>(std::greater_equal<Result>(), gamma, 0, "gamma");

  if (objective == "l2_entropy_topk") {
    add_field_scalar("k", k, context);
    make_solver_solve(context,
      l2_entropy_topk<Data, Result, Summation>(k, C, sum));
  } else if (objective == "l2_topk_hinge") {
    add_field_scalar("k", k, context);
    add_field_scalar("gamma", gamma, context);
    if (gamma > 0) {
      make_solver_solve(context,
        l2_topk_hinge_smooth<Data, Result, Summation>(k, C, gamma, sum));
    } else {
      make_solver_solve(context,
        l2_topk_hinge<Data, Result, Summation>(k, C, sum));
    }
  } else if (objective == "l2_hinge_topk") {
    add_field_scalar("k", k, context);
    add_field_scalar("gamma", gamma, context);
    if (gamma > 0) {
      make_solver_solve(context,
        l2_hinge_topk_smooth<Data, Result, Summation>(k, C, gamma, sum));
    } else {
      make_solver_solve(context,
        l2_hinge_topk<Data, Result, Summation>(k, C, sum));
    }
  } else {
    mexErrMsgIdAndTxt(
      err_id[err_objective], err_msg[err_objective], objective.c_str());
  }

  plhs[0] = mxCreateStruct(context.fields, "model");
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

template <typename Data>
inline void
mex_main(
    mxArray* plhs[],
    const int nrhs,
    const mxArray* prhs[]
    ) {
  const mxArray* opts = (nrhs > 2) ? prhs[2] : nullptr;
  mxCheckStruct(opts, "opts");
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

    mxCheckNotSparse(prhs[0], "data");
    mxCheckNotEmpty(prhs[0], "data");
    mxCheckReal(prhs[0], "data");

    mxCheckNotSparse(prhs[1], "labels");
    mxCheckNotEmpty(prhs[1], "labels");
    mxCheckDouble(prhs[1], "labels");

    logging::format_push();
    mat_cout_hijack mat_cout;
    if (mxIsDouble(prhs[0])) {
       mex_main<double>(plhs, nrhs, prhs);
    } else if (mxIsSingle(prhs[0])) {
       mex_main<float>(plhs, nrhs, prhs);
    }
    mat_cout.release();
    logging::format_pop();
  }
}
