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
  mexPrintf("Usage: model = %s(data, labels, opts);\n"
            "  See %s('help') and %s('version') for more information.\n",
            MEX_SOLVE, MEX_SOLVE, MEX_SOLVE);
}

inline void
printVersion() {
  mexPrintf("%s version %s.\n", MEX_SOLVE, LIBSDCA_VERSION);
}

inline void
printHelp(const mxArray* opts) {
  if (opts == nullptr) {
    mexPrintf(
"Usage: model = %s(data, labels, opts);\n"
"  Optimizes an objective given in opts.objective using the data and labels.\n"
"\n"
"  opts is a struct with the following fields (defaults in [brackets]):\n"
"\n"
"    objective ['topk_svm']  - the objective to optimize;\n"
"    C         [1]           - the regularization parameter;\n"
"    k         [1]           - the k in top-k optimization;\n"
"    gamma     [1]           - the smoothing parameter for hinge losses;\n"
"    is_dual   [false]       - whether data is given as Gram matrix;\n"
"\n"
"    check_on_start [false]  - whether to check the duality gap on start;\n"
"    check_epoch    [10]     - how often to check the gap;\n"
"    max_epoch      [1000]   - epochs limit;\n"
"    max_cpu_time   [0]      - CPU time limit (0: no limit);\n"
"    max_wall_time  [0]      - wall time limit (0: no limit);\n"
"    epsilon        [1e-3]   - relative duality gap bound, stop if\n"
"      (primal - dual) <= epsilon * max(abs(primal), abs(dual))\n"
"\n"
"    log_level  ['info']     - logging verbosity\n"
"                              ('none', 'info', 'verbose', 'debug')\n"
"    log_format ['short_e']  - numeric format (4/15 digits; float/exp format)\n"
"                              ('short_f', 'short_e', 'long_f', 'long_e')\n"
"    precision  ['double']   - intermediate floating-point precision;\n"
"    summation  ['standard'] - summation method (Kahan or standard);\n"
"\n"
"    A [none] - initial dual variables for warm restart;\n"
"    W [none] - initial primal variables (only if opts.is_dual=false);\n"
"\n"
"  Prediction scores can be computed as:\n"
"    scores = model.W' * X;\n"
"    scores = model.A * Xtrn' * Xtst;\n"
"\n"
"  See %s('help', 'data') for more information on\n"
"  supported input data formats.\n"
"  See %s('help', 'objective') for more information on\n"
"  currently supported training objectives.\n",
    MEX_SOLVE, MEX_SOLVE, MEX_SOLVE);
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
"        Ktst = Xtrn' * Xtst.\n"
"\n"
"  labels can be:\n"
"    - a n-by-1 or a 1-by-n vector of class labels;\n"
"      labels must be in the range 0:(m-1) or 1:m,\n"
"      where m is the number of classes.\n"
"    - a cell array with the same number of elements as in data\n"
"      containing vectors of labels as above.\n"
"\n"
"  data matrices must be non-sparse and of type single or double\n"
"\n"
"  labels vectors must be non-sparse and of type double\n"
      );
  } else if (arg == "obj" || arg == "objective") {
    mexPrintf(
"opts.objective - the training objective to optimize.\n"
"  Possible values:\n"
"    msvm (synonym: multi_svm)\n"
"      - multiclass SVM of Crammer and Singer\n"
"    l2_hinge_topk (synonyms: topk_svm, topk_hinge_alpha)\n"
"      - l2 regularized hinge-of-top-k loss (top-k hinge alpha)\n"
"    l2_topk_hinge (synonym: topk_hinge_beta)\n"
"      - l2 regularized top-k-of-hinge loss (top-k hinge beta)\n"
"    l2_entropy_topk (synonyms: softmax)\n"
"      - l2 regularized entropy-on-top-k-simplex loss\n"
"        (reduced to the usual softmax loss for k=1)\n"
"  Default value:\n"
"    topk_svm\n"
      );
  } else {
    mexErrMsgIdAndTxt(
      err_id[err_help_arg], err_msg[err_help_arg], arg.c_str());
  }
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
    trn_data.num_classes, trn_data.num_examples, mex_class<Data>::id());
  context.dual_variables = static_cast<Data*>(mxGetData(mxA));
  info.add("A", mxA);

  if (!context.is_dual) {
    mxArray *mxW = mxDuplicateFieldOrCreateMatrix(opts, "W",
      trn_data.num_dimensions, trn_data.num_classes, mex_class<Data>::id());
    context.primal_variables = static_cast<Data*>(mxGetData(mxW));
    info.add("W", mxW);
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
  info.add("num_classes", mxCreateScalar(trn_data.num_classes));

  std::string objective = mxGetFieldValueOrDefault(
    opts, "objective", std::string("topk_svm"));
  info.add("objective", mxCreateString(objective.c_str()));

  auto C = mxGetFieldValueOrDefault<Result>(opts, "C", 1);
  mxCheck<Result>(std::greater<Result>(), C, 0, "C");
  info.add("C", mxCreateScalar(C));

  auto k = mxGetFieldValueOrDefault<size_type>(opts, "k", 1);
  mxCheckRange<size_type>(k, 1, trn_data.num_classes - 1, "k");

  auto gamma = mxGetFieldValueOrDefault<Result>(opts, "gamma", 1);
  mxCheck<Result>(std::greater_equal<Result>(), gamma, 0, "gamma");

  if (objective == "msvm" ||
      objective == "multi_svm") {
    mxCheckRange<size_type>(k, 1, 1, "k");
    make_solver_solve(context, info,
      l2_topk_hinge<Data, Result, Summation>(k, C, sum));
  } else if (objective == "topk_svm" ||
             objective == "l2_hinge_topk" ||
             objective == "topk_hinge_alpha") {
    info.add("k", mxCreateScalar(k));
    info.add("gamma", mxCreateScalar(gamma));
    if (gamma > 0) {
      make_solver_solve(context, info,
        l2_hinge_topk_smooth<Data, Result, Summation>(k, C, gamma, sum));
    } else {
      make_solver_solve(context, info,
        l2_hinge_topk<Data, Result, Summation>(k, C, sum));
    }
  } else if (objective == "l2_topk_hinge" ||
             objective == "topk_hinge_beta") {
    info.add("k", mxCreateScalar(k));
    info.add("gamma", mxCreateScalar(gamma));
    if (gamma > 0) {
      make_solver_solve(context, info,
        l2_topk_hinge_smooth<Data, Result, Summation>(k, C, gamma, sum));
    } else {
      make_solver_solve(context, info,
        l2_topk_hinge<Data, Result, Summation>(k, C, sum));
    }
  } else if (objective == "softmax" ||
      objective == "l2_entropy_topk") {
    info.add("k", mxCreateScalar(k));
    make_solver_solve(context, info,
      l2_entropy_topk<Data, Result, Summation>(k, C, sum));
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
    const mxArray* opts = (nrhs > 1) ? prhs[1] : nullptr;
    if (command == "help") {
      printHelp(opts);
    } else if (command == "version") {
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
