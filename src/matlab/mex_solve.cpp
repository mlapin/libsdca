#include "mex_util.h"
#include "sdca/solver.h"

#include "sdca/utility/logging.cpp"

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
"    eval_on_start [false]   - whether to check the duality gap on start;\n"
"    eval_epoch    [10]      - how often to check the gap;\n"
"    max_epoch      [1000]   - epochs limit;\n"
"    max_cpu_time   [0]      - CPU time limit (0: no limit);\n"
"    max_wall_time  [0]      - wall time limit (0: no limit);\n"
"    epsilon        [1e-3]   - relative duality gap bound, stop if\n"
"      (primal - dual) <= epsilon * max(abs(primal), abs(dual))\n"
"\n"
"    log_level  ['info']     - logging verbosity\n"
"                              ('none', 'info', 'verbose', 'debug')\n"
"    log_format ['short_e']  - numeric format (4/16 digits; float/exp format)\n"
"                              ('short_f', 'short_e', 'long_f', 'long_e')\n"
"    precision  ['double']   - intermediate floating-point precision;\n"
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
"  labels vectors must be of type double\n"
      );
  } else if (arg == "obj" || arg == "objective") {
    mexPrintf(
"opts.objective - the training objective to optimize.\n"
"  Multiclass:\n"
"    msvm (synonym: l2_multiclass_hinge)\n"
"      - l2 regularized multiclass SVM of Crammer and Singer\n"
"    msvm_smooth (synonym: l2_multiclass_hinge_smooth)\n"
"      - l2 regularized multiclass SVM with smoothed hinge loss\n"
"    softmax (synonym: l2_multiclass_entropy)\n"
"      - l2 regularized multiclass cross-entropy loss\n"
"    l2_hinge_topk (synonyms: topk_hinge_alpha)\n"
"      - l2 regularized hinge-of-top-k loss (top-k hinge alpha)\n"
"        (both smooth and non-smooth depending on gamma)\n"
"    l2_topk_hinge (synonym: topk_hinge_beta)\n"
"      - l2 regularized top-k-of-hinge loss (top-k hinge beta)\n"
"        (both smooth and non-smooth depending on gamma)\n"
"    l2_entropy_topk\n"
"      - l2 regularized entropy-on-top-k-simplex loss\n"
"        (reduces to the usual softmax loss for k=1)\n"
"\n"
"  Multilabel:\n"
"    mlsvm (synonym: l2_multilabel_hinge)\n"
"      - l2 regularized multilabel SVM of Crammer and Singer\n"
"    mlsvm_smooth (synonym: l2_multilabel_hinge_smooth)\n"
"      - l2 regularized multilabel SVM with smoothed hinge loss\n"
"    mlsoftmax (synonym: l2_multilabel_entropy)\n"
"      - l2 regularized multilabel cross-entropy loss\n"
"\n"
"  Default value:\n"
"    mlsvm_smooth\n"
      );
  } else {
    mexErrMsgIdAndTxt(
      err_id[err_help_arg], err_msg[err_help_arg], arg.c_str());
  }
}

template <typename Data>
inline void
validate_data(
    const mxArray* data,
    const bool is_dual
  ) {
  mxCheckNotSparse(data, "data");
  mxCheckNotEmpty(data, "data");
  mxCheckReal(data, "data");
  mxCheckClass(data, "data", mex_class<Data>::id());
  if (is_dual) {
    mxCheckSquare(data, "data");
  }
}

inline void
validate_labels(
    const mxArray* labels,
    const size_type num_examples
  ) {
  mxCheckNotEmpty(labels, "labels");
  mxCheckDouble(labels, "labels");
  if (mxIsSparse(labels)) {
    mxCheck(std::equal_to<size_type>(),
            mxGetN(labels), num_examples, "num_examples");
  } else {
    mxCheckVector(labels, "labels", num_examples);
  }
}

template <typename Data>
inline feature_input<Data>
make_test_input(
    const feature_input<Data>& trn_in,
    const mxArray* data,
    const size_type id
  ) {
  if (trn_in.num_dimensions != mxGetM(data)) {
    mexErrMsgIdAndTxt(err_id[err_num_dim], err_msg[err_num_dim], id);
  }
  return sdca::make_input_feature(mxGetM(data), mxGetN(data),
    static_cast<Data*>(mxGetData(data)));
}

template <typename Data>
inline kernel_input<Data>
make_test_input(
    const kernel_input<Data>& trn_in,
    const mxArray* data,
    const size_type id
  ) {
  if (trn_in.num_examples != mxGetM(data)) {
    mexErrMsgIdAndTxt(err_id[err_num_examples], err_msg[err_num_examples], id);
  }
  return sdca::make_input_kernel(mxGetM(data), mxGetN(data),
    static_cast<Data*>(mxGetData(data)));
}

template <typename Output>
inline Output
make_output(const mxArray*, const size_type) {
  return Output();
}

template <>
inline multiclass_output
make_output<multiclass_output>(
    const mxArray* labels,
    const size_type num_examples
  ) {
  mxCheckNotSparse(labels, "labels");
  auto first = mxGetPr(labels);
  auto last = mxGetPr(labels) + num_examples;
  return make_output_multiclass(first, last);
}

template <>
inline multilabel_output
make_output<multilabel_output>(
    const mxArray* labels,
    const size_type num_examples
  ) {
  if (mxIsSparse(labels)) {
    auto l_first = mxGetIr(labels);
    auto l_last = mxGetIr(labels) + mxGetNzmax(labels);
    auto o_first = mxGetJc(labels);
    auto o_last = mxGetJc(labels) + num_examples + 1;
    return make_output_multilabel(l_first, l_last, o_first, o_last);
  } else {
    auto first = mxGetPr(labels);
    auto last = mxGetPr(labels) + num_examples;
    return make_output_multilabel(first, last);
  }
}

template <typename Data,
          typename Result,
          template <typename> class Input,
          typename Output,
          template <typename, typename> class Objective>
inline void
set_test_data(
    const mxArray* all_data,
    const mxArray* all_labels,
    solver_context<Data, Result, Input, Output, Objective>& ctx
  ) {
  size_type num_datasets = mxGetNumberOfElements(all_data);
  for (size_type i = 1; i < num_datasets; ++i) {
    auto data = mxGetCell(all_data, i);
    auto labels = mxGetCell(all_labels, i);
    validate_data<Data>(data, false);
    validate_labels(labels, mxGetN(data));

    Input<Data> in = make_test_input(ctx.train.in, data, i + 1);
    Output out = make_output<Output>(labels, in.num_examples);
    if (out.num_classes != ctx.train.out.num_classes) {
      mexErrMsgIdAndTxt(
        err_id[err_num_classes], err_msg[err_num_classes], i + 1);
    }

    ctx.add_test(std::move(in), std::move(out));
  }
}

template <typename Context>
inline void
set_stopping_criteria(
    const mxArray* opts,
    Context& ctx
  ) {
  auto c = &ctx.criteria;
  mxSetFieldValue(opts, "eval_on_start", c->eval_on_start);
  mxSetFieldValue(opts, "eval_epoch", c->eval_epoch);
  mxSetFieldValue(opts, "max_epoch", c->max_epoch);
  mxSetFieldValue(opts, "max_cpu_time", c->max_cpu_time);
  mxSetFieldValue(opts, "max_wall_time", c->max_wall_time);
  mxSetFieldValue(opts, "epsilon", c->epsilon);

  mxCheck<size_type>(std::greater_equal<size_type>(),
    c->eval_epoch, 0, "eval_epoch");
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
          typename Output>
inline void
set_variables(
    const mxArray* opts,
    feature_input<Data>& in,
    Output& out,
    model_info<mxArray*>& info,
    Data*& A,
    Data*& W
    ) {
  mxArray *mxA = mxDuplicateFieldOrCreateMatrix(opts, "A",
    out.num_classes, in.num_examples, mex_class<Data>::id());
  info.add("A", mxA);
  A = static_cast<Data*>(mxGetData(mxA));

  mxArray *mxW = mxDuplicateFieldOrCreateMatrix(opts, "W",
    in.num_dimensions, out.num_classes, mex_class<Data>::id());
  info.add("W", mxW);
  W = static_cast<Data*>(mxGetData(mxW));
}

template <typename Data,
          template <typename> class Input,
          typename Output>
inline void
set_variables(
    const mxArray* opts,
    Input<Data>& in,
    Output& out,
    model_info<mxArray*>& info,
    Data*& A,
    Data*&
    ) {
  mxArray *mxA = mxDuplicateFieldOrCreateMatrix(opts, "A",
    out.num_classes, in.num_examples, mex_class<Data>::id());
  info.add("A", mxA);
  A = static_cast<Data*>(mxGetData(mxA));
}

template <typename Data,
          typename Result,
          template <typename> class Input,
          typename Output,
          template <typename, typename> class Objective>
inline void
set_context(
    mxArray* plhs[],
    const mxArray* prhs[],
    const mxArray* opts,
    Input<Data>&& in,
    Output&& out,
    Objective<Data, Result>&& obj
    ) {
  Data *A(0), *W(0);
  model_info<mxArray*> info;
  set_variables(opts, in, out, info, A, W);

  auto ctx = make_context(std::move(in), std::move(out), std::move(obj), A, W);

  if (mxIsCell(prhs[0]) && mxIsCell(prhs[1])) {
    set_test_data(prhs[0], prhs[1], ctx);
  }

  set_stopping_criteria(opts, ctx);

  auto solver = sdca::make_solver(ctx);
  solver.solve();

  plhs[0] = mxCreateStruct(info.fields, "model");
}

template <typename Data,
          typename Result,
          template <typename> class Input>
inline void
set_objective(
    mxArray* plhs[],
    const mxArray* prhs[],
    const mxArray* opts,
    const std::string& obj,
    Input<Data>&& in,
    multiclass_output&& out
  ) {
  auto c = mxGetFieldValueOrDefault<Result>(opts, "C", 1);
  mxCheck<Result>(std::greater<Result>(), c, 0, "C");

  auto k = mxGetFieldValueOrDefault<size_type>(opts, "k", 1);
  mxCheckRange<size_type>(k, 1, out.num_classes - 1, "k");

  auto gamma = mxGetFieldValueOrDefault<Result>(opts, "gamma", 1);
  mxCheck<Result>(std::greater_equal<Result>(), gamma, 0, "gamma");

  if (obj == "msvm" ||
      obj == "l2_multiclass_hinge") {
    mxCheckRange<size_type>(k, 1, 1, "k");
    set_context(plhs, prhs, opts, std::move(in), std::move(out),
      make_objective_l2_topk_hinge<Data>(c, k));
  } else if (obj == "msvm_smooth" ||
             obj == "l2_multiclass_hinge_smooth") {
    mxCheckRange<size_type>(k, 1, 1, "k");
    set_context(plhs, prhs, opts, std::move(in), std::move(out),
      make_objective_l2_topk_hinge_smooth<Data>(c, gamma, k));
  } else if (obj == "softmax" ||
             obj == "l2_multiclass_entropy") {
    mxCheckRange<size_type>(k, 1, 1, "k");
    set_context(plhs, prhs, opts, std::move(in), std::move(out),
      make_objective_l2_entropy_topk<Data>(c, k));
  } else if (obj == "l2_hinge_topk" ||
             obj == "topk_hinge_alpha") {
    if (gamma > 0) {
      set_context(plhs, prhs, opts, std::move(in), std::move(out),
        make_objective_l2_hinge_topk_smooth<Data>(c, gamma, k));
    } else {
      set_context(plhs, prhs, opts, std::move(in), std::move(out),
        make_objective_l2_hinge_topk<Data>(c, k));
    }
  } else if (obj == "l2_topk_hinge" ||
             obj == "topk_hinge_beta") {
    if (gamma > 0) {
      set_context(plhs, prhs, opts, std::move(in), std::move(out),
        make_objective_l2_topk_hinge_smooth<Data>(c, gamma, k));
    } else {
      set_context(plhs, prhs, opts, std::move(in), std::move(out),
        make_objective_l2_topk_hinge<Data>(c, k));
    }
  } else if (obj == "l2_entropy_topk") {
    set_context(plhs, prhs, opts, std::move(in), std::move(out),
      make_objective_l2_entropy_topk<Data>(c, k));
  } else {
    mexErrMsgIdAndTxt(
      err_id[err_objective], err_msg[err_objective], obj.c_str());
  }
}

template <typename Data,
          typename Result,
          template <typename> class Input>
inline void
set_objective(
    mxArray* plhs[],
    const mxArray* prhs[],
    const mxArray* opts,
    const std::string& obj,
    Input<Data>&& in,
    multilabel_output&& out
  ) {
  auto c = mxGetFieldValueOrDefault<Result>(opts, "C", 1);
  mxCheck<Result>(std::greater<Result>(), c, 0, "C");

  auto gamma = mxGetFieldValueOrDefault<Result>(opts, "gamma", 1);
  mxCheck<Result>(std::greater_equal<Result>(), gamma, 0, "gamma");

  if (obj == "mlsvm" ||
      obj == "l2_multilabel_hinge") {
    set_context(plhs, prhs, opts, std::move(in), std::move(out),
      make_objective_l2_multilabel_hinge<Data>(c));
  } else if (obj == "mlsvm_smooth" ||
             obj == "l2_multilabel_hinge_smooth") {
    set_context(plhs, prhs, opts, std::move(in), std::move(out),
      make_objective_l2_multilabel_hinge_smooth<Data>(c, gamma));
  } else if (obj == "mlsoftmax" ||
             obj == "l2_multilabel_entropy") {
    set_context(plhs, prhs, opts, std::move(in), std::move(out),
      make_objective_l2_multilabel_entropy<Data>(c));
  } else {
    mexErrMsgIdAndTxt(
      err_id[err_objective], err_msg[err_objective], obj.c_str());
  }
}

template <typename Data,
          typename Result,
          template <typename> class Input>
inline void
set_output(
    mxArray* plhs[],
    const mxArray* prhs[],
    const mxArray* opts,
    const mxArray* labels,
    Input<Data>&& in
  ) {
  validate_labels(labels, in.num_examples);
  std::string obj = mxGetFieldValueOrDefault(
    opts, "objective", std::string("mlsvm_smooth"));
  if ((obj == "mlsoftmax") || (obj == "l2_multilabel_entropy")
      || (obj == "mlsvm") || (obj == "l2_multilabel_hinge")
      || (obj == "mlsvm_smooth") || (obj == "l2_multilabel_hinge_smooth")) {
    // Multilabel
    set_objective<Data, Result>(plhs, prhs, opts, obj, std::move(in),
      make_output<multilabel_output>(labels, in.num_examples));
  } else {
    // Multiclass
    set_objective<Data, Result>(plhs, prhs, opts, obj, std::move(in),
      make_output<multiclass_output>(labels, in.num_examples));
  }
}

template <typename Data,
          typename Result>
inline void
set_input(
    mxArray* plhs[],
    const mxArray* prhs[],
    const mxArray* opts
  ) {
  bool is_dual = mxGetFieldValueOrDefault(opts, "is_dual", false);
  const mxArray* data;
  const mxArray* labels;
  if (mxIsNumeric(prhs[0]) && mxIsNumeric(prhs[1])) {
    data = prhs[0];
    labels = prhs[1];
  } else {
    mxCheckCellArrays(prhs[0], prhs[1]);
    data = mxGetCell(prhs[0], 0);
    labels = mxGetCell(prhs[1], 0);
  }

  validate_data<Data>(data, is_dual);
  Data* p_data = static_cast<Data*>(mxGetData(data));
  if (is_dual) {
    set_output<Data, Result>(plhs, prhs, opts, labels,
      sdca::make_input_kernel(mxGetN(data), p_data));
  } else {
    set_output<Data, Result>(plhs, prhs, opts, labels,
      sdca::make_input_feature(mxGetM(data), mxGetN(data), p_data));
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
    set_input<Data, double>(plhs, prhs, opts);
  } else if (precision == "single" || precision == "float") {
    set_input<Data, float>(plhs, prhs, opts);
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
    try {
      if (mxIsDouble(prhs[0])
          || (mxIsCell(prhs[0]) && !mxIsEmpty(prhs[0])
              && mxIsDouble(mxGetCell(prhs[0], 0)))) {
         mex_main<double>(plhs, nrhs, prhs);
      } else if (mxIsSingle(prhs[0])
          || (mxIsCell(prhs[0]) && !mxIsEmpty(prhs[0])
              && mxIsSingle(mxGetCell(prhs[0], 0)))) {
         mex_main<float>(plhs, nrhs, prhs);
      } else {
        mexErrMsgIdAndTxt(err_id[err_arg], err_msg[err_arg]);
      }
    } catch(const std::exception& e) {
      mexErrMsgIdAndTxt(
        err_id[err_exception], err_msg[err_exception], e.what());
    }
    mat_cout.release();
    logging::format_pop();
  }
}
