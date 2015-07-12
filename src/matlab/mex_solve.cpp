
#include "mex_util.h"
#include "solve/solve.h"

#ifndef MEX_SOLVE
#define MEX_SOLVE "mex_solve"
#endif

using namespace sdca;

void
printUsage() {
  mexPrintf("Usage: model = %s(X, Y);\n"
            "       model = %s(data, labels [, opts]);\n"
            , MEX_SOLVE, MEX_SOLVE);
}

template <typename Data>
struct result {
  bool is_dual;
  problem_data<Data> problem;
  stopping_criteria criteria;
  std::vector<size_type> labels;
  std::vector<std::pair<std::string, mxArray*>> fields;
};

template <typename data_type>
inline
void
set_labels(
    const mxArray* p_labels,
    result<data_type>& model
  ) {
  auto n = model.problem.num_examples;
  mxCheckVector(n, p_labels, "labels");

  std::vector<size_type> labels(mxGetPr(p_labels), mxGetPr(p_labels) + n);
  auto minmax = std::minmax_element(labels.begin(), labels.end());
  if (*minmax.first == 1) {
    std::for_each(labels.begin(), labels.end(), [](size_type &x){ x -= 1; });
    *minmax.second -= 1;
  } else if (*minmax.first != 0) {
    mexErrMsgIdAndTxt(err_id[err_labels_range], err_msg[err_labels_range]);
  }

  model.labels = std::move(labels);
  model.problem.labels = &model.labels[0];
  model.problem.num_tasks = static_cast<size_type>(*minmax.second) + 1;
}

template <typename data_type>
inline
void
set_problem_data(
    const mxArray* p_data,
    const mxArray* p_labels,
    const mxArray* opts,
    result<data_type>& model
  ) {
  model.problem.data = static_cast<data_type*>(mxGetData(p_data));
  model.problem.num_examples = mxGetN(p_data);
  set_labels(p_labels, model);

  model.is_dual = mxGetFieldValueOrDefault<bool>(opts, "dual", false);
  if (model.is_dual) {
    // Work in the dual
    model.problem.num_dimensions = 0;
    mxCheckSquare(p_data, "data");
  } else {
    // Work in the primal
    model.problem.num_dimensions = mxGetM(p_data);
    mxArray *mxW = mxCreateNumericMatrix(model.problem.num_dimensions,
      model.problem.num_tasks, mxGetClassID(p_data), mxREAL);
    mxCheckCreated(mxW, "W");
    model.problem.primal_variables = static_cast<data_type*>(mxGetData(mxW));
    model.fields.emplace_back(std::make_pair("W", mxW));
  }

  mxArray *mxA = mxCreateNumericMatrix(model.problem.num_tasks,
    model.problem.num_examples, mxGetClassID(p_data), mxREAL);
  mxCheckCreated(mxA, "A");
  model.problem.dual_variables = static_cast<data_type*>(mxGetData(mxA));
  model.fields.emplace_back(std::make_pair("A", mxA));
}

template <typename data_type>
inline
void
set_stopping_criteria(
    const mxArray* opts,
    result<data_type>& model
  ) {
  mxSetFieldValue(opts, "check_epoch", model.criteria.check_epoch);
  mxSetFieldValue(opts, "max_num_epoch", model.criteria.max_num_epoch);
  mxSetFieldValue(opts, "max_cpu_time", model.criteria.max_cpu_time);
  mxSetFieldValue(opts, "max_wall_time", model.criteria.max_wall_time);
  mxSetFieldValue(opts, "epsilon", model.criteria.epsilon);
}

template <typename Objective, typename Data>
inline
void
make_solver_solve(
    const result<Data> model,
    const Objective objective
  ) {
  if (model.is_dual) {
    mexErrMsgIdAndTxt(err_id[err_not_implemented], err_msg[err_not_implemented],
      "Dual solver");
  } else {
    auto solver = make_primal_solver(model.problem, model.criteria, objective);
    solver.solve();
  }
}

template <typename data_type>
void
mex_main(
    mxArray* plhs[],
    const int nrhs,
    const mxArray* prhs[]
    ) {
  const mxArray* p_data = prhs[0];
  const mxArray* p_labels = prhs[1];
  const mxArray* opts = (nrhs > 2) ? prhs[2] : nullptr;
  mxCheckStruct(opts, "opts");

  result<data_type> model;
  set_problem_data(p_data, p_labels, opts, model);
  set_stopping_criteria(opts, model);

  auto c = mxGetFieldValueOrDefault<data_type>(opts, "c", 1);
  mxCheck<data_type>(std::greater_equal<data_type>(), c, 0, "c");
  auto k = mxGetFieldValueOrDefault<size_type>(opts, "k", 1);
  mxCheckRange<size_type>(k, 1, model.problem.num_tasks, "k");

  std::string objective = mxGetFieldValueOrDefault(
    opts, "objective", std::string("l2_hinge_topk"));
  if (objective == "l2_hinge_topk") {
    make_solver_solve(model, make_l2_hinge_topk(k, c));
  } else {
    mexErrMsgIdAndTxt(
      err_id[err_obj_type], err_msg[err_obj_type], objective.c_str());
  }

  plhs[0] = mxCreateStruct(model.fields, "model");
}

void
mexFunction(
    const int nlhs,
    mxArray* plhs[],
    const int nrhs,
    const mxArray* prhs[]
    ) {
  mxCheckArgNum(nrhs, 2, 3, printUsage);
  mxCheckArgNum(nlhs, 0, 1, printUsage);

  mxCheckNotSparse(prhs[0], "data");
  mxCheckNotEmpty(prhs[0], "data");
  mxCheckReal(prhs[0], "data");

  mxCheckNotSparse(prhs[1], "labels");
  mxCheckNotEmpty(prhs[1], "labels");
  mxCheckDouble(prhs[1], "labels");

  if (mxIsDouble(prhs[0])) {
     mex_main<double>(plhs, nrhs, prhs);
  } else if (mxIsSingle(prhs[0])) {
     mex_main<float>(plhs, nrhs, prhs);
  }
}
