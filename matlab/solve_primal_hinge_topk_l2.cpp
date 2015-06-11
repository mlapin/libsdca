#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <memory>
#include <vector>

#include <mex.h>
#include <matrix.h>

#include "solvers.hpp"
#include "mex_util.hpp"

using namespace sdca;

template <typename T>
mxArray * createInfoStruct(std::unique_ptr<Solver<T>> &solver,
    double C, double lambda, double gamma) {
  void const *fields [] = {
    "Solver", mxCreateString(solver->get_solver_name().c_str()),
    "Status", mxCreateScalar(static_cast<double>(solver->get_status())),
    "StatusName", mxCreateString(solver->get_status_name().c_str()),
    "CpuTime", mxCreateScalar(solver->get_cpu_time()),
    "WallTime", mxCreateScalar(solver->get_wall_time()),
    "NumExamples", mxCreateScalar(solver->get_num_examples()),
    "NumTasks", mxCreateScalar(solver->get_num_tasks()),
    "C", mxCreateScalar(C),
    "Lambda", mxCreateScalar(lambda),
    "Gamma", mxCreateScalar(gamma),
    "Primal", mxCreateScalar(solver->get_primal_objective()),
    "Dual", mxCreateScalar(solver->get_dual_objective()),
    "AbsoluteGap", mxCreateScalar(solver->get_absolute_gap()),
    "RelativeGap", mxCreateScalar(solver->get_relative_gap()),
    "Epsilon", mxCreateScalar(solver->get_epsilon()),
    "NumEpoch", mxCreateScalar(solver->get_num_epoch()),
    "MaxNumEpoch", mxCreateScalar(solver->get_max_num_epoch()),
    "MaxCpuTime", mxCreateScalar(solver->get_max_cpu_time()),
    "MaxWallTime", mxCreateScalar(solver->get_max_wall_time()),
    "CheckGapFrequency", mxCreateScalar(
      solver->get_check_gap_frequency()),
    "Seed", mxCreateScalar(solver->get_seed()),
    "Precision", mxCreatePrecisionString<T>(),
    0, 0
  };
  return createScalarStructArray(fields);
}

template <typename T>
std::unique_ptr<Solver<T>> createSolver(
    const SizeType num_dimensions, const SizeType num_examples,
    const SizeType num_tasks, const SizeType top_k,
    const double lambda, const double gamma,
    const T *Xptr, const SizeType *labels, T *Wptr, T *Aptr) {

  if (gamma > 0) {
    mexErrMsgIdAndTxt("LIBSDCA:notimplemented",
      "Smooth HingeTopK loss solver is not implemented yet.");
    return nullptr;
  } else {
    HingeTopKLossL2RegularizerDualVariablesHelper<T>
      solver_helper(num_examples, top_k, static_cast<T>(lambda));
    return std::unique_ptr<Solver<T>>(new
      PrimalSolver<T, HingeTopKLossL2RegularizerDualVariablesHelper<T>>(
        solver_helper, num_dimensions, num_examples, num_tasks,
        Xptr, &labels[0], Wptr, Aptr));
  }
}

void printUsage(const std::vector<double> &params) {
  mexPrintf(
    "Usage: W = solve_primal_hinge_topk_l2(Y,X);\n"
    "       [W,info,A] = solve_primal_hinge_topk_l2(Y,X,<parameters>);\n"
    "Parameters can be given in this order (default value in parentheses):\n"
    "  top_k (%g)\n"
    "  svm_c (%g)\n"
    "  gamma (%g)\n"
    "  epsilon (%g)\n"
    "  check_gap_frequency (%g)\n"
    "  max_num_epoch (%g)\n"
    "  max_wall_time (%g)\n"
    "  max_cpu_time (%g)\n"
    "  seed (%g)\n"
    "\n"
    "Matrix W is a num_dimensions-by-num_tasks matrix of primal variables,\n"
    "matrix A is a num_tasks-by-num_examples matrix of dual variables and\n"
    "  W = Xtrn * A';        %% dim-by-num_examples matrix of predictors\n"
    "  S = A * (Xtrn'*Xtst); %% num_tasks-by-num_examples matrix of scores\n"
    "  S = W' * Xtst;        %% (same as above)\n"
    "\n",
    params[0], params[1], params[2], params[3], params[4], params[5],
    params[6], params[7], params[8]);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // Default values of parameters:
  std::vector<double> params = {1, 1, 0, 1e-3, 10, 100, 0, 0, 1};

  if (nrhs < 2 || nrhs > 2 + static_cast<int>(params.size())) {
    printUsage(params);
    mexErrMsgIdAndTxt("LIBSDCA:inputmismatch",
      "Wrong number of input arguments.");
  }

  if (nlhs < 1 || nlhs > 3) {
    printUsage(params);
    mexErrMsgIdAndTxt("LIBSDCA:outputmismatch",
      "Wrong number of output arguments.");
  }

  const mxArray *mxY = prhs[0];
  const mxArray *mxX = prhs[1];

  mxVerifyNotSparseNotEmpty(mxX, "X");
  mxVerifySingleOrDouble(mxX, "X");

  SizeType num_dimensions = static_cast<SizeType>(mxGetM(mxX));
  SizeType num_examples = static_cast<SizeType>(mxGetN(mxX));
  mxVerifyVectorDimension(mxY, num_examples, "Y");
  mxVerifyNotSparseNotEmpty(mxY, "Y");
  mxVerifyDouble(mxY, "Y");

  std::vector<SizeType> labels(mxGetPr(mxY), mxGetPr(mxY) + num_examples);
  auto minmax = std::minmax_element(mxGetPr(mxY), mxGetPr(mxY) + num_examples);
  if (*minmax.first == 1) {
    std::for_each(labels.begin(), labels.end(), [](SizeType &x){ x -= 1; });
    *minmax.second -= 1;
  } else if (*minmax.first != 0) {
    mexErrMsgIdAndTxt(errInvalidArgument, "Labels must be in the range 1:T.");
  }

  const SizeType num_tasks = static_cast<SizeType>(*minmax.second) + 1;

  for (int i = 2; i < nrhs; ++i) {
    params[i-2] = mxGetScalar(prhs[i]);
  }

  SizeType top_k = static_cast<SizeType>(params[0]);
  double svm_c = params[1];
  double gamma = params[2];
  double epsilon = params[3];
  SizeType check_gap_frequency = static_cast<SizeType>(params[4]);
  SizeType max_num_epoch = static_cast<SizeType>(params[5]);
  double max_wall_time = static_cast<double>(params[6]);
  double max_cpu_time = static_cast<double>(params[7]);
  SizeType seed = static_cast<SizeType>(params[8]);

  double lambda = 1.0 / (static_cast<double>(num_examples) * svm_c);

  std::cout << "solve_primal_hinge_topk_l2 [" <<
    "top_k: " << top_k <<
    ", svm_c: " << svm_c <<
    ", lambda: " << lambda <<
    ", gamma: " << gamma <<
    ", epsilon: " << epsilon <<
    ", check_gap_frequency: " << check_gap_frequency <<
    ", max_num_epoch: " << max_num_epoch <<
    ", max_wall_time: " << max_wall_time <<
    ", max_cpu_time: " << max_cpu_time <<
    ", seed: " << seed <<
    "]" << std::endl;

  mwSize mxDims[2] = {num_dimensions, num_tasks};
  mxArray *mxW = mxCreateNumericArray(2, mxDims, mxGetClassID(mxX), mxREAL);
  if (mxW == NULL) {
    mexErrMsgIdAndTxt(errOutOfMemory, "Failed to allocate memory for W.");
  }

  mxDims[0] = num_tasks;
  mxDims[1] = num_examples;
  mxArray *mxA = mxCreateNumericArray(2, mxDims, mxGetClassID(mxX), mxREAL);
  if (mxA == NULL) {
    mexErrMsgIdAndTxt(errOutOfMemory, "Failed to allocate memory for A.");
  }

  mxArray *mxInfo;
  if (mxIsDouble(mxX)) {
    const double *Xptr = static_cast<double*>(mxGetData(mxX));
    double *Wptr = static_cast<double*>(mxGetData(mxW));
    double *Aptr = static_cast<double*>(mxGetData(mxA));

    std::unique_ptr<Solver<double>> solver = createSolver<double>(
      num_dimensions, num_examples, num_tasks, top_k,
      lambda, gamma, Xptr, &labels[0], Wptr, Aptr);

    solver->set_epsilon(static_cast<double>(epsilon));
    solver->set_check_gap_frequency(check_gap_frequency);
    solver->set_max_num_epoch(max_num_epoch);
    solver->set_max_wall_time(max_wall_time);
    solver->set_max_cpu_time(max_cpu_time);
    solver->set_seed(seed);
    solver->Solve();

    mxInfo = createInfoStruct<double>(solver, svm_c, lambda, gamma);

  } else if (mxIsSingle(mxX)) {
    const float *Xptr = static_cast<float*>(mxGetData(mxX));
    float *Wptr = static_cast<float*>(mxGetData(mxW));
    float *Aptr = static_cast<float*>(mxGetData(mxA));

    std::unique_ptr<Solver<float>> solver = createSolver<float>(
      num_dimensions, num_examples, num_tasks, top_k,
      lambda, gamma, Xptr, &labels[0], Wptr, Aptr);

    solver->set_epsilon(static_cast<float>(epsilon));
    solver->set_check_gap_frequency(check_gap_frequency);
    solver->set_max_num_epoch(max_num_epoch);
    solver->set_max_wall_time(max_wall_time);
    solver->set_max_cpu_time(max_cpu_time);
    solver->set_seed(seed);
    solver->Solve();

    mxInfo = createInfoStruct<float>(solver, svm_c, lambda, gamma);

  }

  if (nlhs > 0) plhs[0] = mxW;
  if (nlhs > 1) plhs[1] = mxInfo;
  if (nlhs > 2) plhs[2] = mxA;
}

