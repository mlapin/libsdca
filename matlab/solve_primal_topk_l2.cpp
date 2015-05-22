#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <vector>

#include <mex.h>
#include <matrix.h>

#include "solvers.hpp"
#include "mex_util.hpp"

using namespace sdca;

void printUsage(const std::vector<double> &params) {
  mexPrintf(
    "Usage: W = solve_primal_topk_l2(Y,X);\n"
    "       [W,info,A] = solve_primal_topk_l2(Y,X,<parameters>);\n"
    "Parameters can be given in this order (default value in parentheses):\n"
    "  top_k (%g)\n"
    "  svm_c (%g)\n"
    "  check_gap_frequency (%g)\n"
    "  max_num_epoch (%g)\n"
    "  epsilon (%g)\n"
    "  seed (%g)\n"
    "\n"
    "Matrix W is a num_dimensions-by-num_tasks matrix of primal variables,\n"
    "matrix A is a num_tasks-by-num_examples matrix of dual variables and\n"
    "  W = Xtrn * A';        %% dim-by-num_examples matrix of predictors\n"
    "  S = A * (Xtrn'*Xtst); %% num_tasks-by-num_examples matrix of scores\n"
    "  S = W' * Xtst;        %% (same as above)\n"
    "\n",
    params[0], params[1], params[2], params[3], params[4], params[5], params[6]
    );
}

template <typename T>
mxArray * createInfoStruct(sdca::Solver<T> &solver, double C, double lambda) {
  void const *fields [] = {
    "Solver", mxCreateString(solver.get_solver_name().c_str()),
    "Status", mxCreateScalar(static_cast<double>(solver.get_status())),
    "StatusName", mxCreateString(solver.get_status_name().c_str()),
    "CpuTime", mxCreateScalar(solver.get_cpu_time()),
    "WallTime", mxCreateScalar(solver.get_wall_time()),
    "NumExamples", mxCreateScalar(solver.get_num_examples()),
    "NumTasks", mxCreateScalar(solver.get_num_tasks()),
    "C", mxCreateScalar(C),
    "Lambda", mxCreateScalar(lambda),
    "Primal", mxCreateScalar(solver.get_primal_objective()),
    "Dual", mxCreateScalar(solver.get_dual_objective()),
    "AbsoluteGap", mxCreateScalar(solver.get_absolute_gap()),
    "RelativeGap", mxCreateScalar(solver.get_relative_gap()),
    "Epsilon", mxCreateScalar(solver.get_epsilon()),
    "NumEpoch", mxCreateScalar(solver.get_num_epoch()),
    "MaxNumEpoch", mxCreateScalar(solver.get_max_num_epoch()),
    "CheckGapFrequency", mxCreateScalar(
      solver.get_check_gap_frequency()),
    "Seed", mxCreateScalar(solver.get_seed()),
    "Precision", mxCreatePrecisionString<T>(),
    0, 0
  };
  return createScalarStructArray(fields);
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // Default values of parameters:
  std::vector<double> params = {1, 1, 10, 100, 1e-2, 1};

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
  SizeType check_gap_frequency = static_cast<SizeType>(params[2]);
  SizeType max_num_epoch = static_cast<SizeType>(params[3]);
  double epsilon = params[4];
  SizeType seed = static_cast<SizeType>(params[5]);

  double lambda = 1.0 / (static_cast<double>(num_examples) * svm_c);

  std::cout << "solve_primal_topk_l2[" <<
    "top_k: " << top_k << ", svm_c: " << svm_c << ", lambda: " << lambda <<
    ", check_gap_frequency: " << check_gap_frequency <<
    ", max_num_epoch: " << max_num_epoch <<
    ", epsilon: " << epsilon << ", seed: " << seed << "]" << std::endl;

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

    TopKLossL2RegularizerDualVariablesHelper<double>
      solver_helper(top_k, static_cast<double>(lambda), num_examples);

    PrimalSolver<double, TopKLossL2RegularizerDualVariablesHelper<double>>
      solver(solver_helper, num_dimensions, num_examples, num_tasks,
             Xptr, &labels[0], Wptr, Aptr);

    solver.set_check_gap_frequency(check_gap_frequency);
    solver.set_max_num_epoch(max_num_epoch);
    solver.set_epsilon(static_cast<double>(epsilon));
    solver.set_seed(seed);
    solver.Solve();

    mxInfo = createInfoStruct<double>(solver, svm_c, lambda);

  } else if (mxIsSingle(mxX)) {
    const float *Xptr = static_cast<float*>(mxGetData(mxX));
    float *Wptr = static_cast<float*>(mxGetData(mxW));
    float *Aptr = static_cast<float*>(mxGetData(mxA));

    TopKLossL2RegularizerDualVariablesHelper<float>
      solver_helper(top_k, static_cast<float>(lambda), num_examples);

    PrimalSolver<float, TopKLossL2RegularizerDualVariablesHelper<float>>
      solver(solver_helper, num_dimensions, num_examples, num_tasks,
             Xptr, &labels[0], Wptr, Aptr);

    solver.set_check_gap_frequency(check_gap_frequency);
    solver.set_max_num_epoch(max_num_epoch);
    solver.set_epsilon(static_cast<float>(epsilon));
    solver.set_seed(seed);
    solver.Solve();

    mxInfo = createInfoStruct<float>(solver, svm_c, lambda);

  }

  if (nlhs > 0) plhs[0] = mxW;
  if (nlhs > 1) plhs[1] = mxInfo;
  if (nlhs > 2) plhs[2] = mxA;
}

