#include <algorithm>
#include <cstddef>
#include <vector>

#include <mex.h>
#include <matrix.h>

#include "solvers.hpp"
#include "mex_util.hpp"

using namespace sdca;

void printUsage() {
  mexPrintf("Usage: A = solve_dual_topk_l2(K,Y); (k = 1, lambda = 1)\n"
            "       [A] = solve_dual_topk_l2(K,Y,k,lambda);\n");
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 1 || nrhs > 4) {
    printUsage();
    mexErrMsgIdAndTxt("LIBSDCA:inputmismatch",
      "Wrong number of input arguments.");
  }

  if (nlhs != 1) {
    printUsage();
    mexErrMsgIdAndTxt("LIBSDCA:outputmismatch",
      "Wrong number of output arguments.");
  }

  SizeType k = 1;
  if (nrhs >= 3) {
    k = static_cast<SizeType>(mxGetScalar(prhs[2]));
  }

  double lambda = 1.0;
  if (nrhs >= 4) {
    lambda = mxGetScalar(prhs[3]);
  }

  const mxArray *mxK = prhs[0];
  const mxArray *mxY = prhs[1];

  SizeType num_examples = static_cast<SizeType>(mxGetM(mxK));

  mxVerifyNotSparseNotEmpty(mxK, "K");
  mxVerifySingleOrDouble(mxK, "K");
  mxVerifyMatrixSquare(mxK, "K");
  mxVerifyNotSparseNotEmpty(mxY, "Y");
  mxVerifyVectorDimension(mxY, num_examples, "Y");

  SizeType min_label, max_label;
  const SizeType *Yptr = mxCreateLabelsVector(mxY, min_label, max_label);
  const SizeType num_tasks = max_label + 1;

  mwSize mxDims[2] = {num_examples, num_tasks};
  mxArray *mxA = mxCreateNumericArray(
    (mwSize) 2, mxDims, mxGetClassID(mxK), mxREAL);
  if (mxA == NULL) {
    mexErrMsgIdAndTxt(errOutOfMemory, "Failed to allocate memory for A.");
  }
  plhs[0] = mxA;

  if (mxIsDouble(mxK)) {
    const double *Kptr = static_cast<double*>(mxGetData(mxK));
    double *Aptr = static_cast<double*>(mxGetData(mxA));

    TopKLossL2RegularizerDualSolverHelper<double>
      solver_helper(k, lambda, num_examples);
    DualSolver<double, TopKLossL2RegularizerDualSolverHelper<double>>
      solver(solver_helper, num_examples, num_tasks, Kptr, Yptr, Aptr);
    solver.Solve();

  } else if (mxIsSingle(mxK)) {
    const float *Kptr = static_cast<float*>(mxGetData(mxK));
    float *Aptr = static_cast<float*>(mxGetData(mxA));

    TopKLossL2RegularizerDualSolverHelper<float>
      solver_helper(k, static_cast<float>(lambda), num_examples);
    DualSolver<float, TopKLossL2RegularizerDualSolverHelper<float>>
      solver(solver_helper, num_examples, num_tasks, Kptr, Yptr, Aptr);
    solver.Solve();

  }
}

