#include <algorithm>
#include <cstddef>
#include <vector>

#include <mex.h>
#include "matrix.h"

#include "knapsack_le_biased_projector.hpp"

void printUsage() {
  mexPrintf("Usage: project_knapsack_le_biased(X);"
            " (lo = 0, hi = 1, rhs = 1, rho = 1)\n"
            "       [X_proj] = project_knapsack_le_biased(X,lo,hi,rhs,rho);\n");
}

void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
  if (nrhs < 1 || nrhs > 5) {
    printUsage();
    mexErrMsgIdAndTxt("LIBSDCA:inputmismatch",
      "Wrong number of input arguments.");
  }

  if (nlhs > 1) {
    printUsage();
    mexErrMsgIdAndTxt("LIBSDCA:outputmismatch",
      "Wrong number of output arguments.");
  }

  double lo = 0, hi = 1, rhs = 1, rho = 1;
  if (nrhs > 1) {
    lo = mxGetScalar(prhs[1]);
  }
  if (nrhs > 2) {
    hi = mxGetScalar(prhs[2]);
  }
  if (nrhs > 3) {
    rhs = mxGetScalar(prhs[3]);
  }
  if (nrhs > 4) {
    rho = mxGetScalar(prhs[4]);
  }

  std::size_t m = static_cast<std::size_t>(mxGetM(prhs[0]));
  std::size_t n = static_cast<std::size_t>(mxGetN(prhs[0]));

  mxArray *mxX;
  if (nlhs == 0) {
    mxX = const_cast<mxArray*>(prhs[0]); // in-place
  } else {
    mxX = mxDuplicateArray(prhs[0]);
    plhs[0] = mxX;
  }

  if (mxIsDouble(mxX)) {
    sdca::KnapsackLEBiasedProjector<double> proj(lo, hi, rhs, rho);
    proj.Project(m, n, static_cast<double*>(mxGetData(mxX)));
  } else if (mxIsSingle(mxX)) {
    sdca::KnapsackLEBiasedProjector<float> proj(
      static_cast<float>(lo),
      static_cast<float>(hi),
      static_cast<float>(rhs),
      static_cast<float>(rho));
    proj.Project(m, n, static_cast<float*>(mxGetData(mxX)));
  }
}

