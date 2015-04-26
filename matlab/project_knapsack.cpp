#include <algorithm>
#include <cstddef>
#include <vector>

#include <mex.h>
#include "matrix.h"

#include "knapsackprojector.hpp"

void printUsage() {
  mexPrintf("Usage: projknapsack(X); (lo = 0, hi = 1, rhs = 1)\n"
            "       [X_proj] = projknapsack(X,lo,hi,rhs);\n");
}

void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
  if (nrhs < 1 || nrhs > 4) {
    printUsage();
    mexErrMsgIdAndTxt("LIBSDCA:inputmismatch",
      "Wrong number of input arguments.");
  }

  if (nlhs > 1) {
    printUsage();
    mexErrMsgIdAndTxt("LIBSDCA:outputmismatch",
      "Wrong number of output arguments.");
  }

  double lo = 0, hi = 1, rhs = 1;
  if (nrhs >= 2) {
    lo = mxGetScalar(prhs[1]);
  }
  if (nrhs >= 3) {
    hi = mxGetScalar(prhs[2]);
  }
  if (nrhs >= 4) {
    rhs = mxGetScalar(prhs[3]);
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
    sdca::KnapsackProjector<double> proj(lo, hi, rhs);
    proj.Project(static_cast<double*>(mxGetData(mxX)), m, n);
  } else if (mxIsSingle(mxX)) {
    sdca::KnapsackProjector<float> proj(
      static_cast<float>(lo),
      static_cast<float>(hi),
      static_cast<float>(rhs));
    proj.Project(static_cast<float*>(mxGetData(mxX)), m, n);
  }
}

