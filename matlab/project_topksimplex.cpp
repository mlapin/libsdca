#include <algorithm>
#include <cstddef>
#include <vector>

#include <mex.h>
#include "matrix.h"

#include "topksimplexprojector.hpp"

void printUsage() {
  mexPrintf("Usage: projtopksimplex(X); (k = 1)\n"
            "       [X_proj] = projtopksimplex(X,k);\n");
}

void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
  if (nrhs < 1 || nrhs > 2) {
    printUsage();
    mexErrMsgIdAndTxt("LIBSDCA:inputmismatch",
      "Wrong number of input arguments.");
  }

  if (nlhs > 1) {
    printUsage();
    mexErrMsgIdAndTxt("LIBSDCA:outputmismatch",
      "Wrong number of output arguments.");
  }

  std::size_t k = 1;
  if (nrhs >= 2) {
    k = static_cast<std::size_t>(mxGetScalar(prhs[1]));
  }

  std::size_t m = static_cast<std::size_t>(mxGetM(prhs[0]));
  std::size_t n = static_cast<std::size_t>(mxGetN(prhs[0]));
  if (k < 1 || k > m) {
    mexErrMsgIdAndTxt("LIBSDCA:project:kbounds",
      "Argument k is out of bounds (must be in [1,size(X,1)]).");
  }

  mxArray *mxX;
  if (nlhs == 0) {
    mxX = const_cast<mxArray*>(prhs[0]); // in-place
  } else {
    mxX = mxDuplicateArray(prhs[0]);
    plhs[0] = mxX;
  }

  if (mxIsDouble(mxX)) {
    sdca::TopKSimplexProjector<double> proj(k);
    proj.Project(static_cast<double*>(mxGetData(mxX)), m, n);
  } else if (mxIsSingle(mxX)) {
    sdca::TopKSimplexProjector<float> proj(k);
    proj.Project(static_cast<float*>(mxGetData(mxX)), m, n);
  }
}

