#include <cstddef>

#include <mex.h> 
#include "matrix.h"

#include "projector.hh"

void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
  if ( nrhs != 2) {
    mexErrMsgIdAndTxt("LIBSDCA:project:inputmismatch",
      "Two input arguments expected: X, k.");
  }
  if (nlhs > 1) {
    mexErrMsgIdAndTxt("LIBSDCA:project:outputmismatch",
      "At most one output argument expected: X_proj (otherwise in-place).");
  }

  std::size_t k = static_cast<std::size_t>(mxGetScalar(prhs[1]));
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
    double *X = static_cast<double*>(mxGetData(mxX));
    sdca::Projector<double> proj;
    proj.MatrixToKSimplex(k, m, n, X);
  } else if (mxIsSingle(mxX)) {
    float *X = static_cast<float*>(mxGetData(mxX));
    sdca::Projector<float> proj;
    proj.MatrixToKSimplex(k, m, n, X);
  }
}
