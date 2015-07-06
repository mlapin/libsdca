
#include "mex_util.h"
#include "prox/prox.h"

using namespace sdca;

void printUsage() {
  mexPrintf("Usage: \n");
}

template <typename Type>
void
mex_main(
    const int nlhs,
    mxArray* plhs[],
    const int nrhs,
    const mxArray* prhs[]
    ) {
  if (nrhs > 1) {
    std::string bla = mxGetFieldValueOrDefault(prhs[1], "bla", std::string("foo"));
    mexPrintf("bla value = %s\n", bla.c_str());
  }
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  if (nrhs < 1 || nrhs > 2) {
    printUsage();
    mexErrMsgIdAndTxt(err_id[err_argnum_input], err_msg[err_argnum_input]);
  }

  if (nlhs > 1) {
    mexErrMsgIdAndTxt(err_id[err_argnum_output], err_msg[err_argnum_output]);
  }

  if (nrhs > 1 && !mxIsStruct(prhs[1])) {
    mexErrMsgIdAndTxt(err_id[err_argopt_struct], err_msg[err_argopt_struct]);
  }

  if (mxIsDouble(prhs[0])) {
     mex_main<double>(nlhs, plhs, nrhs, prhs);
  } else if (mxIsSingle(prhs[0])) {
     mex_main<float>(nlhs, plhs, nrhs, prhs);
  } else {
    mexErrMsgIdAndTxt(err_id[err_argtype_real], err_msg[err_argtype_real]);
  }
}

