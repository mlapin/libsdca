
#include "mex_util.h"
#include "solve/solve.h"

#ifndef MEX_SOLVE
#define MEX_SOLVE "mex_solve"
#endif

using namespace sdca;

void printUsage() {
  mexPrintf("Usage: model = %s(X, Y);\n"
            "       model = %s(data, labels [, opts]);\n"
            , MEX_SOLVE, MEX_SOLVE);
}

template <typename data_type>
void
mex_main(
    const int nlhs,
    mxArray* plhs[],
    const int nrhs,
    const mxArray* prhs[]
    ) {
  const mxArray* p_data = prhs[0];
  const mxArray* p_labels = prhs[1];
  const mxArray* opts = (nrhs > 2) ? prhs[2] : nullptr;
  mxCheckStruct(opts, "opts");

  auto m = mxGetM(p_data);
  auto n = mxGetN(p_data);

  mxCheckVector(n, p_labels, "labels");
  std::vector<std::size_t> labels(mxGetPr(p_labels), mxGetPr(p_labels) + n);
  auto minmax = std::minmax_element(mxGetPr(p_labels), mxGetPr(p_labels) + n);
  if (*minmax.first == 1) {
    std::for_each(labels.begin(), labels.end(), [](std::size_t &x){ x -= 1; });
    *minmax.second -= 1;
  } else if (*minmax.first != 0) {
    mexErrMsgIdAndTxt(err_id[err_labels_range], err_msg[err_labels_range]);
  }

  auto num_tasks = static_cast<std::size_t>(*minmax.second) + 1;
  auto k = mxGetFieldValueOrDefault<std::size_t>(opts, "k", 1);
  auto c = mxGetFieldValueOrDefault<data_type>(opts, "c", 1);

  stopping_criteria criteria;
  auto computer = make_l2_hinge_topk(k, c);
  auto solver = make_primal_solver(criteria, computer, m, n, num_tasks,
    &labels[0], static_cast<data_type*>(mxGetData(p_data)),
    static_cast<data_type*>(mxGetData(p_data)),
    static_cast<data_type*>(mxGetData(p_data)));
}

void mexFunction(
    const int nlhs,
    mxArray* plhs[],
    const int nrhs,
    const mxArray* prhs[]
    ) {
  mxCheckArgNum(nrhs, 2, 3, printUsage);
  mxCheckArgNum(nlhs, 1, 1, printUsage);
  mxCheckNotSparse(prhs[0], "data");
  mxCheckNotEmpty(prhs[0], "data");
  mxCheckReal(prhs[0], "data");
  mxCheckNotSparse(prhs[1], "labels");
  mxCheckNotEmpty(prhs[1], "labels");
  mxCheckDouble(prhs[1], "labels");

  if (mxIsDouble(prhs[0])) {
     mex_main<double>(nlhs, plhs, nrhs, prhs);
  } else if (mxIsSingle(prhs[0])) {
     mex_main<float>(nlhs, plhs, nrhs, prhs);
  }
}
