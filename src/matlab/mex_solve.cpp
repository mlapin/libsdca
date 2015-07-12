
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
  std::vector<size_type> labels(mxGetPr(p_labels), mxGetPr(p_labels) + n);
  auto minmax = std::minmax_element(mxGetPr(p_labels), mxGetPr(p_labels) + n);
  if (*minmax.first == 1) {
    std::for_each(labels.begin(), labels.end(), [](size_type &x){ x -= 1; });
    *minmax.second -= 1;
  } else if (*minmax.first != 0) {
    mexErrMsgIdAndTxt(err_id[err_labels_range], err_msg[err_labels_range]);
  }

  auto num_tasks = static_cast<size_type>(*minmax.second) + 1;
  auto k = mxGetFieldValueOrDefault<size_type>(opts, "k", 1);
  mxCheckRange<size_type>(k, 0, num_tasks, "k");
  auto c = mxGetFieldValueOrDefault<data_type>(opts, "c", 1);
  mxCheck<data_type>(std::greater_equal<data_type>(), c, 0, "c");
//  auto is_dual = mxGetFieldValueOrDefault<bool>(opts, "dual", false);
  std::string obj_name = mxGetFieldValueOrDefault(
    opts, "objective", std::string("l2_hinge_topk"));

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
