
#include "mex_util.h"
#include "prox/prox.h"

#ifndef MEX_PROX
#define MEX_PROX "mex_prox"
#endif

using namespace sdca;

void printUsage() {
  mexPrintf("Usage: X = %s(A);\n"
            "       X = %s(A, opts);\n", MEX_PROX, MEX_PROX);
}

template <typename Type>
void
mex_main(
    const int nlhs,
    mxArray* plhs[],
    const int nrhs,
    const mxArray* prhs[]
    ) {

  mxArray *mxX;
  if (nlhs == 0) {
    mxX = const_cast<mxArray*>(prhs[0]); // in-place
  } else {
    mxX = mxDuplicateArray(prhs[0]);
    plhs[0] = mxX;
  }

  const mxArray* opts = (nrhs > 1) ? prhs[1] : nullptr;
  mxCheckStruct(opts, "opts");

  auto lo = mxGetFieldValueOrDefault<Type>(opts, "lo", 0);
  auto hi = mxGetFieldValueOrDefault<Type>(opts, "hi", 1);
  auto rhs = mxGetFieldValueOrDefault<Type>(opts, "rhs", 1);
  auto rho = mxGetFieldValueOrDefault<Type>(opts, "rho", 1);
  auto k = mxGetFieldValueOrDefault<std::ptrdiff_t>(opts, "k", 1);

  std::ptrdiff_t m = static_cast<std::ptrdiff_t>(mxGetM(mxX));
  std::ptrdiff_t n = static_cast<std::ptrdiff_t>(mxGetN(mxX));

  mxCheckRange<Type>(rhs, 0, std::numeric_limits<Type>::infinity(), "rhs");
  mxCheckRange<Type>(rho, 0, std::numeric_limits<Type>::infinity(), "rho");
  mxCheckRange<std::ptrdiff_t>(k, 1, m, "k");

  std::vector<Type> aux(static_cast<std::size_t>(m));
  Type* first = static_cast<Type*>(mxGetData(mxX));
  Type* last = first + m*n;
  Type* aux_first = &aux[0];
  Type* aux_last = aux_first + m;

  std::string proj = mxGetFieldValueOrDefault(
    opts, "proj", std::string("knapsack_eq"));
  if (proj == "knapsack_eq") {
    project_knapsack_eq(
      m, first, last, aux_first, aux_last, lo, hi, rhs);
  } else if (proj == "knapsack_le") {
    project_knapsack_le(
      m, first, last, aux_first, aux_last, lo, hi, rhs);
  } else if (proj == "knapsack_le_biased") {
    project_knapsack_le_biased(
      m, first, last, aux_first, aux_last, lo, hi, rhs, rho);
  } else if (proj == "topk_cone") {
    project_topk_cone(
      m, first, last, aux_first, aux_last, k);
  } else if (proj == "topk_cone_biased") {
    project_topk_cone_biased(
      m, first, last, aux_first, aux_last, k, rho);
  } else if (proj == "topk_simplex") {
    project_topk_simplex(
      m, first, last, aux_first, aux_last, k, rhs);
  } else if (proj == "topk_simplex_biased") {
    project_topk_simplex_biased(
      m, first, last, aux_first, aux_last, k, rhs, rho);
  } else {
    mexErrMsgIdAndTxt(
      err_id[err_proj_type], err_msg[err_proj_type], proj.c_str());
  }
}

void mexFunction(
    const int nlhs,
    mxArray* plhs[],
    const int nrhs,
    const mxArray* prhs[]
    ) {
  mxCheckArgNum(nrhs, 1, 2, printUsage);
  mxCheckArgNum(nlhs, 0, 1, printUsage);
  mxCheckNotSparse(prhs[0], "A");
  mxCheckReal(prhs[0], "A");

  if (mxIsDouble(prhs[0])) {
     mex_main<double>(nlhs, plhs, nrhs, prhs);
  } else if (mxIsSingle(prhs[0])) {
     mex_main<float>(nlhs, plhs, nrhs, prhs);
  }
}
