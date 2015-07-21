#include "mex_util.h"
#include "prox/prox.h"

#ifndef MEX_PROX
#define MEX_PROX "mex_prox"
#endif

using namespace sdca;

void
printUsage() {
  mexPrintf("Usage: X = %s(A);\n"
            "       X = %s(A, opts);\n", MEX_PROX, MEX_PROX);
}

template <typename Data,
          typename Result,
          typename Summation>
void
mex_main(
    const int nlhs,
    mxArray* plhs[],
    const mxArray* prhs[],
    const mxArray* opts,
    Summation sum
    ) {

  mxArray* mxX;
  if (nlhs == 0) {
    mxX = const_cast<mxArray*>(prhs[0]); // in-place
  } else {
    mxX = mxDuplicateArray(prhs[0]);
    plhs[0] = mxX;
  }

  auto lo = mxGetFieldValueOrDefault<Result>(opts, "lo", 0);
  auto hi = mxGetFieldValueOrDefault<Result>(opts, "hi", 1);
  auto rhs = mxGetFieldValueOrDefault<Result>(opts, "rhs", 1);
  auto rho = mxGetFieldValueOrDefault<Result>(opts, "rho", 1);
  auto k = mxGetFieldValueOrDefault<std::ptrdiff_t>(opts, "k", 1);

  std::ptrdiff_t m = static_cast<std::ptrdiff_t>(mxGetM(mxX));
  std::ptrdiff_t n = static_cast<std::ptrdiff_t>(mxGetN(mxX));

  mxCheck<Result>(std::greater_equal<Result>(), rhs, 0, "rhs");
  mxCheck<Result>(std::greater_equal<Result>(), rho, 0, "rho");
  mxCheckRange<std::ptrdiff_t>(k, 1, m, "k");

  std::vector<Data> aux(static_cast<std::size_t>(m));
  Data* first = static_cast<Data*>(mxGetData(mxX));
  Data* last = first + m*n;
  Data* aux_first = &aux[0];
  Data* aux_last = aux_first + m;

  std::string proj = mxGetFieldValueOrDefault(
    opts, "proj", std::string("knapsack_eq"));
  if (proj == "knapsack_eq") {
    project_knapsack_eq<Data*, Result, Summation>(
      m, first, last, aux_first, aux_last, lo, hi, rhs, sum);
  } else if (proj == "knapsack_le") {
    project_knapsack_le<Data*, Result, Summation>(
      m, first, last, aux_first, aux_last, lo, hi, rhs, sum);
  } else if (proj == "knapsack_le_biased") {
    project_knapsack_le_biased<Data*, Result, Summation>(
      m, first, last, aux_first, aux_last, lo, hi, rhs, rho, sum);
  } else if (proj == "topk_cone") {
    project_topk_cone<Data*, Result, Summation>(
      m, first, last, aux_first, aux_last, k, sum);
  } else if (proj == "topk_cone_biased") {
    project_topk_cone_biased<Data*, Result, Summation>(
      m, first, last, aux_first, aux_last, k, rho, sum);
  } else if (proj == "topk_simplex") {
    project_topk_simplex<Data*, Result, Summation>(
      m, first, last, aux_first, aux_last, k, rhs, sum);
  } else if (proj == "topk_simplex_biased") {
    project_topk_simplex_biased<Data*, Result, Summation>(
      m, first, last, aux_first, aux_last, k, rhs, rho, sum);
  } else {
    mexErrMsgIdAndTxt(
      err_id[err_proj_type], err_msg[err_proj_type], proj.c_str());
  }
}

template <typename Data,
          typename Result>
inline void
mex_main(
    const int nlhs,
    mxArray* plhs[],
    const mxArray* prhs[],
    const mxArray* opts
    ) {
  std::string summation = mxGetFieldValueOrDefault(
    opts, "summation", std::string("standard"));
  if (summation == "standard") {
    std_sum<Data*, Result> sum;
    mex_main<Data, Result, std_sum<Data*, Result>>(
      nlhs, plhs, prhs, opts, sum);
  } else if (summation == "kahan") {
    kahan_sum<Data*, Result> sum;
    mex_main<Data, Result, kahan_sum<Data*, Result>>(
      nlhs, plhs, prhs, opts, sum);
  } else {
    mexErrMsgIdAndTxt(
      err_id[err_sum_type], err_msg[err_sum_type], summation.c_str());
  }
}

template <typename Data>
inline void
mex_main(
    const int nlhs,
    mxArray* plhs[],
    const int nrhs,
    const mxArray* prhs[]
    ) {
  const mxArray* opts = (nrhs > 1) ? prhs[1] : nullptr;
  mxCheckStruct(opts, "opts");
  std::string precision = mxGetFieldValueOrDefault(
    opts, "precision", std::string("double"));
  if (precision == "double") {
    mex_main<Data, double>(nlhs, plhs, prhs, opts);
  } else if (precision == "single" || precision == "float") {
    mex_main<Data, float>(nlhs, plhs, prhs, opts);
  } else if (precision == "long_double" || precision == "long double") {
    mex_main<Data, long double>(nlhs, plhs, prhs, opts);
  } else {
    mexErrMsgIdAndTxt(
      err_id[err_prec_type], err_msg[err_prec_type], precision.c_str());
  }
}

void
mexFunction(
    const int nlhs,
    mxArray* plhs[],
    const int nrhs,
    const mxArray* prhs[]
    ) {
  mxCheckArgNum(nrhs, 1, 2, printUsage);
  mxCheckArgNum(nlhs, 0, 1, printUsage);
  mxCheckNotSparse(prhs[0], "A");
  mxCheckNotEmpty(prhs[0], "A");
  mxCheckReal(prhs[0], "A");

  if (mxIsDouble(prhs[0])) {
     mex_main<double>(nlhs, plhs, nrhs, prhs);
  } else if (mxIsSingle(prhs[0])) {
     mex_main<float>(nlhs, plhs, nrhs, prhs);
  }
}
