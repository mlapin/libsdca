#include "mex_util.h"
#include "prox/prox.h"

#ifndef MEX_PROX
#define MEX_PROX "mex_prox"
#endif
#ifndef LIBSDCA_VERSION
#define LIBSDCA_VERSION "0.0.0"
#endif

using namespace sdca;

inline void
printUsage() {
  mexPrintf("Usage: X = %s(A);\n"
            "       X = %s(A, OPTS);\n"
            "  See %s('version') and %s('help') for more information.\n",
            MEX_PROX, MEX_PROX, MEX_PROX, MEX_PROX);
}

inline void
printVersion() {
  mexPrintf("%s version %s.\n", MEX_PROX, LIBSDCA_VERSION);
}

inline void
printHelp() {
  mexPrintf("Usage: X = %s(A);\n"
            "       X = %s(A, OPTS);\n"
            "       %s(A, OPTS); %% A is modified in-place\n"
            "  OPTS is a struct with the following fields:\n"
            "  (default values in [brackets], synonyms in (parenthesis))\n"
            "    'prox' : 'entropy', ['knapsack'] ('knapsack_eq'),\n"
            "             'knapsack_le', 'knapsack_le_biased',\n"
            "             'topk_simplex', 'topk_simplex_biased',\n"
            "             'topk_cone', 'topk_cone_biased';\n"
            "    'lo'  : [0], prox: knapsack*;\n"
            "    'hi'  : [1], prox: entropy, knapsack*;\n"
            "    'rhs' : [1], prox: entropy, knapsack*, topk_simplex*;\n"
            "    'rho' : [1], prox: *_biased;\n"
            "    'k'   : [1], prox: topk_*.\n"
            "  Options that influence the accuracy of computations:\n"
            "    'precision' : 'single' ('float'), ['double'],\n"
            "                  'long double' ('long_double');\n"
            "    'summation' : ['standard'] ('default'), 'kahan'.\n",
            MEX_PROX, MEX_PROX, MEX_PROX);
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
    const Summation sum
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

  std::string prox = mxGetFieldValueOrDefault(
    opts, "prox", std::string("knapsack"));
  if (prox == "entropy") {
    prox_entropy<Data*, Result, Summation>(
      m, first, last, aux_first, aux_last, hi, rhs, sum);
  } else if (prox == "knapsack" || prox == "knapsack_eq") {
    prox_knapsack_eq<Data*, Result, Summation>(
      m, first, last, aux_first, aux_last, lo, hi, rhs, sum);
  } else if (prox == "knapsack_le") {
    prox_knapsack_le<Data*, Result, Summation>(
      m, first, last, aux_first, aux_last, lo, hi, rhs, sum);
  } else if (prox == "knapsack_le_biased") {
    prox_knapsack_le_biased<Data*, Result, Summation>(
      m, first, last, aux_first, aux_last, lo, hi, rhs, rho, sum);
  } else if (prox == "topk_simplex") {
    prox_topk_simplex<Data*, Result, Summation>(
      m, first, last, aux_first, aux_last, k, rhs, sum);
  } else if (prox == "topk_simplex_biased") {
    prox_topk_simplex_biased<Data*, Result, Summation>(
      m, first, last, aux_first, aux_last, k, rhs, rho, sum);
  } else if (prox == "topk_cone") {
    prox_topk_cone<Data*, Result, Summation>(
      m, first, last, aux_first, aux_last, k, sum);
  } else if (prox == "topk_cone_biased") {
    prox_topk_cone_biased<Data*, Result, Summation>(
      m, first, last, aux_first, aux_last, k, rho, sum);
  } else if (prox == "lambert_w_exp") {
    apply(m, first, last, lambert_w_exp_functor<Data, Result>());
  } else {
    mexErrMsgIdAndTxt(
      err_id[err_prox], err_msg[err_prox], prox.c_str());
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
  if (summation == "standard" || summation == "default") {
    std_sum<Data*, Result> sum;
    mex_main<Data, Result, std_sum<Data*, Result>>(
      nlhs, plhs, prhs, opts, sum);
  } else if (summation == "kahan") {
    kahan_sum<Data*, Result> sum;
    mex_main<Data, Result, kahan_sum<Data*, Result>>(
      nlhs, plhs, prhs, opts, sum);
  } else {
    mexErrMsgIdAndTxt(
      err_id[err_summation], err_msg[err_summation], summation.c_str());
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
  } else if (precision == "long double" || precision == "long_double") {
    mex_main<Data, long double>(nlhs, plhs, prhs, opts);
  } else {
    mexErrMsgIdAndTxt(
      err_id[err_precision], err_msg[err_precision], precision.c_str());
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

  if (mxIsChar(prhs[0])) {
    std::string command = mxGetString(prhs[0], "command");
    if (command == "help" || command == "--help" || command == "-h") {
      printHelp();
    } else if (command == "version" || command == "--version") {
      printVersion();
    } else {
      mexErrMsgIdAndTxt(
        err_id[err_command], err_msg[err_command], command.c_str());
    }
  } else {
    mxCheckNotSparse(prhs[0], "A");
    mxCheckNotEmpty(prhs[0], "A");
    mxCheckReal(prhs[0], "A");

    if (mxIsDouble(prhs[0])) {
       mex_main<double>(nlhs, plhs, nrhs, prhs);
    } else if (mxIsSingle(prhs[0])) {
       mex_main<float>(nlhs, plhs, nrhs, prhs);
    }
  }
}
