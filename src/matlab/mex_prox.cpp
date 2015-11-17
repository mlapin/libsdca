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
  mexPrintf("Usage: X = %s(A, opts);\n"
            "  See %s('help') and %s('version') for more information.\n",
            MEX_PROX, MEX_PROX, MEX_PROX);
}

inline void
printVersion() {
  mexPrintf("%s version %s.\n", MEX_PROX, LIBSDCA_VERSION);
}

inline void
printHelp(const mxArray* opts) {
  if (opts == nullptr) {
    mexPrintf(
"Usage: X = %s(A, opts);\n"
"  Applies a proximal operator opts.prox to the input matrix A columnwise.\n"
"  If X is omitted, A is modified in-place.\n"
"\n"
"  opts is a struct with the following fields (defaults in [brackets]):\n"
"    prox ['knapsack'] - the proximal operator to apply;\n"
"    lo   [0]          - the lower bound;\n"
"    hi   [1]          - the upper bound;\n"
"    rhs  [1]          - the right hand side in the sum constraint;\n"
"    rho  [1]          - the regularization parameter;\n"
"    k    [1]          - the k in the top-k cone and the top-k simplex;\n"
"    precision ['double']   - intermediate floating-point precision;\n"
"    summation ['standard'] - summation method (Kahan or standard).\n"
"\n"
"  See %s('help', <field>) for more information on a particular field, e.g.\n"
"  %s('help', 'prox')\n",
      MEX_PROX, MEX_PROX, MEX_PROX);
    return;
  }

  std::string arg = mxGetString(opts, "help argument");
  if (arg == "prox") {
    mexPrintf(
"opts.prox - the proximal operator or the projection to apply.\n"
"  Possible values:\n"
"    knapsack (synonym: knapsack_eq)\n"
"      - continuous quadratic knapsack problem with the equality constraint\n"
"    knapsack_le\n"
"      - knapsack problem with the inequality constraint\n"
"    knapsack_le_biased\n"
"      - regularized (biased) knapsack problem\n"
"    topk_simplex\n"
"      - projection onto the top-k simplex\n"
"    topk_simplex_biased\n"
"      - regularized (biased) projection onto the top-k simplex\n"
"    topk_cone\n"
"      - projection onto the top-k cone\n"
"    topk_cone_biased\n"
"      - regularized (biased) projection onto the top-k cone\n"
"  Default value:\n"
"    knapsack\n"
      );
  } else if (arg == "lo" || arg == "hi") {
    mexPrintf(
"opts.lo - the lower bound constraint;\n"
"opts.hi - the upper bound constraint.\n"
"  Possible values:\n"
"    any real number such that opts.lo <= opts.hi holds\n"
"  Default values:\n"
"    opts.lo = 0\n"
"    opts.hi = 1\n"
"  Applies to prox operators:\n"
"    knapsack (synonym: knapsack_eq)\n"
"    knapsack_le\n"
"    knapsack_le_biased\n"
      );
  } else if (arg == "rhs") {
    mexPrintf(
"opts.rhs - the right hand side in the sum constraint, the radius of the set.\n"
"  Possible values:\n"
"    any nonnegative real number\n"
"  Default value:\n"
"    opts.rhs = 1\n"
"  Applies to prox operators:\n"
"    knapsack (synonym: knapsack_eq)\n"
"    knapsack_le\n"
"    knapsack_le_biased\n"
"    topk_simplex\n"
"    topk_simplex_biased\n"
      );
  } else if (arg == "rho") {
    mexPrintf(
"opts.rho - the regularization parameter in the biased projection.\n"
"  Possible values:\n"
"    any nonnegative real number\n"
"  Default value:\n"
"    opts.rho = 1\n"
"  Applies to prox operators:\n"
"    knapsack_le_biased\n"
"    topk_simplex_biased\n"
"    topk_cone_biased\n"
      );
  } else if (arg == "k") {
    mexPrintf(
"opts.k - the k in the top-k cone and the top-k simplex.\n"
"  Possible values:\n"
"    any integer in [1,d], where d is the problem dimension\n"
"  Default value:\n"
"    opts.k = 1\n"
"  Applies to prox operators:\n"
"    topk_simplex\n"
"    topk_simplex_biased\n"
"    topk_cone\n"
"    topk_cone_biased\n"
      );
  } else if (arg == "precision") {
    mexPrintf(
"opts.precision - precision in intermediate floating-point operations.\n"
"  (e.g. can be used to increase the accuracy of computation on float data)\n"
"  Possible values:\n"
"    float (synonym: single)\n"
"    double\n"
"    long double (synonym: long_double)\n"
"  Default value:\n"
"    double\n"
"  Applies to prox operators:\n"
"    <all>\n"
      );
  } else if (arg == "summation") {
    mexPrintf(
"opts.summation - summation method to use.\n"
"  (Kahan summation is more accurate, but slower than standard)\n"
"  Possible values:\n"
"    standard (synonym: default)\n"
"    kahan\n"
"  Default value:\n"
"    standard\n"
"  Applies to prox operators:\n"
"    <all>\n"
      );
  } else {
    mexErrMsgIdAndTxt(
      err_id[err_help_arg], err_msg[err_help_arg], arg.c_str());
  }
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
  if (prox == "knapsack" || prox == "knapsack_eq") {
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
    const mxArray* opts = (nrhs > 1) ? prhs[1] : nullptr;
    if (command == "help") {
      printHelp(opts);
    } else if (command == "version") {
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
