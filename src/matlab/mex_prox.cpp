#include "mex_util.h"
#include "sdca/prox.h"

#ifndef MEX_PROX
#define MEX_PROX "mex_prox"
#endif
#ifndef LIBSDCA_VERSION
#ifdef SDCA_VERSION
#define LIBSDCA_VERSION SDCA_VERSION
#else
#define LIBSDCA_VERSION "0.0.0"
#endif
#endif

using namespace sdca;

inline void printUsage() {
  mexPrintf(
      "Usage: X = %s(A, opts);\n"
      "  See %s('help') and %s('version') for more information.\n",
      MEX_PROX, MEX_PROX, MEX_PROX);
}

inline void printVersion() { mexPrintf("%s version %s.\n", MEX_PROX, LIBSDCA_VERSION); }

inline void printHelp(const mxArray* opts) {
  if (opts == nullptr) {
    mexPrintf(
        "Usage: X = %s(A, opts);\n"
        "  Applies a proximal operator opts.prox to the input matrix A columnwise.\n"
        "  An output X is required; A is never modified.\n"
        "\n"
        "  opts is a struct with the following fields (defaults in [brackets]):\n"
        "    prox  ['knapsack'] - the proximal operator to apply;\n"
        "    lo    [0] - the lower bound;\n"
        "    hi    [1] - the upper bound;\n"
        "    rhs   [1] - the right hand side in the sum constraint;\n"
        "    rho   [1] - the regularization parameter in biased projections;\n"
        "    alpha [1] - the alpha parameter in entropic projections;\n"
        "    k     [1] - the k in the top-k cone and the top-k simplex;\n"
        "    p     [1] - the length of x in two_simplex and two_simplex_sort;\n"
        "    precision ['double'] - working arithmetic: float32/fp32/single,\n"
        "                           float64/fp64/double, or long_double.\n"
        "\n"
        "  See %s('help', 'prox') for more information on possible operators.\n",
        MEX_PROX, MEX_PROX);
    return;
  }

  std::string arg = mxGetString(opts, "help argument");
  if (arg == "prox") {
    mexPrintf(
        "opts.prox - an operator to apply.\n"
        "  Proximal and projection operators (columnwise):\n"
        "    entropy\n"
        "        min_x <x, log(x)> - <a, x>\n"
        "        s.t.  <1, x> = rhs, 0 <= x_i <= hi\n"
        "    entropy_norm\n"
        "        min_x 0.5 * <x, x> + <x, log(x)> - <a, x>\n"
        "        s.t.  <1, x> = rhs, 0 <= x_i <= hi\n"
        "    knapsack (synonym: knapsack_eq)\n"
        "        min_x 0.5 * <x, x> - <a, x>\n"
        "        s.t.  <1, x> = rhs, lo <= x_i <= hi\n"
        "    knapsack_le\n"
        "        min_x 0.5 * <x, x> - <a, x>\n"
        "        s.t.  <1, x> <= rhs, lo <= x_i <= hi\n"
        "    knapsack_le_biased\n"
        "        min_x 0.5 * (<x, x> + rho * <1, x>^2) - <a, x>\n"
        "        s.t.  <1, x> <= rhs, lo <= x_i <= hi\n"
        "    topk_cone\n"
        "        min_x 0.5 * <x, x> - <a, x>\n"
        "        s.t.  0 <= x_i <= <1, x> / k\n"
        "    topk_cone_biased\n"
        "        min_x 0.5 * (<x, x> + rho * <1, x>^2) - <a, x>\n"
        "        s.t.  0 <= x_i <= <1, x> / k\n"
        "    topk_entropy\n"
        "        min_{x,s} <x, log(x)> + (1 - s) * log(1 - s) - <a, x>\n"
        "        s.t.      <1, x> = s, s <= 1, 0 <= x_i <= s / k\n"
        "    topk_entropy_biased\n"
        "        min_{x,s} 0.5 * alpha * (<x, x> + s * s) - <a, x>\n"
        "                  + <x, log(x)> + (1 - s) * log(1 - s)\n"
        "        s.t.      <1, x> = s, s <= 1, 0 <= x_i <= s / k\n"
        "    topk_simplex\n"
        "        min_x 0.5 * <x, x> - <a, x>\n"
        "        s.t.  <1, x> <= rhs, 0 <= x_i <= <1, x> / k\n"
        "    topk_simplex_biased\n"
        "        min_x 0.5 * (<x, x> + rho * <1, x>^2) - <a, x>\n"
        "        s.t.  <1, x> <= rhs, 0 <= x_i <= <1, x> / k\n"
        "    two_simplex (two_simplex_sort: the same problem, solved by sorting)\n"
        "        min_{x,y} ||x - a||^2 + ||y - b||^2, [a; b] a column, a of length p\n"
        "        s.t.      <1, x> = <1, y> <= rhs, 0 <= x_i, 0 <= y_j\n"
        "\n"
        "  Elementwise operators (not proximal):\n"
        "    lambert_w_exp\n"
        "      - applies the Lambert W function of exponent, i.e. W(exp(x)).\n"
        "        Computed w satisfies the equation\n"
        "        w + log(w) = x\n"
        "\n"
        "  Default value:\n"
        "    knapsack\n");
  } else {
    mex_error(err_id[err_help_arg], err_msg[err_help_arg], arg.c_str());
  }
}

template <typename Data, typename Result>
void mex_main(const int nlhs, mxArray* plhs[], const mxArray* prhs[], const mxArray* opts) {
  (void)nlhs;
  mxArray* mxX = mxDuplicateArray(prhs[0]);
  plhs[0] = mxX;

  auto lo = mxGetFieldValueOrDefault<Result>(opts, "lo", 0);
  auto hi = mxGetFieldValueOrDefault<Result>(opts, "hi", 1);
  auto rhs = mxGetFieldValueOrDefault<Result>(opts, "rhs", 1);
  auto rho = mxGetFieldValueOrDefault<Result>(opts, "rho", 1);
  auto alpha = mxGetFieldValueOrDefault<Result>(opts, "alpha", 1);
  auto k = mxGetFieldValueOrDefault<std::ptrdiff_t>(opts, "k", 1);
  auto p = mxGetFieldValueOrDefault<std::ptrdiff_t>(opts, "p", 1);

  std::ptrdiff_t m = static_cast<std::ptrdiff_t>(mxGetM(mxX));
  std::ptrdiff_t n = static_cast<std::ptrdiff_t>(mxGetN(mxX));

  std::string prox = mxGetFieldValueOrDefault(opts, "prox", std::string("knapsack"));
  const bool signed_rhs = (prox == "knapsack" || prox == "knapsack_eq" || prox == "knapsack_le" ||
                           prox == "knapsack_le_biased");
  if (!signed_rhs) mxCheck<Result>(std::greater_equal<Result>(), rhs, 0, "rhs");
  mxCheck<Result>(std::greater_equal<Result>(), rho, 0, "rho");
  mxCheck<Result>(std::greater<Result>(), alpha, 0, "alpha");
  mxCheckRange<std::ptrdiff_t>(k, 1, m, "k");

  std::vector<Data> aux(static_cast<std::size_t>(m));
  Data* first = static_cast<Data*>(mxGetData(mxX));
  Data* last = first + m * n;
  Data* aux_first = &aux[0];

  if (prox == "two_simplex" || prox == "two_simplex_sort")
    mxCheckRange<std::ptrdiff_t>(p, 1, m - 1, "p");
  if (lo > hi || !std::isfinite(lo) || !std::isfinite(hi))
    throw std::invalid_argument("Invalid projection bounds");
  for (Data* x = first; x != last; ++x)
    if (!std::isfinite(*x)) throw std::invalid_argument("Proximal inputs must be finite");
  if ((prox == "knapsack" || prox == "knapsack_eq") &&
      (rhs < lo * Result(m) || rhs > hi * Result(m)))
    throw std::invalid_argument("Infeasible knapsack sum");
  if ((prox == "entropy" || prox == "entropy_norm") && (hi <= 0 || rhs > hi * Result(m)))
    throw std::invalid_argument("Infeasible entropy sum");
  if (prox == "knapsack" || prox == "knapsack_eq") {
    prox_knapsack_eq(m, first, last, aux_first, lo, hi, rhs);
  } else if (prox == "knapsack_le") {
    prox_knapsack_le(m, first, last, aux_first, lo, hi, rhs);
  } else if (prox == "knapsack_le_biased") {
    prox_knapsack_le_biased(m, first, last, aux_first, lo, hi, rhs, rho);
  } else if (prox == "topk_simplex") {
    prox_topk_simplex(m, first, last, aux_first, k, rhs);
  } else if (prox == "topk_simplex_biased") {
    prox_topk_simplex_biased(m, first, last, aux_first, k, rhs, rho);
  } else if (prox == "two_simplex") {
    prox_two_simplex(m, p, first, last, aux_first, rhs);
  } else if (prox == "two_simplex_sort") {
    prox_two_simplex_sort(m, p, first, last, aux_first, rhs);
  } else if (prox == "topk_entropy") {
    prox_topk_entropy<Result>(m, first, last, aux_first, k);
  } else if (prox == "topk_entropy_biased") {
    prox_topk_entropy_biased(m, first, last, aux_first, k, alpha);
  } else if (prox == "entropy") {
    prox_entropy(m, first, last, aux_first, hi, rhs);
  } else if (prox == "entropy_norm") {
    prox_entropy_norm(m, first, last, aux_first, hi, rhs);
  } else if (prox == "topk_cone") {
    prox_topk_cone<Result>(m, first, last, aux_first, k);
  } else if (prox == "topk_cone_biased") {
    prox_topk_cone_biased(m, first, last, aux_first, k, rho);
  } else if (prox == "lambert_w_exp") {
    apply(m, first, last,
          [](Data x) { return static_cast<Data>(lambert_w_exp(static_cast<Result>(x))); });
  } else {
    mex_error(err_id[err_prox], err_msg[err_prox], prox.c_str());
  }
}

template <typename Data>
inline void mex_main(const int nlhs, mxArray* plhs[], const int nrhs, const mxArray* prhs[]) {
  const mxArray* opts = (nrhs > 1) ? prhs[1] : nullptr;
  mxCheckStruct(opts, "opts");
  std::string precision = mxGetFieldValueOrDefault(opts, "precision", std::string("double"));
  if (precision == "double" || precision == "float64" || precision == "fp64") {
    mex_main<Data, double>(nlhs, plhs, prhs, opts);
  } else if (precision == "single" || precision == "float32" || precision == "fp32") {
    mex_main<Data, float>(nlhs, plhs, prhs, opts);
  } else if (precision == "long double" || precision == "long_double") {
    mex_main<Data, long double>(nlhs, plhs, prhs, opts);
  } else {
    mex_error(err_id[err_precision], err_msg[err_precision], precision.c_str());
  }
}

void mexFunction(const int nlhs, mxArray* plhs[], const int nrhs, const mxArray* prhs[]) {
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
      mex_error(err_id[err_command], err_msg[err_command], command.c_str());
    }
  } else {
    mxCheckArgNum(nlhs, 1, 1, printUsage);
    mxCheckNotSparse(prhs[0], "A");
    mxCheckNotEmpty(prhs[0], "A");
    mxCheckReal(prhs[0], "A");

    try {
      if (mxIsDouble(prhs[0])) {
        mex_main<double>(nlhs, plhs, nrhs, prhs);
      } else if (mxIsSingle(prhs[0])) {
        mex_main<float>(nlhs, plhs, nrhs, prhs);
      } else {
        throw std::invalid_argument("A must be single or double");
      }
    } catch (const std::exception& e) {
      mex_error(err_id[err_exception], err_msg[err_exception], e.what());
    }
  }
}
