#include "mex_util.h"
#include "sdca/loss.h"
#include "sdca/utility/parallel.h"

#ifndef MEX_LOSS
#define MEX_LOSS "matsdca_loss"
#endif
#ifndef LIBSDCA_VERSION
#ifdef SDCA_VERSION
#define LIBSDCA_VERSION SDCA_VERSION
#else
#define LIBSDCA_VERSION "0.0.0"
#endif
#endif

namespace {
// The reference of matsdca_loss('help').
void print_help() {
  mexPrintf(
      "Usage: [values, gradients] = %s(scores, targets, opts);\n"
      "  Loss values (1-by-n) and their gradients with respect to the m-by-n\n"
      "  scores (m classes, n examples), without weight regularization;\n"
      "  the gradients have the class of the scores (single or double).\n"
      "\n"
      "  targets are 1-based double labels (n elements) or a full logical\n"
      "  m-by-n mask; entropy with a mask puts equal target probability on the\n"
      "  positive classes.\n"
      "\n"
      "  opts is a struct with the following fields (defaults in [brackets]):\n"
      "    loss  ['cross_entropy'] - cross_entropy, topk_cross_entropy,\n"
      "            multiclass_hinge, topk_hinge_alpha, topk_hinge_beta,\n"
      "            multilabel_hinge, multilabel_cross_entropy or\n"
      "            proximal_cross_entropy (alias objective);\n"
      "    k     [1] - the k of the top-k losses, in [1, m-1];\n"
      "    gamma [0] - the smoothing of the hinge losses (0: nonsmooth);\n"
      "    alpha [1] - the step of proximal_cross_entropy;\n"
      "    precision ['double'] - working arithmetic of single scores:\n"
      "            'single' (or 'float32', 'fp32'), 'double' (or 'float64', 'fp64');\n"
      "    projection_precision ['float64'] - the smooth multilabel_hinge projection:\n"
      "            'float64' (double, checked) or 'extended' (always extended precision);\n"
      "    num_threads [SDCA_NUM_THREADS, else 0] - threads for the examples\n"
      "            (0: automatic, 1: serial); the results do not depend on it.\n",
      MEX_LOSS);
}

// The losses of examples [begin, end) of the classes-by-n scores, with
// 1-based labels or a logical mask; no mx calls, so workers may run it.
template <typename Data>
struct loss_rows {
  sdca::loss_kind kind;
  std::size_t classes, k;
  double gamma, alpha;
  bool single;
  sdca::projection_precision mode;
  const Data* input;
  const mxLogical* mask;
  const double* labels;
  double* values;
  Data* gradients;

  void operator()(std::size_t begin, std::size_t end) const {
    using namespace sdca;
    std::vector<std::size_t> order(classes);
    std::vector<bool> positive(classes);
    std::vector<Data> z(classes), g(classes);
    for (std::size_t i = begin; i < end; ++i) {
      std::fill(positive.begin(), positive.end(), false);
      if (mask) {
        for (std::size_t j = 0; j < classes; ++j) positive[j] = mask[i * classes + j];
      } else {
        double label = labels[i];
        if (!std::isfinite(label) || std::floor(label) != label || label < 1 ||
            label > double(classes))
          throw std::invalid_argument("Labels must be integers in [1,classes]");
        positive[static_cast<std::size_t>(label) - 1] = true;
      }
      std::size_t p = 0, q = classes;
      for (std::size_t j = 0; j < classes; ++j)
        if (positive[j]) order[p++] = j;
      for (std::size_t j = classes; j-- > 0;)
        if (!positive[j]) order[--q] = j;
      for (std::size_t j = 0; j < classes; ++j) {
        z[j] = input[i * classes + order[j]];
        if (!std::isfinite(z[j])) throw std::invalid_argument("Scores must be finite");
      }
      values[i] = single ? loss_and_gradient<Data, float>(kind, classes, p, z.data(), g.data(), k,
                                                          gamma, alpha, mode)
                         : loss_and_gradient<Data, double>(kind, classes, p, z.data(), g.data(), k,
                                                           gamma, alpha, mode);
      if (gradients)
        for (std::size_t j = 0; j < classes; ++j) gradients[i * classes + order[j]] = g[j];
    }
  }
};

template <typename Data>
void evaluate(mxArray* plhs[], int nlhs, const mxArray* scores, const mxArray* targets,
              const mxArray* opts) {
  using namespace sdca;
  const std::size_t classes = mxGetM(scores), n = mxGetN(scores);
  if (classes < 2 || n == 0 || mxGetNumberOfDimensions(scores) != 2)
    throw std::invalid_argument("Scores must be a nonempty classes-by-examples matrix");
  // opts.loss, as in the other interfaces, or opts.objective as before.
  if (mxIsField(opts, "loss") && mxIsField(opts, "objective"))
    throw std::invalid_argument("Specify only opts.loss or opts.objective");
  std::string name = mxGetFieldValueOrDefault(opts, "objective", std::string("cross_entropy"));
  name = mxGetFieldValueOrDefault(opts, "loss", name);
  if (name == "topk_hinge") name = "topk_hinge_beta";
  int kind = -1;
  for (int i = 0; i < num_shared_names; ++i)
    if (name == shared_names[i]) kind = i;
  if (kind < 0) throw std::invalid_argument("Unknown loss '" + name + "'");
  auto k = mxGetFieldValueOrDefault<std::size_t>(opts, "k", 1);
  auto gamma = mxGetFieldValueOrDefault<double>(opts, "gamma", 0);
  auto alpha = mxGetFieldValueOrDefault<double>(opts, "alpha", 1);
  // As matsdca_fit: 'single'/'float32' computes single data in float, 'double'/'float64' in double.
  const std::string precision = mxGetFieldValueOrDefault(opts, "precision", std::string("double"));
  const bool single = precision == "single" || precision == "float32" || precision == "fp32";
  if (!single && precision != "double" && precision != "float64" && precision != "fp64")
    throw std::invalid_argument("Unknown precision '" + precision + "'");
  // As matsdca_fit: 'float64' (default) or 'extended' for the smooth multilabel hinge.
  const std::string projection =
      mxGetFieldValueOrDefault(opts, "projection_precision", std::string("float64"));
  projection_precision mode = projection_precision::float64;
  if (!projection_precision_of(projection, mode))
    throw std::invalid_argument("Unknown projection_precision '" + projection + "'");
  const bool mask = mxIsLogical(targets) && mxGetM(targets) == classes && mxGetN(targets) == n;
  if (mxIsSparse(targets) || mxIsComplex(targets) ||
      (!mask && (!mxIsDouble(targets) || mxGetNumberOfElements(targets) != n)))
    throw std::invalid_argument("Targets must be 1-based double labels or a full logical mask");
  // Without opts.num_threads, SDCA_NUM_THREADS, else automatic.
  const auto num_threads =
      mxGetFieldValueOrDefault<std::size_t>(opts, "num_threads", detail::default_threads);
  plhs[0] = mxCreateDoubleMatrix(1, n, mxREAL);
  Data* gradients = nullptr;
  if (nlhs > 1) {
    plhs[1] = mxCreateNumericMatrix(classes, n, mex_class<Data>::id(), mxREAL);
    gradients = static_cast<Data*>(mxGetData(plhs[1]));
  }
  const loss_rows<Data> rows = {loss_kind(kind),
                                classes,
                                k,
                                gamma,
                                alpha,
                                single,
                                mode,
                                static_cast<const Data*>(mxGetData(scores)),
                                mask ? static_cast<const mxLogical*>(mxGetData(targets)) : nullptr,
                                mask ? nullptr : mxGetPr(targets),
                                mxGetPr(plhs[0]),
                                gradients};
  // Examples run on their threads (0: automatic), each independently.
  detail::parallel_rows(n, classes, num_threads, rows);
}
}  // namespace

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  try {
    if (nrhs == 1 && nlhs == 0 && mxIsChar(prhs[0])) {
      const std::string command = sdca::mxGetString(prhs[0], "command");
      if (command == "help")
        print_help();
      else if (command == "version")
        mexPrintf("%s version %s.\n", MEX_LOSS, LIBSDCA_VERSION);
      else
        throw std::invalid_argument("Unknown command '" + command + "'");
      return;
    }
    if (nrhs < 2 || nrhs > 3 || nlhs < 1 || nlhs > 2)
      throw std::invalid_argument(std::string("Usage: [values,gradients] = ") + MEX_LOSS +
                                  "(scores,targets,opts); see " + MEX_LOSS + "('help')");
    sdca::mxCheckNotSparse(prhs[0], "scores");
    sdca::mxCheckReal(prhs[0], "scores");
    const mxArray* opts = nrhs == 3 ? prhs[2] : nullptr;
    sdca::mxCheckStruct(opts, "opts");
    if (mxIsDouble(prhs[0]))
      evaluate<double>(plhs, nlhs, prhs[0], prhs[1], opts);
    else if (mxIsSingle(prhs[0]))
      evaluate<float>(plhs, nlhs, prhs[0], prhs[1], opts);
    else
      throw std::invalid_argument("Scores must be single or double");
  } catch (const std::exception& e) {
    sdca::mex_error("LIBSDCA:exception", "%s", e.what());
  }
}
