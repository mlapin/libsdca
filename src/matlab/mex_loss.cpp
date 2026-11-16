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
      "  scores (m classes, n examples), without weight regularization; the\n"
      "  gradients have the class of the scores (single or double).\n"
      "\n"
      "  targets, by loss:\n"
      "    labels in 1:m (a double vector of n entries): cross_entropy,\n"
      "      topk_cross_entropy, multiclass_hinge, topk_hinge_alpha, topk_hinge_beta,\n"
      "      proximal_cross_entropy;\n"
      "    label sets (an m-by-n logical, or sparse double or logical, matrix):\n"
      "      multilabel_hinge (needs a negative class per example),\n"
      "      multilabel_cross_entropy;\n"
      "    teacher logits (m-by-n, the class of the scores): distillation,\n"
      "      proximal_distillation.\n"
      "\n"
      "  opts is a struct with the following fields (defaults in [brackets]);\n"
      "  an unknown field, or a parameter the loss does not take, is an error:\n"
      "    loss  ['cross_entropy'] - one of the losses above;\n"
      "    k     [none] - the k of the top-k losses (required), in 1:m-1;\n"
      "    gamma [0] - the smoothing of the hinge losses (0: nonsmooth);\n"
      "    alpha [1] - the step of proximal_cross_entropy and proximal_distillation;\n"
      "    temperature [1] - the temperature of the distillation losses;\n"
      "    projection_precision ['float64'] - multilabel_hinge only: 'float64'\n"
      "            (double, checked) or 'extended' (always extended precision);\n"
      "    precision [class of the scores] - working precision: 'float32' (single\n"
      "            scores only) or 'float64'; proximal_distillation computes in double;\n"
      "    num_threads [SDCA_NUM_THREADS, else the CPUs] - threads for the examples\n"
      "            (1: serial); the results do not depend on it.\n",
      MEX_LOSS);
}

// The parsed options of a call.
struct loss_options {
  std::string name;
  int id = 0;
  std::size_t k = 1;
  double gamma = 0, alpha = 1, temperature = 1;
  bool single = false;
  sdca::projection_precision mode = sdca::projection_precision::float64;
  std::size_t num_threads = static_cast<std::size_t>(-1);
};

loss_options read_options(const mxArray* opts, const std::size_t classes, const bool single_data) {
  using namespace sdca;
  check_option_names(opts,
                     {"loss", "k", "gamma", "alpha", "temperature", "projection_precision",
                      "precision", "num_threads"},
                     MEX_LOSS);
  loss_options o;
  o.name = mxGetFieldValueOrDefault(opts, "loss", std::string("cross_entropy"));
  o.id = shared_loss_id(o.name);
  if (o.id < 0) throw std::invalid_argument("Unknown loss '" + o.name + "'");
  if (loss_takes_k(o.id)) {
    if (!mxIsField(opts, "k"))
      throw std::invalid_argument("The loss '" + o.name + "' needs opts.k");
    o.k = mxGetFieldValueOrDefault<std::size_t>(opts, "k", 1);
    if (o.k < 1 || o.k >= classes)
      throw std::invalid_argument("k must be an integer in 1:num_classes-1");
  } else {
    check_not_given(opts, "k", o.name);
  }
  if (loss_takes_gamma(o.id)) {
    o.gamma = mxGetFieldValueOrDefault<double>(opts, "gamma", 0);
    if (!(o.gamma >= 0) || !std::isfinite(o.gamma))
      throw std::invalid_argument("gamma must be finite and nonnegative");
  } else {
    check_not_given(opts, "gamma", o.name);
  }
  if (loss_takes_alpha(o.id)) {
    o.alpha = mxGetFieldValueOrDefault<double>(opts, "alpha", 1);
    if (!(o.alpha > 0) || !std::isfinite(o.alpha))
      throw std::invalid_argument("alpha must be finite and positive");
  } else {
    check_not_given(opts, "alpha", o.name);
  }
  if (loss_takes_temperature(o.id)) {
    o.temperature = mxGetFieldValueOrDefault<double>(opts, "temperature", 1);
    if (!(o.temperature > 0) || !std::isfinite(o.temperature))
      throw std::invalid_argument("temperature must be finite and positive");
  } else {
    check_not_given(opts, "temperature", o.name);
  }
  if (o.id == 5) {
    const std::string projection =
        mxGetFieldValueOrDefault(opts, "projection_precision", std::string("float64"));
    if (!projection_precision_of(projection, o.mode))
      throw std::invalid_argument("Unknown projection_precision '" + projection +
                                  "' (use 'float64' or 'extended')");
  } else {
    check_not_given(opts, "projection_precision", o.name);
  }
  o.single = float32_work(opts, single_data);
  o.num_threads = num_threads_option(opts);
  return o;
}

// The classes-by-n 0/1 mask of the n 1-based labels targets of a single-label loss.
std::vector<unsigned char> label_mask(const mxArray* targets, const std::size_t classes,
                                      const std::size_t n) {
  std::vector<unsigned char> mask(classes * n, 0);
  if (mxIsSparse(targets) || mxIsComplex(targets) || !mxIsDouble(targets) ||
      mxGetNumberOfElements(targets) != n)
    throw std::invalid_argument("targets must be n labels in 1:m (double) for this loss");
  const double* labels = mxGetPr(targets);
  for (std::size_t i = 0; i < n; ++i) {
    const double label = labels[i];
    if (!std::isfinite(label) || std::floor(label) != label || label < 1 || label > double(classes))
      throw std::invalid_argument("Labels must be integers in 1:num_classes");
    mask[i * classes + static_cast<std::size_t>(label) - 1] = 1;
  }
  return mask;
}

// The classes-by-n 0/1 mask of the logical or sparse label matrix targets of a
// multilabel loss.
std::vector<unsigned char> label_matrix_mask(const mxArray* targets, const std::size_t classes,
                                             const std::size_t n) {
  std::vector<unsigned char> mask(classes * n, 0);
  const bool sparse = mxIsSparse(targets);
  if (mxIsComplex(targets) || (!mxIsLogical(targets) && !(sparse && mxIsDouble(targets))) ||
      mxGetM(targets) != classes || mxGetN(targets) != n)
    throw std::invalid_argument(
        "targets must be an m-by-n logical or sparse label matrix for the multilabel losses");
  if (sparse) {
    const mwIndex* ir = mxGetIr(targets);
    const mwIndex* jc = mxGetJc(targets);
    const bool logical = mxIsLogical(targets);
    for (std::size_t i = 0; i < n; ++i)
      for (mwIndex j = jc[i]; j < jc[i + 1]; ++j) {
        const bool set = logical ? static_cast<const mxLogical*>(mxGetData(targets))[j] != 0
                                 : mxGetPr(targets)[j] != 0;
        if (set) mask[i * classes + ir[j]] = 1;
      }
  } else {
    const mxLogical* flags = static_cast<const mxLogical*>(mxGetData(targets));
    for (std::size_t j = 0; j < classes * n; ++j) mask[j] = flags[j] ? 1 : 0;
  }
  return mask;
}

// The positive classes of every example as an m-by-n 0/1 mask, from 1-based
// labels or a label matrix, checked for the loss.
std::vector<unsigned char> positive_mask(const loss_options& o, const mxArray* targets,
                                         const std::size_t classes, const std::size_t n) {
  using namespace sdca;
  if (!loss_is_multilabel(o.id)) return label_mask(targets, classes, n);
  std::vector<unsigned char> mask = label_matrix_mask(targets, classes, n);
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t p = 0;
    for (std::size_t j = 0; j < classes; ++j) p += mask[i * classes + j];
    if (p == 0) throw std::invalid_argument("Every example needs a label");
    if (o.id == 5 && p == classes)
      throw std::invalid_argument("multilabel_hinge needs a negative class in every example");
  }
  return mask;
}

// The label losses of examples [begin, end) of the classes-by-n scores; no mx
// calls, so workers may run it.
template <typename Data>
struct label_rows {
  sdca::loss_kind kind;
  std::size_t classes, k;
  double gamma, alpha;
  bool single;
  sdca::projection_precision mode;
  const Data* input;
  const unsigned char* mask;
  double* values;
  Data* gradients;

  void operator()(std::size_t begin, std::size_t end) const {
    using namespace sdca;
    std::vector<std::size_t> order(classes);
    std::vector<Data> z(classes), g(classes);
    for (std::size_t i = begin; i < end; ++i) {
      const unsigned char* positive = mask + i * classes;
      std::size_t p = 0, q = classes;
      for (std::size_t j = 0; j < classes; ++j)
        if (positive[j]) order[p++] = j;
      for (std::size_t j = classes; j-- > 0;)
        if (!positive[j]) order[--q] = j;
      for (std::size_t j = 0; j < classes; ++j) {
        z[j] = input[i * classes + order[j]];
        if (!std::isfinite(z[j])) throw std::invalid_argument("Scores must be finite");
      }
      values[i] = single ? detail::permuted_loss_and_gradient<Data, float>(
                               kind, classes, p, z.data(), g.data(), k, gamma, alpha, mode)
                         : detail::permuted_loss_and_gradient<Data, double>(
                               kind, classes, p, z.data(), g.data(), k, gamma, alpha, mode);
      if (gradients)
        for (std::size_t j = 0; j < classes; ++j) gradients[i * classes + order[j]] = g[j];
    }
  }
};

// The distillation losses of examples [begin, end) of the student and teacher
// logits (classes-by-n).
template <typename Data>
struct distillation_rows {
  bool proximal, single;
  std::size_t classes;
  double temperature, alpha;
  const Data *student, *teacher;
  double* values;
  Data* gradients;

  void operator()(std::size_t begin, std::size_t end) const {
    using namespace sdca;
    for (std::size_t i = begin; i < end; ++i) {
      const Data *s = student + i * classes, *t = teacher + i * classes;
      Data* g = gradients ? gradients + i * classes : nullptr;
      if (proximal)
        values[i] = proximal_distillation_loss<Data, double>(classes, s, t, temperature, alpha, g);
      else if (single)
        values[i] = distillation_loss<Data, float>(classes, s, t, temperature, g);
      else
        values[i] = distillation_loss<Data, double>(classes, s, t, temperature, g);
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
  const loss_options o = read_options(opts, classes, std::is_same<Data, float>::value);
  const Data* z = static_cast<const Data*>(mxGetData(scores));
  std::vector<unsigned char> mask;
  const Data* teacher = nullptr;
  if (loss_takes_temperature(o.id)) {
    if (mxIsSparse(targets) || mxIsComplex(targets) ||
        mxGetClassID(targets) != mex_class<Data>::id() || mxGetM(targets) != classes ||
        mxGetN(targets) != n)
      throw std::invalid_argument(
          "targets must be teacher logits of the size and class of the scores");
    teacher = static_cast<const Data*>(mxGetData(targets));
    for (std::size_t j = 0; j < classes * n; ++j)
      if (!std::isfinite(z[j]) || !std::isfinite(teacher[j]))
        throw std::invalid_argument("Logits must be finite");
  } else {
    mask = positive_mask(o, targets, classes, n);
  }
  plhs[0] = mxCreateDoubleMatrix(1, n, mxREAL);
  Data* gradients = nullptr;
  if (nlhs > 1) {
    plhs[1] = mxCreateNumericMatrix(classes, n, mex_class<Data>::id(), mxREAL);
    gradients = static_cast<Data*>(mxGetData(plhs[1]));
  }
  // Examples run on their threads, each independently.
  if (teacher) {
    const distillation_rows<Data> rows = {o.id == 9,     o.single,         classes,
                                          o.temperature, o.alpha,          z,
                                          teacher,       mxGetPr(plhs[0]), gradients};
    detail::parallel_rows(n, o.id == 9 ? 8 * classes : 2 * classes, o.num_threads, rows);
    return;
  }
  const label_rows<Data> rows = {loss_kind(o.id), classes,          o.k,      o.gamma,
                                 o.alpha,         o.single,         o.mode,   z,
                                 mask.data(),     mxGetPr(plhs[0]), gradients};
  detail::parallel_rows(n, classes, o.num_threads, rows);
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
