// The C ABI of include/sdca/c_api.h and the internal entry points of
// src/c_api_internal.h over the headers. SDCA_BASELINE (for benchmarks against
// earlier releases) builds the training entry points against earlier headers
// through src/c_api_baseline.h; the loss entry points then fail.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#ifdef __linux__
#include <sys/mman.h>
#endif
#include "c_api_internal.h"
#include "sdca/c_api.h"
#ifdef SDCA_BASELINE
#include "c_api_baseline.h"
#else
#include "sdca/loss.h"
#include "sdca/solver.h"
#include "sdca/utility/scratch.h"
#include "tensor_losses.h"
#endif
#include "sdca/utility/fpenv.h"
#include "sdca/utility/parallel.h"

#ifndef SDCA_VERSION
#define SDCA_VERSION "unknown"
#endif

namespace {
thread_local std::string last_error;

// Every computing entry point: IEEE arithmetic (no flush-to-zero, round to
// nearest, no traps) inside, the caller's modes restored on return.
template <typename F>
int protect(F f) {
  sdca::detail::ieee_environment ieee;
  try {
    f();
    last_error.clear();
    return SDCA_OK;
  } catch (const std::invalid_argument& e) {
    last_error = e.what();
    return SDCA_INVALID_ARGUMENT;
  } catch (const std::exception& e) {
    last_error = e.what();
    return SDCA_FAILURE;
  } catch (...) {
    last_error = "Unknown C++ exception";
    return SDCA_FAILURE;
  }
}

void invalid(const std::string& message) { throw std::invalid_argument(message); }

// The extent a * b of a caller's buffer, which must not wrap around.
std::size_t extent(std::size_t a, std::size_t b) {
  if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
    invalid("Buffer extent overflows size_t");
  return a * b;
}

void check_buffers(std::initializer_list<const void*> buffers) {
  for (const void* buffer : buffers)
    if (!buffer) invalid("Required buffers must not be NULL");
}

std::size_t max_extent() {
  return static_cast<std::size_t>(std::numeric_limits<sdca::blas_int>::max());
}

void check_blas_sizes(std::initializer_list<std::size_t> sizes) {
  const std::size_t limit = max_extent();
  for (std::size_t size : sizes)
    if (size > limit) invalid("Dimensions exceed the native BLAS integer limit");
}

std::size_t storage_size(int dtype) {
  if (dtype == SDCA_DTYPE_FLOAT32) return 4;
  if (dtype == SDCA_DTYPE_FLOAT64) return 8;
  if (dtype == SDCA_INTERNAL_DTYPE_FLOAT16 || dtype == SDCA_INTERNAL_DTYPE_BFLOAT16) return 2;
  throw std::invalid_argument("Invalid dtype");
}

// The num_threads of the headers for that of the ABI (0: the default).
std::size_t threads_of(std::size_t num_threads) {
  return num_threads == 0 ? sdca::detail::default_threads : num_threads;
}

// s = a + b and its rounding error (Knuth's TwoSum).
double two_sum(double a, double b, double& error) {
  const double s = a + b, bb = s - a;
  error = (a - (s - bb)) + (b - bb);
  return s;
}

template <typename D>
bool all_finite(const D* x, std::size_t size) {
  return std::all_of(x, x + size, [](D v) { return std::isfinite(v); });
}

// Whether the rows of x (n, k) are finite, on the threads of a call.
template <typename D>
void check_finite_rows(const D* x, std::size_t n, std::size_t k, std::size_t threads,
                       const char* message) {
  std::atomic<bool> finite(true);
  sdca::detail::parallel_rows(n, k, threads, [&](std::size_t begin, std::size_t end) {
    if (!all_finite(x + begin * k, (end - begin) * k)) finite = false;
  });
  if (!finite) invalid(message);
}

template <typename D>
void copy_rows(const D* from, std::size_t n, std::size_t k, std::size_t threads, D* to) {
  sdca::detail::parallel_rows(n, k, threads, [=](std::size_t begin, std::size_t end) {
    std::copy(from + begin * k, from + end * k, to + begin * k);
  });
}

// A caller's struct: its size between that of the first header of this ABI
// version (every struct's present size) and this one's; the fields beyond the
// caller's size take their defaults.
template <typename T>
T read_struct(const T* value, const char* name, T defaults) {
  if (!value) invalid(std::string("struct sdca_") + name + " must not be NULL");
  const std::size_t first = sizeof(T);
  if (value->size < first || value->size > sizeof(T))
    invalid(std::string("struct sdca_") + name + " has a size this library does not know");
  std::memcpy(static_cast<void*>(&defaults), value, value->size);
  defaults.size = sizeof(T);
  return defaults;
}

const sdca_loss default_loss = SDCA_LOSS_INIT;
const sdca_execution default_execution = SDCA_EXECUTION_INIT;
const sdca_targets default_targets = SDCA_TARGETS_INIT;
const sdca_input default_input = SDCA_INPUT_INIT;
const sdca_fit_options default_fit_options = SDCA_FIT_OPTIONS_INIT;
const sdca_fit_result default_fit_result = SDCA_FIT_RESULT_INIT;

// ---------------------------------------------------------------------------
// Losses, targets and settings.

const char* const loss_names[] = {
    "cross_entropy",   "topk_cross_entropy",   "multiclass_hinge",         "topk_hinge_alpha",
    "topk_hinge_beta", "multilabel_hinge",     "multilabel_cross_entropy", "proximal_cross_entropy",
    "distillation",    "proximal_distillation"};

bool is_topk(int kind) {
  return kind == SDCA_LOSS_TOPK_CROSS_ENTROPY || kind == SDCA_LOSS_TOPK_HINGE_ALPHA ||
         kind == SDCA_LOSS_TOPK_HINGE_BETA;
}
bool is_hinge(int kind) {
  return kind == SDCA_LOSS_MULTICLASS_HINGE || kind == SDCA_LOSS_TOPK_HINGE_ALPHA ||
         kind == SDCA_LOSS_TOPK_HINGE_BETA || kind == SDCA_LOSS_MULTILABEL_HINGE;
}
bool is_proximal(int kind) {
  return kind == SDCA_LOSS_PROXIMAL_CROSS_ENTROPY || kind == SDCA_LOSS_PROXIMAL_DISTILLATION;
}
bool is_distillation(int kind) {
  return kind == SDCA_LOSS_DISTILLATION || kind == SDCA_LOSS_PROXIMAL_DISTILLATION;
}
bool is_multilabel(int kind) {
  return kind == SDCA_LOSS_MULTILABEL_HINGE || kind == SDCA_LOSS_MULTILABEL_CROSS_ENTROPY;
}
// The losses that sdca_dual_update and sdca_model_fit train.
bool is_objective(int kind) {
  return kind >= SDCA_LOSS_CROSS_ENTROPY && kind <= SDCA_LOSS_MULTILABEL_CROSS_ENTROPY;
}

// A real parameter of the loss named name: finite and positive (nonnegative if
// zero_ok) if the loss takes it, else its default.
void check_real_parameter(const std::string& name, const char* parameter, double value,
                          double fallback, bool takes, bool zero_ok) {
  if (takes) {
    if (!(zero_ok ? value >= 0 : value > 0) || !std::isfinite(value))
      invalid(std::string(parameter) + " must be finite and " +
              (zero_ok ? "nonnegative" : "positive"));
  } else if (value != fallback) {
    invalid(name + " takes no parameter " + parameter);
  }
}

// The loss with its own parameters only, for num_classes classes.
sdca_loss check_loss(const sdca_loss* given, std::size_t num_classes) {
  const sdca_loss loss = read_struct(given, "loss", default_loss);
  if (loss.kind < SDCA_LOSS_CROSS_ENTROPY || loss.kind > SDCA_LOSS_PROXIMAL_DISTILLATION)
    invalid("Unknown loss kind");
  const std::string name = loss_names[loss.kind];
  if (is_topk(loss.kind)) {
    if (loss.k < 1 || loss.k >= num_classes) invalid("k must be an integer in [1, num_classes-1]");
  } else if (loss.k != default_loss.k) {
    invalid(name + " takes no parameter k");
  }
  check_real_parameter(name, "gamma", loss.gamma, default_loss.gamma, is_hinge(loss.kind), true);
  check_real_parameter(name, "alpha", loss.alpha, default_loss.alpha, is_proximal(loss.kind),
                       false);
  check_real_parameter(name, "temperature", loss.temperature, default_loss.temperature,
                       is_distillation(loss.kind), false);
  if (loss.projection_precision != SDCA_PROJECTION_FLOAT64 &&
      loss.projection_precision != SDCA_PROJECTION_EXTENDED)
    invalid("Invalid projection_precision");
  if (loss.kind != SDCA_LOSS_MULTILABEL_HINGE &&
      loss.projection_precision != default_loss.projection_precision)
    invalid(name + " takes no parameter projection_precision");
  return loss;
}

void check_dtype(int dtype) {
  if (dtype != SDCA_DTYPE_FLOAT32 && dtype != SDCA_DTYPE_FLOAT64) invalid("Invalid dtype");
}

// The execution settings for a storage dtype: float work, and the threads of the headers.
struct execution {
  bool float_work;
  std::size_t threads;
};

execution check_execution(const sdca_execution* given, int dtype) {
  const sdca_execution e = read_struct(given, "execution", default_execution);
  if (e.precision < SDCA_PRECISION_DEFAULT || e.precision > SDCA_PRECISION_FLOAT64)
    invalid("Invalid precision");
  if (e.precision == SDCA_PRECISION_FLOAT32 && dtype == SDCA_DTYPE_FLOAT64)
    invalid("float32 work needs float32 storage; convert the data to float32");
  const bool float_work = e.precision == SDCA_PRECISION_FLOAT32 ||
                          (e.precision == SDCA_PRECISION_DEFAULT && dtype == SDCA_DTYPE_FLOAT32);
  const execution out = {float_work, threads_of(e.num_threads)};
  return out;
}

// Validated targets of n examples.
struct target_rows {
  int kind;
  const int64_t* labels;
  const int64_t* offsets;
  const uint8_t* mask;
  const double* probabilities;
  const void* logits;
};

void check_labels(std::size_t n, std::size_t k, const int64_t* labels) {
  for (std::size_t i = 0; i < n; ++i)
    if (labels[i] < 0 || static_cast<uint64_t>(labels[i]) >= k)
      invalid("Labels must lie in [0, num_classes)");
}

// A label set of every class: the losses are defined, but the SDCA steps of
// the multilabel objectives need a class outside each set, and the losses
// follow them.
const char* const full_set = "Every label set needs a class outside it";

// Label sets: offsets (n + 1) from zero; each a nonempty strictly increasing
// list of labels in [0, k) without every class.
void check_label_sets(std::size_t n, std::size_t k, const int64_t* labels, const int64_t* offsets) {
  if (offsets[0] != 0) invalid("Label offsets must start at zero");
  for (std::size_t i = 0; i < n; ++i) {
    if (offsets[i + 1] <= offsets[i]) invalid("Every label set needs a label");
    const uint64_t size = static_cast<uint64_t>(offsets[i + 1] - offsets[i]);
    if (size > k) invalid("Label sets must be sorted without duplicates");
    if (size == k) invalid(full_set);
    check_labels(static_cast<std::size_t>(size), k, labels + offsets[i]);
    for (int64_t j = offsets[i] + 1; j < offsets[i + 1]; ++j)
      if (labels[j] <= labels[j - 1]) invalid("Label sets must be sorted without duplicates");
  }
}

void check_mask(std::size_t n, std::size_t k, const uint8_t* mask) {
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t count = 0;
    for (std::size_t j = 0; j < k; ++j) {
      if (mask[i * k + j] > 1) invalid("A label mask holds 0 or 1");
      count += mask[i * k + j];
    }
    if (count == 0) invalid("Every label set needs a label");
    if (count == k) invalid(full_set);
  }
}

// Probabilities: finite, nonnegative and on the simplex within 64 k eps per row.
void check_probabilities(std::size_t n, std::size_t k, const double* p) {
  const double tol = 64 * std::numeric_limits<double>::epsilon() * static_cast<double>(k);
  for (std::size_t i = 0; i < n; ++i) {
    double sum = 0, carry = 0;
    for (std::size_t j = 0; j < k; ++j) {
      const double q = p[i * k + j];
      if (!(q >= 0) || !std::isfinite(q))
        invalid("Target probabilities must be finite and nonnegative");
      double error;
      sum = two_sum(sum, q - carry, error);
      carry = -error;
    }
    if (!(std::abs(sum - carry - 1) <= tol)) invalid("Target probability rows must sum to one");
  }
}

enum class entry { loss, training };

// The targets of a loss (see c_api.h): only the fields of their kind.
target_rows check_targets(const sdca_targets* given, int loss, entry use, std::size_t n,
                          std::size_t k) {
  const sdca_targets t = read_struct(given, "targets", default_targets);
  const std::string name = loss_names[loss];
  bool allowed = false;
  switch (t.kind) {
    case SDCA_TARGET_LABELS:
      allowed = !is_multilabel(loss) && !is_distillation(loss);
      break;
    case SDCA_TARGET_LABEL_SETS:
    case SDCA_TARGET_MASK:
      allowed = is_multilabel(loss);
      break;
    case SDCA_TARGET_PROBABILITIES:
      allowed = loss == SDCA_LOSS_CROSS_ENTROPY && use == entry::training;
      break;
    case SDCA_TARGET_LOGITS:
      allowed = is_distillation(loss);
      break;
    default:
      invalid("Unknown target kind");
  }
  if (!allowed) invalid(name + " does not take targets of this kind");
  const bool labels = t.kind == SDCA_TARGET_LABELS || t.kind == SDCA_TARGET_LABEL_SETS;
  if ((t.labels != nullptr) != labels ||
      (t.offsets != nullptr) != (t.kind == SDCA_TARGET_LABEL_SETS) ||
      (t.mask != nullptr) != (t.kind == SDCA_TARGET_MASK) ||
      (t.probabilities != nullptr) != (t.kind == SDCA_TARGET_PROBABILITIES) ||
      (t.logits != nullptr) != (t.kind == SDCA_TARGET_LOGITS))
    invalid("Targets need the fields of their kind, and NULL in the others");
  switch (t.kind) {
    case SDCA_TARGET_LABELS:
      extent(n, sizeof(int64_t));
      check_labels(n, k, t.labels);
      break;
    case SDCA_TARGET_LABEL_SETS:
      extent(n + 1, sizeof(int64_t));
      check_label_sets(n, k, t.labels, t.offsets);
      break;
    case SDCA_TARGET_MASK:
      extent(n, k);
      check_mask(n, k, t.mask);
      break;
    case SDCA_TARGET_PROBABILITIES:
      extent(extent(n, k), sizeof(double));
      check_probabilities(n, k, t.probabilities);
      break;
    default:
      break;  // logits: checked with the scores
  }
  const target_rows rows = {t.kind, t.labels, t.offsets, t.mask, t.probabilities, t.logits};
  return rows;
}

// Label sets of a mask: the labels and the offsets (n + 1).
void sets_of_mask(std::size_t n, std::size_t k, const uint8_t* mask, std::vector<int64_t>& labels,
                  std::vector<int64_t>& offsets) {
  offsets.assign(1, 0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < k; ++j)
      if (mask[i * k + j]) labels.push_back(static_cast<int64_t>(j));
    offsets.push_back(static_cast<int64_t>(labels.size()));
  }
}

// Label-set targets as labels and offsets (a mask converted into sets).
struct label_sets {
  std::vector<int64_t> labels, offsets;
  target_rows rows;

  label_sets(const target_rows& t, std::size_t n, std::size_t k) : rows(t) {
    if (t.kind != SDCA_TARGET_MASK) return;
    sets_of_mask(n, k, t.mask, labels, offsets);
    rows.kind = SDCA_TARGET_LABEL_SETS;
    rows.labels = labels.data();
    rows.offsets = offsets.data();
    rows.mask = nullptr;
  }
};

// ---------------------------------------------------------------------------
// Dual blocks: feasibility of a caller's duals (to the tolerances of
// docs/API.md), the label entry that makes single-label rows sum to zero, and
// the primal of the duals.

// The positive classes of row i of the targets (labels, label sets or a mask).
std::size_t positives_of(const target_rows& t, std::size_t i, std::size_t k, char* mark) {
  std::fill(mark, mark + k, char(0));
  if (t.kind == SDCA_TARGET_LABELS) {
    mark[static_cast<std::size_t>(t.labels[i])] = 1;
  } else if (t.kind == SDCA_TARGET_LABEL_SETS) {
    for (int64_t j = t.offsets[i]; j < t.offsets[i + 1]; ++j)
      mark[static_cast<std::size_t>(t.labels[j])] = 1;
  } else {
    for (std::size_t j = 0; j < k; ++j) mark[j] = t.mask[i * k + j] ? 1 : 0;
  }
  return static_cast<std::size_t>(std::count(mark, mark + k, char(1)));
}

// The distillation constraints of check_feasible on a dual row v with the
// teacher q: c (q - 1) <= v <= c q to slack.
void check_distillation_row(const std::vector<double>& v, const double* q, double c, double slack,
                            const std::string& what) {
  for (std::size_t j = 0; j < v.size(); ++j)
    if (v[j] > c * q[j] + slack || v[j] < c * (q[j] - 1) - slack)
      invalid(what + " is outside the feasible set of the distillation objective");
}

// The hinge constraints of check_feasible on a dual row v of storage type D with
// the positive classes mark of mass (their sum): the positive entries are
// nonnegative, the mass at most c, and the top-k cap or cone holds.
template <typename D>
void check_hinge_row(const std::vector<double>& v, const char* mark, double mass,
                     const sdca_loss& loss, double c, double slack, const std::string& what) {
  const std::size_t k = v.size();
  for (std::size_t j = 0; j < k; ++j)
    if (mark[j] && v[j] < -slack) invalid(what + " is outside the feasible set");
  if (mass > c + slack) invalid(what + " is outside the feasible set");
  const std::size_t topk = is_topk(loss.kind) ? loss.k : 1;
  if (loss.kind == SDCA_LOSS_TOPK_HINGE_BETA) {
    for (std::size_t j = 0; j < k; ++j)
      if (-v[j] > c / double(topk) + slack) invalid(what + " violates the top-k cap");
  } else if (loss.kind != SDCA_LOSS_MULTILABEL_HINGE) {
    const double cone = double(std::max<std::size_t>(1, k / topk)) * 64 *
                        double(std::numeric_limits<D>::epsilon()) * c;
    for (std::size_t j = 0; j < k; ++j)
      if (-v[j] > mass / double(topk) + cone) invalid(what + " violates the top-k cone");
  }
}

// The constraints of check_feasible on a dual row v of storage type D with the
// count positive classes of mark: the other entries are nonpositive, and the
// cross-entropy box or the hinge constraints hold to slack.
template <typename D>
void check_label_row(const std::vector<double>& v, const char* mark, std::size_t count,
                     const sdca_loss& loss, double c, double slack, const std::string& what) {
  const std::size_t k = v.size();
  double mass = 0;
  for (std::size_t j = 0; j < k; ++j) {
    if (!mark[j] && v[j] > 0)
      invalid(what + ": the entries of the other classes must be nonpositive");
    if (mark[j]) mass += v[j];
  }
  if (loss.kind == SDCA_LOSS_CROSS_ENTROPY || loss.kind == SDCA_LOSS_MULTILABEL_CROSS_ENTROPY) {
    for (std::size_t j = 0; j < k; ++j)
      if (mark[j] && v[j] > c / double(count) + slack)
        invalid(what + " is outside the feasible set of the cross-entropy objective");
    return;
  }
  check_hinge_row<D>(v, mark, mass, loss, c, slack, what);
}

// Whether the dual blocks a (n, k) of storage type D are feasible for the
// objective with weight c: rows sum to zero relative to their largest entry,
// and each objective's sign, box and cone constraints hold to slack.
template <typename D>
void check_feasible(const D* a, std::size_t n, std::size_t k, const sdca_loss& loss, double c,
                    const target_rows& t, const char* name) {
  const double eps = 32 * double(std::numeric_limits<D>::epsilon()), slack = eps * c;
  const std::string what(name);
  std::vector<double> v(k);
  std::vector<char> mark(k);
  for (std::size_t i = 0; i < n; ++i) {
    double largest = 0;
    for (std::size_t j = 0; j < k; ++j) {
      v[j] = double(a[i * k + j]);
      largest = std::max(largest, std::abs(v[j]));
    }
    const double scale = largest > 0 ? largest : 1;
    double sum = 0, total = 0;
    for (double x : v) {
      sum += x / scale;
      total += std::abs(x / scale);
    }
    if (std::abs(sum) > eps * total) invalid(what + " rows must sum to zero to storage precision");
    if (t.kind == SDCA_TARGET_PROBABILITIES) {
      check_distillation_row(v, t.probabilities + i * k, c, slack, what);
      continue;
    }
    const std::size_t count = positives_of(t, i, k, mark.data());
    check_label_row<D>(v, mark.data(), count, loss, c, slack, what);
  }
}

// The label entry of each single-label row: minus the sum of the others in
// double, so that the row sums to zero as stored.
template <typename D>
void rebuild_label_entries(D* a, std::size_t n, std::size_t k, const int64_t* labels) {
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t y = static_cast<std::size_t>(labels[i]);
    double others = 0;
    for (std::size_t j = 0; j < k; ++j)
      if (j != y) others += double(a[i * k + j]);
    a[i * k + y] = static_cast<D>(-others);
  }
}

// The weights (k, d), row c = center_c + sum_i a_ic x_i, accumulated in double and rounded
// once.
template <typename D>
void primal_of_dual(const D* x, const D* a, const D* center, std::size_t n, std::size_t d,
                    std::size_t k, std::size_t threads, D* w) {
  std::vector<double> acc(extent(d, k), 0.0);
  double* sum = acc.data();
  sdca::detail::parallel_rows(d, extent(n, k), threads, [=](std::size_t begin, std::size_t end) {
    for (std::size_t i = 0; i < n; ++i) {
      const D* xi = x + i * d;
      for (std::size_t c = 0; c < k; ++c) {
        const double aic = double(a[i * k + c]);
        if (aic == 0) continue;
        double* column = sum + c * d;
        for (std::size_t j = begin; j < end; ++j) column[j] += aic * double(xi[j]);
      }
    }
  });
  for (std::size_t j = 0; j < d * k; ++j)
    w[j] = static_cast<D>(center ? double(center[j]) + acc[j] : acc[j]);
}

// ---------------------------------------------------------------------------
// Training: the objectives over the solver (sdca_dual_update, sdca_model_fit).

const std::size_t history_fields = SDCA_HISTORY_FIELDS;

template <typename R>
double metric(const sdca::eval_train<R, sdca::multiclass_output>& e) {
  return e.accuracy.empty() ? 0 : static_cast<double>(e.accuracy[0]);
}
#ifndef SDCA_BASELINE
template <typename R>
double metric(const sdca::eval_train<R, sdca::distribution_output>& e) {
  return e.accuracy.empty() ? 0 : static_cast<double>(e.accuracy[0]);
}
#endif
template <typename R>
double metric(const sdca::eval_train<R, sdca::multilabel_output>& e) {
  return static_cast<double>(e.rank_loss);
}

int status_of(sdca::solver_status status) {
  switch (status) {
    case sdca::solver_status::solved:
      return SDCA_STATUS_SOLVED;
    case sdca::solver_status::no_progress:
      return SDCA_STATUS_NO_PROGRESS;
    case sdca::solver_status::max_epoch:
      return SDCA_STATUS_MAX_EPOCH;
    case sdca::solver_status::failed:
      return SDCA_STATUS_FAILED;
    default:
      throw std::logic_error("The solver stopped in an unexpected state");
  }
}

#ifndef SDCA_BASELINE
// The stopping criteria of the fit options.
void configure(sdca::stopping_criteria& criteria, const sdca_fit_options& o, const sdca_loss& loss,
               std::size_t threads) {
  criteria.max_epoch = o.max_epoch;
  criteria.eval_epoch = o.eval_epoch;
  criteria.tol = o.tol;
  criteria.eval_on_start = o.eval_on_start != 0;
  criteria.recompute_primal = o.recompute_primal != 0;
  criteria.projection = static_cast<sdca::projection_precision>(loss.projection_precision);
  criteria.num_threads = threads;
  criteria.shrinking = static_cast<sdca::shrinking_mode>(o.shrinking);
  criteria.seed = o.seed;
}

// One coordinate step of row i in place.
template <typename D, typename Objective, typename Output>
void update_block(std::size_t i, std::size_t k, D norm2, const Output& out, const Objective& obj,
                  D* block, D* scores, int projection) {
  sdca::update_dual_variables(i, k, norm2, out, obj, block, scores,
                              static_cast<sdca::projection_precision>(projection));
}

static_assert(SDCA_SHRINKING_AUTO == static_cast<int>(sdca::shrinking_mode::automatic) &&
                  SDCA_SHRINKING_OFF == static_cast<int>(sdca::shrinking_mode::off) &&
                  SDCA_SHRINKING_HEURISTIC == static_cast<int>(sdca::shrinking_mode::heuristic) &&
                  SDCA_SHRINKING_SAFE == static_cast<int>(sdca::shrinking_mode::safe) &&
                  SDCA_SHRINKING_BOTH == static_cast<int>(sdca::shrinking_mode::both),
              "The C shrinking ids follow sdca::shrinking_mode");
static_assert(SDCA_PROJECTION_FLOAT64 == static_cast<int>(sdca::projection_precision::float64) &&
                  SDCA_PROJECTION_EXTENDED ==
                      static_cast<int>(sdca::projection_precision::extended),
              "The C projection ids follow sdca::projection_precision");
static_assert(SDCA_LOSS_PROXIMAL_DISTILLATION ==
                      static_cast<int>(sdca::loss_kind::proximal_distillation) &&
                  SDCA_LOSS_MULTICLASS_HINGE == static_cast<int>(sdca::loss_kind::multiclass_hinge),
              "The C loss ids follow sdca::loss_kind");
#endif

// request.run(objective, output) for the objective of a loss and its targets
// (labels, label sets or probabilities).
template <typename D, typename R, typename Request>
void dispatch(const sdca_loss& loss, double c, std::size_t n, std::size_t k, const target_rows& t,
              Request& request) {
  const R cc = static_cast<R>(c), g = static_cast<R>(loss.gamma);
  // gamma = 0 selects the nonsmooth hinge; a positive gamma must be normal.
  const bool smooth = loss.gamma > 0;
  if (!std::isnormal(cc) || (smooth && !std::isnormal(g)))
    invalid("C and gamma must be normal in the working precision");
  if (t.kind == SDCA_TARGET_PROBABILITIES) {
#ifndef SDCA_BASELINE
    std::vector<double> q(t.probabilities, t.probabilities + n * k);
    request.run(sdca::make_objective_l2_distillation<D, R>(cc),
                sdca::distribution_output(k, n, std::move(q)));
    return;
#else
    throw std::logic_error("Target probabilities are not built against these headers");
#endif
  }
  if (t.kind == SDCA_TARGET_LABELS) {
    std::vector<sdca::size_type> y(t.labels, t.labels + n);
    sdca::multiclass_output out(k, y);
    switch (loss.kind) {
      case SDCA_LOSS_CROSS_ENTROPY:
        request.run(sdca::make_objective_l2_cross_entropy<D, R>(cc), std::move(out));
        return;
      case SDCA_LOSS_TOPK_CROSS_ENTROPY:
        request.run(sdca::make_objective_l2_topk_cross_entropy<D, R>(cc, loss.k), std::move(out));
        return;
      case SDCA_LOSS_MULTICLASS_HINGE:
      case SDCA_LOSS_TOPK_HINGE_ALPHA: {
        const std::size_t topk = loss.kind == SDCA_LOSS_MULTICLASS_HINGE ? 1 : loss.k;
        if (smooth)
          request.run(sdca::make_objective_l2_topk_hinge_alpha_smooth<D, R>(cc, topk, g),
                      std::move(out));
        else
          request.run(sdca::make_objective_l2_topk_hinge_alpha<D, R>(cc, topk), std::move(out));
        return;
      }
      case SDCA_LOSS_TOPK_HINGE_BETA:
        if (smooth)
          request.run(sdca::make_objective_l2_topk_hinge_beta_smooth<D, R>(cc, loss.k, g),
                      std::move(out));
        else
          request.run(sdca::make_objective_l2_topk_hinge_beta<D, R>(cc, loss.k), std::move(out));
        return;
      default:
        break;
    }
  } else {
    std::vector<sdca::size_type> u(t.offsets, t.offsets + n + 1);
    std::vector<sdca::size_type> y(t.labels, t.labels + u.back());
    sdca::multilabel_output out(k, y, u);
    switch (loss.kind) {
      case SDCA_LOSS_MULTILABEL_HINGE:
        if (smooth)
          request.run(sdca::make_objective_l2_multilabel_hinge_smooth<D, R>(cc, g), std::move(out));
        else
          request.run(sdca::make_objective_l2_multilabel_hinge<D, R>(cc), std::move(out));
        return;
      case SDCA_LOSS_MULTILABEL_CROSS_ENTROPY:
        request.run(sdca::make_objective_l2_multilabel_cross_entropy<D, R>(cc), std::move(out));
        return;
      default:
        break;
    }
  }
  invalid("Unknown objective");
}

// One coordinate step per row: scores z_i and block previous_i (or zero) to out_i.
template <typename D>
struct update_request {
  std::size_t n, k;
  const D* scores;
  const D* previous;
  const double* norm2;
  int projection;
  std::size_t threads;
  D* out;

  template <typename Objective, typename Output>
  void run(Objective obj, Output output) {
    sdca::detail::parallel_rows(n, k, threads, [&](std::size_t begin, std::size_t end) {
      std::vector<D> z(k);
      for (std::size_t i = begin; i < end; ++i) {
        std::copy(scores + i * k, scores + (i + 1) * k, z.begin());
        D* block = out + i * k;
        if (previous)
          std::copy(previous + i * k, previous + (i + 1) * k, block);
        else
          std::fill(block, block + k, D(0));
        update_block(i, k, static_cast<D>(norm2[i]), output, obj, block, z.data(), projection);
      }
    });
  }
};

template <typename D, typename R>
void dual_update(const sdca_loss& loss, double c, const execution& e, std::size_t n, std::size_t k,
                 const D* scores, const target_rows& t, const D* previous, const double* norm2,
                 D* out) {
  check_finite_rows(scores, n, k, e.threads, "Scores must be finite");
  if (previous) {
    check_finite_rows(previous, n, k, e.threads, "previous must be finite");
    check_feasible(previous, n, k, loss, c, t, "previous");
  }
  for (std::size_t i = 0; i < n; ++i) {
    const double s = norm2[i];
    if (!(s > 0) || !std::isfinite(s)) invalid("norm2 must be finite and positive");
    if (!std::isnormal(static_cast<D>(c)) || !std::isnormal(static_cast<D>(s)) ||
        !std::isnormal(static_cast<D>(c * s)))
      invalid("C, norm2, and C*norm2 must be normal in the storage type");
  }
  const label_sets sets(t, n, k);
  std::unique_ptr<D[]> blocks(new D[n * k]);
  update_request<D> request = {
      n, k, scores, previous, norm2, loss.projection_precision, e.threads, blocks.get()};
  dispatch<D, R>(loss, c, n, k, sets.rows, request);
  copy_rows(blocks.get(), n, k, e.threads, out);
}

// The solve of a fit, on the library's own copies of W and the duals.
template <typename D>
struct fit_request {
  std::size_t n, d, k;
  const D* x;
  bool kernel;
  const D* center;
  D* weights;
  D* dual;
  sdca_fit_options options;
  sdca_loss loss;
  std::size_t threads;
  double* history;
  std::size_t history_rows;
  sdca_fit_result* result;

  template <typename Context>
  void solve(Context& ctx) {
    configure(ctx.criteria, options, loss, threads);
    sdca::make_solver(ctx).solve();
    std::size_t rows = 0;
    for (const auto& e : ctx.train.evals) {
      if (!history) break;
      if (rows == history_rows) throw std::logic_error("The history outgrew its bound");
      const double row[history_fields] = {double(e.epoch),
                                          double(e.primal),
                                          double(e.dual),
                                          double(e.primal_loss),
                                          double(e.dual_loss),
                                          double(e.primal_regularizer),
                                          double(e.relative_gap()),
                                          metric(e),
                                          e.cpu_time(),
                                          e.wall_time()};
      std::copy(row, row + history_fields, history + history_fields * rows++);
    }
    result->status = status_of(ctx.status);
    result->epochs = ctx.epoch;
    result->history_rows = rows;
  }

  template <typename Objective, typename Output>
  void run(Objective obj, Output out) {
    if (kernel) {
      auto ctx =
          sdca::make_context(sdca::make_input_kernel(n, x), std::move(out), std::move(obj), dual);
      solve(ctx);
    } else {
      auto ctx = sdca::make_context(sdca::make_input_feature(n, d, x), std::move(out),
                                    std::move(obj), weights, dual, center);
      solve(ctx);
    }
  }
};

template <typename D, typename R>
void model_fit(const sdca_loss& loss, const sdca_fit_options& o, const execution& e,
               const sdca_input& in, std::size_t k, const target_rows& t, const D* center,
               const D* initial, D* weights, D* dual, double* history, std::size_t history_rows,
               sdca_fit_result* result) {
  const bool kernel = in.kind == SDCA_INPUT_KERNEL;
  const std::size_t n = in.n, d = in.d;
  const D* x = static_cast<const D*>(in.x);
  check_finite_rows(x, n, kernel ? n : d, e.threads, "Inputs must be finite");
  if (center && !all_finite(center, d * k)) invalid("The center must be finite");
  // The library's copies of the iterate: the caller's outputs are written at the end.
  std::vector<D> a(n * k, D(0)), w(kernel ? 0 : d * k, D(0));
  if (initial) {
    check_finite_rows(initial, n, k, e.threads, "initial_dual must be finite");
    check_feasible(initial, n, k, loss, o.c, t, "initial_dual");
    std::copy(initial, initial + n * k, a.begin());
    if (t.kind == SDCA_TARGET_LABELS) rebuild_label_entries(a.data(), n, k, t.labels);
    if (!kernel) primal_of_dual(x, a.data(), center, n, d, k, e.threads, w.data());
  } else if (center && !kernel) {
    std::copy(center, center + d * k, w.begin());
  }
  const label_sets sets(t, n, k);
  sdca_fit_result outcome = *result;
  fit_request<D> request = {n,        d, k,    x,         kernel,  center,       w.data(),
                            a.data(), o, loss, e.threads, history, history_rows, &outcome};
  dispatch<D, R>(loss, o.c, n, k, sets.rows, request);
  std::copy(a.begin(), a.end(), dual);
  if (!kernel) std::copy(w.begin(), w.end(), weights);
  *result = outcome;
}

sdca_fit_options check_fit_options(const sdca_fit_options* given) {
  const sdca_fit_options o = read_struct(given, "fit_options", default_fit_options);
  if (!(o.c > 0) || !std::isfinite(o.c)) invalid("C must be finite and positive");
  if (!(o.tol >= 0) || !std::isfinite(o.tol)) invalid("tol must be finite and nonnegative");
  if ((o.eval_on_start != 0 && o.eval_on_start != 1) ||
      (o.recompute_primal != 0 && o.recompute_primal != 1))
    invalid("eval_on_start and recompute_primal must be 0 or 1");
  if (o.shrinking < SDCA_SHRINKING_AUTO || o.shrinking > SDCA_SHRINKING_BOTH)
    invalid("Invalid shrinking");
  return o;
}

// The evaluations of a fit: at most max_epoch / eval_epoch + 2, the start and
// the end included (2 for eval_epoch 0).
// The evaluations of a fit: every eval_epoch epochs, the start and the end (eval_epoch 0:
// the start and the end only). Saturates instead of wrapping for huge max_epoch.
std::size_t history_bound(const sdca_fit_options& o) {
  const std::size_t scheduled = o.eval_epoch == 0 ? 0 : o.max_epoch / o.eval_epoch;
  const std::size_t max = std::numeric_limits<std::size_t>::max();
  return scheduled > max - 2 ? max : scheduled + 2;
}

#ifndef SDCA_BASELINE
// ---------------------------------------------------------------------------
// Losses: the row kernels of src/tensor_losses.h.

namespace tensor = sdca::tensor;

// F<format>(arguments) for the storage dtype.
#define STORAGE(F, ...)                             \
  switch (dtype) {                                  \
    case SDCA_DTYPE_FLOAT32:                        \
      F<tensor::float32>(__VA_ARGS__);              \
      break;                                        \
    case SDCA_DTYPE_FLOAT64:                        \
      F<tensor::float64>(__VA_ARGS__);              \
      break;                                        \
    case SDCA_INTERNAL_DTYPE_FLOAT16:               \
      F<tensor::float16>(__VA_ARGS__);              \
      break;                                        \
    case SDCA_INTERNAL_DTYPE_BFLOAT16:              \
      F<tensor::bfloat16>(__VA_ARGS__);             \
      break;                                        \
    default:                                        \
      throw std::invalid_argument("Invalid dtype"); \
  }

// Asks Linux for transparent huge pages (THP in madvise mode) over a large
// output the parallel row loops touch first: 2 MiB faults, not 4 KiB ones.
void advise_huge_pages(void* data, std::size_t bytes) {
#ifdef __linux__
  const std::uintptr_t huge = std::uintptr_t(1) << 21;
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(data), end = begin + bytes;
  const std::uintptr_t first = (begin + huge - 1) & ~(huge - 1), last = end & ~(huge - 1);
  if (bytes >= (std::size_t(32) << 20) && last > first)
    madvise(reinterpret_cast<void*>(first), last - first, MADV_HUGEPAGE);
#else
  (void)data;
  (void)bytes;
#endif
}

// The label losses (kinds 0 to 7) of rows: labels or a 0/1 mask.
struct loss_rows_call {
  sdca::loss_kind kind;
  std::size_t n, k, topk;
  double gamma, alpha;
  sdca::projection_precision projection;
  std::size_t threads;
  const void* z;
  const int64_t* labels;
  const uint8_t* positive;
  double* values;
  void* gradients;
};

template <typename Format, typename Data, typename Result>
void tensor_loss_rows(const loss_rows_call& c) {
  const tensor::loss_gradient_rows<Format, Data, Result> rows = {
      c.kind,
      c.k,
      c.topk,
      c.gamma,
      c.alpha,
      c.projection,
      static_cast<const typename Format::type*>(c.z),
      c.labels,
      c.positive,
      c.values,
      static_cast<Data*>(c.gradients)};
  sdca::detail::parallel_rows(c.n, c.k, c.threads, rows);
}

// Float rows (16-bit and float32 storage) in float or double arithmetic;
// float64 rows in double.
template <typename Format>
void loss_rows_of(const loss_rows_call& c, bool fp32) {
  if (fp32)
    tensor_loss_rows<Format, float, float>(c);
  else
    tensor_loss_rows<Format, float, double>(c);
}
template <>
void loss_rows_of<tensor::float64>(const loss_rows_call& c, bool) {
  tensor_loss_rows<tensor::float64, double, double>(c);
}

struct distillation_call {
  std::size_t n, k;
  const void *student, *teacher;
  double temperature;
  bool fp32;
  std::size_t threads;

  template <typename Format>
  tensor::distillation_logits<Format> logits() const {
    const tensor::distillation_logits<Format> x = {
        k, static_cast<const typename Format::type*>(student),
        static_cast<const typename Format::type*>(teacher), temperature};
    return x;
  }
};

// Whether float work holds a temperature.
bool float_temperature(double temperature) {
  const double tiny = double(std::numeric_limits<float>::min());
  return tiny <= temperature && temperature <= 1 / tiny;
}

template <typename Format>
void distillation_forward(const distillation_call& c, double* values, double* stats) {
  const tensor::distillation_forward_rows<Format> rows = {c.logits<Format>(), c.fp32, values,
                                                          stats};
  sdca::detail::parallel_rows(c.n, 2 * c.k, c.threads, rows);
}

template <typename Format>
void distillation_backward(const distillation_call& c, const double* stats, const double* scale,
                           std::size_t stride, void* gradients) {
  const tensor::distillation_backward_rows<Format> rows = {
      c.logits<Format>(),
      c.fp32,
      stats,
      scale,
      stride,
      static_cast<typename Format::type*>(gradients)};
  sdca::detail::parallel_rows(c.n, 2 * c.k, c.threads, rows);
}

// Float gradients for float and 16-bit storage, double ones for float64.
template <typename Format>
void proximal_distillation(const distillation_call& c, double alpha, double* values,
                           void* gradients) {
  typedef
      typename std::conditional<std::is_same<Format, tensor::float64>::value, double, float>::type
          Data;
  const tensor::proximal_distillation_rows<Format, Data> rows = {c.logits<Format>(), alpha, values,
                                                                 static_cast<Data*>(gradients)};
  // Halley steps: several passes per row.
  sdca::detail::parallel_rows(c.n, 8 * c.k, c.threads, rows);
}

// sdca_loss_evaluate on validated arguments; every row is checked before any output.
template <typename D>
void loss_evaluate(const sdca_loss& loss, const execution& e, int dtype, std::size_t n,
                   std::size_t k, const D* scores, const target_rows& t, double* values,
                   D* gradients) {
  check_finite_rows(scores, n, k, e.threads, "Scores must be finite");
  if (is_distillation(loss.kind)) {
    const D* teacher = static_cast<const D*>(t.logits);
    check_finite_rows(teacher, n, k, e.threads, "Teacher logits must be finite");
    const distillation_call call = {n,
                                    k,
                                    scores,
                                    teacher,
                                    loss.temperature,
                                    e.float_work && float_temperature(loss.temperature),
                                    e.threads};
    if (loss.kind == SDCA_LOSS_PROXIMAL_DISTILLATION) {
      std::unique_ptr<D[]> scratch(gradients ? nullptr : new D[n * k]);
      void* out = gradients ? static_cast<void*>(gradients) : static_cast<void*>(scratch.get());
      STORAGE(proximal_distillation, call, loss.alpha, values, out)
      return;
    }
    std::unique_ptr<double[]> stats(gradients ? new double[n * tensor::distillation_stats]
                                              : nullptr);
    STORAGE(distillation_forward, call, values, stats.get())
    if (gradients) {
      const double one = 1;
      STORAGE(distillation_backward, call, stats.get(), &one, 0, gradients)
    }
    return;
  }
  std::vector<uint8_t> mask;
  const uint8_t* positive = t.mask;
  if (t.kind == SDCA_TARGET_LABEL_SETS) {
    mask.assign(n * k, 0);
    for (std::size_t i = 0; i < n; ++i)
      for (int64_t j = t.offsets[i]; j < t.offsets[i + 1]; ++j)
        mask[i * k + static_cast<std::size_t>(t.labels[j])] = 1;
    positive = mask.data();
  }
  const bool labels = t.kind == SDCA_TARGET_LABELS;
  const loss_rows_call call = {sdca::loss_kind(loss.kind),
                               n,
                               k,
                               is_topk(loss.kind) ? loss.k : 1,
                               loss.gamma,
                               loss.alpha,
                               static_cast<sdca::projection_precision>(loss.projection_precision),
                               e.threads,
                               scores,
                               labels ? t.labels : nullptr,
                               labels ? nullptr : positive,
                               values,
                               gradients};
  STORAGE(loss_rows_of, call, e.float_work)
}
#endif
}  // namespace

extern "C" {
int sdca_abi_version() { return SDCA_ABI_VERSION; }
const char* sdca_version() { return SDCA_VERSION; }
const char* sdca_last_error() { return last_error.c_str(); }

size_t sdca_max_extent() { return max_extent(); }

size_t sdca_history_rows(const sdca_fit_options* options) {
  std::size_t rows = 0;
  protect([&] { rows = history_bound(check_fit_options(options)); });
  return rows;
}

int sdca_dual_update(const sdca_loss* loss, double c, const sdca_execution* execution_settings,
                     int dtype, size_t n, size_t k, const void* scores, const sdca_targets* targets,
                     const void* previous, const double* norm2, void* out) {
  return protect([&] {
    check_dtype(dtype);
    if (k < 2) invalid("num_classes must be >= 2");
    const sdca_loss l = check_loss(loss, k);
    if (!is_objective(l.kind)) invalid(std::string(loss_names[l.kind]) + " is not trainable");
    if (!(c > 0) || !std::isfinite(c)) invalid("C must be finite and positive");
    const execution exec = check_execution(execution_settings, dtype);
    extent(extent(n, k), sizeof(double));
    const target_rows t = check_targets(targets, l.kind, entry::training, n, k);
    if (n == 0) return;
    check_buffers({scores, norm2, out});
    if (dtype == SDCA_DTYPE_FLOAT64)
      dual_update<double, double>(l, c, exec, n, k, static_cast<const double*>(scores), t,
                                  static_cast<const double*>(previous), norm2,
                                  static_cast<double*>(out));
    else if (exec.float_work)
      dual_update<float, float>(l, c, exec, n, k, static_cast<const float*>(scores), t,
                                static_cast<const float*>(previous), norm2,
                                static_cast<float*>(out));
    else
      dual_update<float, double>(l, c, exec, n, k, static_cast<const float*>(scores), t,
                                 static_cast<const float*>(previous), norm2,
                                 static_cast<float*>(out));
  });
}

int sdca_model_fit(const sdca_loss* loss, const sdca_fit_options* options,
                   const sdca_execution* execution_settings, const sdca_input* input, size_t k,
                   const sdca_targets* targets, const void* center, const void* initial_dual,
                   void* weights, void* dual, double* history, size_t history_rows,
                   sdca_fit_result* result) {
  return protect([&] {
    const sdca_input in = read_struct(input, "input", default_input);
    if (in.kind != SDCA_INPUT_FEATURES && in.kind != SDCA_INPUT_KERNEL)
      invalid("Unknown input kind");
    const int dtype = in.dtype;
    check_dtype(dtype);
    const bool kernel = in.kind == SDCA_INPUT_KERNEL;
    if (in.n < 1 || in.d < 1 || k < 2) invalid("Invalid fit dimensions");
    if (kernel && in.d != in.n) invalid("A Gram matrix is square: d must equal n");
    const sdca_loss l = check_loss(loss, k);
    if (!is_objective(l.kind)) invalid(std::string(loss_names[l.kind]) + " is not trainable");
    const sdca_fit_options o = check_fit_options(options);
    if (!is_hinge(l.kind) && o.shrinking != SDCA_SHRINKING_AUTO)
      invalid(std::string(loss_names[l.kind]) +
              " takes no shrinking: its steps update every class of every example");
    const execution exec = check_execution(execution_settings, dtype);
    sdca_fit_result r = read_struct(result, "fit_result", default_fit_result);
    const std::size_t n = in.n, d = in.d, size = storage_size(dtype);
    check_blas_sizes({n, d, k, extent(n, k), extent(d, k)});
    extent(extent(n, d), size);
    extent(extent(n, k), size);
    extent(extent(d, k), size);
    extent(extent(history_rows, history_fields), sizeof(double));
    check_buffers({in.x, dual});
    if (kernel && (weights || center)) invalid("A Gram matrix takes no weights and no center");
    if (!kernel) check_buffers({weights});
    if (!history != (history_rows == 0)) invalid("Pass history with history_rows, or neither");
    if (history && history_rows < history_bound(o))
      invalid("The history needs sdca_history_rows(options) rows");
    const target_rows t = check_targets(targets, l.kind, entry::training, n, k);
    if (dtype == SDCA_DTYPE_FLOAT64)
      model_fit<double, double>(l, o, exec, in, k, t, static_cast<const double*>(center),
                                static_cast<const double*>(initial_dual),
                                static_cast<double*>(weights), static_cast<double*>(dual), history,
                                history_rows, &r);
    else if (exec.float_work)
      model_fit<float, float>(l, o, exec, in, k, t, static_cast<const float*>(center),
                              static_cast<const float*>(initial_dual), static_cast<float*>(weights),
                              static_cast<float*>(dual), history, history_rows, &r);
    else
      model_fit<float, double>(l, o, exec, in, k, t, static_cast<const float*>(center),
                               static_cast<const float*>(initial_dual),
                               static_cast<float*>(weights), static_cast<float*>(dual), history,
                               history_rows, &r);
    result->status = r.status;
    result->epochs = r.epochs;
    result->history_rows = r.history_rows;
  });
}

int sdca_model_scores(const sdca_input* input, size_t k, const void* coefficients, void* scores) {
  return protect([&] {
    const sdca_input in = read_struct(input, "input", default_input);
    if (in.kind != SDCA_INPUT_FEATURES && in.kind != SDCA_INPUT_KERNEL)
      invalid("Unknown input kind");
    check_dtype(in.dtype);
    if (in.d < 1 || k < 1) invalid("Invalid score dimensions");
    const std::size_t n = in.n, d = in.d, size = storage_size(in.dtype);
    // One product: each dimension, not the element counts, must fit the BLAS integer.
    check_blas_sizes({n, d, k});
    extent(extent(n, d), size);
    extent(extent(n, k), size);
    extent(extent(d, k), size);
    check_buffers({coefficients});
    if (n == 0) return;
    check_buffers({in.x, scores});
    const bool kernel = in.kind == SDCA_INPUT_KERNEL;
    if (in.dtype == SDCA_DTYPE_FLOAT64)
      sdca::batch_scores(kernel, n, d, k, static_cast<const double*>(in.x),
                         static_cast<const double*>(coefficients), static_cast<double*>(scores));
    else
      sdca::batch_scores(kernel, n, d, k, static_cast<const float*>(in.x),
                         static_cast<const float*>(coefficients), static_cast<float*>(scores));
  });
}

#ifndef SDCA_BASELINE
int sdca_loss_evaluate(const sdca_loss* loss, const sdca_execution* execution_settings, int dtype,
                       size_t n, size_t k, const void* scores, const sdca_targets* targets,
                       double* values, void* gradients) {
  return protect([&] {
    check_dtype(dtype);
    if (k < 2) invalid("num_classes must be >= 2");
    const sdca_loss l = check_loss(loss, k);
    const execution exec = check_execution(execution_settings, dtype);
    extent(extent(n, k), sizeof(double));
    const target_rows t = check_targets(targets, l.kind, entry::loss, n, k);
    if (n == 0) return;
    check_buffers({scores, values});
    if (gradients) advise_huge_pages(gradients, n * k * storage_size(dtype));
    if (dtype == SDCA_DTYPE_FLOAT64)
      loss_evaluate<double>(l, exec, dtype, n, k, static_cast<const double*>(scores), t, values,
                            static_cast<double*>(gradients));
    else
      loss_evaluate<float>(l, exec, dtype, n, k, static_cast<const float*>(scores), t, values,
                           static_cast<float*>(gradients));
  });
}
#else
int sdca_loss_evaluate(const sdca_loss*, const sdca_execution*, int, size_t, size_t, const void*,
                       const sdca_targets*, double*, void*) {
  return protect([] { throw std::logic_error("Losses are not built against these headers"); });
}
#endif
}

// ---------------------------------------------------------------------------
// The internal entry points (src/c_api_internal.h).

namespace {
#ifndef SDCA_BASELINE
// The internal options and the checks of every internal tensor call.
const sdca_internal_options& check_internal(const sdca_internal_options* options, int dtype,
                                            std::size_t n, std::size_t k) {
  if (!options) invalid("Internal options must not be NULL");
  if (options->size != sizeof(sdca_internal_options))
    invalid("struct sdca_internal_options is not this library's");
  if (k < 2) invalid("Invalid dimensions");
  storage_size(dtype);
  if (options->float_work && dtype == SDCA_DTYPE_FLOAT64)
    invalid("float64 storage needs double arithmetic");
  extent(extent(n, k), sizeof(double));
  extent(extent(n, tensor::entropy_stats + 1), sizeof(double));
  return *options;
}

struct cross_entropy_call {
  std::size_t n, k;
  const void* z;
  const int64_t* labels;
  int64_t ignore;
  const double* weights;
  bool fp32;
  std::size_t threads;
};

template <typename Format>
void cross_entropy_forward(const cross_entropy_call& c, double* values, double* stats,
                           double* totals) {
  // Into scratch: a failed row leaves the outputs as given.
  const std::size_t width = 1 + tensor::entropy_stats;
  std::unique_ptr<double[]> scratch(new double[c.n * width]);
  double *v = scratch.get(), *s = v + c.n;
  const tensor::cross_entropy_forward_rows<Format> rows = {
      c.k, static_cast<const typename Format::type*>(c.z), c.labels, c.ignore, c.weights, c.fp32, v,
      s};
  sdca::detail::parallel_rows(c.n, c.k, c.threads, rows);
  if (totals) {
    // In row order, compensated: the same for every thread count.
    double sum[2] = {0, 0}, carry[2] = {0, 0};
    for (std::size_t i = 0; i < c.n; ++i) {
      if (c.labels[i] == c.ignore) continue;
      const double term[2] = {v[i], c.weights ? c.weights[c.labels[i]] : 1.0};
      for (int t = 0; t < 2; ++t) {
        double error;
        sum[t] = sdca::detail::two_sum(sum[t], term[t] - carry[t], error);
        carry[t] = -error;
      }
    }
    totals[0] = sum[0] - carry[0];
    totals[1] = sum[1] - carry[1];
  }
  std::copy(v, v + c.n, values);
  if (stats) std::copy(s, s + c.n * tensor::entropy_stats, stats);
}

template <typename Format>
void cross_entropy_backward(const cross_entropy_call& c, const double* stats, const double* scale,
                            std::size_t stride, void* gradients) {
  // Every label first: a failed call writes no gradient.
  for (std::size_t i = 0; i < c.n; ++i)
    if (c.labels[i] != c.ignore) tensor::checked_label(c.labels[i], c.k);
  const tensor::cross_entropy_backward_rows<Format> rows = {
      c.k,       static_cast<const typename Format::type*>(c.z),
      c.labels,  c.ignore,
      c.weights, stats,
      scale,     stride,
      c.fp32,    static_cast<typename Format::type*>(gradients)};
  sdca::detail::parallel_rows(c.n, c.k, c.threads, rows);
}

void check_temperature(double temperature, double alpha) {
  if (!(std::isfinite(temperature) && temperature > 0))
    invalid("temperature must be finite and positive");
  if (!(std::isfinite(alpha) && alpha > 0)) invalid("alpha must be finite and positive");
}
#endif
}  // namespace

extern "C" {
#ifndef SDCA_BASELINE
int sdca_internal_loss_rows(const sdca_internal_options* options, int dtype, size_t n, size_t k,
                            const void* z, const int64_t* labels, const uint8_t* positive,
                            double* values, void* gradients) {
  return protect([&] {
    const sdca_internal_options& o = check_internal(options, dtype, n, k);
    if (o.loss < SDCA_LOSS_CROSS_ENTROPY || o.loss > SDCA_LOSS_PROXIMAL_CROSS_ENTROPY)
      invalid("Unknown loss");
    if (o.projection_precision != SDCA_PROJECTION_FLOAT64 &&
        o.projection_precision != SDCA_PROJECTION_EXTENDED)
      invalid("Invalid projection precision");
    if (!sdca::detail::valid_loss_arguments(k, 1, o.k, o.gamma, o.alpha))
      invalid("Invalid loss dimensions or parameters");
    if (!labels == !positive) invalid("Pass either labels or positive");
    if (n == 0) return;
    check_buffers({z, values});
    if (gradients) advise_huge_pages(gradients, n * k * (dtype == SDCA_DTYPE_FLOAT64 ? 8 : 4));
    const loss_rows_call call = {sdca::loss_kind(o.loss),
                                 n,
                                 k,
                                 o.k,
                                 o.gamma,
                                 o.alpha,
                                 static_cast<sdca::projection_precision>(o.projection_precision),
                                 threads_of(o.num_threads),
                                 z,
                                 labels,
                                 positive,
                                 values,
                                 gradients};
    STORAGE(loss_rows_of, call, o.float_work != 0)
  });
}

int sdca_internal_cross_entropy_forward(const sdca_internal_options* options, int dtype, size_t n,
                                        size_t k, const void* z, const int64_t* labels,
                                        int64_t ignore_index, const double* class_weights,
                                        double* values, double* stats, double* totals) {
  return protect([&] {
    const sdca_internal_options& o = check_internal(options, dtype, n, k);
    if (n == 0) {
      if (totals) totals[0] = totals[1] = 0;
      return;
    }
    check_buffers({z, labels, values});
    const cross_entropy_call call = {
        n, k, z, labels, ignore_index, class_weights, o.float_work != 0, threads_of(o.num_threads)};
    STORAGE(cross_entropy_forward, call, values, stats, totals)
  });
}

int sdca_internal_cross_entropy_backward(const sdca_internal_options* options, int dtype, size_t n,
                                         size_t k, const void* z, const int64_t* labels,
                                         int64_t ignore_index, const double* class_weights,
                                         const double* stats, const double* scale,
                                         size_t scale_stride, void* gradients) {
  return protect([&] {
    const sdca_internal_options& o = check_internal(options, dtype, n, k);
    if (scale_stride > 1) invalid("scale_stride must be 0 or 1");
    if (n == 0) return;
    check_buffers({z, labels, stats, scale, gradients});
    advise_huge_pages(gradients, n * k * storage_size(dtype));
    const cross_entropy_call call = {
        n, k, z, labels, ignore_index, class_weights, o.float_work != 0, threads_of(o.num_threads)};
    STORAGE(cross_entropy_backward, call, stats, scale, scale_stride, gradients)
  });
}

int sdca_internal_distillation_forward(const sdca_internal_options* options, int dtype, size_t n,
                                       size_t k, const void* student, const void* teacher,
                                       double temperature, double* values, double* stats) {
  return protect([&] {
    const sdca_internal_options& o = check_internal(options, dtype, n, k);
    extent(extent(n, tensor::distillation_stats), sizeof(double));
    check_temperature(temperature, 1);
    if (n == 0) return;
    check_buffers({student, teacher, values});
    const distillation_call call = {n,
                                    k,
                                    student,
                                    teacher,
                                    temperature,
                                    o.float_work != 0 && float_temperature(temperature),
                                    threads_of(o.num_threads)};
    STORAGE(distillation_forward, call, values, stats)
  });
}

int sdca_internal_distillation_backward(const sdca_internal_options* options, int dtype, size_t n,
                                        size_t k, const void* student, const void* teacher,
                                        double temperature, const double* stats,
                                        const double* scale, size_t scale_stride, void* gradients) {
  return protect([&] {
    const sdca_internal_options& o = check_internal(options, dtype, n, k);
    extent(extent(n, tensor::distillation_stats), sizeof(double));
    check_temperature(temperature, 1);
    if (scale_stride > 1) invalid("scale_stride must be 0 or 1");
    if (n == 0) return;
    check_buffers({student, teacher, stats, scale, gradients});
    advise_huge_pages(gradients, n * k * storage_size(dtype));
    const distillation_call call = {n,
                                    k,
                                    student,
                                    teacher,
                                    temperature,
                                    o.float_work != 0 && float_temperature(temperature),
                                    threads_of(o.num_threads)};
    STORAGE(distillation_backward, call, stats, scale, scale_stride, gradients)
  });
}

int sdca_internal_proximal_distillation_rows(const sdca_internal_options* options, int dtype,
                                             size_t n, size_t k, const void* student,
                                             const void* teacher, double temperature, double alpha,
                                             double* values, void* gradients) {
  return protect([&] {
    const sdca_internal_options& o = check_internal(options, dtype, n, k);
    check_temperature(temperature, alpha);
    if (n == 0) return;
    check_buffers({student, teacher, values, gradients});
    advise_huge_pages(gradients, n * k * (dtype == SDCA_DTYPE_FLOAT64 ? 8 : 4));
    const distillation_call call = {
        n, k, student, teacher, temperature, false, threads_of(o.num_threads)};
    STORAGE(proximal_distillation, call, alpha, values, gradients)
  });
}
#else
// The bindings resolve every entry point; the earlier headers have no losses.
#define SDCA_NO_LOSSES \
  return protect([] { throw std::logic_error("Losses are not built against these headers"); })
int sdca_internal_loss_rows(const sdca_internal_options*, int, size_t, size_t, const void*,
                            const int64_t*, const uint8_t*, double*, void*) {
  SDCA_NO_LOSSES;
}
int sdca_internal_cross_entropy_forward(const sdca_internal_options*, int, size_t, size_t,
                                        const void*, const int64_t*, int64_t, const double*,
                                        double*, double*, double*) {
  SDCA_NO_LOSSES;
}
int sdca_internal_cross_entropy_backward(const sdca_internal_options*, int, size_t, size_t,
                                         const void*, const int64_t*, int64_t, const double*,
                                         const double*, const double*, size_t, void*) {
  SDCA_NO_LOSSES;
}
int sdca_internal_distillation_forward(const sdca_internal_options*, int, size_t, size_t,
                                       const void*, const void*, double, double*, double*) {
  SDCA_NO_LOSSES;
}
int sdca_internal_distillation_backward(const sdca_internal_options*, int, size_t, size_t,
                                        const void*, const void*, double, const double*,
                                        const double*, size_t, void*) {
  SDCA_NO_LOSSES;
}
int sdca_internal_proximal_distillation_rows(const sdca_internal_options*, int, size_t, size_t,
                                             const void*, const void*, double, double, double*,
                                             void*) {
  SDCA_NO_LOSSES;
}
#endif

int sdca_internal_run_rows(size_t rows, size_t work_per_row, size_t num_threads,
                           sdca_internal_row_callback callback, void* context) {
  return protect([&] {
    if (!callback) invalid("row callback must not be null");
    if (rows == 0) return;
    const std::size_t requested = sdca::detail::requested_threads(threads_of(num_threads));
    const std::size_t team =
        std::min<std::size_t>(requested ? requested : sdca::detail::automatic_threads(), 4096);
    const std::size_t cpus = requested ? team : team / (sdca::detail::parallel_calls().load() + 1);
    const std::size_t threads = std::min(rows, cpus);
    const auto apply = [&](std::size_t begin, std::size_t end) { callback(context, begin, end); };
    // Fused kernels use the same small-call cutoff as the tensor adapter. Their
    // callbacks can batch adjacent rows into SIMD lanes, so probing single rows
    // overestimates their cost and gives unstable thread decisions.
    if (threads < 2 || static_cast<double>(rows) * work_per_row < 4096 ||
        sdca::detail::in_parallel_rows() || (requested == 0 && sdca::detail::in_openmp_parallel()))
      apply(0, rows);
    else
      sdca::detail::parallel_tasks(0, rows, threads, team, apply);
  });
}
}

#ifdef SDCA_PARALLEL_OPENMP
// From the library's load on: a child forked after it never enters the
// parent's OpenMP team.
__attribute__((unused)) static const bool watching_forks = sdca::detail::watch_forks();
#endif
