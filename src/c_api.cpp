// The C ABI of include/sdca/c_api.h: argument checks, then the batch driver
// (sdca/kernel.h) for losses and the solver (sdca/solver.h) for training.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sdca/utility/row_reduction.h"
#include "sdca/c_api.h"
#include "sdca/kernel.h"
#include "sdca/solver.h"
#include "sdca/solver/duals.h"
#include "sdca/utility/fpenv.h"
#include "sdca/utility/parallel.h"

#ifndef SDCA_VERSION
#define SDCA_VERSION "unknown"
#endif

namespace {
namespace solver = sdca::native;
namespace kernel = sdca::kernel;

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
  for (std::size_t size : sizes)
    if (size > max_extent()) invalid("Dimensions exceed the native BLAS integer limit");
}

void check_dtype(int dtype) {
  if (dtype != SDCA_DTYPE_FLOAT32 && dtype != SDCA_DTYPE_FLOAT64) invalid("Invalid dtype");
}

std::size_t storage_size(int dtype) { return dtype == SDCA_DTYPE_FLOAT32 ? 4 : 8; }

template <typename D>
bool all_finite(const D* x, std::size_t size) {
  return std::all_of(x, x + size, [](D v) { return std::isfinite(v); });
}

// A caller's struct: its size between that of the first header of this ABI
// version (first) and this one's; the fields beyond the caller's size take
// their defaults.
template <typename T>
T read_struct(const T* value, const char* name, T defaults, std::size_t first = sizeof(T)) {
  if (!value) invalid(std::string("struct sdca_") + name + " must not be NULL");
  if (value->size < first || value->size > sizeof(T))
    invalid(std::string("struct sdca_") + name + " has a size this library does not know");
  std::memcpy(static_cast<void*>(&defaults), value, value->size);
  defaults.size = sizeof(T);
  return defaults;
}

const sdca_loss default_loss = SDCA_LOSS_INIT(SDCA_LOSS_CROSS_ENTROPY);
const sdca_execution default_execution = SDCA_EXECUTION_INIT;
const sdca_targets default_targets = SDCA_TARGETS_INIT;
const sdca_input default_input = SDCA_INPUT_INIT;
const sdca_fit_options default_fit_options = SDCA_FIT_OPTIONS_INIT;
const sdca_fit_result default_fit_result = SDCA_FIT_RESULT_INIT;

// ---------------------------------------------------------------------------
// Losses, targets and settings.

const kernel::Traits& traits(int kind) { return kernel::traits(kind); }
const char* name_of(int kind) { return traits(kind).name; }

enum class entry { loss, training, rows };

const char* const ignored_probabilities = "ignore_index is not supported with class probabilities";

void check_class_weights(const double* weights, std::size_t num_classes) {
  extent(num_classes, sizeof(double));
  for (std::size_t j = 0; j < num_classes; ++j)
    if (!(weights[j] >= 0) || !std::isfinite(weights[j]))
      invalid("weight must be finite and nonnegative");
}

// The options of sdca_loss_evaluate (weight, label_smoothing, ignore_index; c_api.h): only
// those the loss takes, as PyTorch's and JAX's losses take them, with valid values.
void check_options(const sdca_loss& loss, std::size_t num_classes, entry use) {
  const bool weight = loss.weight != nullptr, smoothing = loss.label_smoothing != 0;
  const bool ignore = loss.ignore != 0 || loss.ignore_index != default_loss.ignore_index;
  if (use == entry::training && (weight || smoothing || ignore))
    invalid("weight, label_smoothing and ignore_index are options of sdca_loss_evaluate");
  const kernel::Traits& t = traits(loss.kind);
  const bool single_label = !t.multilabel && !t.distillation;
  const char* const own = !single_label && weight                             ? "weight"
                          : loss.kind != SDCA_LOSS_CROSS_ENTROPY && smoothing ? "label_smoothing"
                          : !single_label && ignore                           ? "ignore_index"
                                                                              : nullptr;
  if (own) invalid(std::string(t.name) + " takes no parameter " + own);
  if (!(loss.label_smoothing >= 0 && loss.label_smoothing <= 1))
    invalid("label_smoothing must be in [0, 1], got " +
            kernel::detail::real_text(loss.label_smoothing));
  if (loss.ignore != 0 && loss.ignore != 1) invalid("ignore must be 0 or 1");
  if (!weight || use == entry::rows) return;
  check_class_weights(loss.weight, num_classes);
}

// The loss with its own parameters and options only, for num_classes classes.
sdca_loss check_loss(const sdca_loss* given, std::size_t num_classes, entry use) {
  const sdca_loss loss = read_struct(given, "loss", default_loss);
  kernel::check_parameters(loss.kind, num_classes, loss.k, loss.gamma, loss.temperature);
  check_options(loss, num_classes, use);
  return loss;
}

// The execution settings for a storage dtype.
struct execution {
  int precision;
  std::size_t num_threads;
  bool check_rows;
  sdca_loss_report* report;

  // float work on float32 storage (the solver's <float, float>).
  bool float_work(int dtype) const {
    return dtype == SDCA_DTYPE_FLOAT32 && precision != SDCA_PRECISION_FLOAT64;
  }
};

execution check_execution(const sdca_execution* given, int dtype) {
  const sdca_execution e =
      read_struct(given, "execution", default_execution, offsetof(sdca_execution, report));
  if (e.precision < SDCA_PRECISION_DEFAULT || e.precision > SDCA_PRECISION_FLOAT64)
    invalid("Invalid precision");
  if (e.check_rows != 0 && e.check_rows != 1) invalid("check_rows must be 0 or 1");
  if (e.precision == SDCA_PRECISION_FLOAT32 && dtype == SDCA_DTYPE_FLOAT64)
    invalid("float32 work needs float32 storage; convert the data to float32");
  if (e.report && e.report->size != sizeof(sdca_loss_report))
    invalid("struct sdca_loss_report has a size this library does not know");
  const execution out = {e.precision, e.num_threads, e.check_rows != 0, e.report};
  return out;
}

// Validated targets of n examples. Training probabilities are the caller's rows renormalized
// (normalized); probabilities() gives the rows in use.
struct target_rows {
  int kind;
  const int64_t* labels;
  const int64_t* offsets;
  const uint8_t* mask;
  const double* given;
  const void* logits;
  std::vector<double> normalized;

  const double* probabilities() const { return normalized.empty() ? given : normalized.data(); }
};

const char* const full_set = "Every label set needs a class outside it";

// Label sets: offsets (n + 1) from zero, each set strictly increasing, every label in [0, m);
// for training also nonempty and without every class (sdca_loss_evaluate counts such rows
// as invalid).
void check_label_sets(std::size_t n, std::size_t m, const int64_t* labels, const int64_t* offsets,
                      entry use) {
  if (offsets[0] != 0) invalid("Label offsets must start at zero");
  for (std::size_t i = 0; i < n; ++i) {
    if (offsets[i + 1] < offsets[i]) invalid("Label offsets must not decrease");
    for (int64_t j = offsets[i] + 1; j < offsets[i + 1]; ++j)
      if (labels[j] <= labels[j - 1]) invalid("Label sets must be sorted without duplicates");
  }
  kernel::check_labels(labels, static_cast<std::size_t>(offsets[n]), m);
  if (use != entry::training) return;
  for (std::size_t i = 0; i < n; ++i) {
    const uint64_t size = static_cast<uint64_t>(offsets[i + 1] - offsets[i]);
    if (size == 0) invalid("Every label set needs a label");
    if (size == m) invalid(full_set);
  }
}

void check_mask(std::size_t n, std::size_t m, const uint8_t* mask, entry use) {
  for (std::size_t i = 0; i < n; ++i) {
    const uint8_t* row = mask + i * m;
    uint8_t top = 0;
    std::size_t count = 0;
    for (std::size_t j = 0; j < m; ++j) {
      top = row[j] > top ? row[j] : top;
      count += row[j];
    }
    if (top > 1) invalid("A label mask holds 0 or 1");
    if (use != entry::training) continue;
    if (count == 0) invalid("Every label set needs a label");
    if (count == m) invalid(full_set);
  }
}

// Training probabilities: finite and nonnegative, each row summing to one within the rounding
// of a row normalized in the storage dtype (the m - 1 roundings of its sum, the division and
// the rounding to the dtype: gamma_{m+1}), renormalized into normalized (n, m). A loss evaluation takes them as
// given, as PyTorch's cross_entropy.
void check_probabilities(std::size_t n, std::size_t m, const double* p, entry use, int dtype,
                         std::vector<double>& normalized) {
  if (use != entry::training) return;
  const double tol = dtype == SDCA_DTYPE_FLOAT32 ? solver::rounding_bound<float>(m + 1)
                                                 : solver::rounding_bound<double>(m + 1);
  normalized.resize(n * m);
  for (std::size_t i = 0; i < n; ++i) {
    solver::core::compensated_sum<double> sum;
    for (std::size_t j = 0; j < m; ++j) {
      const double q = p[i * m + j];
      if (!(q >= 0) || !std::isfinite(q))
        invalid("Target probabilities must be finite and nonnegative");
      sum.add(q);
    }
    const double total = sum.value();
    if (!(std::abs(total - 1) <= tol)) invalid("Target probability rows must sum to one");
    for (std::size_t j = 0; j < m; ++j) normalized[i * m + j] = p[i * m + j] / total;
  }
}

const char* const target_names[] = {"labels", "label sets", "a label mask", "probabilities",
                                    "teacher logits"};

// The targets that a loss takes, in words.
const char* targets_of(int loss) {
  if (traits(loss).multilabel) return "label sets or a label mask";
  if (traits(loss).distillation) return "teacher logits";
  return loss == SDCA_LOSS_CROSS_ENTROPY ? "labels or probabilities" : "labels";
}

void check_target_fields(const sdca_targets& t, int loss) {
  bool allowed = false;
  switch (t.kind) {
    case SDCA_TARGET_LABELS:
      allowed = !traits(loss).multilabel && !traits(loss).distillation;
      break;
    case SDCA_TARGET_LABEL_SETS:
    case SDCA_TARGET_MASK:
      allowed = traits(loss).multilabel;
      break;
    case SDCA_TARGET_PROBABILITIES:
      allowed = loss == SDCA_LOSS_CROSS_ENTROPY;
      break;
    case SDCA_TARGET_LOGITS:
      allowed = traits(loss).distillation;
      break;
    default:
      invalid("Unknown target kind");
  }
  if (!allowed)
    invalid(std::string(name_of(loss)) + " takes " + targets_of(loss) + ", not " +
            target_names[t.kind]);
  const bool labels = t.kind == SDCA_TARGET_LABELS || t.kind == SDCA_TARGET_LABEL_SETS;
  if ((t.labels != nullptr) != labels ||
      (t.offsets != nullptr) != (t.kind == SDCA_TARGET_LABEL_SETS) ||
      (t.mask != nullptr) != (t.kind == SDCA_TARGET_MASK) ||
      (t.probabilities != nullptr) != (t.kind == SDCA_TARGET_PROBABILITIES) ||
      (t.logits != nullptr) != (t.kind == SDCA_TARGET_LOGITS))
    invalid("Targets need the fields of their kind, and NULL in the others");
}

// The targets of a loss (see c_api.h): only the fields of their kind. ignored: the label of
// ignored rows (sdca_loss_evaluate's ignore_index), if any.
target_rows check_targets(const sdca_targets* given, int loss, entry use, int dtype, std::size_t n,
                          std::size_t m, const int64_t* ignored = nullptr) {
  const sdca_targets t = read_struct(given, "targets", default_targets);
  check_target_fields(t, loss);
  target_rows rows = {t.kind, t.labels, t.offsets, t.mask, t.probabilities, t.logits, {}};
  switch (t.kind) {
    case SDCA_TARGET_LABELS:
      extent(n, sizeof(int64_t));
      if (use == entry::rows) break;
      if (ignored)
        kernel::check_labels(t.labels, n, m, *ignored);
      else
        kernel::check_labels(t.labels, n, m);
      break;
    case SDCA_TARGET_LABEL_SETS:
      extent(n + 1, sizeof(int64_t));
      check_label_sets(n, m, t.labels, t.offsets, use);
      break;
    case SDCA_TARGET_MASK:
      if (use != entry::rows) check_mask(n, m, t.mask, use);
      break;
    case SDCA_TARGET_PROBABILITIES:
      if (ignored && *ignored >= 0) invalid(ignored_probabilities);
      extent(extent(n, m), sizeof(double));
      check_probabilities(n, m, t.probabilities, use, dtype, rows.normalized);
      break;
    default:
      break;
  }
  return rows;
}

// ---------------------------------------------------------------------------
// sdca_loss_evaluate on the batch driver (sdca/kernel.h), which picks the instruction set.

const sdca_row_options default_row_options = SDCA_ROW_OPTIONS_INIT;

sdca_row_options check_row_options(const sdca_row_options* given, int target, int dtype) {
  const sdca_row_options o = read_struct(given, "row_options", default_row_options);
  const int p = o.probability_dtype;
  if (target == SDCA_TARGET_PROBABILITIES && p != SDCA_DTYPE_FLOAT32 && p != SDCA_DTYPE_FLOAT64 &&
      p != dtype)
    invalid("Probabilities must be float32, float64 or of the scores' dtype");
  return o;
}

template <typename S, typename Targets>
kernel::Result rows_of(const kernel::Loss& f, const kernel::Execution& e, std::size_t n,
                       std::size_t m, const void* scores, const Targets& t, double* values,
                       void* gradients, const sdca_row_options& o) {
  return kernel::evaluate(f, e, n, m, static_cast<const S*>(scores), t, values,
                          static_cast<S*>(gradients), o.upstream, o.row_upstream);
}

template <typename S>
kernel::Result loss_rows_of(const sdca_loss& l, const kernel::Loss& f, const kernel::Execution& e,
                            std::size_t n, std::size_t m, const void* scores, const target_rows& t,
                            const sdca_row_options& o, double* values, void* gradients) {
  switch (t.kind) {
    case SDCA_TARGET_LABELS: {
      const kernel::Labels labels = {t.labels, l.weight, l.ignore_index, l.ignore != 0};
      return rows_of<S>(f, e, n, m, scores, labels, values, gradients, o);
    }
    case SDCA_TARGET_LABEL_SETS: {
      const kernel::LabelSets sets = {t.labels, t.offsets};
      return rows_of<S>(f, e, n, m, scores, sets, values, gradients, o);
    }
    case SDCA_TARGET_MASK: {
      const kernel::Mask mask = {t.mask};
      return rows_of<S>(f, e, n, m, scores, mask, values, gradients, o);
    }
    case SDCA_TARGET_PROBABILITIES: {
      const void* p = t.given;
      if (o.probability_dtype == SDCA_DTYPE_FLOAT64) {
        const kernel::Probabilities<double> q = {static_cast<const double*>(p), l.weight};
        return rows_of<S>(f, e, n, m, scores, q, values, gradients, o);
      }
      if (o.probability_dtype == SDCA_DTYPE_FLOAT32) {
        const kernel::Probabilities<float> q = {static_cast<const float*>(p), l.weight};
        return rows_of<S>(f, e, n, m, scores, q, values, gradients, o);
      }
      const kernel::Probabilities<S> q = {static_cast<const S*>(p), l.weight};
      return rows_of<S>(f, e, n, m, scores, q, values, gradients, o);
    }
    default: {
      const kernel::Teacher<S> teacher = {static_cast<const S*>(t.logits)};
      return rows_of<S>(f, e, n, m, scores, teacher, values, gradients, o);
    }
  }
}

// The rows of sdca_loss_rows; returns the description of the invalid rows, "" if there is none.
std::string loss_rows(const sdca_loss& l, const sdca_row_options& o, const execution& e, int dtype,
                      std::size_t n, std::size_t m, const void* scores, const target_rows& t,
                      double* values, void* gradients) {
  const kernel::Loss f = {l.kind, l.k, l.gamma, l.temperature, l.label_smoothing};
  const kernel::Execution settings = {e.precision, e.num_threads, e.check_rows};
  kernel::Result r;
  switch (dtype) {
    case SDCA_DTYPE_FLOAT64:
      r = loss_rows_of<double>(l, f, settings, n, m, scores, t, o, values, gradients);
      break;
    case SDCA_DTYPE_FLOAT16:
      r = loss_rows_of<sdca::half>(l, f, settings, n, m, scores, t, o, values, gradients);
      break;
    case SDCA_DTYPE_BFLOAT16:
      r = loss_rows_of<sdca::bfloat16>(l, f, settings, n, m, scores, t, o, values, gradients);
      break;
    default:
      r = loss_rows_of<float>(l, f, settings, n, m, scores, t, o, values, gradients);
  }
  if (r.status == kernel::status_unsupported)
    throw std::runtime_error(std::string(name_of(l.kind)) +
                             " has no kernel for these targets in this build");
  if (e.report) {
    e.report->invalid_rows = r.invalid_rows;
    e.report->first_invalid_row = r.invalid_rows ? r.first_invalid : n;
  }
  if (r.status == kernel::status_ok && r.invalid_rows == 0) return std::string();
  const std::string rows = std::to_string(r.invalid_rows) + " of " + std::to_string(n) +
                           " rows are invalid (first: row " + std::to_string(r.first_invalid) +
                           "): scores must not be NaN or +inf, and labels must be in range "
                           "with a positive class";
  if (r.status != kernel::status_ok) invalid(rows);
  return rows;
}

// ---------------------------------------------------------------------------
// Training: the objectives over the solver (sdca_dual_update, sdca_model_fit).

const std::size_t history_fields = SDCA_HISTORY_FIELDS;

template <typename R>
double metric(const solver::eval_train<R, solver::multiclass_output>& e) {
  return e.accuracy.empty() ? 0 : static_cast<double>(e.accuracy[0]);
}
template <typename R>
double metric(const solver::eval_train<R, solver::distribution_output>& e) {
  return e.accuracy.empty() ? 0 : static_cast<double>(e.accuracy[0]);
}
template <typename R>
double metric(const solver::eval_train<R, solver::multilabel_output>& e) {
  return static_cast<double>(e.rank_loss);
}

int status_of(solver::solver_status status) {
  switch (status) {
    case solver::solver_status::solved:
      return SDCA_STATUS_SOLVED;
    case solver::solver_status::no_progress:
      return SDCA_STATUS_NO_PROGRESS;
    case solver::solver_status::max_epoch:
      return SDCA_STATUS_MAX_EPOCH;
    case solver::solver_status::failed:
      return SDCA_STATUS_FAILED;
    default:
      throw std::logic_error("The solver stopped in an unexpected state");
  }
}

static_assert(SDCA_SHRINKING_AUTO == static_cast<int>(solver::shrinking_mode::automatic) &&
                  SDCA_SHRINKING_OFF == static_cast<int>(solver::shrinking_mode::off) &&
                  SDCA_SHRINKING_HEURISTIC == static_cast<int>(solver::shrinking_mode::heuristic) &&
                  SDCA_SHRINKING_SAFE == static_cast<int>(solver::shrinking_mode::safe) &&
                  SDCA_SHRINKING_BOTH == static_cast<int>(solver::shrinking_mode::both),
              "The C shrinking ids follow solver::shrinking_mode");

// The solver objective of a training loss and its weight c.
solver::objective_spec objective_of(const sdca_loss& l, double c) {
  const solver::objective_spec spec = {static_cast<kernel::Kind>(l.kind), c, l.k, l.gamma};
  return spec;
}

// The positive classes of row i of label targets (labels, label sets or a mask).
struct positives_of {
  const target_rows* t;
  std::size_t m;

  std::size_t operator()(std::size_t i, char* mark) const {
    std::size_t count = 0;
    if (t->kind == SDCA_TARGET_LABELS) {
      mark[static_cast<std::size_t>(t->labels[i])] = 1;
      return 1;
    }
    if (t->kind == SDCA_TARGET_LABEL_SETS) {
      for (int64_t j = t->offsets[i]; j < t->offsets[i + 1]; ++j, ++count)
        mark[static_cast<std::size_t>(t->labels[j])] = 1;
      return count;
    }
    for (std::size_t j = 0; j < m; ++j) count += (mark[j] = t->mask[i * m + j] ? 1 : 0);
    return count;
  }
};

template <typename D>
void check_duals(const D* a, std::size_t n, std::size_t m, const sdca_loss& l, double c,
                 const target_rows& t, const char* name) {
  const positives_of positives = {&t, m};
  solver::duals::check_feasible(a, n, m, objective_of(l, c), t.probabilities(), positives, name);
}

// Label sets of label targets: those of LABEL_SETS, or of a mask.
struct label_sets {
  std::vector<sdca::size_type> labels, offsets;

  label_sets(const target_rows& t, std::size_t n, std::size_t m) : offsets(1, 0) {
    if (t.kind == SDCA_TARGET_LABEL_SETS) {
      labels.assign(t.labels, t.labels + t.offsets[n]);
      offsets.assign(t.offsets, t.offsets + n + 1);
      return;
    }
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < m; ++j)
        if (t.mask[i * m + j]) labels.push_back(j);
      offsets.push_back(labels.size());
    }
  }
};

// request.run(objective, output) of a solver objective visitor.
template <typename Request, typename Output>
struct run_on {
  Request& request;
  Output& out;
  template <typename Objective>
  void operator()(Objective obj) {
    request.run(std::move(obj), std::move(out));
  }
};
template <typename Request, typename Output>
run_on<Request, Output> run_on_of(Request& request, Output& out) {
  const run_on<Request, Output> visitor = {request, out};
  return visitor;
}

// request.run(objective, output) for the objective of a loss and its targets.
template <typename D, typename R, typename Request>
void dispatch(const sdca_loss& loss, double c, std::size_t n, std::size_t m, const target_rows& t,
              Request& request) {
  if (!std::isnormal(static_cast<R>(c)) ||
      (loss.gamma > 0 && !std::isnormal(static_cast<R>(loss.gamma))))
    invalid("C and gamma must be normal in the working precision");
  if (t.kind == SDCA_TARGET_PROBABILITIES) {
    std::vector<double> q(t.probabilities(), t.probabilities() + n * m);
    request.run(solver::l2_distillation<D, R>(static_cast<R>(c)),
                solver::distribution_output(m, n, std::move(q)));
    return;
  }
  const solver::objective_spec spec = objective_of(loss, c);
  if (t.kind == SDCA_TARGET_LABELS) {
    solver::multiclass_output out(m, std::vector<sdca::size_type>(t.labels, t.labels + n));
    solver::visit_multiclass_objective<D, R>(spec, run_on_of(request, out));
  } else {
    label_sets sets(t, n, m);
    solver::multilabel_output out(m, std::move(sets.labels), std::move(sets.offsets));
    solver::visit_multilabel_objective<D, R>(spec, run_on_of(request, out));
  }
}

// One coordinate step per row: scores z_i and block previous_i (or zero) into blocks_i.
template <typename D>
struct update_request {
  std::size_t n, m;
  const D* scores;
  const D* previous;
  const double* norm2;
  std::size_t num_threads;
  D* blocks;

  template <typename Objective, typename Output>
  void run(Objective obj, Output output) {
    solver::core::driver::parallel_for(n, m, num_threads, [&](std::size_t begin, std::size_t end) {
      std::vector<D> z(m);
      for (std::size_t i = begin; i < end; ++i) {
        std::copy(scores + i * m, scores + (i + 1) * m, z.begin());
        D* block = blocks + i * m;
        if (previous)
          std::copy(previous + i * m, previous + (i + 1) * m, block);
        else
          std::fill(block, block + m, D(0));
        solver::update_dual_variables(i, m, static_cast<D>(norm2[i]), output, obj, block, z.data());
      }
    });
  }
};

template <typename D, typename R>
void dual_update(const sdca_loss& loss, double c, const execution& e, std::size_t n, std::size_t m,
                 const D* scores, const target_rows& t, const D* previous, const double* norm2,
                 D* out) {
  if (!all_finite(scores, n * m)) invalid("Scores must be finite");
  if (previous) check_duals(previous, n, m, loss, c, t, "previous");
  for (std::size_t i = 0; i < n; ++i) {
    const double s = norm2[i];
    if (!(s > 0) || !std::isfinite(s)) invalid("norm2 must be finite and positive");
    if (!std::isnormal(static_cast<D>(c)) || !std::isnormal(static_cast<D>(s)) ||
        !std::isnormal(static_cast<D>(c * s)))
      invalid("C, norm2, and C*norm2 must be normal in the storage type");
  }
  std::vector<D> blocks(n * m);
  update_request<D> request = {n, m, scores, previous, norm2, e.num_threads, blocks.data()};
  dispatch<D, R>(loss, c, n, m, t, request);
  if (!all_finite(blocks.data(), n * m))
    throw std::runtime_error("A coordinate step gave a nonfinite dual block");
  std::copy(blocks.begin(), blocks.end(), out);
}

// The solve of a fit, on the library's own copies of W and the duals.
template <typename D>
struct fit_request {
  std::size_t n, d;
  const D* x;
  bool kernel;
  const D* center;
  D* weights;
  D* dual;
  solver::stopping_criteria criteria;
  double* history;
  std::size_t history_rows;
  sdca_fit_result* result;

  template <typename Context>
  void solve(Context& ctx) {
    ctx.criteria = criteria;
    solver::make_solver(ctx).solve();
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
      auto ctx = solver::make_context(solver::make_input_kernel(n, x), std::move(out),
                                      std::move(obj), dual);
      solve(ctx);
    } else {
      auto ctx = solver::make_context(solver::make_input_feature(n, d, x), std::move(out),
                                      std::move(obj), weights, dual, center);
      solve(ctx);
    }
  }
};

// The solver's stopping criteria of fit options; threads 0 keeps the solver's default.
solver::stopping_criteria criteria_of(const sdca_fit_options& o, std::size_t num_threads) {
  solver::stopping_criteria criteria;
  criteria.max_epoch = o.max_epoch;
  criteria.eval_epoch = o.eval_epoch;
  criteria.tol = o.tol;
  criteria.eval_on_start = o.eval_on_start != 0;
  criteria.recompute_primal = o.recompute_primal != 0;
  if (num_threads != 0) criteria.num_threads = num_threads;
  criteria.shrinking = static_cast<solver::shrinking_mode>(o.shrinking);
  criteria.seed = o.seed;
  return criteria;
}

template <typename D, typename R>
void model_fit(const sdca_loss& loss, const sdca_fit_options& o, const execution& e,
               const sdca_input& in, std::size_t m, const target_rows& t, const D* center,
               const D* initial, D* weights, D* dual, double* history, std::size_t history_rows,
               sdca_fit_result* result) {
  const bool kernel = in.kind == SDCA_INPUT_KERNEL;
  const std::size_t n = in.n, d = in.d;
  const solver::stopping_criteria criteria = criteria_of(o, e.num_threads);
  const D* x = static_cast<const D*>(in.x);
  if (!all_finite(x, n * (kernel ? n : d))) invalid("Inputs must be finite");
  if (center && !all_finite(center, d * m)) invalid("The center must be finite");
  // The library's copies of the iterate: the caller's outputs are written at the end.
  std::vector<D> a(n * m, D(0)), w(kernel ? 0 : d * m, D(0));
  if (initial) {
    check_duals(initial, n, m, loss, o.c, t, "initial_dual");
    std::copy(initial, initial + n * m, a.begin());
    if (t.kind == SDCA_TARGET_LABELS)
      solver::duals::rebuild_label_entries(
          a.data(), n, m, [&](std::size_t i) { return static_cast<std::size_t>(t.labels[i]); });
    if (!kernel)
      solver::duals::primal_of_dual(x, a.data(), center, n, d, m, criteria.num_threads, w.data());
  } else if (center && !kernel) {
    std::copy(center, center + d * m, w.begin());
  }
  sdca_fit_result outcome = *result;
  fit_request<D> request = {n,        d,        x,       kernel,       center,  w.data(),
                            a.data(), criteria, history, history_rows, &outcome};
  dispatch<D, R>(loss, o.c, n, m, t, request);
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

// The evaluations of a fit: every eval_epoch epochs, the start and the end (eval_epoch 0:
// the start and the end only). Saturates instead of wrapping for huge max_epoch.
std::size_t history_bound(const sdca_fit_options& o) {
  const std::size_t scheduled = o.eval_epoch == 0 ? 0 : o.max_epoch / o.eval_epoch;
  const std::size_t max = std::numeric_limits<std::size_t>::max();
  return scheduled > max - 2 ? max : scheduled + 2;
}

// f<D, R>() for the storage dtype and the working precision of e.
template <template <typename, typename> class F, typename... Args>
void by_precision(int dtype, const execution& e, Args&&... args) {
  if (dtype == SDCA_DTYPE_FLOAT64)
    F<double, double>::run(std::forward<Args>(args)...);
  else if (e.float_work(dtype))
    F<float, float>::run(std::forward<Args>(args)...);
  else
    F<float, double>::run(std::forward<Args>(args)...);
}

template <typename D, typename R>
struct dual_update_of {
  static void run(const sdca_loss& l, double c, const execution& e, std::size_t n, std::size_t m,
                  const void* scores, const target_rows& t, const void* previous,
                  const double* norm2, void* out) {
    dual_update<D, R>(l, c, e, n, m, static_cast<const D*>(scores), t,
                      static_cast<const D*>(previous), norm2, static_cast<D*>(out));
  }
};

template <typename D, typename R>
struct model_fit_of {
  static void run(const sdca_loss& l, const sdca_fit_options& o, const execution& e,
                  const sdca_input& in, std::size_t m, const target_rows& t, const void* center,
                  const void* initial, void* weights, void* dual, double* history, std::size_t rows,
                  sdca_fit_result* result) {
    model_fit<D, R>(l, o, e, in, m, t, static_cast<const D*>(center),
                    static_cast<const D*>(initial), static_cast<D*>(weights), static_cast<D*>(dual),
                    history, rows, result);
  }
};
}  // namespace

extern "C" {
int sdca_abi_version() { return SDCA_ABI_VERSION; }
const char* sdca_version() { return SDCA_VERSION; }
const char* sdca_blas_backend() { return sdca::blas_backend(); }
const char* sdca_last_error() { return last_error.c_str(); }

size_t sdca_max_extent() { return max_extent(); }

size_t sdca_history_rows(const sdca_fit_options* options) {
  std::size_t rows = 0;
  protect([&] { rows = history_bound(check_fit_options(options)); });
  return rows;
}

int sdca_loss_evaluate(const sdca_loss* loss, const sdca_execution* execution_settings, int dtype,
                       size_t n, size_t m, const void* scores, const sdca_targets* targets,
                       double* values, void* gradients) {
  std::string rows;
  const int code = protect([&] {
    check_dtype(dtype);
    if (m < 2) invalid("At least two classes are required");
    const sdca_loss l = check_loss(loss, m, entry::loss);
    const execution e = check_execution(execution_settings, dtype);
    extent(extent(n, m), sizeof(double));
    const target_rows t = check_targets(targets, l.kind, entry::loss, dtype, n, m,
                                        l.ignore ? &l.ignore_index : nullptr);
    if (n > 0) check_buffers({scores, values});
    rows = loss_rows(l, default_row_options, e, dtype, n, m, scores, t, values, gradients);
  });
  if (code != SDCA_OK || rows.empty()) return code;
  last_error = rows;
  return SDCA_INVALID_ROWS;
}

int sdca_loss_rows(const sdca_loss* loss, const sdca_row_options* options,
                   const sdca_execution* execution_settings, int dtype, size_t n, size_t m,
                   const void* scores, const sdca_targets* targets, double* values,
                   void* gradients) {
  std::string rows;
  const int code = protect([&] {
    if (dtype < SDCA_DTYPE_FLOAT32 || dtype > SDCA_DTYPE_BFLOAT16) invalid("Invalid dtype");
    if (m < 2) invalid("At least two classes are required");
    const sdca_loss l = check_loss(loss, m, entry::rows);
    const execution e = check_execution(execution_settings, dtype);
    extent(extent(n, m), sizeof(double));
    const target_rows t = check_targets(targets, l.kind, entry::rows, dtype, n, m);
    const sdca_row_options o = check_row_options(options, t.kind, dtype);
    if (n > 0) check_buffers({scores});
    rows = loss_rows(l, o, e, dtype, n, m, scores, t, values, gradients);
  });
  if (code != SDCA_OK || rows.empty()) return code;
  last_error = rows;
  return SDCA_INVALID_ROWS;
}

int sdca_reduce_rows(int dtype, int reduction, size_t n, const double* values, const double* terms,
                     void* out) {
  return protect([&] {
    check_dtype(dtype);
    if (reduction < 0 || reduction > 2) invalid("Invalid reduction");
    extent(n, sizeof(double));
    if (n) check_buffers({values});
    if (n || reduction) check_buffers({out});
    if (dtype == SDCA_DTYPE_FLOAT64)
      sdca::detail::store_row_values(static_cast<double*>(out), values, terms, n, reduction, n);
    else
      sdca::detail::store_row_values(static_cast<float*>(out), values, terms, n, reduction, n);
  });
}

int sdca_reduce_row_groups(int dtype, int reduction, size_t groups, size_t rows,
                           const double* values, const double* terms, void* out) {
  return protect([&] {
    check_dtype(dtype);
    if (reduction != 1 && reduction != 2) invalid("Invalid grouped reduction");
    extent(extent(groups, rows), sizeof(double));
    extent(groups, dtype == SDCA_DTYPE_FLOAT64 ? sizeof(double) : sizeof(float));
    if (groups && rows) check_buffers({values});
    if (groups) check_buffers({out});
    if (dtype == SDCA_DTYPE_FLOAT64)
      sdca::detail::store_row_groups(static_cast<double*>(out), values, terms, groups, rows,
                                     reduction);
    else
      sdca::detail::store_row_groups(static_cast<float*>(out), values, terms, groups, rows,
                                     reduction);
  });
}

int sdca_dual_update(const sdca_loss* loss, double c, const sdca_execution* execution_settings,
                     int dtype, size_t n, size_t m, const void* scores, const sdca_targets* targets,
                     const void* previous, const double* norm2, void* out) {
  return protect([&] {
    check_dtype(dtype);
    if (m < 2) invalid("At least two classes are required");
    const sdca_loss l = check_loss(loss, m, entry::training);
    kernel::check_objective(l.kind);
    if (!(c > 0) || !std::isfinite(c)) invalid("C must be finite and positive");
    const execution e = check_execution(execution_settings, dtype);
    extent(extent(n, m), sizeof(double));
    const target_rows t = check_targets(targets, l.kind, entry::training, dtype, n, m);
    if (n == 0) return;
    check_buffers({scores, norm2, out});
    by_precision<dual_update_of>(dtype, e, l, c, e, n, m, scores, t, previous, norm2, out);
  });
}

int sdca_model_fit(const sdca_loss* loss, const sdca_fit_options* options,
                   const sdca_execution* execution_settings, const sdca_input* input, size_t m,
                   const sdca_targets* targets, const void* center, const void* initial_dual,
                   void* weights, void* dual, double* history, size_t history_rows,
                   sdca_fit_result* result) {
  return protect([&] {
    const sdca_input in = read_struct(input, "input", default_input);
    if (in.kind != SDCA_INPUT_FEATURES && in.kind != SDCA_INPUT_KERNEL)
      invalid("Unknown input kind");
    check_dtype(in.dtype);
    const bool kernel = in.kind == SDCA_INPUT_KERNEL;
    if (in.n < 1 || in.d < 1)
      invalid("The input must have n >= 1 and d >= 1, not " + std::to_string(in.n) + "x" +
              std::to_string(in.d));
    if (m < 2) invalid("At least two classes are required");
    if (kernel && in.d != in.n) invalid("A Gram matrix is square: d must equal n");
    const sdca_loss l = check_loss(loss, m, entry::training);
    kernel::check_objective(l.kind);
    const sdca_fit_options o = check_fit_options(options);
    if (!traits(l.kind).hinge && o.shrinking != SDCA_SHRINKING_AUTO)
      invalid(std::string(name_of(l.kind)) +
              " takes no shrinking: its steps update every class of every example");
    const execution e = check_execution(execution_settings, in.dtype);
    sdca_fit_result r = read_struct(result, "fit_result", default_fit_result);
    const std::size_t n = in.n, d = in.d, size = storage_size(in.dtype);
    check_blas_sizes({n, d, m, extent(n, m), extent(d, m)});
    extent(extent(n, d), size);
    extent(extent(n, m), size);
    extent(extent(d, m), size);
    extent(extent(history_rows, history_fields), sizeof(double));
    check_buffers({in.x, dual});
    if (kernel && (weights || center)) invalid("A Gram matrix takes no weights and no center");
    if (!kernel) check_buffers({weights});
    if (!history != (history_rows == 0)) invalid("Pass history with history_rows, or neither");
    if (history && history_rows < history_bound(o))
      invalid("The history needs sdca_history_rows(options) rows");
    const target_rows t = check_targets(targets, l.kind, entry::training, in.dtype, n, m);
    by_precision<model_fit_of>(in.dtype, e, l, o, e, in, m, t, center, initial_dual, weights, dual,
                               history, history_rows, &r);
    result->status = r.status;
    result->epochs = r.epochs;
    result->history_rows = r.history_rows;
  });
}

int sdca_model_scores(const sdca_input* input, size_t m, const void* coefficients, void* scores) {
  return protect([&] {
    const sdca_input in = read_struct(input, "input", default_input);
    if (in.kind != SDCA_INPUT_FEATURES && in.kind != SDCA_INPUT_KERNEL)
      invalid("Unknown input kind");
    check_dtype(in.dtype);
    if (in.d < 1 || m < 1) invalid("Invalid score dimensions");
    const std::size_t n = in.n, d = in.d, size = storage_size(in.dtype);
    check_blas_sizes({n, d, m});
    extent(extent(n, d), size);
    extent(extent(n, m), size);
    extent(extent(d, m), size);
    check_buffers({coefficients});
    if (n == 0) return;
    check_buffers({in.x, scores});
    const bool kernel = in.kind == SDCA_INPUT_KERNEL;
    if (in.dtype == SDCA_DTYPE_FLOAT64)
      solver::batch_scores(kernel, n, d, m, static_cast<const double*>(in.x),
                           static_cast<const double*>(coefficients), static_cast<double*>(scores));
    else
      solver::batch_scores(kernel, n, d, m, static_cast<const float*>(in.x),
                           static_cast<const float*>(coefficients), static_cast<float*>(scores));
  });
}
}

#ifdef SDCA_PARALLEL_OPENMP
// From the library's load on: a child forked after it never enters the
// parent's OpenMP team.
__attribute__((unused)) static const bool watching_forks = sdca::detail::watch_forks();
#endif
