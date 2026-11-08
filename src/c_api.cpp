// The C ABI of include/sdca/c_api.h over the headers. SDCA_BASELINE builds
// the solver entry points against headers without sdca/loss.h; its loss
// entry point always fails.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdint>
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
#include "sdca/c_api.h"
#include "sdca/solver.h"
#ifndef SDCA_BASELINE
#include "sdca/loss.h"
#include "sdca/utility/scratch.h"
#include "tensor_losses.h"
#endif
#include "sdca/utility/parallel.h"
#include "sdca/utility/fpenv.h"
#include "sdca/utility/logging.cpp"

#ifndef SDCA_VERSION
#define SDCA_VERSION "unknown"
#endif

namespace {
thread_local std::string last_error;
const bool quiet_logger = [] {
  sdca::logging::set_level(sdca::logging::level::none);
  return true;
}();

// Every computing entry point: IEEE arithmetic (no flush-to-zero, round to
// nearest, no traps) inside, the caller's modes restored on return.
template <typename F>
int protect(F f) {
  sdca::detail::ieee_environment ieee;
  try {
    f();
    last_error.clear();
    return 0;
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

void check_labels(std::size_t n, std::size_t k, const int64_t* labels) {
  for (std::size_t i = 0; i < n; ++i)
    if (labels[i] < 0 || static_cast<std::size_t>(labels[i]) >= k)
      throw std::invalid_argument("Labels must lie in [0, num_classes)");
}

void check_sizes(std::size_t n, std::size_t k) {
  if (n < 1 || k < 2) throw std::invalid_argument("Invalid dimensions");
}

// The extent a * b of a caller's buffer, which must not wrap around.
std::size_t extent(std::size_t a, std::size_t b) {
  if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::invalid_argument("Buffer extent overflows size_t");
  return a * b;
}

void check_buffers(std::initializer_list<const void*> buffers) {
  for (const void* buffer : buffers)
    if (!buffer) throw std::invalid_argument("Required buffers must not be NULL");
}

void check_blas_sizes(std::initializer_list<std::size_t> sizes) {
  const auto limit = static_cast<std::size_t>(std::numeric_limits<sdca::blas_int>::max());
  for (std::size_t size : sizes)
    if (size > limit)
      throw std::invalid_argument("Dimensions exceed the native BLAS integer limit");
}

template <typename D>
void check_finite(const void* x, std::size_t size, const char* message) {
  const D* first = static_cast<const D*>(x);
  if (!std::all_of(first, first + size, [](D v) { return std::isfinite(v); }))
    throw std::invalid_argument(message);
}

// The loss and update entry points compute into scratch and write their
// outputs only on success; a failed call leaves the caller's buffers as given.
// Rows run on the threads of num_threads (see c_api.h), each independently.
template <typename D>
void check_finite_rows(const D* x, std::size_t n, std::size_t k, std::size_t num_threads,
                       const char* message) {
  sdca::detail::parallel_rows(n, k, num_threads, [=](std::size_t begin, std::size_t end) {
    check_finite<D>(x + begin * k, (end - begin) * k, message);
  });
}

template <typename D>
void copy_rows(const D* from, std::size_t n, std::size_t k, std::size_t num_threads, D* to) {
  sdca::detail::parallel_rows(n, k, num_threads, [=](std::size_t begin, std::size_t end) {
    std::copy(from + begin * k, from + end * k, to + begin * k);
  });
}

#ifndef SDCA_BASELINE
using projection_mode = sdca::projection_precision;
static_assert(SDCA_PROJECTION_FLOAT64 == static_cast<int>(projection_mode::float64) &&
                  SDCA_PROJECTION_EXTENDED == static_cast<int>(projection_mode::extended),
              "The C projection ids follow sdca::projection_precision");
#else
enum class projection_mode { float64, extended };  // validated, unused by the baseline headers
#endif

// The projection_precision argument of an entry point.
projection_mode check_projection(int mode) {
  if (mode != SDCA_PROJECTION_FLOAT64 && mode != SDCA_PROJECTION_EXTENDED)
    throw std::invalid_argument("Invalid projection precision");
  return static_cast<projection_mode>(mode);
}

#ifndef SDCA_BASELINE
using shrinking_mode = sdca::shrinking_mode;
static_assert(SDCA_SHRINKING_AUTO == static_cast<int>(shrinking_mode::automatic) &&
                  SDCA_SHRINKING_OFF == static_cast<int>(shrinking_mode::off) &&
                  SDCA_SHRINKING_HEURISTIC == static_cast<int>(shrinking_mode::heuristic) &&
                  SDCA_SHRINKING_SAFE == static_cast<int>(shrinking_mode::safe) &&
                  SDCA_SHRINKING_BOTH == static_cast<int>(shrinking_mode::both),
              "The C shrinking ids follow sdca::shrinking_mode");
#else
enum class shrinking_mode { automatic, off, heuristic, safe, both };  // unused by the baseline
#endif

// The shrinking of sdca_fit: AUTO for a struct of a header without the field.
shrinking_mode check_shrinking(const sdca_options& o) {
  if (o.size < offsetof(sdca_options, shrinking) + sizeof(o.shrinking))
    return shrinking_mode::automatic;
  if (o.shrinking < SDCA_SHRINKING_AUTO || o.shrinking > SDCA_SHRINKING_BOTH)
    throw std::invalid_argument("Invalid shrinking");
  return static_cast<shrinking_mode>(o.shrinking);
}

#ifndef SDCA_BASELINE
static_assert(SDCA_PROXIMAL_CROSS_ENTROPY ==
                      static_cast<int>(sdca::loss_kind::proximal_cross_entropy) &&
                  SDCA_MULTICLASS_HINGE == static_cast<int>(sdca::loss_kind::multiclass_hinge),
              "The C loss ids follow sdca::loss_kind");

// The index of the first nonzero of p[from, n), else n; eight bytes at a time.
std::size_t first_nonzero(const uint8_t* p, std::size_t n, std::size_t from) {
  std::size_t j = from;
  for (; j + 8 <= n; j += 8) {
    uint64_t word;
    std::memcpy(&word, p + j, sizeof word);
    if (word) break;
  }
  while (j < n && !p[j]) ++j;
  return j;
}

// The losses of rows [begin, end): z (n, k) and its positive mask in, the
// values and gradients out.
template <typename D, typename R>
struct loss_rows {
  sdca::loss_kind kind;
  std::size_t k, topk;
  double gamma, alpha;
  projection_mode mode;
  const D* z;
  const uint8_t* positive;
  double* values;
  D* gradients;

  void operator()(std::size_t begin, std::size_t end) const {
    // Row scratch on the stack for up to 64 classes.
    sdca::detail::entropy_buffer<D> scores(k), grad(k);
    sdca::detail::entropy_buffer<std::size_t> order_buffer(k);
    std::size_t* order = order_buffer.data();
    if (kind == sdca::loss_kind::cross_entropy ||
        kind == sdca::loss_kind::multilabel_cross_entropy) {
      // In place, as sdca_loss_gradient_rows (the same bits).
      sdca::detail::entropy_buffer<double> terms(k);
      for (std::size_t i = begin; i < end; ++i) {
        const uint8_t* row = positive + i * k;
        const std::size_t p = sdca::tensor::mask_positions(k, row, order);
        if (p == 0) throw std::invalid_argument("Invalid loss dimensions or parameters");
        values[i] = sdca::tensor::mask_entropy_row<D, R>(k, row, z + i * k, p, order, terms.data(),
                                                         gradients + i * k);
      }
      return;
    }
    for (std::size_t i = begin; i < end; ++i) {
      const uint8_t* row = positive + i * k;
      const bool topk_entropy =
          kind == sdca::loss_kind::topk_cross_entropy && topk > 1 && topk + 1 < k;
      if (kind == sdca::loss_kind::cross_entropy || topk_entropy) {
        // One label: the row itself, no reordering.
        const std::size_t label = first_nonzero(row, k, 0);
        if (label < k && first_nonzero(row, k, label + 1) == k) {
          values[i] = topk_entropy
                          ? sdca::topk_entropy_label_loss<D, R>(k, label, z + i * k, topk,
                                                                gradients + i * k)
                          : sdca::entropy_label_loss<D, R>(k, label, z + i * k, gradients + i * k);
          continue;
        }
      }
      // Positive labels first, as loss_and_gradient takes them.
      std::size_t p = 0, q = k;
      for (std::size_t j = 0; j < k; ++j) {
        if (row[j]) order[p++] = j;
      }
      for (std::size_t j = k; j-- > 0;) {
        if (!row[j]) order[--q] = j;
      }
      for (std::size_t j = 0; j < k; ++j) scores.data()[j] = z[i * k + order[j]];
      values[i] = sdca::loss_and_gradient<D, R>(kind, k, p, scores.data(), grad.data(), topk, gamma,
                                                alpha, mode);
      for (std::size_t j = 0; j < k; ++j) gradients[i * k + order[j]] = grad.data()[j];
    }
  }
};

template <typename D, typename R>
void batch_loss_gradient(int kind, std::size_t n, std::size_t k, const void* logits,
                         const uint8_t* positive, std::size_t topk, double gamma, double alpha,
                         projection_mode mode, std::size_t num_threads, double* values,
                         void* gradients) {
  if (kind < 0 || kind > SDCA_PROXIMAL_CROSS_ENTROPY) throw std::invalid_argument("Unknown loss");
  const D* z = static_cast<const D*>(logits);
  const std::size_t size = extent(n, k);
  extent(size, sizeof(D));
  extent(n, sizeof(double));
  check_buffers({logits, positive, values, gradients});
  check_finite_rows(z, n, k, num_threads, "Logits must be finite");
  if (!sdca::detail::valid_loss_arguments(k, 1, topk, gamma, alpha))
    throw std::invalid_argument("Invalid loss dimensions or parameters");
  if (kind == SDCA_CROSS_ENTROPY) {
    // Valid once every row has a label: compute into the outputs.
    sdca::detail::parallel_rows(n, k, num_threads, [=](std::size_t begin, std::size_t end) {
      for (std::size_t i = begin; i < end; ++i)
        if (first_nonzero(positive + i * k, k, 0) == k)
          throw std::invalid_argument("Invalid loss dimensions or parameters");
    });
    const loss_rows<D, R> rows = {
        sdca::loss_kind(kind),     k, topk, gamma, alpha, mode, z, positive, values,
        static_cast<D*>(gradients)};
    sdca::detail::parallel_rows(n, k, num_threads, rows);
    return;
  }
  // Uninitialized: the workers touch the pages of their own rows first.
  std::unique_ptr<double[]> value(new double[n]);
  std::unique_ptr<D[]> g(new D[size]);
  const loss_rows<D, R> rows = {
      sdca::loss_kind(kind), k, topk, gamma, alpha, mode, z, positive, value.get(), g.get()};
  sdca::detail::parallel_rows(n, k, num_threads, rows);
  std::copy(value.get(), value.get() + n, values);
  copy_rows(g.get(), n, k, num_threads, static_cast<D*>(gradients));
}

// Statistics of sdca_entropy_forward per row: maximum, lowest, sum, mass, tiny.
const std::size_t entropy_stats = 5;

template <typename D, typename R>
void entropy_forward(std::size_t n, std::size_t k, const void* logits, const int64_t* labels,
                     std::size_t num_threads, double* values, double* stats) {
  const D* z = static_cast<const D*>(logits);
  extent(extent(n, k), sizeof(D));
  extent(extent(n, entropy_stats), sizeof(double));
  check_buffers({logits, labels, values});
  check_labels(n, k, labels);
  // Into scratch: a nonfinite row fails the call after the pass.
  std::unique_ptr<double[]> value(new double[n]), stat(new double[n * entropy_stats]);
  std::atomic<bool> finite(true);
  double* v = value.get();
  double* st = stat.get();
  sdca::detail::parallel_rows(n, k, num_threads, [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      const std::size_t label = static_cast<std::size_t>(labels[i]);
      const D* row = z + i * k;
      const sdca::entropy_row<double> r =
          sdca::single_entropy_row<double>(k, label, row, static_cast<double*>(nullptr));
      if (!std::isfinite(r.lowest) || !std::isfinite(r.sum) || !std::isfinite(double(row[label])))
        finite = false;
      // As loss_and_gradient in R: float rounds the value.
      v[i] = double(R(r.loss));
      double* s = st + i * entropy_stats;
      s[0] = r.maximum;
      s[1] = r.lowest;
      s[2] = r.sum;
      s[3] = r.mass;
      s[4] = sdca::entropy_row_tiny(r) ? 1 : 0;
    }
  });
  if (!finite) throw std::invalid_argument("Logits must be finite");
  std::copy(v, v + n, values);
  if (stats) std::copy(st, st + n * entropy_stats, stats);
}

template <typename D, typename R>
void entropy_gradient(std::size_t n, std::size_t k, const void* logits, const int64_t* labels,
                      const double* stats, const double* scale, std::size_t num_threads,
                      void* gradients) {
  const D* z = static_cast<const D*>(logits);
  extent(extent(n, k), sizeof(D));
  extent(extent(n, entropy_stats), sizeof(double));
  check_buffers({logits, labels, stats, scale, gradients});
  check_labels(n, k, labels);
  D* g = static_cast<D*>(gradients);
  sdca::detail::parallel_rows(n, k, num_threads, [=](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      const double* s = stats + i * entropy_stats;
      const sdca::entropy_row<double> r = {s[0], s[1], s[2], s[3], 0.0};
      sdca::single_entropy_gradient(k, static_cast<std::size_t>(labels[i]), z + i * k, r, scale[i],
                                    static_cast<const double*>(nullptr), g + i * k);
    }
  });
}
#endif

static_assert(SDCA_THREADS_DEFAULT == sdca::detail::default_threads,
              "The C default thread count follows parallel.h");

#ifdef SDCA_PARALLEL_OPENMP
// From the library's load on: a child forked after it never enters the
// parent's OpenMP team.
__attribute__((unused)) static const bool watching_forks = sdca::detail::watch_forks();
#endif

// History rows: epoch, primal, dual, primal_loss, dual_loss, regularizer,
// relative_gap, metric (top-1 accuracy or multilabel rank loss), cpu, wall.
const std::size_t history_fields = 10;

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

struct fit_options {
  std::size_t epochs, eval_epoch, history_rows;
  double tol;
  bool eval_on_start, recompute_primal;
  projection_mode projection;
  std::size_t num_threads;
  shrinking_mode shrinking;
};

template <typename Context>
void solve(Context& ctx, const fit_options& opt, double* history, sdca_fit_result* result) {
  ctx.criteria.max_epoch = opt.epochs;
  ctx.criteria.eval_epoch = opt.eval_epoch;
  ctx.criteria.epsilon = opt.tol;
  ctx.criteria.eval_on_start = opt.eval_on_start;
#ifndef SDCA_BASELINE  // v0.2.0 never recomputes the primal
  ctx.criteria.recompute_primal = opt.recompute_primal;
  ctx.criteria.projection = opt.projection;
  ctx.criteria.num_threads = opt.num_threads;
  ctx.criteria.shrinking = opt.shrinking;
#endif
  sdca::make_solver(ctx).solve();
  std::size_t i = 0;
  for (const auto& e : ctx.train.evals) {
    if (i == opt.history_rows) throw std::length_error("The history buffer is too small");
    double row[history_fields] = {double(e.epoch),
                                  double(e.primal),
                                  double(e.dual),
                                  double(e.primal_loss),
                                  double(e.dual_loss),
                                  double(e.primal_regularizer),
                                  double(e.relative_gap()),
                                  metric(e),
                                  e.cpu_time(),
                                  e.wall_time()};
    std::copy(row, row + history_fields, history + history_fields * i++);
  }
  result->status = static_cast<int>(ctx.status);
  result->epochs = ctx.epoch;
  result->history_used = i;
}

// Objective ids (sdca_loss): the first five take one label per example, the
// last two sorted label lists with offsets.
const int last_single_label = SDCA_TOPK_HINGE_BETA, last_objective = SDCA_MULTILABEL_CROSS_ENTROPY;
static_assert(SDCA_STATUS_FAILED == static_cast<int>(sdca::solver_status::failed),
              "The C status codes follow sdca::solver_status");

// Offsets (n+1) from zero; each row a strictly increasing list of between 1
// and k-1 labels in [0, k).
void check_label_sets(std::size_t n, std::size_t k, const int64_t* labels, const int64_t* offsets) {
  if (!offsets) throw std::invalid_argument("Multilabel objectives need label offsets");
  if (offsets[0] != 0) throw std::invalid_argument("Label offsets must start at zero");
  for (std::size_t i = 0; i < n; ++i) {
    if (offsets[i + 1] <= offsets[i])
      throw std::invalid_argument("Label offsets must be strictly increasing");
    const int64_t size = offsets[i + 1] - offsets[i];
    if (size < 1 || static_cast<std::size_t>(size) >= k)
      throw std::invalid_argument("Each example needs between 1 and num_classes-1 labels");
    check_labels(static_cast<std::size_t>(size), k, labels + offsets[i]);
    for (int64_t j = offsets[i] + 1; j < offsets[i + 1]; ++j)
      if (labels[j] <= labels[j - 1])
        throw std::invalid_argument("Label sets must be sorted without duplicates");
  }
}

// The checks of a direct ABI caller, before any output is written.
void check_objective(int objective, std::size_t n, std::size_t k, const int64_t* labels,
                     const int64_t* offsets, double c, std::size_t topk, double gamma) {
  if (objective < 0 || objective > last_objective) throw std::invalid_argument("Unknown objective");
  if (n < 1 || k < 2 || topk < 1 || topk >= k)
    throw std::invalid_argument("Invalid objective dimensions");
  if (!(c > 0) || !std::isfinite(c) || !(gamma >= 0) || !std::isfinite(gamma))
    throw std::invalid_argument("C must be finite and positive, gamma finite and nonnegative");
  if (objective <= last_single_label)
    check_labels(n, k, labels);
  else
    check_label_sets(n, k, labels, offsets);
}

// A smooth top-k hinge from (c, k, gamma); the baseline headers take gamma before k.
template <typename Objective, typename R>
Objective smooth_topk(R c, std::size_t k, R gamma) {
#ifdef SDCA_BASELINE
  return Objective(c, gamma, k);
#else
  return Objective(c, k, gamma);
#endif
}

// request.run(objective, output) does the work for the selected pair.
template <typename D, typename R, typename Request>
void dispatch(int objective, std::size_t n, std::size_t k, const int64_t* labels,
              const int64_t* offsets, double c, std::size_t topk, double gamma, Request& request) {
  const R cc = static_cast<R>(c), g = static_cast<R>(gamma);
  // gamma = 0 selects the nonsmooth hinge; a positive gamma must be normal.
  const bool smooth = gamma > 0;
  if (!std::isnormal(cc) || (smooth && !std::isnormal(g)))
    throw std::invalid_argument("C and gamma must be normal in the working precision");
  if (objective <= last_single_label) {
    std::vector<sdca::size_type> y(labels, labels + n);
    sdca::multiclass_output out(k, y);
    switch (objective) {
      case SDCA_CROSS_ENTROPY:
        request.run(sdca::l2_entropy<D, R>(cc), std::move(out));
        break;
      case SDCA_TOPK_CROSS_ENTROPY:
        request.run(sdca::l2_entropy_topk<D, R>(cc, topk), std::move(out));
        break;
      case SDCA_MULTICLASS_HINGE:
      case SDCA_TOPK_HINGE_ALPHA: {
        const std::size_t kk = objective == SDCA_MULTICLASS_HINGE ? 1 : topk;
        if (smooth)
          request.run(smooth_topk<sdca::l2_hinge_topk_smooth<D, R>>(cc, kk, g), std::move(out));
        else
          request.run(sdca::l2_hinge_topk<D, R>(cc, kk), std::move(out));
        break;
      }
      case SDCA_TOPK_HINGE_BETA:
        if (smooth)
          request.run(smooth_topk<sdca::l2_topk_hinge_smooth<D, R>>(cc, topk, g), std::move(out));
        else
          request.run(sdca::l2_topk_hinge<D, R>(cc, topk), std::move(out));
        break;
      default:
        throw std::invalid_argument("Unknown objective");
    }
  } else {
    std::vector<sdca::size_type> u(offsets, offsets + n + 1);
    std::vector<sdca::size_type> y(labels, labels + u.back());
    sdca::multilabel_output out(k, y, u);
    switch (objective) {
      case SDCA_MULTILABEL_HINGE:
        if (smooth)
          request.run(sdca::l2_multilabel_hinge_smooth<D, R>(cc, g), std::move(out));
        else
          request.run(sdca::l2_multilabel_hinge<D, R>(cc), std::move(out));
        break;
      case SDCA_MULTILABEL_CROSS_ENTROPY:
        request.run(sdca::l2_multilabel_entropy<D, R>(cc), std::move(out));
        break;
      default:
        throw std::invalid_argument("Unknown objective");
    }
  }
}

// Dual (and primal) variables are used as given when warm is set; the caller
// keeps W = center + X' A consistent. Otherwise both start from zero/center.
template <typename D>
struct fit_request {
  std::size_t n, d, k;
  const D* input;
  bool kernel, warm;
  const D* center;
  D* weights;
  D* dual;
  fit_options options;
  double* history;
  sdca_fit_result* result;

  template <typename Objective, typename Output>
  void run(Objective obj, Output out) {
    if (!warm) std::fill(dual, dual + n * k, D(0));
    if (kernel) {
      auto ctx = sdca::make_context(sdca::make_input_kernel(n, input), std::move(out),
                                    std::move(obj), dual);
      solve(ctx, options, history, result);
    } else {
      if (!warm) {
        std::fill(weights, weights + d * k, D(0));
        if (center) std::copy(center, center + d * k, weights);
      }
      auto ctx = sdca::make_context(sdca::make_input_feature(d, n, input), std::move(out),
                                    std::move(obj), dual, weights);
      ctx.primal_initial = center;
      solve(ctx, options, history, result);
    }
  }
};

// One coordinate step per row: scores z_i, current block a_i -> new block.
template <typename D>
struct update_request {
  std::size_t n, k;
  const D* logits;
  const D* previous;
  D norm2;
  projection_mode mode;
  std::size_t num_threads;
  D* result;

  template <typename Objective, typename Output>
  void run(Objective obj, Output out) {
    sdca::detail::parallel_rows(n, k, num_threads, [&](std::size_t begin, std::size_t end) {
      std::vector<D> z(k);
      for (std::size_t i = begin; i < end; ++i) {
        std::copy(logits + i * k, logits + (i + 1) * k, z.begin());
        std::copy(previous + i * k, previous + (i + 1) * k, result + i * k);
#ifndef SDCA_BASELINE
        sdca::update_dual_variables(i, k, norm2, out, obj, result + i * k, z.data(), mode);
#else
        sdca::update_dual_variables(i, k, norm2, out, obj, result + i * k, z.data());
#endif
      }
    });
  }
};

template <typename D, typename R>
void fit(int objective, std::size_t n, std::size_t d, std::size_t k, const void* x,
         const int64_t* labels, const int64_t* offsets, double c, std::size_t topk, double gamma,
         const fit_options& opt, bool kernel, bool warm, const void* center, void* w, void* a,
         double* history, sdca_fit_result* result) {
  check_blas_sizes({n, d, k, extent(n, k), extent(d, k)});
  extent(extent(n, k), sizeof(D));
  extent(extent(n, kernel ? n : d), sizeof(D));
  extent(n, sizeof(int64_t));
  check_buffers({x, labels, a, history});
  if (!kernel) check_buffers({w});
  check_objective(objective, n, k, labels, offsets, c, topk, gamma);
  if (d < 1 || opt.eval_epoch < 1) throw std::invalid_argument("Invalid fit dimensions");
  if (!(opt.tol >= 0) || !std::isfinite(opt.tol))
    throw std::invalid_argument("tol must be finite and nonnegative");
  extent(extent(d, k), sizeof(D));
  check_finite<D>(x, extent(n, kernel ? n : d), "Inputs must be finite");
  if (center && !kernel) check_finite<D>(center, extent(d, k), "The center must be finite");
  if (warm) {
    check_finite<D>(a, extent(n, k), "Warm-start duals must be finite");
    if (!kernel) check_finite<D>(w, d * k, "Warm-start weights must be finite");
  }
  // At most epochs / eval_epoch + 2 evaluations (stopping_criteria::eval_epoch).
  if (opt.history_rows < 2 || opt.epochs / opt.eval_epoch > opt.history_rows - 2)
    throw std::invalid_argument("The history needs epochs / eval_epoch + 2 rows");
  fit_request<D> request = {n,
                            d,
                            k,
                            static_cast<const D*>(x),
                            kernel,
                            warm,
                            static_cast<const D*>(center),
                            static_cast<D*>(w),
                            static_cast<D*>(a),
                            opt,
                            history,
                            result};
  dispatch<D, R>(objective, n, k, labels, offsets, c, topk, gamma, request);
}

template <typename D, typename R>
void update_objective(int objective, std::size_t n, std::size_t k, const void* z,
                      const int64_t* labels, const int64_t* offsets, const void* previous, double c,
                      double norm2, std::size_t topk, double gamma, projection_mode mode,
                      std::size_t num_threads, void* result) {
  const std::size_t size = extent(n, k);
  extent(size, sizeof(D));
  extent(n, sizeof(int64_t));
  check_buffers({z, labels, previous, result});
  check_objective(objective, n, k, labels, offsets, c, topk, gamma);
  const D *logits = static_cast<const D*>(z), *old = static_cast<const D*>(previous);
  check_finite_rows(logits, n, k, num_threads, "Logits must be finite");
  check_finite_rows(old, n, k, num_threads, "Previous duals must be finite");
  if (!(norm2 > 0) || !std::isfinite(norm2))
    throw std::invalid_argument("norm2 must be finite and positive");
  if (!std::isnormal(static_cast<D>(c)) || !std::isnormal(static_cast<D>(norm2)) ||
      !std::isnormal(static_cast<D>(c * norm2)))
    throw std::invalid_argument("C, norm2, and C*norm2 must be normal in the data type");
  std::unique_ptr<D[]> blocks(new D[size]);
  update_request<D> request = {n,    k,           logits,      old, static_cast<D>(norm2),
                               mode, num_threads, blocks.get()};
  dispatch<D, R>(objective, n, k, labels, offsets, c, topk, gamma, request);
  copy_rows(blocks.get(), n, k, num_threads, static_cast<D*>(result));
}

#ifndef SDCA_BASELINE
template <typename D, typename R>
void fit_distillation(std::size_t n, std::size_t d, std::size_t k, const void* x,
                      const double* targets, double c, const fit_options& opt, bool kernel,
                      bool warm, const void* center, void* w, void* a, double* history,
                      sdca_fit_result* result) {
  check_blas_sizes({n, d, k, extent(n, k), extent(d, k)});
  extent(extent(n, k), sizeof(D));
  extent(extent(n, k), sizeof(double));
  extent(extent(n, kernel ? n : d), sizeof(D));
  check_buffers({x, targets, a, history});
  if (!kernel) check_buffers({w});
  if (n < 1 || k < 2 || d < 1 || opt.eval_epoch < 1)
    throw std::invalid_argument("Invalid fit dimensions");
  if (!(c > 0) || !std::isfinite(c))
    throw std::invalid_argument("C must be finite and positive, gamma finite and nonnegative");
  const R cc = static_cast<R>(c);
  if (!std::isnormal(cc))
    throw std::invalid_argument("C and gamma must be normal in the working precision");
  if (!(opt.tol >= 0) || !std::isfinite(opt.tol))
    throw std::invalid_argument("tol must be finite and nonnegative");
  extent(extent(d, k), sizeof(D));
  check_finite<D>(x, extent(n, kernel ? n : d), "Inputs must be finite");
  if (center && !kernel) check_finite<D>(center, extent(d, k), "The center must be finite");
  if (warm) {
    check_finite<D>(a, extent(n, k), "Warm-start duals must be finite");
    if (!kernel) check_finite<D>(w, d * k, "Warm-start weights must be finite");
  }
  if (opt.history_rows < 2 || opt.epochs / opt.eval_epoch > opt.history_rows - 2)
    throw std::invalid_argument("The history needs epochs / eval_epoch + 2 rows");
  std::vector<double> q(targets, targets + n * k);
  sdca::distribution_output out(k, n, std::move(q));
  fit_request<D> request = {n,
                            d,
                            k,
                            static_cast<const D*>(x),
                            kernel,
                            warm,
                            static_cast<const D*>(center),
                            static_cast<D*>(w),
                            static_cast<D*>(a),
                            opt,
                            history,
                            result};
  request.run(sdca::l2_distillation<D, R>(cc), std::move(out));
}

template <typename D, typename R>
void update_distillation(std::size_t n, std::size_t k, const void* z, const double* targets,
                         const void* previous, double c, double norm2, projection_mode mode,
                         std::size_t num_threads, void* result) {
  const std::size_t size = extent(n, k);
  extent(size, sizeof(D));
  extent(size, sizeof(double));
  check_buffers({z, targets, previous, result});
  if (n < 1 || k < 2) throw std::invalid_argument("Invalid objective dimensions");
  if (!(c > 0) || !std::isfinite(c))
    throw std::invalid_argument("C must be finite and positive, gamma finite and nonnegative");
  const R cc = static_cast<R>(c);
  if (!std::isnormal(cc))
    throw std::invalid_argument("C and gamma must be normal in the working precision");
  const D *logits = static_cast<const D*>(z), *old = static_cast<const D*>(previous);
  check_finite_rows(logits, n, k, num_threads, "Logits must be finite");
  check_finite_rows(old, n, k, num_threads, "Previous duals must be finite");
  if (!(norm2 > 0) || !std::isfinite(norm2))
    throw std::invalid_argument("norm2 must be finite and positive");
  if (!std::isnormal(static_cast<D>(c)) || !std::isnormal(static_cast<D>(norm2)) ||
      !std::isnormal(static_cast<D>(c * norm2)))
    throw std::invalid_argument("C, norm2, and C*norm2 must be normal in the data type");
  std::vector<double> q(targets, targets + size);
  sdca::distribution_output out(k, n, std::move(q));
  std::unique_ptr<D[]> blocks(new D[size]);
  update_request<D> request = {n,    k,           logits,      old, static_cast<D>(norm2),
                               mode, num_threads, blocks.get()};
  request.run(sdca::l2_distillation<D, R>(cc), std::move(out));
  copy_rows(blocks.get(), n, k, num_threads, static_cast<D*>(result));
}
#endif
// The caller's struct: its size must be this header's. A later library that
// appends fields will also take the sizes of the earlier headers.
template <typename Struct>
void check_struct(const Struct* value, const char* name) {
  if (!value) throw std::invalid_argument(std::string(name) + " must not be NULL");
  if (value->size != sizeof(Struct))
    throw std::invalid_argument(std::string(name) + " size must be sizeof(struct sdca_" + name +
                                ") of this header");
}

// sdca_options also takes the size of the headers before shrinking.
template <>
void check_struct(const sdca_options* value, const char* name) {
  if (!value) throw std::invalid_argument(std::string(name) + " must not be NULL");
  const std::size_t align = alignof(sdca_options),
                    before = (offsetof(sdca_options, shrinking) + align - 1) / align * align;
  if (value->size != sizeof(sdca_options) && value->size != before)
    throw std::invalid_argument(std::string(name) + " size must be sizeof(struct sdca_" + name +
                                ") of this header");
}
}  // namespace

// Precision (sdca_precision): float/float, float/double, double/double.
#define PRECISION(F, ...)                     \
  if (precision == SDCA_FLOAT32)              \
    F<float, float>(__VA_ARGS__);             \
  else if (precision == SDCA_FLOAT32_FLOAT64) \
    F<float, double>(__VA_ARGS__);            \
  else if (precision == SDCA_FLOAT64)         \
    F<double, double>(__VA_ARGS__);           \
  else                                        \
    throw std::invalid_argument("Invalid precision");

extern "C" {
#ifndef SDCA_BASELINE
int sdca_loss_gradient(const sdca_options* options, std::size_t n, std::size_t k, const void* z,
                       const uint8_t* positive, double* values, void* gradients) {
  return protect([&] {
    check_struct(options, "options");
    check_sizes(n, k);
    const sdca_options& o = *options;
    const projection_mode mode = check_projection(o.projection_precision);
    const int precision = o.precision;
    PRECISION(batch_loss_gradient, o.loss, n, k, z, positive, o.k, o.gamma, o.alpha, mode,
              o.num_threads, values, gradients)
  });
}

int sdca_entropy_forward(const sdca_options* options, std::size_t n, std::size_t k, const void* z,
                         const int64_t* labels, double* values, double* stats) {
  return protect([&] {
    check_struct(options, "options");
    check_sizes(n, k);
    const int precision = options->precision;
    PRECISION(entropy_forward, n, k, z, labels, options->num_threads, values, stats)
  });
}

int sdca_entropy_gradient(const sdca_options* options, std::size_t n, std::size_t k, const void* z,
                          const int64_t* labels, const double* stats, const double* scale,
                          void* gradients) {
  return protect([&] {
    check_struct(options, "options");
    check_sizes(n, k);
    const int precision = options->precision;
    PRECISION(entropy_gradient, n, k, z, labels, stats, scale, options->num_threads, gradients)
  });
}
#else
// The bindings resolve every entry point; the baseline headers have no losses.
int sdca_loss_gradient(const sdca_options*, std::size_t, std::size_t, const void*, const uint8_t*,
                       double*, void*) {
  return protect([] { throw std::logic_error("Losses are not built against these headers"); });
}
int sdca_entropy_forward(const sdca_options*, std::size_t, std::size_t, const void*, const int64_t*,
                         double*, double*) {
  return protect([] { throw std::logic_error("Losses are not built against these headers"); });
}
int sdca_entropy_gradient(const sdca_options*, std::size_t, std::size_t, const void*,
                          const int64_t*, const double*, const double*, void*) {
  return protect([] { throw std::logic_error("Losses are not built against these headers"); });
}
#endif
int sdca_abi_version() { return SDCA_ABI_VERSION; }
const char* sdca_version() { return SDCA_VERSION; }
const char* sdca_last_error() { return last_error.c_str(); }
std::size_t sdca_history_fields() { return history_fields; }
int sdca_update_objective(const sdca_options* options, std::size_t n, std::size_t k, const void* z,
                          const int64_t* labels, const int64_t* offsets, const void* previous,
                          double norm2, void* out) {
  return protect([&] {
    check_struct(options, "options");
    const sdca_options& o = *options;
    const projection_mode mode = check_projection(o.projection_precision);
    const int precision = o.precision;
    PRECISION(update_objective, o.loss, n, k, z, labels, offsets, previous, o.c, norm2, o.k,
              o.gamma, mode, o.num_threads, out)
  });
}
int sdca_fit(const sdca_options* options, std::size_t n, std::size_t d, std::size_t k,
             const void* x, const int64_t* labels, const int64_t* offsets, const void* center,
             void* w, void* a, double* history, std::size_t history_rows, sdca_fit_result* result) {
  return protect([&] {
    check_struct(options, "options");
    check_struct(result, "fit_result");
    extent(extent(history_rows, history_fields), sizeof(double));
    const sdca_options& o = *options;
    const fit_options opt = {o.max_epoch,
                             o.eval_epoch,
                             history_rows,
                             o.tol,
                             o.eval_on_start != 0,
                             o.recompute_primal != 0,
                             check_projection(o.projection_precision),
                             o.num_threads,
                             check_shrinking(o)};
    const int precision = o.precision;
    PRECISION(fit, o.loss, n, d, k, x, labels, offsets, o.c, o.k, o.gamma, opt, o.kernel != 0,
              o.warm != 0, center, w, a, history, result)
  });
}
#ifndef SDCA_BASELINE
int sdca_update_distillation(const sdca_options* options, std::size_t n, std::size_t k,
                             const void* z, const double* targets, const void* previous,
                             double norm2, void* out) {
  return protect([&] {
    check_struct(options, "options");
    const sdca_options& o = *options;
    const projection_mode mode = check_projection(o.projection_precision);
    const int precision = o.precision;
    PRECISION(update_distillation, n, k, z, targets, previous, o.c, norm2, mode, o.num_threads, out)
  });
}
int sdca_fit_distillation(const sdca_options* options, std::size_t n, std::size_t d, std::size_t k,
                          const void* x, const double* targets, const void* center, void* w,
                          void* a, double* history, std::size_t history_rows,
                          sdca_fit_result* result) {
  return protect([&] {
    check_struct(options, "options");
    check_struct(result, "fit_result");
    extent(extent(history_rows, history_fields), sizeof(double));
    const sdca_options& o = *options;
    const fit_options opt = {o.max_epoch,
                             o.eval_epoch,
                             history_rows,
                             o.tol,
                             o.eval_on_start != 0,
                             o.recompute_primal != 0,
                             check_projection(o.projection_precision),
                             o.num_threads,
                             check_shrinking(o)};
    const int precision = o.precision;
    PRECISION(fit_distillation, n, d, k, x, targets, o.c, opt, o.kernel != 0, o.warm != 0, center,
              w, a, history, result)
  });
}
#else
int sdca_update_distillation(const sdca_options*, std::size_t, std::size_t, const void*,
                             const double*, const void*, double, void*) {
  return protect([] { throw std::logic_error("Losses are not built against these headers"); });
}
int sdca_fit_distillation(const sdca_options*, std::size_t, std::size_t, std::size_t, const void*,
                          const double*, const void*, void*, void*, double*, std::size_t,
                          sdca_fit_result*) {
  return protect([] { throw std::logic_error("Losses are not built against these headers"); });
}
#endif
}

// The tensor entry points: storage dtypes, ignored rows and class weights
// (src/tensor_losses.h).
namespace {
#ifndef SDCA_BASELINE
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
    case SDCA_DTYPE_FLOAT16:                        \
      F<tensor::float16>(__VA_ARGS__);              \
      break;                                        \
    case SDCA_DTYPE_BFLOAT16:                       \
      F<tensor::bfloat16>(__VA_ARGS__);             \
      break;                                        \
    default:                                        \
      throw std::invalid_argument("Invalid dtype"); \
  }

std::size_t storage_size(int dtype) {
  if (dtype == SDCA_DTYPE_FLOAT32) return 4;
  if (dtype == SDCA_DTYPE_FLOAT64) return 8;
  if (dtype == SDCA_DTYPE_FLOAT16 || dtype == SDCA_DTYPE_BFLOAT16) return 2;
  throw std::invalid_argument("Invalid dtype");
}

// Whether options select float arithmetic (SDCA_FLOAT32) for this storage.
bool float_arithmetic(int precision, int dtype) {
  if (precision == SDCA_FLOAT32_FLOAT64 || precision == SDCA_FLOAT64) return false;
  if (precision != SDCA_FLOAT32) throw std::invalid_argument("Invalid precision");
  if (dtype == SDCA_DTYPE_FLOAT64)
    throw std::invalid_argument("float64 storage needs double arithmetic");
  return true;
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

// The checks of every tensor call: options, sizes and extents of z (n, k).
void check_tensor_call(const sdca_options* options, int dtype, std::size_t n, std::size_t k) {
  check_struct(options, "options");
  if (k < 2) throw std::invalid_argument("Invalid dimensions");
  storage_size(dtype);
  extent(extent(n, k), sizeof(double));
  extent(extent(n, tensor::entropy_stats + 1), sizeof(double));
}

struct cross_entropy_call {
  std::size_t n, k;
  const void* z;
  const int64_t* labels;
  int64_t ignore;
  const double* weights;
  bool fp32;
  std::size_t num_threads;
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
  sdca::detail::parallel_rows(c.n, c.k, c.num_threads, rows);
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
  sdca::detail::parallel_rows(c.n, c.k, c.num_threads, rows);
}

struct loss_rows_call {
  const sdca_options* options;
  std::size_t n, k;
  const void* z;
  const int64_t* labels;
  const uint8_t* positive;
  double* values;
  void* gradients;
};

template <typename Format, typename Data, typename Result>
void tensor_loss_rows(const loss_rows_call& c) {
  const sdca_options& o = *c.options;
  const tensor::loss_gradient_rows<Format, Data, Result> rows = {
      sdca::loss_kind(o.loss),
      c.k,
      o.k,
      o.gamma,
      o.alpha,
      check_projection(o.projection_precision),
      static_cast<const typename Format::type*>(c.z),
      c.labels,
      c.positive,
      c.values,
      static_cast<Data*>(c.gradients)};
  sdca::detail::parallel_rows(c.n, c.k, o.num_threads, rows);
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
  std::size_t num_threads;

  template <typename Format>
  tensor::distillation_logits<Format> logits() const {
    const tensor::distillation_logits<Format> x = {
        k, static_cast<const typename Format::type*>(student),
        static_cast<const typename Format::type*>(teacher), temperature};
    return x;
  }
};

// The checks of a distillation call; whether it takes float arithmetic.
bool check_distillation(const sdca_options* options, int dtype, std::size_t n, std::size_t k,
                        double temperature, double alpha) {
  check_tensor_call(options, dtype, n, k);
  extent(extent(n, tensor::distillation_stats), sizeof(double));
  if (!(std::isfinite(temperature) && temperature > 0))
    throw std::invalid_argument("temperature must be finite and positive");
  if (!(std::isfinite(alpha) && alpha > 0))
    throw std::invalid_argument("alpha must be finite and positive");
  // Float work needs a temperature float holds.
  const double tiny = double(std::numeric_limits<float>::min());
  return float_arithmetic(options->precision, dtype) && tiny <= temperature &&
         temperature <= 1 / tiny;
}

template <typename Format>
void distillation_forward(const distillation_call& c, double* values, double* stats) {
  const tensor::distillation_forward_rows<Format> rows = {c.logits<Format>(), c.fp32, values,
                                                          stats};
  sdca::detail::parallel_rows(c.n, 2 * c.k, c.num_threads, rows);
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
  sdca::detail::parallel_rows(c.n, 2 * c.k, c.num_threads, rows);
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
  sdca::detail::parallel_rows(c.n, 8 * c.k, c.num_threads, rows);
}
#endif
}  // namespace

extern "C" {
#ifndef SDCA_BASELINE
int sdca_cross_entropy_forward(const sdca_options* options, int dtype, std::size_t n, std::size_t k,
                               const void* z, const int64_t* labels, int64_t ignore_index,
                               const double* class_weights, double* values, double* stats,
                               double* totals) {
  return protect([&] {
    check_tensor_call(options, dtype, n, k);
    const bool fp32 = float_arithmetic(options->precision, dtype);
    if (n == 0) {
      if (totals) totals[0] = totals[1] = 0;
      return;
    }
    check_buffers({z, labels, values});
    const cross_entropy_call call = {
        n, k, z, labels, ignore_index, class_weights, fp32, options->num_threads};
    STORAGE(cross_entropy_forward, call, values, stats, totals)
  });
}

int sdca_cross_entropy_backward(const sdca_options* options, int dtype, std::size_t n,
                                std::size_t k, const void* z, const int64_t* labels,
                                int64_t ignore_index, const double* class_weights,
                                const double* stats, const double* scale, std::size_t scale_stride,
                                void* gradients) {
  return protect([&] {
    check_tensor_call(options, dtype, n, k);
    const bool fp32 = float_arithmetic(options->precision, dtype);
    if (scale_stride > 1) throw std::invalid_argument("scale_stride must be 0 or 1");
    if (n == 0) return;
    check_buffers({z, labels, stats, scale, gradients});
    advise_huge_pages(gradients, n * k * storage_size(dtype));
    const cross_entropy_call call = {
        n, k, z, labels, ignore_index, class_weights, fp32, options->num_threads};
    STORAGE(cross_entropy_backward, call, stats, scale, scale_stride, gradients)
  });
}

int sdca_loss_gradient_rows(const sdca_options* options, int dtype, std::size_t n, std::size_t k,
                            const void* z, const int64_t* labels, const uint8_t* positive,
                            double* values, void* gradients) {
  return protect([&] {
    check_tensor_call(options, dtype, n, k);
    const sdca_options& o = *options;
    const bool fp32 = float_arithmetic(o.precision, dtype);
    if (o.loss < 0 || o.loss > SDCA_PROXIMAL_CROSS_ENTROPY)
      throw std::invalid_argument("Unknown loss");
    check_projection(o.projection_precision);
    if (!sdca::detail::valid_loss_arguments(k, 1, o.k, o.gamma, o.alpha))
      throw std::invalid_argument("Invalid loss dimensions or parameters");
    if (!labels == !positive) throw std::invalid_argument("Pass either labels or positive");
    if (n == 0) return;
    check_buffers({z, values});
    if (gradients) advise_huge_pages(gradients, n * k * (dtype == SDCA_DTYPE_FLOAT64 ? 8 : 4));
    const loss_rows_call call = {options, n, k, z, labels, positive, values, gradients};
    STORAGE(loss_rows_of, call, fp32)
  });
}

int sdca_distillation_forward(const sdca_options* options, int dtype, std::size_t n, std::size_t k,
                              const void* student, const void* teacher, double temperature,
                              double* values, double* stats) {
  return protect([&] {
    const bool fp32 = check_distillation(options, dtype, n, k, temperature, 1);
    if (n == 0) return;
    check_buffers({student, teacher, values});
    const distillation_call call = {
        n, k, student, teacher, temperature, fp32, options->num_threads};
    STORAGE(distillation_forward, call, values, stats)
  });
}

int sdca_distillation_backward(const sdca_options* options, int dtype, std::size_t n, std::size_t k,
                               const void* student, const void* teacher, double temperature,
                               const double* stats, const double* scale, std::size_t scale_stride,
                               void* gradients) {
  return protect([&] {
    const bool fp32 = check_distillation(options, dtype, n, k, temperature, 1);
    if (scale_stride > 1) throw std::invalid_argument("scale_stride must be 0 or 1");
    if (n == 0) return;
    check_buffers({student, teacher, stats, scale, gradients});
    advise_huge_pages(gradients, n * k * storage_size(dtype));
    const distillation_call call = {
        n, k, student, teacher, temperature, fp32, options->num_threads};
    STORAGE(distillation_backward, call, stats, scale, scale_stride, gradients)
  });
}

int sdca_proximal_distillation_rows(const sdca_options* options, int dtype, std::size_t n,
                                    std::size_t k, const void* student, const void* teacher,
                                    double temperature, double alpha, double* values,
                                    void* gradients) {
  return protect([&] {
    check_distillation(options, dtype, n, k, temperature, alpha);
    if (n == 0) return;
    check_buffers({student, teacher, values, gradients});
    advise_huge_pages(gradients, n * k * (dtype == SDCA_DTYPE_FLOAT64 ? 8 : 4));
    const distillation_call call = {
        n, k, student, teacher, temperature, false, options->num_threads};
    STORAGE(proximal_distillation, call, alpha, values, gradients)
  });
}
#else
int sdca_cross_entropy_forward(const sdca_options*, int, std::size_t, std::size_t, const void*,
                               const int64_t*, int64_t, const double*, double*, double*, double*) {
  return protect([] { throw std::logic_error("Losses are not built against these headers"); });
}
int sdca_cross_entropy_backward(const sdca_options*, int, std::size_t, std::size_t, const void*,
                                const int64_t*, int64_t, const double*, const double*,
                                const double*, std::size_t, void*) {
  return protect([] { throw std::logic_error("Losses are not built against these headers"); });
}
int sdca_loss_gradient_rows(const sdca_options*, int, std::size_t, std::size_t, const void*,
                            const int64_t*, const uint8_t*, double*, void*) {
  return protect([] { throw std::logic_error("Losses are not built against these headers"); });
}
int sdca_distillation_forward(const sdca_options*, int, std::size_t, std::size_t, const void*,
                              const void*, double, double*, double*) {
  return protect([] { throw std::logic_error("Losses are not built against these headers"); });
}
int sdca_distillation_backward(const sdca_options*, int, std::size_t, std::size_t, const void*,
                               const void*, double, const double*, const double*, std::size_t,
                               void*) {
  return protect([] { throw std::logic_error("Losses are not built against these headers"); });
}
int sdca_proximal_distillation_rows(const sdca_options*, int, std::size_t, std::size_t, const void*,
                                    const void*, double, double, double*, void*) {
  return protect([] { throw std::logic_error("Losses are not built against these headers"); });
}
#endif
}
