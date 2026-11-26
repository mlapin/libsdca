#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_CORE_DRIVER_H)
#if SDCA_ISA_ID == 1
#define SDCA_CORE_DRIVER_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_CORE_DRIVER_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_CORE_DRIVER_H3 1
#else
#define SDCA_CORE_DRIVER_H4 1
#endif

// The batch driver of a loss functor (sdca/loss/row.h) on the lanes of this instruction set:
// n rows on up to num_threads threads with the same results for every thread count. It picks
// the work type from the storage type, the precision and the loss's parameters, reruns in
// double work the float-work rows that ask for it, ignores and weighs label rows, takes blocks of small rows
// from losses that have them, and writes NaN values and NaN gradient rows for invalid rows,
// which it counts.
#include <atomic>
#include <cstddef>
#include <cstdint>

#include "sdca/core/execution.h"
#include "sdca/core/lanes.h"
#include "sdca/core/storage.h"
#include "sdca/loss/row.h"
#include "sdca/utility/parallel.h"

SDCA_ISA_BEGIN
namespace core {
namespace driver {

template <typename S>
struct Work {
  typedef float type;
};
template <>
struct Work<double> {
  typedef double type;
};

template <typename Rows>
void parallel_for(std::size_t n, std::size_t row_size, std::size_t num_threads, const Rows& rows) {
  static std::atomic<double> cost(0);
  struct Call {
    static void run(void* context, std::size_t begin, std::size_t end) {
      (*static_cast<const Rows*>(context))(begin, end);
    }
  };
  const std::size_t threads = num_threads == 0 ? static_cast<std::size_t>(-1) : num_threads;
  ::sdca::detail::parallel_rows_erased(n, row_size, threads, &cost, &Call::run,
                                       const_cast<void*>(static_cast<const void*>(&rows)));
}

// Ignored rows and class weights of label targets.
template <typename Targets>
SDCA_LANE bool ignored(const Targets&, std::size_t) {
  return false;
}
SDCA_LANE bool ignored(const loss::Labels& t, std::size_t i) { return t.ignored(i); }
template <typename Targets>
SDCA_LANE double class_weight(const Targets&, std::size_t, std::size_t) {
  return 1;
}
SDCA_LANE double class_weight(const loss::Labels& t, std::size_t i, std::size_t n) {
  const int64_t y = t.labels[i];
  return t.class_weight && y >= 0 && static_cast<uint64_t>(y) < n ? t.class_weight[y] : 1.0;
}

template <typename W, typename Loss, typename S, typename G, typename Targets>
struct Rows {
  typedef Lanes<W, isa::native> A;
  const Loss& loss;
  std::size_t n, classes;
  const S* scores;
  const Targets& targets;
  double* values;
  G* grads;
  double upstream;
  const double* row_upstream;
  bool write;
  std::atomic<std::size_t>* invalid;
  std::atomic<std::size_t>* first;

  double scale(std::size_t i) const { return row_upstream ? upstream * row_upstream[i] : upstream; }

  void row(std::size_t i, loss::Scratch& scratch, std::size_t& count, std::size_t& least) const {
    G* g = write && grads ? grads + i * classes : 0;
    if (ignored(targets, i)) {
      if (g)
        for (std::size_t j = 0; j < classes; ++j) g[j] = to_storage<G>(0.0f);
      if (write && values) values[i] = 0;
      return;
    }
    const double w = loss::ClassWeights<Loss>::value ? 1.0 : class_weight(targets, i, classes);
    const loss::RowArgs<S, G> args = {scores + i * classes, g, classes, scale(i) * w};
    scratch.needs_double = false;
    double v = loss.template operator()<W, isa::native>(args, targets.row(i, classes), scratch);
    if (scratch.needs_double) {
      scratch.needs_double = false;
      v = loss.template operator()<double, isa::native>(args, targets.row(i, classes), scratch);
      if (scratch.needs_double) v = loss::Row::invalid();
    }
    if (w != 1) v *= w;
    if (v != v) {
      if (least == n) least = i;
      ++count;
      if (g)
        for (std::size_t j = 0; j < classes; ++j) g[j] = storage_nan<G>();
    }
    if (write && values) values[i] = v;
  }

  void rows(std::size_t begin, std::size_t end, loss::Scratch& scratch, std::size_t& count,
            std::size_t& least, std::false_type) const {
    for (std::size_t i = begin; i < end; ++i) row(i, scratch, count, least);
  }
  void rows(std::size_t begin, std::size_t end, loss::Scratch& scratch, std::size_t& count,
            std::size_t& least, std::true_type) const {
    if (classes > loss::BlockClasses<Loss>::value)
      return rows(begin, end, scratch, count, least, std::false_type());
    const std::size_t L = static_cast<std::size_t>(A::L);
    double scales[A::L], block_values[A::L] = {};
    bool done[A::L];
    for (std::size_t i = begin; i < end; i += L) {
      const std::size_t m = end - i < L ? end - i : L;
      for (std::size_t r = 0; r < m; ++r) scales[r] = scale(i + r), done[r] = false;
      const loss::BlockArgs<S, G> b = {
          scores + i * classes, write && grads ? grads + i * classes : 0, m, classes, scales};
      const bool taken =
          loss.template block<W, isa::native>(b, targets, i, block_values, done, scratch);
      for (std::size_t r = 0; r < m; ++r) {
        if (!taken || !done[r])
          row(i + r, scratch, count, least);
        else if (write && values)
          values[i + r] = block_values[r];
      }
    }
  }

  void operator()(std::size_t begin, std::size_t end) const {
    loss::Scratch scratch;
    std::size_t count = 0, least = n;
    rows(begin, end, scratch, count, least,
         std::integral_constant<bool, (loss::BlockClasses<Loss>::value > 0)>());
    if (count == 0) return;
    invalid->fetch_add(count);
    std::size_t seen = first->load();
    while (least < seen && !first->compare_exchange_weak(seen, least)) {
    }
  }
};

template <typename W, typename Loss, typename S, typename G, typename Targets>
kernel::Result run(const Loss& loss, const kernel::Execution& e, std::size_t n, std::size_t classes,
                   const S* scores, const Targets& t, double* values, G* grads, double upstream,
                   const double* row_upstream) {
  std::atomic<std::size_t> invalid(0), first(n);
  Rows<W, Loss, S, G, Targets> rows = {loss,  n,        classes,      scores, t,        values,
                                       grads, upstream, row_upstream, true,   &invalid, &first};
  if (e.check_rows) {
    rows.write = false;
    parallel_for(n, classes, e.num_threads, rows);
    if (invalid.load() > 0) {
      const kernel::Result r = {kernel::status_invalid_argument, invalid.load(), first.load()};
      return r;
    }
    rows.write = true;
  }
  parallel_for(n, classes, e.num_threads, rows);
  const kernel::Result r = {kernel::status_ok, invalid.load(), first.load()};
  return r;
}

}  // namespace driver

// Values (n, unless null) and gradients (n x classes, unless null) of n rows of a loss functor
// at row-major scores; the gradient of row i scaled by upstream (times row_upstream[i] unless
// null) before its one rounding into G (S, or float for 16-bit S). grads may be scores (G = S).
template <typename Loss, typename S, typename G, typename Targets>
kernel::Result evaluate(const Loss& loss, const kernel::Execution& e, std::size_t n,
                        std::size_t classes, const S* scores, const Targets& t, double* values,
                        G* grads, double upstream = 1.0, const double* row_upstream = 0) {
  typedef typename driver::Work<S>::type Default;
  if (e.precision == kernel::precision_float64)
    return driver::run<double>(loss, e, n, classes, scores, t, values, grads, upstream,
                               row_upstream);
  if (e.precision == kernel::precision_float32 && sizeof(Default) == sizeof(double)) {
    const kernel::Result r = {kernel::status_invalid_argument, 0, n};
    return r;
  }
  if (!loss::FloatParameters<Loss>::held(loss))
    return driver::run<double>(loss, e, n, classes, scores, t, values, grads, upstream,
                               row_upstream);
  return driver::run<Default>(loss, e, n, classes, scores, t, values, grads, upstream,
                              row_upstream);
}

}  // namespace core
SDCA_ISA_END

#endif  // SDCA_CORE_DRIVER_H
