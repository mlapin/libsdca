#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_LOSS_MULTILABEL_H)
#if SDCA_ISA_ID == 1
#define SDCA_LOSS_MULTILABEL_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_LOSS_MULTILABEL_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_LOSS_MULTILABEL_H3 1
#else
#define SDCA_LOSS_MULTILABEL_H4 1
#endif

// multilabel_hinge rows (sdca/loss/row.h) of a label set Y, as a mask or as sorted labels:
// max(0, 1 + max_{j not in Y} z_j - min_{i in Y} z_i), smoothed by gamma > 0.
//
// Nonsmooth: the lowest positive (from the mask's nonzero words) and the largest negative
// (lanes of the scores with the positives masked out); the gradient is scale at that negative
// and -scale at that positive where the loss is positive, split evenly among ties and halved
// at a zero margin. Smoothed: the pairwise margins 1 + z_j - z_i split as b_j + a_i, a_i = 0.5 -
// (z_i - z_0) and b_j = 0.5 + (z_j - z_0) from the first positive z_0 (or, where the largest
// margin nearly cancels, a_i = low - z_i and b_j = 1 + z_j - low), projected in double onto
// {x, y >= 0, sum x = sum y <= gamma} (core::thresholds_two_simplex); the value is (<a, x> + <b, y> - (<x, x> + <y, y>) / 2) / gamma and the gradient
// -x / gamma at the positives, y / gamma at the negatives.
//
// Masked negatives (-inf) are removed; without a negative left the loss is 0 with a zero
// gradient (masked positives too, unless every class is masked). A masked positive beside a
// negative left gives +inf and the limit of the gradient: the nonsmooth rows' as they are,
// the smoothed rows' at a stand-in score for the masked positives. A row of -inf only, NaN
// or +inf scores, an empty label set, a set of every class and labels out of range make the
// row invalid. The rows never ask for double work.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>

#include "sdca/core/project.h"
#include "sdca/loss/topk.h"

SDCA_ISA_BEGIN
namespace loss {
namespace multilabel {

// The smoothed rows recentre the margins on the lowest positive where the largest pairwise
// margin falls below this fraction of its terms (python/libsdca/_composed: the same).
const double kRecenter = 1.0 / 16;

// f(j) for every j < n with m[j] != 0, zero words of the mask skipped.
template <typename F>
SDCA_LANE void each_positive(const uint8_t* m, std::size_t n, F&& f) {
  std::size_t j = 0;
  for (; j + 8 <= n; j += 8) {
    uint64_t word;
    std::memcpy(&word, m + j, sizeof word);
    if (word == 0) continue;
    for (std::size_t i = 0; i < 8; ++i)
      if (m[j + i] != 0) f(j + i);
  }
  for (; j < n; ++j)
    if (m[j] != 0) f(j);
}

// The positives' count, their first index, how many are masked, their lowest score low (-inf
// if one is masked) at its first index and how many equal it, and the negatives' largest score
// high (-inf if every one is masked).
struct Extremes {
  std::size_t count, first, masked, at_low, lows;
  double low, high;
};

// false for an invalid row.
template <typename W, typename Isa, typename S>
SDCA_FLAT inline bool extremes(const S* z, const uint8_t* m, std::size_t n, Extremes& e) {
  typedef core::Lanes<W, Isa> A;
  const W inf = std::numeric_limits<W>::infinity();
  bool finite = true;
  W low = inf;
  e.count = 0, e.first = n, e.masked = 0, e.at_low = n, e.lows = 0;
  each_positive(m, n, [&](std::size_t j) {
    const W v = core::to_work<W>(z[j]);
    if (e.count++ == 0) e.first = j;
    finite = finite && v < inf;
    e.masked += v == -inf;
    if (v < low) low = v, e.at_low = j, e.lows = 0;
    e.lows += v == low;
  });
  if (!finite || e.count == 0 || e.count == n) return false;
  A top = A::splat(-inf), total = A::zero();
  core::each<A>(z, n, -inf, [&](const A& v, std::size_t j, int count) {
    total += v;
    top = core::max(top, core::select(core::mask_lanes<A>(m + j, count), A::splat(-inf), v));
  });
  if (!core::all(total < inf) && !core::below_infinity<A>(z, n)) return false;
  e.low = static_cast<double>(low);
  e.high = static_cast<double>(core::max_lane(top));
  return true;
}

// Mark every score at the highest negative score, including ties on positive classes.
// When exactly one class is marked its index is exact in float below 2^24 classes.
template <typename W, typename Isa, typename S, typename G>
inline std::size_t mark_highs(const RowArgs<S, G>& row, W high, std::size_t& single) {
  typedef core::Lanes<W, Isa> A;
  W lane[A::L];
  for (int i = 0; i < A::L; ++i) lane[i] = static_cast<W>(i);
  const A iota = A::load(lane);
  A count = A::zero(), at = A::zero();
  core::each<A>(row.scores, row.n, -std::numeric_limits<W>::infinity(),
                [&](const A& v, std::size_t j, int c) {
                  const A hit = core::select(v == high, A::splat(W(1)), A::zero());
                  count += hit;
                  at = core::select(hit > W(0), iota + static_cast<W>(j), at);
                  core::store(row.grad + j, hit, c);
                });
  const std::size_t hits = static_cast<std::size_t>(core::sum(count));
  if (hits == 1 && row.n < (std::size_t(1) << 24)) {
    for (int i = 0; i < A::L; ++i) {
      if (count[i] > 0) {
        single = static_cast<std::size_t>(at[i]);
        break;
      }
    }
  }
  return hits;
}

// A unique negative maximum and positive minimum need only two nonzero stores.
// Locate both winners before zeroing the buffer because the gradient may alias the scores.
template <typename W, typename Isa, typename S, typename G>
inline bool unique_gradient(const RowArgs<S, G>& row, const uint8_t* mask, const Extremes& e,
                            double factor) {
  if (e.lows != 1 || row.n < 4096) return false;
  typedef core::Lanes<W, Isa> A;
  const W high = static_cast<W>(e.high);
  const std::size_t at = core::find_at_least<A>(row.scores, 0, row.n, high);
  if (at == row.n || mask[at] || core::find_at_least<A>(row.scores, at + 1, row.n, high) != row.n)
    return false;
  std::fill_n(row.grad, row.n, core::to_storage<G>(W(0)));
  row.grad[at] = core::to_storage<G>(factor * row.scale + 0.0);
  row.grad[e.at_low] = core::to_storage<G>(-factor * row.scale + 0.0);
  return true;
}

// The gradient splits scale evenly among the negatives tied at high and -scale among the
// positives tied at low (the masked positives, where low is -inf), halved at a zero margin: the
// mean of the one-sided subgradients at a kink.
template <typename W, typename Isa, typename S, typename G>
SDCA_FLAT inline double nonsmooth_row(const RowArgs<S, G>& row, const uint8_t* m, const Extremes& e,
                                      Scratch& s) {
  typedef core::Lanes<W, Isa> A;
  const S* z = row.scores;
  const std::size_t n = row.n;
  const double h = topk::margin(e.high, e.low), value = std::max(h, 0.0);
  if (!row.grad) return value;
  if (!(h >= 0)) {
    std::fill_n(row.grad, n, core::to_storage<G>(W(0)));
    return value;
  }
  const double factor = h > 0 ? 1.0 : 0.5;
  if (std::is_same<Isa, core::isa::neon>::value && unique_gradient<W, Isa>(row, m, e, factor))
    return value;
  // The positives tied at low (from extremes where one is; read first: row.grad may be the
  // scores). One pass zeroes the gradient but for a mark at the classes at high and counts
  // them; one mark's lane holds its class (exact in W below 2^24 classes), more are found from
  // the front and cleared. Several tied classes go to the scratch: the negatives from its
  // front, the positives from its back.
  double* tied = 0;
  std::size_t lows = 0, highs = 0, low_at = e.at_low, high_at = n;
  const W high = static_cast<W>(e.high), low = static_cast<W>(e.low);
  if (e.lows > 1) {
    tied = s.doubles(n, 2);
    each_positive(m, n, [&](std::size_t j) {
      if (core::to_work<W>(z[j]) == low) tied[n - 1 - lows++] = static_cast<double>(j);
    });
  }
  const std::size_t hits = mark_highs<W, Isa>(row, high, high_at);
  if (e.lows == 1) lows = 1;
  if (high_at < n) {
    highs = 1;
    row.grad[high_at] = core::to_storage<G>(W(0));
  } else {
    if (!tied) tied = s.doubles(n, 2);
    for (std::size_t j = core::find_at_least<A>(row.grad, 0, n, W(0.5)), cleared = 0;
         j < n && cleared < hits;
         j = core::find_at_least<A>(row.grad, j + 1, n, W(0.5)), ++cleared) {
      row.grad[j] = core::to_storage<G>(W(0));
      if (m[j] == 0) tied[highs++] = static_cast<double>(j);
    }
  }
  const double up = factor * row.scale / static_cast<double>(highs);
  const double down = -factor * row.scale / static_cast<double>(lows);
  for (std::size_t i = 0; i < highs; ++i)
    row.grad[highs == 1 && high_at < n ? high_at : static_cast<std::size_t>(tied[i])] =
        core::to_storage<G>(up + 0.0);
  for (std::size_t i = 0; i < lows; ++i)
    row.grad[e.lows == 1 ? low_at : static_cast<std::size_t>(tied[n - 1 - i])] =
        core::to_storage<G>(down + 0.0);
  return value;
}

// Short nonsmooth rows fit in registers. Keep the extremal masks until the gradient
// is stored so ties and aliased output need neither scratch space nor a second read.
template <typename W, typename Isa, typename S, typename G>
inline double ten_class_row(const RowArgs<S, G>& row, const uint8_t* mask) {
  static const int chains = (16 + core::Width<W, Isa>::value - 1) / core::Width<W, Isa>::value;
  typedef core::Lanes<W, Isa, chains> A;
  const int n = 10;
  const W inf = std::numeric_limits<W>::infinity();
  const A v = core::load<A>(row.scores, n, -inf);
  const auto positives = core::mask_lanes<A>(mask, n);
  const auto negatives = core::first_lanes<W, Isa, chains>(n) & ~positives;
  if (!core::all(v < inf) || !core::any(positives) || !core::any(negatives)) return Row::invalid();
  const W high = core::max_lane(core::select(negatives, v, A::splat(-inf)));
  if (high == -inf) {
    if (!core::any(v > -inf)) return Row::invalid();
    if (row.grad) ce::zero_row(row.grad, row.n);
    return 0;
  }
  const W low = core::min_lane(core::select(positives, v, A::splat(inf)));
  const double h = topk::margin(static_cast<double>(high), static_cast<double>(low));
  if (row.grad) {
    A g = A::zero();
    if (h >= 0) {
      const double step = (h > 0 ? 1.0 : 0.5) * row.scale;
      const auto highs = negatives & (v == high), lows = positives & (v == low);
      const A ones = A::splat(W(1));
      const double up = step / static_cast<double>(core::sum(core::select(highs, ones, A::zero())));
      const double down =
          -step / static_cast<double>(core::sum(core::select(lows, ones, A::zero())));
      g = core::select(highs, A::splat(static_cast<W>(up + 0.0)),
                       core::select(lows, A::splat(static_cast<W>(down + 0.0)), A::zero()));
    }
    core::store(row.grad, g, n);
  }
  return std::max(h, 0.0);
}

// masked: the score that stands in for the masked positives (e.low).
template <typename S, typename G>
inline double smooth_row(const RowArgs<S, G>& row, const uint8_t* m, const Extremes& e,
                         double gamma, Scratch& s, double masked = -topk::kInf) {
  typedef core::thresholds<double, double*> Thresholds;
  const S* z = row.scores;
  const std::size_t n = row.n;
  auto score = [&](std::size_t j) {
    const double x = core::to_work<double>(z[j]);
    return m[j] != 0 && !(x > -topk::kInf) ? masked : x;
  };
  const double z0 = score(e.first), low = e.low, high = e.high;
  const bool recenter = std::fabs((high - low) + 1.0) <
                        kRecenter * (1.0 + std::fabs(low - z0) + std::fabs(high - z0));
  auto margin = [&](std::size_t j, bool positive) -> double {
    const double x = score(j);
    if (recenter) return positive ? low - x : topk::margin(x, low);
    return positive ? 0.5 - (x - z0) : 0.5 + (x - z0);
  };
  double *a = s.doubles(e.count, 0), *b = s.doubles(n - e.count, 1);
  std::size_t p = 0, q = 0;
  for (std::size_t j = 0; j < n; ++j) {
    if (m[j] != 0)
      a[p++] = margin(j, true);
    else if (core::to_work<double>(z[j]) > -topk::kInf)
      b[q++] = margin(j, false);
  }
  std::pair<Thresholds, Thresholds> t;
  double value;
  try {
    t = core::thresholds_two_simplex(a, a + p, b, b + q, gamma);
    value = core::dot_smooth_hinge(t.first, a, a + p, gamma) +
            core::dot_smooth_hinge(t.second, b, b + q, gamma);
  } catch (const std::exception&) {
    return Row::invalid();
  }
  if (!row.grad) return value;
  const double f = row.scale / gamma;
  for (std::size_t j = 0; j < n; ++j) {
    const bool positive = m[j] != 0;
    const double h = margin(j, positive);
    const double x = h > -topk::kInf ? core::projected(positive ? t.first : t.second, h) : 0.0;
    row.grad[j] = core::to_storage<G>((positive ? -x : x) * f + 0.0);
  }
  return value;
}

}  // namespace multilabel

// multilabel_hinge of a label mask or a label set.
struct MultilabelHinge {
  double gamma;
  MultilabelHinge() : gamma(0) {}

  template <typename W, typename Isa, typename S, typename G>
  double operator()(const RowArgs<S, G>& row, const MaskRow& m, Scratch& s) const {
    // The ten-class kernel pays off on NEON and on AVX2 with float work.
    const bool short_work =
        std::is_same<Isa, core::isa::neon>::value ||
        (std::is_same<Isa, core::isa::avx2>::value && std::is_same<W, float>::value);
    if (short_work && gamma == 0 && row.n == 10 && std::is_same<S, W>::value &&
        std::is_same<G, W>::value)
      return multilabel::ten_class_row<W, Isa>(row, m.mask);
    multilabel::Extremes e;
    if (!multilabel::extremes<W, Isa>(row.scores, m.mask, row.n, e)) return Row::invalid();
    if (!(e.high > -topk::kInf)) {
      if (e.masked == e.count) return Row::invalid();
      if (row.grad) ce::zero_row(row.grad, row.n);
      return 0;
    }
    if (e.masked > 0 && gamma > 0) {
      // The limit as the masked positives go to -inf together: below the lowest score by more
      // than 2 gamma + 1 they take the projection's whole mass, equally, as in the limit.
      W lo, hi;
      bool valid = true;
      core::extrema<core::Lanes<W, Isa>>(row.scores, row.n, lo, hi, valid);
      multilabel::Extremes stand = e;
      stand.low = static_cast<double>(lo) - (2 * gamma + 1);
      const double value = multilabel::smooth_row(row, m.mask, stand, gamma, s, stand.low);
      return value == value ? topk::kInf : value;
    }
    if (gamma > 0) return multilabel::smooth_row(row, m.mask, e, gamma, s);
    // A masked positive makes the lowest positive -inf: the margin, and the loss, +inf.
    return multilabel::nonsmooth_row<W, Isa>(row, m.mask, e, s);
  }

  template <typename W, typename Isa, typename S, typename G>
  double operator()(const RowArgs<S, G>& row, const LabelSet& y, Scratch& s) const {
    if (y.count == 0) return Row::invalid();
    uint8_t* mask = reinterpret_cast<uint8_t*>(s.floats((row.n + 3) / 4, 3));
    std::memset(mask, 0, row.n);
    for (std::size_t i = 0; i < y.count; ++i) {
      if (y.labels[i] < 0 || static_cast<uint64_t>(y.labels[i]) >= row.n) return Row::invalid();
      mask[y.labels[i]] = 1;
    }
    const MaskRow m = {mask};
    return (*this).template operator()<W, Isa>(row, m, s);
  }
};

}  // namespace loss
SDCA_ISA_END

#endif  // SDCA_LOSS_MULTILABEL_H
