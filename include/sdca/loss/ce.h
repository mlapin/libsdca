#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_LOSS_CE_H)
#if SDCA_ISA_ID == 1
#define SDCA_LOSS_CE_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_LOSS_CE_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_LOSS_CE_H3 1
#else
#define SDCA_LOSS_CE_H4 1
#endif

// Cross-entropy rows (sdca/loss/row.h).
//
// Labels: the loss log(1 + sum_{j != y} exp(z_j - z_y)) from the largest competitor m, the
// gap m - z_y and the competitors' sum s = sum_{j != y} exp(z_j - m) (one sweep, s in double),
// times the class weight w_y. Its gradient is -mass at the label and exp(z_j - m) mass / s at
// the competitors, with the competitors' mass 1 / (1 + exp(-gap) / s) formed without
// cancellation.
//
// Targets: the loss -sum_j u_j log softmax(z)_j of target weights u_j (probabilities t_j as
// torch.nn.functional.cross_entropy takes them, w_j ((1 - eps) t_j + eps / C) with class
// weights w and label smoothing eps; for labels with smoothing, w_j eps / C at every class and
// (1 - eps) w_y at the label; for multilabel_cross_entropy 1 / |Y| at the labels). It is
// sum_j u_j (m - z_j) + T log1p(tail) with T = sum_j u_j, m the largest score at its first
// class j0 and tail = sum_{j != j0} exp(z_j - m); the gradient is T softmax(z) - u, at j0
// sum_{j != j0} u_j - T tail / (1 + tail) for confident rows (tail <= 1), else T / (1 + tail)
// - u_j0.
//
// Masked classes (-inf) take the same loops: a masked competitor adds +0.0; a masked label
// beside a finite competitor gives the loss +inf and the mass 1 (a zero class weight makes the
// row invalid); a row without a finite competitor has loss 0 and a zero gradient. A masked
// class of positive target weight gives +inf, a masked label of multilabel_cross_entropy too
// (its gradient -1 / |Y| there). Rows of -inf only, NaN or +inf scores, labels out of range,
// or targets that are not finite and nonnegative are invalid. The driver takes ignored label
// rows (sdca/loss/row.h).
//
// Float work: float exponentials, double row sums. Rows rerun in double where the scaled
// gradient leaves float's range, and (targets) where a finite exp(z - m) would leave float's
// normal range.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "sdca/core/lanes.h"
#include "sdca/core/limits.h"
#include "sdca/core/math.h"
#include "sdca/core/reduce.h"
#include "sdca/core/storage.h"
#include "sdca/loss/row.h"

SDCA_ISA_BEGIN
namespace loss {
namespace ce {

const double kInf = std::numeric_limits<double>::infinity();

template <typename G>
SDCA_LANE void zero_row(G* g, std::size_t n) {
  for (std::size_t j = 0; j < n; ++j) g[j] = core::to_storage<G>(0.0f);
}

// A gradient entry x formed in double, rounded once into G: +0.0 where W rounds it to zero (at
// most half W's least subnormal, whatever the flush mode; docs/API.md, Signed zeros), a 16-bit
// G keeping the sign of a nonzero W entry.
template <typename W, typename G>
SDCA_LANE G work_entry(double x) {
  constexpr double half = static_cast<double>(std::numeric_limits<W>::denorm_min()) / 2;
  return core::to_storage<G>(std::fabs(x) <= half ? 0.0 : x);
}

// The loss log(1 + t) and the competitors' mass t / (1 + t) of t = exp(gap) sum, for sum >= 1
// (the largest competitor's term is one), on double lanes (one per row) or one row. Directly
// where exp(gap) is normal (gap from limits::exp_reduction_min on) and t below max (exp
// saturates there): t within exp's error and a rounding. Below, t = exp(gap / 2) (exp(gap / 2)
// sum), whose factor is normal down to twice that gap (and t +0.0 below it, as sum is at most
// the class count); t beyond double: from relative = gap + log(sum), the loss relative +
// log1p(exp(-relative)) and the mass 1 / (1 + exp(-relative)).
template <typename Isa, int K>
SDCA_LANE void label_statistics(const core::Lanes<double, Isa, K>& gap,
                                const core::Lanes<double, Isa, K>& sum,
                                core::Lanes<double, Isa, K>& value,
                                core::Lanes<double, Isa, K>& mass) {
  typedef core::Lanes<double, Isa, K> D;
  typedef typename D::mask_type M;
  constexpr double low = limits::exp_reduction_min<double>();
  const D t = core::exp(core::max(gap, D::splat(low))) * sum;
  value = core::log1p(t);
  mass = t / (1.0 + t);
  const M direct = (gap >= low) & (t < std::numeric_limits<double>::max());
  if (core::all(direct)) return;
  // over: t beyond double, and NaN gaps (which stay NaN there).
  const M below = gap < low, over = ~direct & ~below;
  if (core::any(below)) {
    const D half = core::exp(gap * 0.5), tail = half * (half * sum);
    value = core::select(below, core::log1p(tail), value);
    mass = core::select(below, tail / (1.0 + tail), mass);
  }
  if (core::any(over)) {
    const D relative = gap + core::log(sum), e = core::exp(-relative);
    value = core::select(over, relative + core::log1p(e), value);
    mass = core::select(over, 1.0 / (1.0 + e), mass);
  }
}
SDCA_LANE void label_statistics(double gap, double sum, double& value, double& mass) {
  core::Scalar64 v, m;
  label_statistics(core::Scalar64::splat(gap), core::Scalar64::splat(sum), v, m);
  value = v[0], mass = m[0];
}

// The row of a label y (0 <= y < n) of class weight w.
template <typename W, typename Isa, typename S, typename G>
SDCA_FLAT inline double label_row(const RowArgs<S, G>& row, std::size_t y, double w, Scratch& s) {
  typedef core::Lanes<W, Isa> A;
  if (!(w >= 0 && w < kInf)) return Row::invalid();
  const S* z = row.scores;
  W top;
  bool valid = true;
  // The gradient of a single block reuses the exponentials of the normalization.
  W* saved = row.grad && row.n <= core::kBlock ? s.buffer<W>(row.n, 0) : 0;
  const double sum = core::max_exp_sum<A>(z, row.n, y, top, valid, saved);
  const double label = static_cast<double>(core::to_work<W>(z[y]));
  if (!valid || !(label < kInf)) return Row::invalid();
  if (!(top > -std::numeric_limits<W>::infinity())) {
    if (!(label > -kInf)) return Row::invalid();
    if (row.grad) zero_row(row.grad, row.n);
    return 0;
  }
  double value, mass;
  label_statistics(static_cast<double>(top) - label, sum, value, mass);
  if (row.grad) {
    const double scale = row.scale * w, factor = mass / sum * scale;
    // A factor beyond float times 2^64: double work.
    if (!core::exp_scaled<A>(z, row.n, top, factor, row.grad, saved)) {
      s.needs_double = true;
      return 0;
    }
    row.grad[y] = work_entry<W, G>(-mass * scale);
  }
  return w * value;
}

template <typename A, typename S, typename G>
SDCA_LANE typename A::mask_type pack_label_block(const BlockArgs<S, G>& b, const Labels& t,
                                                 std::size_t first, typename A::value_type* col) {
  typedef typename A::value_type W;
  typedef typename A::mask_type M;
  typedef typename M::value_type I;
  const int L = A::L;
  const std::size_t c = b.n, rows = b.rows;
  M target = M::splat(-1);
  for (int r = 0; r < L; ++r) {
    const std::size_t row =
        static_cast<std::size_t>(r) < rows ? static_cast<std::size_t>(r) : rows - 1;
    const int64_t y = t.labels[first + row];
    if (static_cast<std::size_t>(r) < rows && !t.ignored(first + row) && y >= 0 &&
        static_cast<uint64_t>(y) < c)
      target.set(r, static_cast<I>(y));
    const S* z = b.scores + row * c;
    for (std::size_t j = 0; j < c; ++j) col[j * L + r] = core::to_work<W>(z[j]);
  }
  return target;
}

template <typename A, typename S, typename G, typename D>
SDCA_LANE void store_label_block(const BlockArgs<S, G>& b, const bool* done,
                                 typename A::value_type* col, const typename A::mask_type& target,
                                 const A& factors, const A& powers, bool unit, const D& mass,
                                 const double* weight) {
  typedef typename A::value_type W;
  const int L = A::L;
  const std::size_t c = b.n, rows = b.rows;
  for (std::size_t j = 0; j < c; ++j) {
    const A e = A::load(col + j * L) * factors;
    (unit ? e + W(0) : e * powers + W(0)).store(col + j * L);
  }
  for (std::size_t r = 0; r < rows; ++r) {
    if (!done[r]) continue;
    const int i = static_cast<int>(r);
    G* g = b.grad + r * c;
    for (std::size_t j = 0; j < c; ++j) g[j] = core::to_storage<G>(col[j * L + r]);
    const double w = weight ? weight[target[i]] : 1.0;
    g[target[i]] = work_entry<W, G>(-mass[i] * (b.scale[r] * w));
  }
}

// label_row of b.rows rows at once, a row per lane: the rows by column into scratch, then
// per class one vector of the rows' scores. done[r] where the block decides row r: a valid
// label (not ignore), finite scores (masked competitors allowed), a finite label beside a
// finite competitor, and a gradient factor within core::Wide's range.
template <typename W, typename Isa, typename S, typename G>
SDCA_FLAT inline void label_block(const BlockArgs<S, G>& b, const Labels& t, std::size_t first,
                                  double* values, bool* done, Scratch& s) {
  const double* weight = t.class_weight;
  typedef core::Lanes<W, Isa> A;
  typedef typename A::mask_type M;
  typedef typename M::value_type I;
  typedef typename core::Accumulator<A>::lanes_type D;
  const W inf = std::numeric_limits<W>::infinity();
  const int L = A::L;
  const std::size_t c = b.n, rows = b.rows;
  W* col = s.buffer<W>(c * static_cast<std::size_t>(L), 2);
  const M target = pack_label_block<A>(b, t, first, col);
  A top = A::splat(-inf), label = A::zero(), bad = A::zero();
  for (std::size_t j = 0; j < c; ++j) {
    const A v = A::load(col + j * L);
    const M at = target == static_cast<I>(j);
    bad += v;
    label = core::select(at, v, label);
    top = core::max(top, core::select(at, A::splat(-inf), v));
  }
  core::Accumulator<A> sum;
  for (std::size_t j = 0; j < c; ++j) {
    const A e = core::exp_nonpositive_wide(
        core::select(target == static_cast<I>(j), A::splat(-inf), A::load(col + j * L)) - top);
    sum.add(e);
    // The gradient scales these same competitor exponentials after normalization.
    if (b.grad) e.store(col + j * L);
  }
  const D sums = sum.lanes() * double(core::Wide<W>::unwide()), tops = core::to_double(top);
  D value, mass;
  label_statistics(tops - core::to_double(label), sums, value, mass);
  A factors = A::zero(), powers = A::splat(W(1));
  bool any_done = false, unit = true;
  for (std::size_t r = 0; r < rows; ++r) {
    const int i = static_cast<int>(r);
    done[r] = target[i] >= 0 && bad[i] < inf && label[i] > -inf && top[i] > -inf;
    if (!done[r]) continue;
    const double w = weight ? weight[target[i]] : 1.0;
    if (!(w >= 0 && w < kInf)) {
      done[r] = false;
      continue;
    }
    values[r] = w * value[i];
    if (!b.grad) continue;
    W f, power;
    if (!core::Wide<W>::split(mass[i] / sums[i] * (b.scale[r] * w), f, power)) {
      done[r] = false;
      continue;
    }
    factors.set(i, f);
    if (power != W(1)) powers.set(i, power), unit = false;
    any_done = true;
  }
  if (!b.grad || !any_done) return;
  store_label_block(b, done, col, target, factors, powers, unit, mass, weight);
}

// Target weights u_j of probabilities t (class weights w unless null, smoothing eps).
// The lowest target and the lane sums of the targets of a row: nonnegative, and finite where
// the sums are (else entry by entry).
template <typename A>
struct TargetCheck {
  A low, total;
  TargetCheck() : low(A::zero()), total(A::zero()) {}
  SDCA_LANE void add(const A& t) {
    low = core::min(low, t);
    total += t;
  }
  bool passed() const {
    return core::min_lane(low) >= 0 &&
           core::all(total < std::numeric_limits<typename A::value_type>::infinity());
  }
};

template <typename W, typename P>
struct ProbabilityWeights {
  const P* t;
  const double* w;
  W keep, even;
  bool smooth;
  static const bool checked = true, anchored = false;
  template <typename A>
  SDCA_LANE A lanes(std::size_t j, int count, TargetCheck<A>* check) const {
    A u = core::load<A>(t + j, count, W(0));
    if (check) check->add(u);
    if (smooth) u = core::head(u * keep + even, count);
    if (w) u = u * core::load<A>(w + j, count, W(0));
    return u;
  }
  // Whether every target is finite and nonnegative.
  bool valid(std::size_t n) const {
    for (std::size_t j = 0; j < n; ++j) {
      const double x = core::to_work<double>(t[j]);
      if (!(x >= 0 && x < kInf)) return false;
    }
    return true;
  }
  SDCA_LANE W at(std::size_t j) const {
    W u = core::to_work<W>(t[j]);
    if (smooth) u = u * keep + even;
    if (w) u = u * static_cast<W>(w[j]);
    return u;
  }
};

// Target weights of a label y with smoothing eps: w_j eps / C, plus (1 - eps) w_y at y.
template <typename W>
struct SmoothedLabelWeights {
  std::size_t y;
  const double* w;
  W keep, even;
  static const bool checked = false, anchored = false;
  bool valid(std::size_t) const { return true; }
  W label() const { return (w ? static_cast<W>(w[y]) : W(1)) * keep; }
  template <typename A, typename C>
  SDCA_LANE A lanes(std::size_t j, int count, C*) const {
    const A u = w ? core::load<A>(w + j, count, W(0)) * even : core::head(A::splat(even), count);
    return core::with_lane(u, j, y, at(y));
  }
  SDCA_LANE W at(std::size_t j) const {
    const W u = (w ? static_cast<W>(w[j]) : W(1)) * even;
    return j == y ? u + label() : u;
  }
};

// expm1(z - a) of z - a = d + r exactly, r taken to first order: the result keeps its relative
// accuracy as z nears a.
SDCA_LANE double anchored_expm1(double z, double a) {
  if (!(z > -kInf)) return -1;
  const double d = z - a, back = d - z;
  const double r = (z - (d - back)) + (-a - back);
  const double m = std::expm1(d);
  return m + (m + 1.0) * r;
}

// f(j) for every j < n with m[j] != 0, skipping zero words of the mask.
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

// Target weights 1 at the labels of a mask (multilabel_cross_entropy, normalized by the
// caller).
template <typename W>
struct MaskWeights {
  const uint8_t* m;
  static const bool checked = false;
  bool valid(std::size_t) const { return true; }
  template <typename A, typename C>
  SDCA_LANE A lanes(std::size_t j, int count, C*) const {
    return core::select(core::mask_lanes<A>(m + j, count), A::splat(W(1)), A::zero());
  }
  SDCA_LANE W at(std::size_t j) const { return m[j] != 0 ? W(1) : W(0); }

  // The gradient entries of the p positives (index, value) into out (2 p doubles of s), each
  // softmax_j - 1 / p times scale without its cancellation: on the scale of the anchor
  // a = top + log(den / p), where u_j = exp(z_j - a) sum to U = p + S near p, the entry is
  // ((u_j - 1) - S / p) / U, with S the sum of the positives' u_j - 1 = expm1(z_j - a) (z_j - a
  // split exactly) and the negatives' u_j. den and negs: the sums of exp(z - top) over all
  // classes and over the negatives.
  static const bool anchored = true;
  template <typename S>
  std::size_t anchored_entries(const S* z, std::size_t n, double top, double den, double negs,
                               double p, double scale, Scratch& s, double*& out) const {
    out = s.doubles(2 * static_cast<std::size_t>(p), 1);
    const double a = top + std::log(den / p);
    double sum = (anchored_expm1(top, a) + 1) * negs, carry = 0;
    std::size_t k = 0;
    each_positive(m, n, [&](std::size_t j) {
      const double e = anchored_expm1(core::to_work<double>(z[j]), a);
      out[2 * k] = static_cast<double>(j), out[2 * k + 1] = e;
      ++k;
      const double y = e - carry, t = sum + y;
      carry = (t - sum) - y;
      sum = t;
    });
    const double w = sum / p, total = p + sum;
    for (std::size_t i = 0; i < k; ++i) out[2 * i + 1] = (out[2 * i + 1] - w) / total * scale + 0.0;
    return k;
  }
};

template <typename U, typename S>
SDCA_LANE std::size_t anchored_dispatch(const U& u, const S* z, std::size_t n, double top,
                                        double den, double negs, double p, double scale, Scratch& s,
                                        double*& out, std::true_type) {
  return u.anchored_entries(z, n, top, den, negs, p, scale, s, out);
}
template <typename U, typename S>
SDCA_LANE std::size_t anchored_dispatch(const U&, const S*, std::size_t, double, double, double,
                                        double, double, Scratch&, double*&, std::false_type) {
  return 0;
}

// The positives' entries of anchored target weights (MaskWeights), else none.
template <typename U, typename S>
SDCA_LANE std::size_t anchored(const U& u, const S* z, std::size_t n, double top, double den,
                               double negs, double p, double scale, Scratch& s, double*& out) {
  return anchored_dispatch(u, z, n, top, den, negs, p, scale, s, out,
                           std::integral_constant<bool, U::anchored>());
}

// Whether the work type holds a target row's factor (target mass over 1 + tail) and scale,
// float within its range (else double work); f and power: the core::Wide split of q's factor.
template <typename W>
inline bool target_factors(double factor, double scale, double q_factor, W& f, W& power) {
  return core::Wide<W>::split(q_factor, f, power) &&
         (sizeof(W) == sizeof(double) || (core::fits_float(factor) && core::fits_float(scale)));
}

// The first largest class's entry T q - u of a target row, q = 1 / (1 + tail): where q >= 1/2
// (tail <= 1) from the others' mass, others - T tail / (1 + tail), which stays nonzero beyond
// the precision cliff (Barron et al., 2026); elsewhere T q - u, whose terms are at most T / 2 and
// do not cancel the others' mass. Either is within a few roundings of the larger of T q and u.
SDCA_LANE double first_entry(double weight, double others, double tail, double u) {
  const double share = weight / (1 + tail), confident = others - tail * share;
  return tail <= 1 ? confident : share - u;
}

// The targets but the first largest class's (index first) of a float-work row, summed apart:
// T less u_j0 errs by the rounding of T's float sums, which is more than twice that of their own
// sum where u_j0 > T / 2.
template <typename W, typename Isa, typename U>
SDCA_LANE double target_others(const U& u, std::size_t n, std::size_t first) {
  typedef core::Lanes<W, Isa> A;
  ce::TargetCheck<A>* const unchecked = 0;
  core::Accumulator<A> others;
  for (std::size_t j = 0; j < n; j += A::L) {
    const int count = n - j < static_cast<std::size_t>(A::L) ? static_cast<int>(n - j) : A::L;
    others.add(core::with_lane(u.template lanes<A>(j, count, unchecked), j, first, W(0)));
  }
  return others.value();
}

// target_others of a block row of c classes.
template <typename U>
inline double block_others(const U& u, std::size_t c, std::size_t first) {
  double others = 0;
  for (std::size_t j = 0; j < c; ++j)
    if (j != first) others += static_cast<double>(u.at(j));
  return others;
}

// Statistics used in the value and, when requested, its gradient. tail excludes the
// first largest class; rest is the target mass outside that class.
struct TargetSums {
  double tail, weight, rest, gaps, negatives;
};

template <typename A, typename S, typename G, typename U>
SDCA_LANE typename std::enable_if<!U::anchored, bool>::type target_sums(
    const RowArgs<S, G>& row, const U& u, typename A::value_type top, std::size_t first,
    TargetSums& st, Scratch& s, typename A::value_type* saved) {
  typedef typename A::value_type W;
  const W inf = std::numeric_limits<W>::infinity();
  const bool float_work = sizeof(W) < sizeof(double);
  const S* z = row.scores;
  const std::size_t n = row.n;
  // The targets but the first largest class's: their own sum in double work, else T less u_j0
  // (target_others where u_j0 > T / 2).
  core::Accumulator<A> tail, mass, others, gap;
  ce::TargetCheck<A> check;
  // The sums of exp(z - top) over core::Wide::unwide(): exact terms, also below float's normal
  // range.
  // always_inline: GCC's inline-unit-growth limit can keep flatten from inlining the lambdas.
  core::each<A>(
      z, n, -inf, [&](const A& v, std::size_t j, int count) __attribute__((always_inline)) {
        const A t = u.template lanes<A>(j, count, &check);
        const A e = core::exp_nonpositive_wide(v - top);
        if (saved) core::store(saved + j, e, count);
        tail.add(core::with_lane(e, j, first, W(0)));
        mass.add(t);
        if (!float_work) others.add(core::with_lane(t, j, first, W(0)));
        gap.add(core::select(t == W(0), A::zero(), t * (top - v)));
      });
  // Valid targets whose float lane sums overflow: double work.
  if (U::checked && !check.passed()) {
    if (!u.valid(n)) return false;
    if (float_work) {
      s.needs_double = true;
      return false;
    }
  }
  const double unwide = static_cast<double>(core::Wide<W>::unwide());
  st.tail = tail.value() * unwide;
  st.weight = mass.value();
  st.rest = float_work ? st.weight - static_cast<double>(u.at(first)) : others.value();
  double gaps = gap.value();
  if (!(gaps < kInf)) {
    gaps = 0;
    for (std::size_t j = 0; j < n; ++j) {
      const double t = static_cast<double>(u.at(j));
      if (t != 0) {
        const double score = core::to_work<double>(z[j]);
        if (score == -kInf) {
          // A positive target mass on a masked class fixes the loss at infinity.
          gaps = kInf;
          break;
        }
        gaps += t * (static_cast<double>(top) - score);
      }
    }
  }
  st.gaps = gaps;
  return true;
}

// Count the positive classes and sum their gaps, skipping all-zero groups of mask bytes.
// The gradient also needs the negatives' probability mass for the anchored entries.
template <typename A, typename S, typename G, typename W>
SDCA_LANE bool target_sums(const RowArgs<S, G>& row, const MaskWeights<W>& u,
                           typename A::value_type top, std::size_t first, TargetSums& st, Scratch&,
                           typename A::value_type*) {
  const S* z = row.scores;
  core::Accumulator<A> mass, gap;
  core::each<A>(z, row.n, top, [&](const A& v, std::size_t j, int n) {
    static const uint8_t zeros[A::L] = {};
    if (std::memcmp(u.m + j, zeros, static_cast<std::size_t>(n)) != 0) {
      const auto positive = core::mask_lanes<A>(u.m + j, n);
      mass.add(core::select(positive, A::splat(W(1)), A::zero()));
      gap.add(core::select(positive, top - v, A::zero()));
    }
  });
  st.weight = mass.value();
  if (!(st.weight > 0)) return false;
  st.gaps = gap.value();
  if (!(st.gaps < kInf)) {
    // A masked positive gives +inf. Retry finite gaps that overflowed the work type in double.
    bool masked = false;
    core::row_sum<W, double> precise;
    each_positive(u.m, row.n, [&](std::size_t j) {
      const double score = core::to_work<double>(z[j]);
      if (score == -kInf)
        masked = true;
      else
        precise.add(static_cast<double>(top) - score);
    });
    st.gaps = masked ? kInf : precise.value();
    if (!masked && !(st.gaps < kInf)) return false;
  }
  st.rest = st.weight - static_cast<double>(u.at(first));
  if (!row.grad) {
    st.tail = core::exp_sum_except<A>(z, row.n, first, top);
    return true;
  }
  core::Accumulator<A> tail, negs;
  core::each<A>(
      z, row.n, -std::numeric_limits<W>::infinity(),
      [&](const A& v, std::size_t j, int n) __attribute__((always_inline)) {
        const A e = core::exp_nonpositive_wide(v - top);
        tail.add(core::with_lane(e, j, first, W(0)));
        negs.add(core::select(core::mask_lanes<A>(u.m + j, n), A::zero(), e));
      });
  const double unwide = static_cast<double>(core::Wide<W>::unwide());
  st.tail = tail.value() * unwide;
  st.negatives = negs.value() * unwide;
  return true;
}

// Replace the entries most vulnerable to cancellation after the vector gradient pass.
template <typename W, typename Isa, typename S, typename G, typename U>
SDCA_LANE void refine_target_gradient(const RowArgs<S, G>& row, const U& u, const TargetSums& st,
                                      std::size_t first, double scale, std::size_t positives,
                                      const double* refined) {
  const std::size_t n = row.n;
  const double weight = st.weight, sum = st.tail;
  const double at_first = static_cast<double>(u.at(first));
  double rest = st.rest;
  if (!U::anchored && sizeof(W) < sizeof(double) && !(rest >= weight / 2))
    rest = target_others<W, Isa>(u, n, first);
  row.grad[first] = work_entry<W, G>(first_entry(weight, rest, sum, at_first) * scale);
  for (std::size_t i = 0; i < positives; ++i)
    row.grad[static_cast<std::size_t>(refined[2 * i])] = work_entry<W, G>(refined[2 * i + 1]);
}

// A scratch row bounded as in label_row. Reusing the float exponentials pays for
// the cancellation test, and delaying the stores leaves aliased scores available.
template <typename W, typename S, typename G, typename U>
SDCA_LANE W* probability_buffer(const RowArgs<S, G>&, const U&, W, Scratch&) {
  return 0;
}
SDCA_LANE float* probability_buffer(const RowArgs<bfloat16>& row,
                                    const ProbabilityWeights<float, bfloat16>& u, float,
                                    Scratch& s) {
  return row.grad && row.n <= core::kBlock && !u.w && !u.smooth ? s.floats(row.n) : 0;
}

// The double partition of a row with a cancelling gradient. Near-equal scores
// keep the perturbation of the partition about the number of finite classes:
// exp(z - m) alone would round a sufficiently small nonzero difference away.
struct ProbabilityPartition {
  double denominator, tail, perturbation, weight, rest, count;
  bool centered;

  template <typename D, typename S, typename G, typename U>
  SDCA_LANE void compute(const RowArgs<S, G>& row, const U& u, double top, std::size_t first) {
    rest = count = perturbation = 0;
    centered = true;
    for (std::size_t j = 0; j < row.n; ++j) {
      if (j != first) rest += static_cast<double>(u.at(j));
      const double z = core::to_work<double>(row.scores[j]);
      if (z == -kInf) continue;
      count += 1.0;
      if (z - top < -0.0625) centered = false;
    }
    weight = rest + static_cast<double>(u.at(first));
    if (centered) {
      for (std::size_t j = 0; j < row.n; ++j) {
        const double z = core::to_work<double>(row.scores[j]);
        if (z != -kInf && z != top) perturbation += std::expm1(z - top);
      }
      denominator = count + perturbation;
    } else {
      tail = core::exp_sum_except<D>(row.scores, row.n, first, top);
      denominator = 1.0 + tail;
    }
  }

  SDCA_LANE double gradient(double shift, double target, bool first) const {
    if (centered) {
      const double centered_mass = std::fma(-count, target, weight);
      const double change = std::fma(weight, std::expm1(shift), -target * perturbation);
      return (centered_mass + change) / denominator;
    }
    if (first) return std::fma(-target, tail, rest) / denominator;
    return std::fma(std::exp(shift), weight / denominator, -target);
  }
};

template <bool Scaled, typename A, typename U>
SDCA_LANE bool probability_differences(const U& u, std::size_t n, std::size_t first,
                                       typename A::value_type factor, typename A::value_type c,
                                       typename A::value_type threshold,
                                       typename A::value_type* saved) {
  typedef typename A::value_type W;
  TargetCheck<A>* const unchecked = 0;
  typename A::mask_type close = A::zero() < W(0);
  core::each<A>(saved, n, W(0), [&](const A& e, std::size_t j, int count) {
    const A t = u.template lanes<A>(j, count, unchecked);
    const A target = Scaled ? t * c : t;
    const A difference = e * factor - target;
    const A magnitude = Scaled ? core::abs(target) : target;
    close =
        close | (core::abs(difference) < core::with_lane(magnitude, j, first, W(0)) * threshold);
    core::store(saved + j, difference, count);
  });
  return core::any(close);
}

// The ordinary gradient keeps its work-type arithmetic. Four float epsilons in
// the softmax contribution can span one BF16 epsilon in the residual once the
// terms agree within 2^-14. Only such coordinates use the double row partition.
template <typename A, typename Isa, typename S, typename G, typename U>
SDCA_LANE void sparse_probability_gradient(const RowArgs<S, G>& row, const U& u,
                                           typename A::value_type top, std::size_t first,
                                           const TargetSums& st, typename A::value_type factor,
                                           double scale, typename A::value_type* saved) {
  typedef typename A::value_type W;
  const W threshold = W(1.0 / 16384), c = static_cast<W>(scale);
  const bool large = std::fabs(scale) > 1;
  const bool close =
      large ? probability_differences<true, A>(u, row.n, first, factor, c, threshold, saved)
            : probability_differences<false, A>(u, row.n, first, factor, c, threshold, saved);
  const double correction_scale = large ? scale : 1.0;
  const W work_scale = static_cast<W>(correction_scale), final_scale = large ? W(1) : c;
  const double at_first = static_cast<double>(u.at(first));
  const double rest = st.rest >= st.weight / 2 ? st.rest : target_others<W, Isa>(u, row.n, first);
  double first_gradient = first_entry(st.weight, rest, st.tail, at_first);
  const double flux = at_first * st.tail;
  const bool close_first = std::fabs(rest - flux) < flux * static_cast<double>(threshold);
  if (close || close_first) {
    ProbabilityPartition precise = {};
    precise.template compute<core::Lanes<double, Isa>>(row, u, static_cast<double>(top), first);
    if (close_first) first_gradient = precise.gradient(0, at_first, true);
    for (std::size_t j = 0; j < row.n; ++j) {
      const W t = u.at(j);
      if (j != first && std::fabs(saved[j]) < std::fabs(t * work_scale) * threshold) {
        const double shift = core::to_work<double>(row.scores[j]) - static_cast<double>(top);
        saved[j] = static_cast<W>(precise.gradient(shift, static_cast<double>(t), false) *
                                  correction_scale);
      }
    }
  }
  core::each<A>(saved, row.n, W(0), [&](const A& value, std::size_t j, int count) {
    core::store(row.grad + j, value * final_scale + W(0), count);
  });
  row.grad[first] = work_entry<W, G>(first_gradient * scale);
}

template <typename W>
SDCA_LANE bool ordinary_probability_scale(W power, double weight, double scale, double value) {
  return power == W(1) && value < kInf &&
         weight * std::fabs(scale) < static_cast<double>(std::numeric_limits<float>::max());
}

// Keep an underflow of the requested work type positive when a wider exponential
// recovers an amplified tail. Larger negative entries can still round to -0 in BF16.
template <typename Floor, typename A>
SDCA_LANE A work_floor(const A& x) {
  if (sizeof(Floor) >= sizeof(typename A::value_type)) return x;
  constexpr double half = static_cast<double>(std::numeric_limits<Floor>::denorm_min()) / 2;
  return core::select(core::abs(x) <= static_cast<typename A::value_type>(half), A::zero(), x);
}

// The row of target weights u: its value, and its gradient times scale.
template <typename W, typename Isa, typename Floor = W, typename S, typename G, typename U>
SDCA_FLAT inline double target_row(const RowArgs<S, G>& row, const U& u, double scale, Scratch& s) {
  typedef core::Lanes<W, Isa> A;
  const S* z = row.scores;
  const std::size_t n = row.n;
  bool valid = true;
  const W top = core::maximum<A>(z, n, valid);
  if (!valid || !(top > -std::numeric_limits<W>::infinity())) return Row::invalid();
  const std::size_t first = core::find_at_least<A>(z, 0, n, top);
  TargetSums st = {};
  W* saved = probability_buffer(row, u, top, s);
  if (!target_sums<A>(row, u, top, first, st, s, saved)) return s.needs_double ? 0 : Row::invalid();
  const double sum = st.tail, weight = st.weight;
  double value = st.gaps + weight * core::log1p(sum);
  if (U::anchored) {
    if (!(value < kInf) && st.gaps < kInf) return Row::invalid();
    value /= weight;
  }
  if (!row.grad || !(value == value)) return value;
  const double den = 1 + sum;
  double* refined = 0;
  const std::size_t positives = U::anchored ? anchored(u, z, n, static_cast<double>(top), den,
                                                       st.negatives, weight, scale, s, refined)
                                            : 0;
  if (U::anchored) scale /= weight;
  const double factor = weight / den;
  // q_j = exp(z_j - top) factor rounded once (core::Wide), times a scale beyond 1 before that
  // rounding.
  const bool large = std::fabs(scale) > 1;
  W fq, pq;
  if (!target_factors(factor, scale, large ? factor * scale : factor, fq, pq)) {
    s.needs_double = true;
    return 0;
  }
  const W c = static_cast<W>(scale);
  ce::TargetCheck<A>* const unchecked = 0;
  if (saved && ordinary_probability_scale(pq, weight, scale, value)) {
    sparse_probability_gradient<A, Isa>(row, u, top, first, st, fq, scale, saved);
    return value;
  }
  if (!large && pq == W(1))
    core::each<A>(
        z, n, top, [&](const A& v, std::size_t j, int count) __attribute__((always_inline)) {
          const A t = u.template lanes<A>(j, count, unchecked);
          const A g = (core::exp_nonpositive_wide(v - top) * fq - t) * c + W(0);
          core::store(row.grad + j, work_floor<Floor>(g), count);
        });
  else
    core::each<A>(
        z, n, top, [&](const A& v, std::size_t j, int count) __attribute__((always_inline)) {
          const A t = u.template lanes<A>(j, count, unchecked);
          const A q = (core::exp_nonpositive_wide(v - top) * fq) * pq;
          const A g = large ? q - t * c + W(0) : (q - t) * c + W(0);
          core::store(row.grad + j, work_floor<Floor>(g), count);
        });
  refine_target_gradient<Floor, Isa>(row, u, st, first, scale, positives, refined);
  return value;
}

template <typename A, typename S, typename G, typename Make>
SDCA_LANE void pack_target_block(const BlockArgs<S, G>& b, const Make& make,
                                 typename A::value_type* col, typename A::value_type* ucol) {
  typedef typename A::value_type W;
  typedef typename Make::type U;
  const int L = A::L;
  const std::size_t c = b.n, rows = b.rows;
  for (int r = 0; r < L; ++r) {
    const std::size_t row =
        static_cast<std::size_t>(r) < rows ? static_cast<std::size_t>(r) : rows - 1;
    const U u = make(row);
    const S* z = b.scores + row * c;
    for (std::size_t j = 0; j < c; ++j) {
      col[j * L + r] = core::to_work<W>(z[j]);
      ucol[j * L + r] = u.at(j);
    }
  }
}

template <typename A, typename D>
struct TargetBlockGradient {
  A top, factor, scale;
  typename A::mask_type first;
  D sums, masses, rest, negatives;
};

template <typename A, typename S, typename G, typename Make, typename D>
SDCA_LANE void store_target_block(const BlockArgs<S, G>& b, const Make& make, bool normalize,
                                  const bool* done, Scratch& s, const typename A::value_type* col,
                                  typename A::value_type* ucol,
                                  const TargetBlockGradient<A, D>& state) {
  typedef typename A::value_type W;
  typedef typename Make::type U;
  const bool float_work = sizeof(W) < sizeof(double);
  const int L = A::L;
  const std::size_t c = b.n, rows = b.rows;
  for (std::size_t j = 0; j < c; ++j)
    ((A::load(col + j * L) * state.factor - A::load(ucol + j * L)) * state.scale + W(0))
        .store(ucol + j * L);
  for (std::size_t r = 0; r < rows; ++r) {
    if (!done[r]) continue;
    const int i = static_cast<int>(r);
    const U u = make(r);
    const double weight = state.masses[i], sum = state.sums[i], den = 1 + sum;
    const double sc = normalize ? b.scale[r] / weight : b.scale[r];
    double* refined = 0;
    const std::size_t positives =
        U::anchored ? anchored(u, b.scores + r * c, c, static_cast<double>(state.top[i]), den,
                               state.negatives[i], weight, b.scale[r], s, refined)
                    : 0;
    G* g = b.grad + r * c;
    for (std::size_t j = 0; j < c; ++j) g[j] = core::to_storage<G>(ucol[j * L + r]);
    const std::size_t at = static_cast<std::size_t>(state.first[i]);
    const double at_u = static_cast<double>(u.at(at));
    double others_r = float_work ? weight - at_u : state.rest[i];
    if (float_work && !(others_r >= weight / 2)) others_r = block_others(u, c, at);
    g[at] = work_entry<W, G>(first_entry(weight, others_r, sum, at_u) * sc);
    for (std::size_t k = 0; k < positives; ++k)
      g[static_cast<std::size_t>(refined[2 * k])] = work_entry<W, G>(refined[2 * k + 1]);
  }
}

template <typename U, typename W>
SDCA_LANE bool valid_target_lane(W total, W top, W low, W targets, double value, double weight,
                                 bool normalize) {
  const W inf = std::numeric_limits<W>::infinity();
  return total < inf && top > -inf && (!U::checked || (low >= 0 && targets < inf)) &&
         value < kInf && (!normalize || weight > 0);
}

// target_row of b.rows rows at once, a row per lane (as label_block), the target weights of
// row r from make(r). done[r] where the block decides row r: finite scores with a finite
// largest one, valid targets, a finite value, a scale within 1 and a factor within core::Wide;
// normalize (multilabel_cross_entropy) also needs a positive
// target total, which divides the value and the gradient.
template <typename W, typename Isa, typename S, typename G, typename Make>
SDCA_FLAT inline void target_block(const BlockArgs<S, G>& b, const Make& make, bool normalize,
                                   double* values, bool* done, Scratch& s) {
  typedef core::Lanes<W, Isa> A;
  typedef typename A::mask_type M;
  typedef typename M::value_type I;
  typedef typename core::Accumulator<A>::lanes_type D;
  typedef typename Make::type U;
  const W inf = std::numeric_limits<W>::infinity();
  const bool float_work = sizeof(W) < sizeof(double);
  const int L = A::L;
  const std::size_t c = b.n, rows = b.rows;
  W* col = s.buffer<W>(2 * c * static_cast<std::size_t>(L), 2);
  W* ucol = col + c * static_cast<std::size_t>(L);
  pack_target_block<A>(b, make, col, ucol);
  A top = A::splat(-inf), total = A::zero(), low = A::zero(), targets = A::zero();
  for (std::size_t j = 0; j < c; ++j) {
    const A v = A::load(col + j * L), t = A::load(ucol + j * L);
    top = core::max(top, v);
    total += v;
    low = core::min(low, t);
    targets += t;
  }
  M at_first = M::splat(-1);
  for (std::size_t j = 0; j < c; ++j)
    at_first = core::select((at_first < 0) & (A::load(col + j * L) == top),
                            M::splat(static_cast<I>(j)), at_first);
  core::Accumulator<A> tail, mass, others, gap, negs;
  for (std::size_t j = 0; j < c; ++j) {
    const A v = A::load(col + j * L), t = A::load(ucol + j * L);
    const A e = core::exp_nonpositive_wide(v - top);
    // The input scores remain in b.scores for the per-row refinements. Reuse their
    // transposed scratch for the same exponentials needed by the gradient.
    if (b.grad) e.store(col + j * L);
    const M at = at_first == static_cast<I>(j);
    tail.add(core::select(at, A::zero(), e));
    mass.add(t);
    if (!float_work) others.add(core::select(at, A::zero(), t));
    gap.add(core::select(t == W(0), A::zero(), t * (top - v)));
    if (U::anchored) negs.add(core::select(t == W(0), e, A::zero()));
  }
  const double unwide = static_cast<double>(core::Wide<W>::unwide());
  const D sums = tail.lanes() * unwide, masses = mass.lanes(), gaps = gap.lanes();
  const D value = gaps + masses * core::log1p(sums);
  const D rest = float_work ? D::zero() : others.lanes(), negatives = negs.lanes() * unwide;
  A f = A::zero(), scale = A::zero();
  bool any_done = false;
  for (std::size_t r = 0; r < rows; ++r) {
    const int i = static_cast<int>(r);
    const double v = value[i], weight = masses[i];
    done[r] = valid_target_lane<U>(total[i], top[i], low[i], targets[i], v, weight, normalize);
    if (!done[r]) continue;
    values[r] = normalize ? v / weight : v;
    if (!b.grad) continue;
    const double sc = normalize ? b.scale[r] / weight : b.scale[r];
    const double factor = weight / (1 + sums[i]);
    // The rows of scales beyond 1 or tiny factors take target_row.
    W fq, pq;
    if (!(target_factors(factor, sc, factor, fq, pq) && pq == W(1) && std::fabs(sc) <= 1)) {
      done[r] = false;
      continue;
    }
    f.set(i, fq);
    scale.set(i, static_cast<W>(sc));
    any_done = true;
  }
  if (!b.grad || !any_done) return;
  const TargetBlockGradient<A, D> state = {top, f, scale, at_first, sums, masses, rest, negatives};
  store_target_block(b, make, normalize, done, s, col, ucol, state);
}

// The target weights of the rows of a block.
template <typename W, typename P>
struct ProbabilityRows {
  typedef ProbabilityWeights<W, P> type;
  const Probabilities<P>& t;
  std::size_t first, n;
  W keep, even;
  bool smooth;
  type operator()(std::size_t r) const {
    const type u = {t.probabilities + (first + r) * n, t.class_weight, keep, even, smooth};
    return u;
  }
};
template <typename W>
struct MaskRows {
  typedef MaskWeights<W> type;
  const Mask& t;
  std::size_t first, n;
  type operator()(std::size_t r) const {
    const type u = {t.mask + (first + r) * n};
    return u;
  }
};

}  // namespace ce

// cross_entropy of labels or probability targets, with their class weights.
struct CrossEntropy {
  static const std::size_t block_classes = 32;
  static const bool class_weights = true;
  double label_smoothing;  // eps in [0, 1]
  CrossEntropy() : label_smoothing(0) {}

  template <typename W, typename Isa, typename S, typename G>
  double operator()(const RowArgs<S, G>& row, const Label& y, Scratch& s) const {
    if (y.index < 0 || static_cast<uint64_t>(y.index) >= row.n) return Row::invalid();
    const std::size_t label = static_cast<std::size_t>(y.index);
    if (label_smoothing == 0)
      return ce::label_row<W, Isa>(row, label, y.class_weight ? y.class_weight[label] : 1.0, s);
    const W eps = static_cast<W>(label_smoothing);
    const ce::SmoothedLabelWeights<W> u = {label, y.class_weight, W(1) - eps,
                                           static_cast<W>(label_smoothing / row.n)};
    return ce::target_row<W, Isa>(row, u, row.scale, s);
  }
  template <typename W, typename Isa, typename S, typename G, typename P>
  double operator()(const RowArgs<S, G>& row, const ProbabilityRow<P>& t, Scratch& s) const {
    const W eps = static_cast<W>(label_smoothing);
    const ce::ProbabilityWeights<W, P> u = {t.p, t.class_weight, W(1) - eps,
                                            static_cast<W>(label_smoothing / row.n),
                                            label_smoothing != 0};
    return ce::target_row<W, Isa>(row, u, row.scale, s);
  }

  template <typename W, typename Isa, typename S, typename G>
  bool block(const BlockArgs<S, G>& b, const Labels& t, std::size_t first, double* values,
             bool* done, Scratch& s) const {
    if (label_smoothing != 0) return false;
    ce::label_block<W, Isa>(b, t, first, values, done, s);
    return true;
  }
  template <typename W, typename Isa, typename S, typename G, typename P>
  bool block(const BlockArgs<S, G>& b, const Probabilities<P>& t, std::size_t first, double* values,
             bool* done, Scratch& s) const {
    const W eps = static_cast<W>(label_smoothing);
    const ce::ProbabilityRows<W, P> make = {
        t, first, b.n, W(1) - eps, static_cast<W>(label_smoothing / b.n), label_smoothing != 0};
    ce::target_block<W, Isa>(b, make, false, values, done, s);
    return true;
  }
  template <typename W, typename Isa, typename S, typename G, typename Targets>
  bool block(const BlockArgs<S, G>&, const Targets&, std::size_t, double*, bool*, Scratch&) const {
    return false;
  }
};

// multilabel_cross_entropy: log sum_j exp(z_j) - mean_{y in Y} z_y for a nonempty label set
// Y, as a mask or as sorted labels.
struct MultilabelCrossEntropy {
  static const std::size_t block_classes = 32;
  template <typename W, typename Isa, typename S, typename G>
  bool block(const BlockArgs<S, G>& b, const Mask& t, std::size_t first, double* values, bool* done,
             Scratch& s) const {
    const ce::MaskRows<W> make = {t, first, b.n};
    ce::target_block<W, Isa>(b, make, true, values, done, s);
    return true;
  }
  // Label sets as the masks of the block's rows, so that both take the same arithmetic; a set
  // with a label out of range as no labels, which the block leaves to operator().
  template <typename W, typename Isa, typename S, typename G>
  bool block(const BlockArgs<S, G>& b, const LabelSets& t, std::size_t first, double* values,
             bool* done, Scratch& s) const {
    const std::size_t n = b.n;
    uint8_t* bytes = reinterpret_cast<uint8_t*>(s.floats((b.rows * n + 3) / 4, 3));
    std::memset(bytes, 0, b.rows * n);
    for (std::size_t r = 0; r < b.rows; ++r) {
      const LabelSet y = t.row(first + r, n);
      uint8_t* m = bytes + r * n;
      for (std::size_t i = 0; i < y.count; ++i) {
        if (y.labels[i] < 0 || static_cast<uint64_t>(y.labels[i]) >= n) {
          std::memset(m, 0, n);
          break;
        }
        m[y.labels[i]] = 1;
      }
    }
    const Mask mask = {bytes};
    const ce::MaskRows<W> make = {mask, 0, n};
    ce::target_block<W, Isa>(b, make, true, values, done, s);
    return true;
  }
  template <typename W, typename Isa, typename S, typename G, typename Targets>
  bool block(const BlockArgs<S, G>&, const Targets&, std::size_t, double*, bool*, Scratch&) const {
    return false;
  }
  template <typename W, typename Isa, typename S, typename G>
  double operator()(const RowArgs<S, G>& row, const MaskRow& m, Scratch& s) const {
    // The widened float exponential reaches about 2^-190. Up to scale 2^55, its
    // discarded tail stays below half a BF16 subnormal (2^-134).
    if (std::is_same<S, core::bfloat16>::value && sizeof(W) == sizeof(float) && row.grad &&
        std::fabs(row.scale) > limits::detail::pow2<double>(55) && std::isfinite(row.scale)) {
      // Only a finite gap near the widened exponential's floor can lose this tail.
      // Ignore structurally masked classes (-inf) when bounding the spread.
      W low, high;
      bool valid = true;
      core::extrema<core::Lanes<W, Isa>>(row.scores, row.n, low, high, valid);
      if (valid &&
          static_cast<double>(high) - static_cast<double>(low) > limits::log_pow2<double>(190)) {
        const ce::MaskWeights<double> u = {m.mask};
        return ce::target_row<double, Isa, W>(row, u, row.scale, s);
      }
    }
    const ce::MaskWeights<W> u = {m.mask};
    return ce::target_row<W, Isa>(row, u, row.scale, s);
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

#endif  // SDCA_LOSS_CE_H
