#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_LOSS_TOPK_H)
#if SDCA_ISA_ID == 1
#define SDCA_LOSS_TOPK_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_LOSS_TOPK_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_LOSS_TOPK_H3 1
#else
#define SDCA_LOSS_TOPK_H4 1
#endif

// Top-k rows (sdca/loss/row.h) of a label y: multiclass_hinge, topk_hinge_alpha and
// topk_hinge_beta, nonsmooth (gamma = 0) or smoothed (gamma > 0), and topk_cross_entropy.
//
// The hinges take the margins h_j = 1 + z_j - z_y of the competitors j != y in double, the
// rounding of z_j - z_y added back where the sum cancels. Nonsmooth: alpha is max(0, mean of
// the k largest margins), beta the mean of their positive parts. Margins rise with the score,
// so the k largest competitor scores give them; the gradient is scale / k at the chosen
// competitors (margins above the k-th, and the remaining places split evenly among those equal
// to it; beta: positive margins, half at a zero one; alpha: half where the mean is zero) and
// minus their sum at the label. Smoothed: the projection a of h onto gamma S,
// S the top-k simplex {p >= 0, sum p <= 1, p_j <= sum p / k} (alpha) or {0 <= p_j <= 1 / k,
// sum p <= 1} (beta; both the simplex at k = 1), solved by sdca/core/project.h on a double
// copy of h (k = 1: by Michelot's method on the margins within gamma of the largest,
// simplex_row); the value is (<h, a> - <a, a> / 2) / gamma and the gradient a / gamma at the
// competitors, minus their sum at the label.
//
// Masked competitors (-inf) are removed. With r < k competitors left, alpha has loss 0 and a
// zero gradient, and beta its loss on the r left, which is r / k times its loss at k = r and
// gamma r / k (docs/API.md, masked classes). A masked label beside a competitor left gives
// +inf and the limit of the gradient (nonsmooth_row by the scores; smoothed, smooth_row at a
// stand-in label below the cap). A row of -inf only, NaN or +inf scores, a label out of range
// and k outside [1, n) make the row invalid. The hinges never ask for double work: their
// arithmetic is double, the scans of the scores in the work type.
//
// topk_cross_entropy at k > 1 selects the k largest competitors, then sums the exponentials
// of those outside the top k relative to the k-th score. Its value and gradient follow from
// these statistics (entropy_value). Fewer than k competitors left give loss 0 and a zero gradient, a masked
// label among them too (the top-k simplex is then {0}). A masked label beside k or more takes
// the same computation: the gap is +inf, so the loss is +inf and the mass 1, and the competitors'
// entries are the capped softmax min(1 / k, exp(z_j - tau)) of the competitors left (which do
// not depend on z_y). Float work asks for double where the scaled gradient leaves float's
// range.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>

#include "sdca/core/limits.h"
#include "sdca/core/project.h"
#include "sdca/core/select.h"
#include "sdca/loss/ce.h"

SDCA_ISA_BEGIN
namespace loss {
namespace topk {

const double kInf = std::numeric_limits<double>::infinity();

// k up to this selects the k largest by insertion (core::top_k), larger k by a partial sort of
// a copy.
const std::size_t kInsertTopK = 64;

// 1 + x - y in double with the rounding errors of x - y and of the sum (TwoSum) added back:
// within an ulp of the exact margin, also where the sum cancels (infinite where it is).
SDCA_LANE double margin(double x, double y) {
  const double d = x - y, h = d + 1.0;
  if (!(std::fabs(h) < kInf)) return h;
  const double rd = d - x, rh = h - d;
  return h + (((x - (d - rd)) + (-y - rd)) + ((d - (h - rh)) + (1.0 - rh)));
}

// The margins 1 + x - y of double lanes x, as margin() (-inf where x is -inf).
template <typename D>
SDCA_LANE D margins(const D& x, double y) {
  const D d = x - y, h = d + 1.0, rd = d - x, rh = h - d;
  const D sum = h + (((x - (d - rd)) + ((D::zero() - y) - rd)) + ((d - (h - rh)) + (1.0 - rh)));
  return core::select(core::abs(h) < kInf, sum, h);
}

// The k largest scores of z[0, n) but z[y], in descending order (-inf where fewer), in
// scratch slot 0; valid turns false on NaN or +inf.
template <typename W, typename Isa, typename S>
inline const W* largest(const S* z, std::size_t n, std::size_t y, std::size_t k, Scratch& s,
                        bool& valid) {
  typedef core::Lanes<W, Isa> A;
  const W inf = std::numeric_limits<W>::infinity();
  W* top = s.buffer<W>(k > kInsertTopK ? n : k, 0);
  if (k <= kInsertTopK) {
    core::top_k<A>(z, n, y, k, top, valid);
  } else {
    A total = A::zero();
    core::each<A>(z, n, W(0), [&](const A& v, std::size_t j, int count) {
      total += v;
      v.store(top + j, count);
    });
    if (!core::all(total < inf)) valid = valid && core::below_infinity<A>(z, n);
    if (valid) {
      top[y] = -inf;
      core::sort_largest(top, k, top + n);
    }
  }
  return top;
}

// Count competitors at the score cutoff and remember one for a unique maximum.
template <typename A, typename S>
SDCA_LANE std::size_t count_cutoff(const S* z, std::size_t n, std::size_t y,
                                   typename A::value_type cutoff, std::size_t& winner) {
  std::size_t tied = 0;
  typedef typename A::value_type W;
  if (std::isfinite(cutoff))
    for (std::size_t j = core::find_at_least<A>(z, 0, n, cutoff); j < n;
         j = core::find_at_least<A>(z, j + 1, n, cutoff)) {
      if (j != y && core::to_work<W>(z[j]) == cutoff) {
        ++tied;
        winner = j;
      }
    }
  return tied;
}

// Only stripes whose SIMD lane attained the row maximum can contain a winner.
template <typename A, typename S>
SDCA_LANE std::size_t count_maxima(const S* z, std::size_t n, std::size_t y,
                                   typename A::value_type cutoff, const A& maxima,
                                   std::size_t& winner) {
  typedef typename A::value_type W;
  std::size_t tied = 0;
  // Read every tied score before an in-place gradient overwrites the input.
  for (int lane = 0; lane < A::L; ++lane) {
    if (maxima[lane] != cutoff) continue;
    for (std::size_t j = static_cast<std::size_t>(lane); j < n; j += A::L) {
      if (j != y && core::to_work<W>(z[j]) == cutoff) {
        ++tied;
        winner = j;
      }
    }
  }
  return tied;
}

SDCA_LANE double hinge_slope(double margin_value) {
  return margin_value > 0 ? 1.0 : (margin_value == 0 ? 0.5 : 0.0);
}

template <typename W>
SDCA_LANE double finite_hinge_mean(const W* top, std::size_t k, double label, bool beta,
                                   double total) {
  const W cutoff = top[k - 1];
  const double kd = static_cast<double>(k);
  if (k > 1) {
    if (beta && total == kInf) {
      // The margin can overflow before taking its share of the mean.
      total = 0;
      for (std::size_t i = 0; i < k; ++i) {
        const double x = static_cast<double>(top[i]), h = margin(x, label);
        total += h == kInf ? x / kd - label / kd + 1.0 / kd : std::max(h, 0.0) / kd;
      }
    } else if (!beta && std::isfinite(cutoff)) {
      // Keep the unit offset when the k large signed margins cancel on averaging.
      const double largest = std::fabs(static_cast<double>(top[0]));
      const double smallest = std::fabs(static_cast<double>(cutoff));
      const double magnitude = std::max(std::fabs(label), std::max(largest, smallest));
      if (!std::isfinite(total) ||
          std::fabs(total) <= kd * std::numeric_limits<double>::epsilon() * magnitude)
        total = static_cast<double>(1.0L + core::topk_entropy::entropy_mean_offset(
                                               k, [&](std::size_t i) { return top[i]; }, -label));
    }
  }
  return total;
}

struct HingeSlots {
  double mean, active;
  std::size_t places;
};

template <typename W>
SDCA_LANE HingeSlots hinge_slots(const W* top, std::size_t k, double label, bool masked,
                                 bool beta) {
  const W cutoff = top[k - 1];
  const double kd = static_cast<double>(k);
  double total = 0, active = 0;
  std::size_t places = 0;
  for (std::size_t i = 0; i < k; ++i) {
    const double h = masked ? (std::isfinite(top[i]) ? kInf : -kInf)
                            : margin(static_cast<double>(top[i]), label);
    if (!masked) total += (beta ? std::max(h, 0.0) : h) / kd;
    active += hinge_slope(h);
    places += top[i] == cutoff;
  }
  return {masked ? total : finite_hinge_mean(top, k, label, beta, total), active, places};
}

template <typename W, typename A, typename S, typename G>
SDCA_LANE void nonsmooth_gradient(const RowArgs<S, G>& row, std::size_t y, W cutoff, double label,
                                  std::size_t k, bool masked, bool beta, const HingeSlots& slots,
                                  double value, const A& maxima) {
  const S* z = row.scores;
  const std::size_t n = row.n, places = slots.places;
  const double total = slots.mean, active = slots.active, kd = static_cast<double>(k);
  const double factor = beta ? 1.0 : (masked ? static_cast<double>(value > 0) : hinge_slope(total));
  if (factor == 0) {
    ce::zero_row(row.grad, n);
    return;
  }
  // Rank by score before rounding the margins.
  std::size_t winner = 0;
  const std::size_t tied = k == 1 ? count_maxima(z, n, y, cutoff, maxima, winner)
                                  : count_cutoff<A>(z, n, y, cutoff, winner);
  const bool flat = !masked && k == 1 && margin(static_cast<double>(cutoff), label) == 0;
  const double share = flat ? 1.0 / static_cast<double>(tied + 1)
                            : (tied ? static_cast<double>(places) / static_cast<double>(tied) : 0);
  const double step = row.scale / kd;
  // Compute the label mass from the k selected slots. This stays accurate even for a
  // cutoff shared by millions of competitors.
  const double weights = flat ? static_cast<double>(tied) / static_cast<double>(tied + 1)
                              : (beta ? active : kd * factor);
  if (k == 1 && tied == 1) {
    // The unique competitor takes the entire label mass, including half a step
    // at a zero margin. Its index was found while checking for other ties.
    ce::zero_row(row.grad, n);
    row.grad[winner] = core::to_storage<G>(weights * step + 0.0);
    row.grad[y] = core::to_storage<G>(-(weights * step) + 0.0);
    return;
  }
  std::size_t begin = 0;
  // The output may alias the scores. Only overwrite entries after reading them.
  for (std::size_t j = core::find_at_least<A>(z, 0, n, cutoff); j < n;
       j = core::find_at_least<A>(z, j + 1, n, cutoff)) {
    const W score = core::to_work<W>(z[j]);
    if (j == y || !std::isfinite(score)) continue;
    const double h = masked ? kInf : margin(static_cast<double>(score), label);
    const double slope = beta ? (hinge_slope(h)) : factor;
    const double w = flat ? share : (score == cutoff ? share : 1.0) * slope;
    ce::zero_row(row.grad + begin, j - begin);
    row.grad[j] = core::to_storage<G>(step * w + 0.0);
    begin = j + 1;
  }
  ce::zero_row(row.grad + begin, n - begin);
  row.grad[y] = core::to_storage<G>(-(weights * step) + 0.0);
}

// The nonsmooth hinge of the label y of score label: finite, or -inf for the limit as z_y goes
// there, where every margin is +inf: the competitors are then ranked by their scores (the keys
// below), the loss is +inf (alpha with fewer than k left: 0), and every one left counts as
// positive.
template <typename W, typename Isa, typename S, typename G>
SDCA_FLAT inline double nonsmooth_row(const RowArgs<S, G>& row, std::size_t y, double label,
                                      std::size_t k, bool beta, Scratch& s) {
  typedef core::Lanes<W, Isa> A;
  const S* z = row.scores;
  const std::size_t n = row.n;
  bool valid = true;
  A maxima{};
  W singleton;
  const W* top;
  if (k == 1) {
    singleton = core::max_except<A>(z, n, y, valid, &maxima);
    top = &singleton;
    beta = false;  // At k == 1 both objectives are the maximum positive margin.
  } else {
    top = largest<W, Isa>(z, n, y, k, s, valid);
  }
  const bool masked = !(label > -kInf);
  if (!valid || (masked && !(top[0] > -std::numeric_limits<W>::infinity()))) return Row::invalid();
  const W cutoff = top[k - 1];
  const HingeSlots slots = hinge_slots(top, k, label, masked, beta);
  const double total = slots.mean;
  const double value =
      masked ? (beta || std::isfinite(cutoff) ? kInf : 0.0) : (!beta && total < 0.0 ? 0.0 : total);
  if (row.grad)
    nonsmooth_gradient<W, A>(row, y, cutoff, label, k, masked, beta, slots, value, maxima);
  return value;
}

// For the beta projection, if k+1 margins reach b then a threshold below b-gamma/k
// would put all k+1 at the cap and exceed gamma. The omitted margins project to zero.
// The extra competitor covers rounding the cap; round the lower bound down as well.
template <typename D>
inline double beta_lower_bound(const double* margins, std::size_t y, std::size_t k, double cap) {
  double top[9];  // k <= 8
  core::top_k<D>(margins, 64, y, k + 1, top);
  return std::max(0.0, std::nextafter(top[k] - cap, -kInf));
}

// Discard the label and inactive competitors in place. A valid row keeps at most
// n-1 margins; n signals a NaN or +inf. floor also screens the output gradient.
template <typename D>
inline std::size_t keep_margins(double* h, std::size_t n, std::size_t y, std::size_t k, bool beta,
                                double gamma, bool finite, double& floor) {
  std::size_t r = n - 1;
  if (beta && n >= 128 && k <= 8 && gamma >= 16 * std::numeric_limits<double>::min()) {
    if (!finite && !core::below_infinity<D>(h, n)) return n;
    floor = beta_lower_bound<D>(h, y, k, gamma / static_cast<double>(k));
    r = 0;
    for (std::size_t j = 0; j < n; ++j) {
      const double v = h[j];
      h[r] = v;
      r += j != y && v > floor;
    }
  } else if (finite) {
    h[y] = h[n - 1];
  } else {
    r = 0;
    for (std::size_t j = 0; j < n; ++j) {
      if (j == y) continue;
      if (!(h[j] < kInf)) return n;
      if (h[j] > -kInf) h[r++] = h[j];
    }
  }
  return r;
}

// The k = 1 smoothed hinge of an ordinary row: the projection a of the margins h onto {a >= 0,
// sum a <= gamma} (the cap gamma never binds there). Its support lies in U = {h > top - gamma}
// of the largest competitor margin top: a box clip(h, 0, gamma) of mass at most gamma has no
// positive margin outside U (one would add to clip(top, 0, gamma) = gamma), and otherwise the
// threshold is at least top - gamma. So the projection of the margins of U is exact: their box
// where its mass is below gamma, else a = max(0, (h - top) - shift) by Michelot's method from
// U, as the public kernels and triton/topk.py take it. Two scans of the scores in the work
// type W (the largest competitor, then the candidates of U, widened by a few roundings, whose
// margins decide) and a gradient of zeros but at U, without a row of margins. False leaves the
// row to the general path: a NaN or +inf score, or no finite competitor.
template <typename W, typename Isa, typename S, typename G>
SDCA_FLAT inline bool simplex_row(const RowArgs<S, G>& row, std::size_t y, double label,
                                  double gamma, Scratch& s, double& value) {
  typedef core::Lanes<W, Isa> A;
  typedef core::thresholds<double, double*> Thresholds;
  const S* z = row.scores;
  const std::size_t n = row.n;
  bool valid = true;
  const double best = static_cast<double>(core::max_except<A>(z, n, y, valid));
  if (!valid || !(best > -kInf)) return false;
  const double top = margin(best, label);
  // The candidates: scores within the cap of the best, widened by the margins' roundings.
  const double magnitude = std::max(std::max(std::fabs(best), std::fabs(label)), 1.0);
  const double wide = (best - gamma) - 8 * limits::eps<double>() * (magnitude + gamma);
  W cut = static_cast<W>(wide);
  if (static_cast<double>(cut) > wide)
    cut = std::nextafter(cut, -std::numeric_limits<W>::infinity());
  double* h = s.doubles(n, 0);
  double* at = s.doubles(n, 1);  // U's indices, exact in double
  std::size_t r = 0;
  double mass = 0;
  for (std::size_t j = core::find_at_least<A>(z, 0, n, cut); j < n;
       j = core::find_at_least<A>(z, j + 1, n, cut)) {
    const double m = margin(static_cast<double>(core::to_work<W>(z[j])), label);
    // By the gap m - top (exact near top, where top - gamma can round to top for a gamma below
    // its ulp), and every positive margin, which the box needs at a rounding of that gap.
    if (j == y || !(m - top > -gamma || m > 0)) continue;
    h[r] = m, at[r++] = static_cast<double>(j);
    mass += std::min(std::max(m, 0.0), gamma);
  }
  // Both branches give the same projection at a mass of gamma (a saturated top alone, for
  // instance), so a rounding of the mass that decides between them moves no entry by more
  // than its own rounding.
  Thresholds t(0.0, 0.0, gamma, h, h + r);
  if (mass > gamma) {
    // The set {h - top > level} only shrinks: a rounded shift cannot readmit a margin and cycle.
    double shift = -gamma, level = -gamma;
    std::size_t count = r + 1;
    for (;;) {
      std::size_t inside = 0;
      double sum = 0;
      for (std::size_t j = 0; j < r; ++j) {
        const double g = h[j] - top;
        if (g > level) ++inside, sum += g;
      }
      if (inside == count) break;
      count = inside;
      shift = (sum - gamma) / static_cast<double>(count);
      level = std::max(level, shift);
    }
    t = core::split_thresholds(top, shift, 0.0, gamma, h, h + r);
  }
  value = core::dot_smooth_hinge(t, h, h + r, gamma);
  if (!row.grad) return true;
  // Every score was read above: the gradient may overwrite them.
  ce::zero_row(row.grad, n);
  core::Neumaier total;
  for (std::size_t j = 0; j < r; ++j) {
    const double g = core::projected(t, h[j]) / gamma * row.scale + 0.0;
    total.add(g);
    row.grad[static_cast<std::size_t>(at[j])] = core::to_storage<G>(g);
  }
  row.grad[y] = core::to_storage<G>(-total.value() + 0.0);
  return true;
}

// The smoothed hinge of the label y of score label (finite), gamma > 0.
template <typename W, typename Isa, typename S, typename G>
SDCA_FLAT inline double smooth_row(const RowArgs<S, G>& row, std::size_t y, double label,
                                   std::size_t k, bool beta, double gamma, Scratch& s) {
  typedef core::Lanes<double, Isa> D;
  typedef core::thresholds<double, double*> Thresholds;
  const S* z = row.scores;
  const std::size_t n = row.n;
  double value;
  if (k == 1 && simplex_row<W, Isa>(row, y, label, gamma, s, value)) return value;
  double* h = s.doubles(n, 0);
  // The margins into h, the label's moved out; the lane sums of the margins are finite
  // unless a margin is -inf (masked), +inf or NaN, which a second pass sorts out.
  D sums = D::zero();
  core::each<D>(z, n, label, [&](const D& x, std::size_t j, int count) {
    const D m = margins(x, label);
    sums += m;
    m.store(h + j, count);
  });
  double floor = -kInf;
  const bool finite = core::all((sums > -kInf) & (sums < kInf));
  const std::size_t r = keep_margins<D>(h, n, y, k, beta, gamma, finite, floor);
  if (r == n) return Row::invalid();
  if (r == 0 || (!beta && k > 1 && r < k)) {
    if (row.grad) ce::zero_row(row.grad, n);
    return 0;
  }
  Thresholds t;
  try {
    t = beta || k == 1
            ? core::thresholds_knapsack_le(h, h + r, 0.0, gamma / static_cast<double>(k), gamma)
            : core::thresholds_topk_simplex(h, h + r, static_cast<std::ptrdiff_t>(k), gamma);
    value = core::dot_smooth_hinge(t, h, h + r, gamma);
  } catch (const std::exception&) {
    return Row::invalid();
  }
  if (!row.grad) return value;
  // Every entry projects to at least t.lo, which can be positive: masked ones are zeroed.
  const bool omitted = r < n - 1;
  core::Accumulator<D> total;
  core::each<D>(z, n, label, [&](const D& x, std::size_t j, int count) {
    const D m = margins(x, label);
    D a = core::max(core::min((m - t.origin) - t.correction, t.hi), t.lo);
    if (omitted) a = core::select(m > floor, a, D::zero());
    const D g = core::head(core::with_lane(a, j, y, 0.0), count) / gamma * row.scale + 0.0;
    total.add(g);
    core::store(row.grad + j, g, count);
  });
  row.grad[y] = core::to_storage<G>(-total.value() + 0.0);
  return value;
}

// The top-k entropy row of a label y, its score label (finite), 1 < k < n: the competitors'
// gradient entries are cap min(1, exp(z_j - first) / spread), cap = mass / k, and the label's
// is -mass.
struct Entropy {
  double first, spread, mass;
};

// The mean score gap (sum_{i < u} top[i] + (k - u) top[u]) / k - label of the u capped
// competitors and the k - u free places at the first free score, and in low the rest of it:
// the sum is exact as a double-double (TwoSum, TwoProduct) and the quotient keeps its
// remainder. A power of two scales a sum that overflows; NaN or +-inf where the mean leaves
// double (a masked label, a spread past finfo.max).
inline double capped_mean(const double* top, std::size_t u, std::size_t k, double label,
                          double& low) {
  const double kd = static_cast<double>(k), places = static_cast<double>(k - u);
  for (int exponent = 0;;) {
    const double scale = std::ldexp(1.0, -exponent);
    double high = 0, error;
    core::Neumaier correction;
    auto add = [&](double v) {
      high = core::sum_error(high, v, error);
      correction.add(error);
    };
    for (std::size_t i = 0; i < u; ++i) add(top[i] * scale);
    double product;
    add(core::two_product(top[u] * scale, places, product));
    add(product);
    add(core::two_product(-label * scale, kd, product));
    add(product);
    const double combined = core::sum_error(high, correction.hi, error);
    const double sum = core::sum_error(combined, error + correction.lo, error);
    if (!std::isfinite(sum) && exponent == 0) {
      exponent = std::ilogb(kd) + 2;
      continue;
    }
    const double mean = sum / kd;
    low = std::ldexp((std::fma(-mean, kd, sum) + error) / kd, exponent);
    return std::ldexp(mean, exponent);
  }
}

// The value and gradient statistics of a top-k entropy row from its largest competitor
// maximum = top[0], the label score, others = sum exp(z_j - maximum) over the competitors but
// the largest, the k largest competitors top (descending, finite) and tail = sum
// exp(z_j - top[k - 1]) over the competitors outside them. Uncapped (the largest competitor
// takes at most a k-th of the mass) it is cross-entropy; else the first u largest are capped
// at mass / k, u found by probing u = 1, 2, 4, ... and bisecting (being capped is monotone).
inline double entropy_value(double maximum, double label, double others, std::size_t k,
                            const double* top, double tail, Entropy& st) {
  // The tests of being capped, t = log(a) - log(b) < 0, take the error of t: the sums of
  // exponentials within 3u (an ulp per term, compensated sums), each logarithm one more
  // rounding, the difference one: below eps (|log a| + |log b| + 2), eps = 2u.
  const double kd = static_cast<double>(k), log_k = std::log(kd), eps = limits::eps<double>();
  const double sum = 1 + others, gap = maximum - label, log_sum = std::log(sum);
  const double spread = log_sum - log_k;
  double value;
  if (!(spread + eps * (std::fabs(log_sum) + log_k + 2) < 0)) {
    ce::label_statistics(gap, sum, value, st.mass);
    st.first = maximum;
    st.spread = sum / kd;
    return value;
  }
  const double last = top[k - 1];
  // The free competitors' sum of exp(z_j - top[u]) when the u largest are capped.
  auto spread_at = [&](std::size_t u) -> double {
    core::Neumaier mid;
    for (std::size_t j = u + 1; j < k; ++j) mid.add(std::exp(top[j] - top[u]));
    return mid.value() + std::exp(last - top[u]) * tail;
  };
  std::size_t lo = 1, hi = k - 1;
  double found = 0, found_log = 0;
  bool have = false;
  auto capped = [&](std::size_t u) -> bool {
    const double f = spread_at(u), l = std::log1p(f);
    const double log_free = std::log(static_cast<double>(k - u));
    if ((l - log_free) + eps * (l + log_free + 2) < 0) return true;
    found = f, found_log = l, have = true;
    return false;
  };
  for (std::size_t probe = 1; probe < hi && !have; probe *= 2) {
    if (capped(probe))
      lo = probe + 1;
    else
      hi = probe;
  }
  while (lo < hi) {
    const std::size_t u = lo + (hi - lo) / 2;
    if (capped(u))
      lo = u + 1;
    else
      hi = u;
  }
  const std::size_t u = lo;
  const double first = top[u], free_sum = have ? found : spread_at(u);
  const double lse = have ? found_log : std::log1p(free_sum);
  const double remaining = static_cast<double>(k - u), log_remaining = std::log(remaining);
  st.first = first;
  st.spread = (1 + free_sum) / remaining;
  // The odds split into the mean score gap and the free group's log-mass, as triton/topk.py
  // takes them: the mean exact to its low part, the log-mass, at most log n, as a multiplier
  // of exp(mean). Rounding their sum would lose |odds| eps / 2 near exp(-700), and the
  // differences from the maximum and their mean round at |gap| where they cancel it.
  double mean_low;
  const double mean = capped_mean(top, u, k, label, mean_low);
  if (std::isfinite(mean)) {
    // Beyond 2 log_max the low part changes neither the loss nor the mass.
    const bool in_range = std::fabs(mean) < 2 * limits::log_max<double>();
    const double factor = kd * (in_range ? std::exp(mean_low) : 1.0) *
                          std::exp(remaining / kd * (lse - log_remaining));
    ce::label_statistics(mean, factor, value, st.mass);
    return value;
  }
  // Scores beyond double's range (a masked label, a spread past finfo.max): by the
  // differences from the maximum.
  core::Neumaier head;
  for (std::size_t i = 1; i < u; ++i) head.add(top[i] - maximum);
  double base = gap;
  double offset = (head.value() + remaining * ((first - maximum) + lse - log_remaining)) / kd;
  // The score terms of the mean nearly cancel the gap where a far competitor is capped and the
  // free ones lie as far below: those terms first, then the log terms.
  const double balance = gap + (head.value() + remaining * (first - maximum)) / kd;
  if (std::fabs(balance) < 0.5 * std::fabs(gap)) {
    base = balance;
    offset = remaining * (lse - log_remaining) / kd;
  }
  const double odds = base + offset;
  double factor = kd;
  if (odds < limits::log_eps<double>() && std::isfinite(odds)) {
    // Preserve the low part of log-odds in its exponential, where the loss is a tail.
    double low;
    core::sum_error(base, offset, low);
    factor *= std::exp(low);
  }
  ce::label_statistics(odds, factor, value, st.mass);
  return value;
}

// With exactly k competitors their normalized weights are all 1/k and only their mean
// score enters the odds. Sum the scores before subtracting k times the label. Keeping the
// error of the correction preserves a small term beside two cancelling large scores.
template <typename W, typename S, typename G>
inline double all_capped(const RowArgs<S, G>& row, std::size_t y, double label) {
  const double k = static_cast<double>(row.n - 1);
  double value = 0, mass = 0;
  for (int exponent = 0;;) {
    double high = 0;
    core::Neumaier correction;
    auto add = [&](double v) {
      double low;
      high = core::sum_error(high, v, low);
      correction.add(low);
    };
    const double scale = std::ldexp(1.0, -exponent);
    std::size_t kept = 0;
    for (std::size_t j = 0; j < row.n; ++j) {
      if (j == y) continue;
      const double z = static_cast<double>(core::to_work<W>(row.scores[j]));
      if (!(z < kInf)) return Row::invalid();
      if (z > -kInf) {
        ++kept;
        add(z * scale);
      }
    }
    if (kept != row.n - 1) {
      if (!kept && label == -kInf) return Row::invalid();
      break;
    }
    if (label == -kInf) {
      value = kInf;
      mass = 1;
      break;
    }
    double product_error, low;
    add(core::two_product(-label * scale, k, product_error));
    add(product_error);
    const double combined = core::sum_error(high, correction.hi, low);
    const double sum = core::sum_error(combined, low + correction.lo, low);
    if (!std::isfinite(sum)) {
      // A power of two makes the sum of the finite scores and k times their label fit.
      exponent = std::ilogb(k) + 2;
      continue;
    }
    const double mean = sum / k;
    const double gap = std::ldexp(mean, exponent);
    // Supply the remainder of the quotient in the multiplier of exp(gap). Rounding a
    // log-odds near -700 to double alone would discard hundreds of relative epsilons.
    const double remainder = (std::fma(-mean, k, sum) + low) / k;
    const double factor = std::fabs(gap) < 2 * limits::log_max<double>()
                              ? k * std::exp(std::ldexp(remainder, exponent))
                              : k;
    ce::label_statistics(gap, factor, value, mass);
    break;
  }
  if (row.grad) {
    const double scaled = mass == 0 ? 0 : mass * row.scale;
    std::fill_n(row.grad, row.n, ce::work_entry<W, G>(scaled / k));
    row.grad[y] = ce::work_entry<W, G>(-scaled);
  }
  return value;
}

// Cache the tail only for sparse top-k selection with bounded worker scratch.
// The half-exponential path also handles larger gradient multipliers.
template <typename W, typename S, typename G>
inline W* gradient_terms(const RowArgs<S, G>& row, std::size_t k, Scratch& s) {
  const bool cache = row.grad && std::is_same<W, float>::value && std::fabs(row.scale) <= 1 &&
                     row.n >= 256 && row.n <= 32768 && k <= row.n / 32 && k <= kInsertTopK;
  return cache ? s.buffer<W>(row.n, 2) : 0;
}

// Reuse the exponentials below the k-th score. At most k-1 competitors
// lie above it; compute those entries from their score and the solved cap.
// Read each group of scores before writing its possibly aliased gradient.
template <typename A, typename W, typename S, typename G>
SDCA_LANE bool cached_gradient(const RowArgs<S, G>& row, const W* terms, W last, const Entropy& st,
                               std::size_t k) {
  const double cap = st.mass * std::fabs(row.scale) / static_cast<double>(k);
  const double factor = cap / st.spread * std::exp(static_cast<double>(last) - st.first);
  W scale, power;
  if (!(factor >= static_cast<double>(std::numeric_limits<W>::min())) ||
      !core::Wide<W>::split(factor, scale, power))
    return false;
  core::each<A>(row.scores, row.n, -std::numeric_limits<W>::infinity(),
                [&](const A& scores, std::size_t j, int count) {
                  A magnitude = core::load<A>(terms + j, count, W(0)) * scale;
                  if (power != W(1)) magnitude = magnitude * power;
                  if (core::any(scores > last)) {
                    for (int lane = 0; lane < count; ++lane) {
                      const double score = static_cast<double>(scores[lane]);
                      if (score > static_cast<double>(last)) {
                        const double g =
                            score > st.first ? cap : cap / st.spread * std::exp(score - st.first);
                        magnitude = core::with_lane(magnitude, j, j + lane, static_cast<W>(g));
                      }
                    }
                  }
                  core::store(row.grad + j, row.scale < 0 ? W(0) - magnitude : magnitude, count);
                });
  return true;
}

// Compute the tail at the final k-th score. The classes tied there but outside the top k
// each contribute one; count them apart so their removal cannot cancel a tiny lower tail.
template <typename W, typename Isa, typename S, typename G>
SDCA_FLAT inline double entropy_row(const RowArgs<S, G>& row, std::size_t y, double label,
                                    std::size_t k, Scratch& s) {
  typedef core::Lanes<W, Isa> A;
  const W inf = std::numeric_limits<W>::infinity();
  const S* z = row.scores;
  const std::size_t n = row.n;
  W* top = s.buffer<W>(k, 0);
  bool valid = true;
  core::top_k<A>(z, n, y, k, top, valid);
  if (!valid || (!(top[0] > -inf) && !(label > -kInf))) return Row::invalid();
  const W last = top[k - 1];
  if (!(last > -inf)) {
    if (row.grad) ce::zero_row(row.grad, n);
    return 0;
  }
  W* terms = gradient_terms<W>(row, k, s);
  std::size_t ties = 0;
  core::Accumulator<A> sums;
  core::each<A>(z, n, -inf, [&](A v, std::size_t j, int count) {
    v = core::with_lane(v, j, y, -inf);
    const auto tie = v == last;
    if (core::any(tie))
      for (int i = 0; i < A::L; ++i) ties += tie[i] != 0;
    const A d = core::select(v < last, v, A::splat(-inf)) - last;
    const A e = core::exp_nonpositive_wide(d);
    sums.add(e);
    if (terms) {
      const W one = static_cast<W>(1.0 / core::Wide<W>::unwide());
      core::store(terms + j, core::select(tie, A::splat(one), e), count);
    }
  });
  const double tail = sums.value() * core::Wide<W>::unwide() +
                      static_cast<double>(ties - core::count_of(top, k, last));
  double* wide = s.doubles(k, 1);
  for (std::size_t i = 0; i < k; ++i) wide[i] = static_cast<double>(top[i]);
  double others = tail * std::exp(wide[k - 1] - wide[0]);
  for (std::size_t i = 1; i < k; ++i) others += std::exp(wide[i] - wide[0]);
  Entropy st;
  const double value = entropy_value(wide[0], label, others, k, wide, tail, st);
  if (!row.grad) return value;
  if (!(st.mass > 0)) {
    ce::zero_row(row.grad, n);
    return value;
  }
  if (terms && cached_gradient<A>(row, terms, last, st, k)) {
    row.grad[y] = core::to_storage<G>(-st.mass * row.scale + 0.0);
    return value;
  }
  // Free competitors use the half-exponential product to preserve subnormal gradients.
  // Scores above the largest free competitor receive the solved cap directly.
  const double root =
      std::sqrt(st.mass / static_cast<double>(k) / st.spread * std::fabs(row.scale));
  if (sizeof(W) < sizeof(double) && !core::fits_float(root)) {
    s.needs_double = true;
    return 0;
  }
  const W center = static_cast<W>(st.first);
  const W cap = static_cast<W>((st.mass * std::fabs(row.scale)) / static_cast<double>(k));
  const W r = static_cast<W>(root);
  typedef core::Lanes<W, Isa, 2> Grad;
  core::each<Grad>(z, n, center, [&](const Grad& v, std::size_t j, int count) {
    const Grad half = core::exp_nonpositive_full((core::min(v, center) - center) * W(0.5)) * r;
    const Grad magnitude = core::select(v > center, Grad::splat(cap), half * half);
    core::store(row.grad + j, row.scale < 0 ? W(0) - magnitude : magnitude, count);
  });
  row.grad[y] = core::to_storage<G>(-st.mass * row.scale + 0.0);
  return value;
}

}  // namespace topk

template <typename W, typename Isa, typename S, typename G>
inline double topk_hinge(const RowArgs<S, G>& row, const Label& y, std::size_t k, bool beta,
                         double gamma, Scratch& s) {
  if (y.index < 0 || static_cast<uint64_t>(y.index) >= row.n || k < 1 || k >= row.n)
    return Row::invalid();
  const std::size_t label = static_cast<std::size_t>(y.index);
  const double score = static_cast<double>(core::to_work<W>(row.scores[label]));
  if (!(score < topk::kInf)) return Row::invalid();
  if (gamma > 0 && score == -topk::kInf) {
    // The limit as z_y goes to -inf: a label below the lowest score left by more than the cap
    // makes every margin exceed it, so the projection holds its full mass, as in the limit
    // (alpha with fewer than k left stays 0); the loss is +inf.
    typedef core::Lanes<W, Isa> A;
    W lo, hi;
    bool valid = true;
    core::extrema<A>(row.scores, row.n, lo, hi, valid);
    if (!valid || !(hi > -std::numeric_limits<W>::infinity())) return Row::invalid();
    const double stand_in = static_cast<double>(lo) - gamma / static_cast<double>(k);
    const double value = topk::smooth_row<W, Isa>(row, label, stand_in, k, beta, gamma, s);
    return value > 0 ? topk::kInf : value;
  }
  if (gamma > 0) return topk::smooth_row<W, Isa>(row, label, score, k, beta, gamma, s);
  return topk::nonsmooth_row<W, Isa>(row, label, score, k, beta, s);
}

// topk_cross_entropy: cross_entropy at k = 1 (which takes a masked label as +inf), else the
// top-k entropy max_{p in Delta_k} <z - z_y, p> + H(1 - sum p, p) over the competitors.
struct TopKCrossEntropy {
  std::size_t k;
  TopKCrossEntropy() : k(1) {}
  template <typename W, typename Isa, typename S, typename G>
  double operator()(const RowArgs<S, G>& row, const Label& y, Scratch& s) const {
    if (y.index < 0 || static_cast<uint64_t>(y.index) >= row.n || k < 1 || k >= row.n)
      return Row::invalid();
    const std::size_t label = static_cast<std::size_t>(y.index);
    if (k == 1) return ce::label_row<W, Isa>(row, label, 1.0, s);
    const double score = static_cast<double>(core::to_work<W>(row.scores[label]));
    if (!(score < topk::kInf)) return Row::invalid();
    if (k == row.n - 1) return topk::all_capped<W>(row, label, score);
    return topk::entropy_row<W, Isa>(row, label, score, k, s);
  }
};

// multiclass_hinge: max(0, max_{j != y} h_j), smoothed by gamma > 0 over the simplex.
struct MulticlassHinge {
  double gamma;
  MulticlassHinge() : gamma(0) {}
  template <typename W, typename Isa, typename S, typename G>
  double operator()(const RowArgs<S, G>& row, const Label& y, Scratch& s) const {
    return topk_hinge<W, Isa>(row, y, 1, true, gamma, s);
  }
};

// topk_hinge_alpha: max(0, mean of the k largest h_j), smoothed by gamma > 0.
struct TopKHingeAlpha {
  std::size_t k;
  double gamma;
  TopKHingeAlpha() : k(1), gamma(0) {}
  template <typename W, typename Isa, typename S, typename G>
  double operator()(const RowArgs<S, G>& row, const Label& y, Scratch& s) const {
    return topk_hinge<W, Isa>(row, y, k, false, gamma, s);
  }
};

// topk_hinge_beta: the mean of max(0, h_j) over the k largest h_j, smoothed by gamma > 0.
struct TopKHingeBeta {
  std::size_t k;
  double gamma;
  TopKHingeBeta() : k(1), gamma(0) {}
  template <typename W, typename Isa, typename S, typename G>
  double operator()(const RowArgs<S, G>& row, const Label& y, Scratch& s) const {
    return topk_hinge<W, Isa>(row, y, k, true, gamma, s);
  }
};

}  // namespace loss
SDCA_ISA_END

#endif  // SDCA_LOSS_TOPK_H
