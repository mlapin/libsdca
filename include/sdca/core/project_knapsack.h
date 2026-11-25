#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_CORE_PROJECT_KNAPSACK_H)
#if SDCA_ISA_ID == 1
#define SDCA_CORE_PROJECT_KNAPSACK_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_CORE_PROJECT_KNAPSACK_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_CORE_PROJECT_KNAPSACK_H3 1
#else
#define SDCA_CORE_PROJECT_KNAPSACK_H4 1
#endif

#include <functional>

#include "sdca/core/project_base.h"
#include "sdca/core/lanes.h"

SDCA_ISA_BEGIN
namespace core {

// Kiwiel's variable fixing for the continuous quadratic knapsack problem (J. Optim. Theory
// Appl. 136, 2008).
namespace knapsack {

// The thresholds of the fixed sets [first, m_first) at hi and [m_last, last) at lo: the free
// mass recomputed in Accum (wider than the data) relative to the smallest free score, within
// the bounds of the fixed sets. remaining is the mass left for the free set.
template <typename Work, typename Accum, typename Result, typename Iterator>
inline thresholds<Result, Iterator> finish(Iterator first, Iterator last, Iterator m_first,
                                           Iterator m_last, const Work origin, const Result lo,
                                           const Result hi, const Accum remaining) {
  const Work stored = m_first != m_last ? Work(*std::min_element(m_first, m_last)) : origin;
  Work t_lo = std::numeric_limits<Work>::lowest(), t_hi = std::numeric_limits<Work>::max();
  if (m_last != last)
    t_lo = Work((Accum(*std::max_element(m_last, last)) - Accum(lo)) - Accum(stored));
  if (first != m_first)
    t_hi = Work((Accum(*std::min_element(first, m_first)) - Accum(hi)) - Accum(stored));
  Work correction;
  if (m_first != m_last) {
    Accum sum(0);
    for (Iterator it = m_first; it != m_last; ++it) sum += Accum(*it) - Accum(stored);
    correction = Work((sum - remaining) / Accum(std::distance(m_first, m_last)));
    correction = std::max(t_lo, std::min(correction, t_hi));
  } else {
    correction = t_lo / 2 + t_hi / 2;
  }
  return split_thresholds(stored, correction, lo, hi, m_first, m_last);
}

// first_sum, if given, is the sum of x - *first over all x.
template <typename Work, typename Accum, typename Result, typename Iterator>
inline thresholds<Result, Iterator> solve(Iterator first, Iterator last, const Result lo,
                                          const Result hi, const Result rhs,
                                          const Work* first_sum = nullptr) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  Iterator m_first = first, m_last = last;
  Work origin(0), t(0);
  // With nothing fixed at hi, lo = +0 and hi >= +0, both products are +0 and the remaining
  // mass is exactly rhs (the simplex).
  const bool simplex = lo == 0 && !std::signbit(lo) && !std::signbit(hi);
  auto remaining = [&]() {
    if (simplex && m_first == first) return Accum(rhs);
    const Accum n_hi = Accum(std::distance(first, m_first));
    const Accum n_lo = Accum(std::distance(m_last, last));
    const Accum p_hi = Accum(hi) * n_hi, p_lo = Accum(lo) * n_lo;
    if (p_hi == 0 || p_lo == 0) return (Accum(rhs) - p_hi) - p_lo;
    Accum e, f;
    const Accum s = bound_mass(Accum(lo), n_lo, Accum(hi), n_hi, e);
    const Accum r = sum_error(Accum(rhs), -s, f);
    return r + (f - e);
  };
  for (;;) {
    // Recentered after each fixing: large saturated scores cannot erase the free mass.
    origin = Work(*m_first);
    const Work rem = Work(remaining());
    Work sum(0);
    if (first_sum != nullptr) {
      sum = *first_sum;
      first_sum = nullptr;
    } else {
      sum = shifted_sum(m_first, m_last, origin, Work(0));
    }
    t = (sum - rem) / Work(std::distance(m_first, m_last));
    Iterator it_lo = partition_monotone(
        m_first, m_last, [=](Data x) { return (Work(x) - origin) - t > Work(lo); },
        (Work(lo) + t) + origin);
    Iterator it_hi = partition_monotone(
        m_first, it_lo, [=](Data x) { return (Work(x) - origin) - t > Work(hi); },
        (Work(hi) + t) + origin);
    // Feasibility from the bounded coordinates, not from cancelling sums of raw scores; eps
    // scales with the bound sets active in this step only.
    Work mass = Work(hi) * Work(std::distance(m_first, it_hi)) +
                Work(lo) * Work(std::distance(it_lo, m_last));
    mass += shifted_sum(it_hi, it_lo, origin, t);
    const Work error = mass - rem;
    const Work scale = std::abs(rem) + std::abs(Work(hi)) * Work(std::distance(m_first, it_hi)) +
                       std::abs(Work(lo)) * Work(std::distance(it_lo, m_last));
    const Work eps = Work(8) * std::numeric_limits<Work>::epsilon() * scale;
    if (it_hi != m_first && (it_lo == m_last || error < -eps)) {
      m_first = it_hi;
    } else if (it_lo != m_last && (it_hi == m_first || error > eps)) {
      m_last = it_lo;
    } else {
      m_first = it_hi;
      m_last = it_lo;
      break;
    }
    if (m_first == m_last) break;
  }
  return finish<Work, Accum>(first, last, m_first, m_last, origin, lo, hi, remaining());
}

// Finite bounds and mass are required before forming the feasible interval.
template <typename T>
inline bool finite_bounds(T lo, T hi, T rhs) {
  return std::isfinite(lo) && std::isfinite(hi) && std::isfinite(rhs) && lo <= hi;
}

// Throws unless lo <= x_i <= hi, <1, x> <= rhs has a solution.
template <typename Work, typename Result, typename Count>
inline void check_le(const Count count, const Result lo, const Result hi, const Result rhs) {
  if (count <= 0 || !knapsack::finite_bounds(lo, hi, rhs))
    throw std::invalid_argument("Infeasible knapsack projection");
  if (lo <= 0 && rhs >= 0) return;
  const Work slack = Work(4) * Work(std::numeric_limits<Result>::epsilon()) * std::abs(Work(lo));
  if (Work(rhs) / Work(count) < Work(lo) - slack)
    throw std::invalid_argument("Infeasible knapsack projection");
}

// Whether rhs / count lies within (lo, hi) by a margin that no rounding of the quotient, in
// double or wider, can close: then no endpoint test on the mean needs extended precision. The
// quotient errs by u |m| in double (less if wider) and the difference by u more; where the
// test passes |m| <= (1 + u) max(|lo|, |hi|), so the margin 2 u (1 + u)^2 (|lo| + |hi|) < 2 eps
// (|lo| + |hi|) separates both quotients from the bounds, and min the subnormal ones.
template <typename T, typename Count>
inline bool interior_mean(const Count count, const T lo, const T hi, const T rhs) {
  if (!std::is_same<T, double>::value && !std::is_same<T, float>::value) return false;
  const double mean = double(rhs) / double(count);
  const double margin = 2 * limits::eps<double>() * (std::abs(double(lo)) + std::abs(double(hi))) +
                        std::numeric_limits<double>::min();
  return mean - double(lo) > margin && double(hi) - mean > margin;
}

}  // namespace knapsack

/**
 * x = max(lo, min(a - t, hi)) minimizing 0.5 <x, x> - <a, x>
 * subject to <1, x> = rhs, lo <= x_i <= hi.
 **/
template <typename Result = double, typename Iterator>
inline thresholds<Result, Iterator> thresholds_knapsack_eq(Iterator first, Iterator last,
                                                           const Result lo = 0, const Result hi = 1,
                                                           const Result rhs = 1) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  typedef typename std::common_type<Result, Data, double>::type FastWork;
  typedef typename std::common_type<Result, Data, long double>::type WideWork;
  typedef typename std::common_type<Result, Data>::type Common;
  typedef typename wider<Common>::type FastAccum;
  const auto count = std::distance(first, last);
  if (count <= 0 || !knapsack::finite_bounds(lo, hi, rhs))
    throw std::invalid_argument("Infeasible knapsack projection");
  const bool interior = knapsack::interior_mean(count, Common(lo), Common(hi), Common(rhs));
  const WideWork mean = interior ? WideWork(0) : WideWork(rhs) / WideWork(count);
  const WideWork slack = WideWork(4) * WideWork(std::numeric_limits<Result>::epsilon());
  if (!interior && (mean < WideWork(lo) - slack * std::abs(WideWork(lo)) ||
                    mean > WideWork(hi) + slack * std::abs(WideWork(hi))))
    throw std::invalid_argument("Infeasible knapsack projection");
  // One pass checks the scores, bounds their magnitude and forms the first iteration's sum.
  FastWork magnitude = std::max(std::abs(FastWork(lo)), std::abs(FastWork(hi)));
  bool finite = true;
  const FastWork sum = scan_scores(first, last, FastWork(*first), magnitude, finite);
  if (!finite) throw std::invalid_argument("Knapsack scores must be finite");
  // Outward rounding of a represented bound (hi = 1/3) is tolerated, but a genuinely interior
  // mean is never snapped to that endpoint.
  if (!interior && mean <= WideWork(lo)) return make_thresholds(Result(0), lo, lo, first, first);
  if (!interior && mean >= WideWork(hi)) return make_thresholds(Result(0), hi, hi, last, last);
  // One interior entry takes the whole mass. Unbounded centered sums or tiny scales take
  // extended working precision: below min / eps the tolerances eps scale of solve are subnormal.
  const FastWork span = std::abs(FastWork(hi) - FastWork(lo)), abs_rhs = std::abs(FastWork(rhs));
  const FastWork safe = std::numeric_limits<FastWork>::max() / (FastWork(4) * FastWork(count));
  const FastWork tiny = limits::min_over_eps<FastWork>();
  if (magnitude <= safe && span >= tiny && (rhs == Result(0) || abs_rhs >= tiny)) {
    if (count == 1)
      return split_thresholds(FastWork(*first), FastWork(0) - FastWork(rhs), lo, hi, first, last);
    return knapsack::solve<FastWork, FastAccum, Result>(first, last, lo, hi, rhs, &sum);
  }
  if (count == 1)
    return split_thresholds(WideWork(*first), WideWork(0) - WideWork(rhs), lo, hi, first, last);
  return knapsack::solve<WideWork, WideWork, Result>(first, last, lo, hi, rhs);
}

/**
 * x = max(lo, min(a - t, hi)) minimizing 0.5 <x, x> - <a, x>
 * subject to <1, x> <= rhs, lo <= x_i <= hi.
 **/
template <typename Result = double, typename Iterator>
inline thresholds<Result, Iterator> thresholds_knapsack_le(Iterator first, Iterator last,
                                                           const Result lo = 0, const Result hi = 1,
                                                           const Result rhs = 1) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  typedef typename std::common_type<Result, Data>::type Compare;
  typedef typename wider<Compare>::type Work;
  knapsack::check_le<Work>(std::distance(first, last), lo, hi, rhs);
  // The box-projected mass is tested, with no absolute epsilon floor (a small rhs such as a
  // smoothed hinge's gamma would lose the constraint); only the free entries are summed.
  bool finite = true;
  const auto n_hi = count_at_least(first, last, Compare(hi), finite);
  if (!finite) throw std::invalid_argument("Knapsack scores must be finite");
  Iterator m_first =
      partition_counted(first, last, [=](const Data x) { return Compare(x) >= Compare(hi); }, n_hi);
  Iterator m_last = partition_counted(
      m_first, last, [=](const Data x) { return Compare(x) > Compare(lo); },
      count_above(m_first, last, Compare(lo)));
  // Opposing saturated masses cancel exactly. The mass is formed plainly in double, and again
  // with compensation in Work where that rounding decides.
  typedef typename std::common_type<Compare, double>::type Fast;
  const Fast at_lo = Fast(std::distance(m_last, last)), at_hi = Fast(n_hi);
  Fast fast_error;
  const Fast fast_bounds = bound_mass(Fast(lo), at_lo, Fast(hi), at_hi, fast_error);
  const Fast free_mass = shifted_sum(m_first, m_last, Fast(0), Fast(0));
  const Fast fast_over = ((fast_bounds + free_mass) - Fast(rhs)) + fast_error;
  const Fast n_free = Fast(std::distance(m_first, m_last));
  const double size = std::abs(double(fast_bounds)) + std::abs(double(rhs)) +
                      double(n_free) * std::max(std::abs(double(lo)), std::abs(double(hi)));
  const Fast tolerance = 2 * (n_free + 3) * std::numeric_limits<Fast>::epsilon() * Fast(size);
  bool over = fast_over > 0;
  if (!(std::abs(fast_over) > tolerance) || !std::isfinite(fast_over)) {
    Work error;
    Work mass = bound_mass(Work(lo), Work(at_lo), Work(hi), Work(at_hi), error);
    for (Iterator it = m_first; it != m_last; ++it) {
      Work e;
      mass = sum_error(mass, Work(*it), e);
      error += e;
    }
    Work f;
    Work exact = sum_error(mass, -Work(rhs), f);
    exact += f + error;
    over = exact > 0;
  }
  if (over) return thresholds_knapsack_eq(first, last, lo, hi, rhs);
  return thresholds<Result, Iterator>(0, lo, hi, m_first, m_last);
}

namespace knapsack {

// The root of t = rho <1, clamp(a - t, lo, hi)>, capped at t_max, in units of 1 / unit: a
// bisection with compensated sums and no tolerance.
template <typename Result, typename Iterator>
inline thresholds<Result, Iterator> biased_root(Iterator first, Iterator last, const Result lo,
                                                const Result hi, const Result rho,
                                                const Result t_max, const Result unit) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  const Result lo_u = lo * unit, hi_u = hi * unit;
  auto test = [=](const Result t) {
    compensated_sum<Result> sum;
    for (Iterator it = first; it != last; ++it)
      sum.add(std::max(lo_u, std::min(Result(*it) * unit - t, hi_u)));
    return t > rho * sum.value();
  };
  const Result t = std::min(monotone_cut<Result>(Result(0), test), t_max * unit);
  Iterator m_first =
      std::partition_point(first, last, [=](const Data x) { return Result(x) * unit - t >= hi_u; });
  Iterator m_last = std::partition_point(m_first, last,
                                         [=](const Data x) { return Result(x) * unit - t > lo_u; });
  return make_thresholds(t / unit, lo, hi, m_first, m_last);
}

// The search with condition (5) as t <= t_max over scores sorted here in decreasing order,
// for U = [first, first + u) at hi, M = [first + u, first + u + m) free and L at lo:
//    t = (lo |L| + hi u + S_M) / (1 / rho + m)
// such that
//  (1)  lo + t >= max_L,  (2)  lo + t <= min_M,  (3)  hi + t >= max_M,
//  (4)  hi + t <= min_U,  (5)  t <= t_max = rho rhs.
// For a fixed u, t(m) is a weighted mean of t(m - 1) and b_m - lo for the m-th score b_m of
// M, so b_m - lo > t(m) holds on a prefix of m whose last m satisfies (1) and (2): it is
// bisected on compensated prefix sums, and only it and its neighbours (rounding at ties) are
// tested, O(n log n). Every test is resolved to the rounding of its own terms, so that tiny
// scores are never taken as zero.
template <typename Result, typename Iterator>
inline thresholds<Result, Iterator> biased_search(Iterator first, Iterator last, const Result lo,
                                                  const Result hi, const Result rho,
                                                  const Result t_max) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  assert(rho > 0);
  std::sort(first, last, std::greater<Data>());
  // Near overflow the search runs in units of a power of two: the sums stay finite.
  const std::size_t n = std::size_t(std::distance(first, last));
  const Result num_X = Result(n);
  const Result magnitude =
      std::max(std::max(std::abs(lo), std::abs(hi)),
               std::max(std::abs(Result(*first)), std::abs(Result(*std::prev(last)))));
  const Result unit = overflow_unit(magnitude, num_X + 2);
  const Result lo_u = lo * unit, hi_u = hi * unit, t_max_u = t_max * unit;
  auto value = [=](const std::size_t i) { return Result(first[i]) * unit; };
  const prefix_sums<Result> sum(n, value);
  const prefix_sums<Result> sum_abs(n, [=](const std::size_t i) { return std::abs(value(i)); });
  const Result eps = std::numeric_limits<Result>::epsilon();
  const Result inf = std::numeric_limits<Result>::infinity();
  // Tolerances: eps |bound|, and eps |terms of t| / den <= eps max(1, rho) terms.
  const Result rho_inverse = Result(1) / rho;
  const Result eps_lo = eps * std::abs(lo_u), eps_hi = eps * std::abs(hi_u);
  const Result eps_t_max = eps * std::abs(t_max_u), eps_den = eps * std::max(Result(1), rho);
  const bool both_bounds = lo_u != 0 && hi_u != 0;
  for (std::size_t u = 0; u <= n; ++u) {
    const Result num_U = Result(u), min_U = u > 0 ? value(u - 1) : inf;
    auto solve_m = [&](const std::size_t m, Result& terms) {
      const Result num_L = num_X - num_U - Result(m), sum_M = sum(u, u + m);
      terms = std::abs(lo_u) * num_L + std::abs(hi_u) * num_U + sum_abs(u, u + m);
      Result num;
      if (both_bounds) {
        Result error;
        num = (bound_mass(lo_u, num_L, hi_u, num_U, error) + sum_M) + error;
      } else {
        num = lo_u * num_L + hi_u * num_U + sum_M;
      }
      return num / (rho_inverse + Result(m));
    };
    // The largest m with b_m - lo > t(m), or zero.
    std::size_t m_lo = 0, m_hi = n - u;
    while (m_lo < m_hi) {
      const std::size_t m = m_hi - (m_hi - m_lo) / 2;
      Result terms;
      if (value(u + m - 1) - lo_u > solve_m(m, terms))
        m_lo = m;
      else
        m_hi = m - 1;
    }
    for (std::size_t m = m_lo > 0 ? m_lo - 1 : 0; m <= std::min(m_lo + 1, n - u); ++m) {
      Result terms;
      const Result t = solve_m(m, terms), den = rho_inverse + Result(m);
      // A violation v of a test against bound b is within the rounding of b and of t,
      // eps (terms of t) / den: no absolute floor. The looser first test spares the
      // division for clear violations.
      auto within = [&](const Result v, const Result e) {
        return v <= 0 || (v <= eps_den * terms + e && v <= eps * (terms / den) + e);
      };
      const Result max_M = m > 0 ? value(u) : -inf, min_M = m > 0 ? value(u + m - 1) : inf;
      const Result max_L = u + m < n ? value(u + m) : -inf;
      if (within(t - t_max_u, eps_t_max) && within(max_M - (hi_u + t), eps_hi) &&
          within((hi_u + t) - min_U, eps_hi) && within((lo_u + t) - min_M, eps_lo) &&
          within(max_L - (lo_u + t), eps_lo))
        return make_thresholds(t / unit, lo, hi, first + u, first + (u + m));
    }
  }
  // No active set passed its tests (rounding at a tie): t is solved for instead.
  return biased_root(first, last, lo, hi, rho, t_max, unit);
}

}  // namespace knapsack

/**
 * x = max(lo, min(a - t, hi)) minimizing 0.5 (<x, x> + rho <1, x>^2) - <a, x>
 * subject to <1, x> <= rhs, lo <= x_i <= hi.
 **/
template <typename Result = double, typename Iterator>
inline thresholds<Result, Iterator> thresholds_knapsack_le_biased(Iterator first, Iterator last,
                                                                  const Result lo = 0,
                                                                  const Result hi = 1,
                                                                  const Result rhs = 1,
                                                                  const Result rho = 1) {
  if (!(rho >= 0)) throw std::invalid_argument("rho must be nonnegative");
  if (rho == 0) return thresholds_knapsack_le(first, last, lo, hi, rhs);
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  typedef typename wider<typename std::common_type<Result, Data>::type>::type Work;
  const auto count = std::distance(first, last);
  knapsack::check_le<Work>(count, lo, hi, rhs);
  if (!(Work(rhs) < Work(hi) * Work(count))) {
    // The sum bound is redundant: no condition (5).
    check_scores(first, last, "Knapsack scores must be finite");
    return knapsack::biased_search(first, last, lo, hi, rho,
                                   std::numeric_limits<Result>::infinity());
  }
  thresholds<Result, Iterator> t = thresholds_knapsack_eq(first, last, lo, hi, rhs);
  // rhs = n lo (or lo = hi) leaves one feasible point, every entry at lo.
  if (t.lo == t.hi) return t;
  // The bound binds unless t < rho rhs beyond the rounding of t (whose terms include its
  // score origin) and of rho rhs: no absolute floor.
  const Result t_max = rho * rhs;
  const Result eps = 4 * std::numeric_limits<Result>::epsilon() *
                     std::max(std::max(std::abs(t.t), std::abs(t.origin)), std::abs(t_max));
  if (t.t >= t_max - eps) return t;
  return knapsack::biased_search(first, last, lo, hi, rho, t_max);
}

}  // namespace core
SDCA_ISA_END

#endif  // SDCA_CORE_PROJECT_KNAPSACK_H
