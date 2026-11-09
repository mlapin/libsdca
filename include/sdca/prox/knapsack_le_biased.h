#ifndef SDCA_PROX_KNAPSACK_LE_BIASED_H
#define SDCA_PROX_KNAPSACK_LE_BIASED_H

#include <functional>

#include "sdca/prox/knapsack_le.h"

namespace sdca {

namespace detail {

// The root of t = rho <1, clamp(a - t, lo, hi)>, capped at t_max, in units
// of 1/unit: a bisection with compensated sums and no tolerance.
template <typename Result, typename Iterator>
inline thresholds<Result, Iterator> knapsack_le_biased_root(Iterator first, Iterator last,
                                                            const Result lo, const Result hi,
                                                            const Result rho, const Result t_max,
                                                            const Result unit) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  const Result lo_u = lo * unit, hi_u = hi * unit;
  auto test = [=](const Result t) {
    Result sum(0), error(0);
    for (auto it = first; it != last; ++it) {
      Result e;
      sum = two_sum(sum, std::max(lo_u, std::min(static_cast<Result>(*it) * unit - t, hi_u)), e);
      error += e;
    }
    return t > rho * (sum + error);
  };
  const Result t = std::min(monotone_root(Result(0), test), t_max * unit);
  auto m_first = std::partition_point(
      first, last, [=](const Data x) { return static_cast<Result>(x) * unit - t >= hi_u; });
  auto m_last = std::partition_point(
      m_first, last, [=](const Data x) { return static_cast<Result>(x) * unit - t > lo_u; });
  return make_thresholds(t / unit, lo, hi, m_first, m_last);
}

// The search with condition (5) as t <= t_max over scores sorted here in
// decreasing order, for U = [first, first + u) at hi, M = [first + u,
// first + u + m) free and L at lo:
//    t = (lo |L| + hi u + S_M) / (1/rho + m)
// such that
//  (1)  lo + t  >= max_L = (first + u + m) or (-Inf)
//  (2)  lo + t  <= min_M = (first + u + m - 1) or (+Inf)
//  (3)  hi + t  >= max_M = (first + u) or (-Inf)
//  (4)  hi + t  <= min_U = (first + u - 1) or (+Inf)
//  (5)       t  <= t_max = rho * rhs
// For a fixed u, t(m) is a weighted mean of t(m - 1) and b_m - lo for the
// m-th score b_m of M, so b_m - lo > t(m) holds on a prefix of m whose last
// m satisfies (1) and (2): it is bisected on compensated prefix sums, and
// only it and its neighbours (rounding at ties) are tested, O(n log n). Every
// test is resolved to the rounding of its own terms, so tiny scores are never
// taken as zero.
template <typename Result, typename Iterator>
inline thresholds<Result, Iterator> knapsack_le_biased_search(Iterator first, Iterator last,
                                                              const Result lo, const Result hi,
                                                              const Result rho,
                                                              const Result t_max) {
  // At this point, rho must be positive
  assert(rho > 0);

  // Sort data to search efficiently
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  std::sort(first, last, std::greater<Data>());

  // Near overflow, search in units of a power of two: sums stay finite.
  const std::size_t n = static_cast<std::size_t>(std::distance(first, last));
  const Result num_X = static_cast<Result>(n);
  const Result magnitude = std::max(std::max(std::abs(lo), std::abs(hi)),
                                    std::max(std::abs(static_cast<Result>(*first)),
                                             std::abs(static_cast<Result>(*std::prev(last)))));
  const Result unit = overflow_unit(magnitude, num_X + 2);
  const Result lo_u = lo * unit, hi_u = hi * unit, t_max_u = t_max * unit;
  auto value = [=](const std::size_t i) { return static_cast<Result>(first[i]) * unit; };
  const prefix_sums<Result> sum(n, value);
  const prefix_sums<Result> sum_abs(n, [=](const std::size_t i) { return std::abs(value(i)); });
  const Result eps = std::numeric_limits<Result>::epsilon();
  const Result inf = std::numeric_limits<Result>::infinity();

  // Tolerances: eps |bound|, and eps |terms of t| / den <= eps max(1, rho) terms.
  const Result rho_inverse = static_cast<Result>(1) / rho;
  const Result eps_lo = eps * std::abs(lo_u), eps_hi = eps * std::abs(hi_u);
  const Result eps_t_max = eps * std::abs(t_max_u), eps_den = eps * std::max(Result(1), rho);
  const bool both_bounds = lo_u != 0 && hi_u != 0;

  for (std::size_t u = 0; u <= n; ++u) {
    const Result num_U = static_cast<Result>(u), min_U = u > 0 ? value(u - 1) : inf;
    // t for M = [u, u + m), and the magnitude of its terms.
    auto solve = [&](const std::size_t m, Result& terms) {
      const Result num_L = num_X - num_U - static_cast<Result>(m), sum_M = sum(u, u + m);
      terms = std::abs(lo_u) * num_L + std::abs(hi_u) * num_U + sum_abs(u, u + m);
      Result num;
      if (both_bounds) {
        Result error;  // opposite bound masses cancel exactly
        num = (bound_mass(lo_u, num_L, hi_u, num_U, error) + sum_M) + error;
      } else {
        num = lo_u * num_L + hi_u * num_U + sum_M;
      }
      return num / (rho_inverse + static_cast<Result>(m));
    };
    // The largest m with b_m - lo > t(m), or zero.
    std::size_t m_lo = 0, m_hi = n - u;
    while (m_lo < m_hi) {
      const std::size_t m = m_hi - (m_hi - m_lo) / 2;
      Result terms;
      if (value(u + m - 1) - lo_u > solve(m, terms))
        m_lo = m;
      else
        m_hi = m - 1;
    }
    for (std::size_t m = m_lo > 0 ? m_lo - 1 : 0; m <= std::min(m_lo + 1, n - u); ++m) {
      Result terms;
      const Result t = solve(m, terms), den = rho_inverse + static_cast<Result>(m);
      // A violation v of a test against bound b is within the rounding of b
      // and of t, eps (terms of t) / den: no absolute floor. The first, looser
      // test spares the division for clear violations.
      auto within = [&](const Result v, const Result e) {
        return v <= 0 || (v <= eps_den * terms + e && v <= eps * (terms / den) + e);
      };
      const Result max_M = m > 0 ? value(u) : -inf, min_M = m > 0 ? value(u + m - 1) : inf;
      const Result max_L = u + m < n ? value(u + m) : -inf;
      if (within(t - t_max_u, eps_t_max) && within(max_M - (hi_u + t), eps_hi) &&
          within((hi_u + t) - min_U, eps_hi) && within((lo_u + t) - min_M, eps_lo) &&
          within(max_L - (lo_u + t), eps_lo)) {
        return make_thresholds(t / unit, lo, hi, first + u, first + (u + m));
      }
    }
  }

  // No active set passed its tests (rounding at a tie): solve for t instead.
  return knapsack_le_biased_root(first, last, lo, hi, rho, t_max, unit);
}

}  // namespace detail

template <typename Result, typename Iterator>
inline thresholds<Result, Iterator> thresholds_knapsack_le_biased_search(
    Iterator first, Iterator last, const Result lo, const Result hi, const Result rhs,
    const Result rho) {
  return detail::knapsack_le_biased_search(first, last, lo, hi, rho, rho * rhs);
}

/**
 * Solve
 *    min_x 0.5 * (<x, x> + rho * <1, x>^2) - <a, x>
 *          <1, x> <= rhs
 *          lo <= x_i <= hi
 *
 * The solution is
 *    x = max(lo, min(a - t, hi))
 **/
template <typename Result = double, typename Iterator>
inline thresholds<Result, Iterator> thresholds_knapsack_le_biased(Iterator first, Iterator last,
                                                                  const Result lo = 0,
                                                                  const Result hi = 1,
                                                                  const Result rhs = 1,
                                                                  const Result rho = 1) {
  if (!(rho >= 0)) throw std::invalid_argument("rho must be nonnegative");
  if (rho == 0) {
    return thresholds_knapsack_le(first, last, lo, hi, rhs);
  }

  using Data = typename std::iterator_traits<Iterator>::value_type;
  using Work = typename detail::wider<typename std::common_type<Result, Data>::type>::type;
  const auto count = std::distance(first, last);
  detail::validate_knapsack_le<Work>(count, lo, hi, rhs);

  // Equality constraint, then the general case, if the sum bound can bind
  if (Work(rhs) < Work(hi) * Work(count)) {
    auto t = thresholds_knapsack_eq(first, last, lo, hi, rhs);
    // rhs = n lo (or lo = hi) leaves one feasible point, every entry at lo.
    if (t.lo == t.hi) return t;
    // The bound binds unless t < rho rhs beyond the rounding of t (whose
    // terms include its score origin) and of rho rhs: no absolute floor.
    const Result t_max = rho * rhs;
    const Result eps = 4 * std::numeric_limits<Result>::epsilon() *
                       std::max(std::max(std::abs(t.t), std::abs(t.origin)), std::abs(t_max));
    if (t.t >= t_max - eps) {
      return t;
    }
    return detail::knapsack_le_biased_search(first, last, lo, hi, rho, t_max);
  }

  // The sum bound is redundant: no condition (5)
  if (!std::all_of(first, last, [](const Data x) { return std::isfinite(x); }))
    throw std::invalid_argument("Knapsack scores must be finite");
  return detail::knapsack_le_biased_search(first, last, lo, hi, rho,
                                           std::numeric_limits<Result>::infinity());
}

// [first, last) in place, through a temporary copy.
template <typename Result = double, typename Iterator>
inline void prox_knapsack_le_biased(Iterator first, Iterator last, const Result lo = 0,
                                    const Result hi = 1, const Result rhs = 1,
                                    const Result rho = 1) {
  prox(first, last,
       thresholds_knapsack_le_biased<
           Result,
           typename std::vector<typename std::iterator_traits<Iterator>::value_type>::iterator>,
       lo, hi, rhs, rho);
}

// [first, last) in place; aux (as long) is scratch.
template <typename Result = double, typename Iterator>
inline void prox_knapsack_le_biased(Iterator first, Iterator last, Iterator aux,
                                    const Result lo = 0, const Result hi = 1, const Result rhs = 1,
                                    const Result rho = 1) {
  prox(first, last, aux, thresholds_knapsack_le_biased<Result, Iterator>, lo, hi, rhs, rho);
}

// Each block of dim consecutive entries of [first, last) in place; aux holds dim
// entries.
template <typename Result = double, typename Iterator>
inline void prox_knapsack_le_biased(
    const typename std::iterator_traits<Iterator>::difference_type dim, Iterator first,
    Iterator last, Iterator aux, const Result lo = 0, const Result hi = 1, const Result rhs = 1,
    const Result rho = 1) {
  prox(dim, first, last, aux, thresholds_knapsack_le_biased<Result, Iterator>, lo, hi, rhs, rho);
}

}  // namespace sdca

#endif
