#ifndef SDCA_PROX_TOPK_CONE_H
#define SDCA_PROX_TOPK_CONE_H

#include <functional>

#include "sdca/prox/proxdef.h"

namespace sdca {

namespace detail {

// 16 eps relative to a score magnitude below one: scale-invariant for tiny
// scores, and the absolute 16 eps from one up.
template <typename Result>
inline Result topk_cone_tolerance(const Result scale) {
  return 16 * std::numeric_limits<Result>::epsilon() * std::min(static_cast<Result>(1), scale);
}

// eps relative to the larger of rhs and the score magnitude (capped at one).
template <typename Result>
inline Result topk_simplex_tolerance(const Result scale, const Result rhs) {
  return std::numeric_limits<Result>::epsilon() *
         std::max(std::min(static_cast<Result>(1), scale), std::abs(rhs));
}

// The candidate active set (|U|, |M|) = (u, m) of a top-k cone search that
// violates its conditions least, for when rounding at a tie rejects all.
template <typename Result>
struct topk_cone_best {
  Result violation = std::numeric_limits<Result>::infinity();
  Result t = 0, hi = 0;
  std::size_t u = 0, m = 0;

  void update(const Result t_, const Result hi_, const Result max_M, const Result min_M,
              const Result min_U, const Result max_L, const std::size_t u_, const std::size_t m_) {
    const Result tt = hi_ + t_;
    const Result v = std::max(std::max(max_M - tt, tt - min_U), std::max(t_ - min_M, max_L - t_));
    if (v < violation) {
      violation = v;
      t = t_;
      hi = hi_;
      u = u_;
      m = m_;
    }
  }
};

template <typename Result>
inline void validate_topk_simplex(const Result rhs) {
  if (!(rhs >= 0) || !std::isfinite(rhs))
    throw std::invalid_argument("Infeasible top-k projection");
}

}  // namespace detail

template <typename Result, typename Iterator>
struct topk_cone_projection {
  sdca::projection projection;
  sdca::thresholds<Result, Iterator> thresholds;
  Result scale;  // the sum of magnitudes of the k largest scores, in units of 1/unit
  Result unit;   // a power of two that keeps the search's sums finite (usually one)
};

template <typename Result, typename Iterator>
inline topk_cone_projection<Result, Iterator> topk_cone_special_cases(
    Iterator first, Iterator last, const typename std::iterator_traits<Iterator>::difference_type k,
    const Result div_const, const Result rhs = 0) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  bool finite = true;
  Result magnitude = std::abs(rhs);
  for (auto it = first; it != last; ++it) {
    finite &= *it - *it == 0;
    magnitude = std::max(magnitude, std::abs(static_cast<Result>(*it)));
  }
  if (!finite) throw std::invalid_argument("Top-k scores must be finite");

  // Near overflow, solve in units of a power of two (exact): the searches'
  // sums and their products with counts and rho k^2 stay finite.
  topk_cone_projection<Result, Iterator> proj;
  const Result num_X = static_cast<Result>(std::distance(first, last));
  proj.unit =
      detail::overflow_unit(magnitude, num_X * (num_X + static_cast<Result>(k) * div_const));
  const Result unit = proj.unit;

  // Partially sort data around the kth element
  auto k_last = std::next(first, k);
  std::nth_element(first, k_last - 1, last, std::greater<Data>());

  // The sums in double at least (float cancels at a large rho).
  using Accum = typename std::common_type<Result, double>::type;
  Accum scale = 0, top = 0, sum_k_largest = 0;
  for (auto it = first; it != k_last; ++it) {
    const Accum x = static_cast<Accum>(*it) * Accum(unit);
    scale += std::abs(x);
    top = std::max(top, std::abs(x));
    sum_k_largest += x;
  }
  proj.scale = Result(scale);

  // Case 1: U empty, M empty, proj = 0
  if (sum_k_largest <= Accum(detail::topk_cone_tolerance(proj.scale))) {
    proj.projection = projection::zero;
    proj.thresholds = thresholds<Result, Iterator>(0, 0, 0, first, first);
    return proj;
  }

  // Case 2: U not empty, M empty, proj = const * sum_k_largest for k largest;
  // the test within the tolerance of the scores it compares (as the search).
  const Accum hi = sum_k_largest / Accum(div_const);
  const Accum t = static_cast<Accum>(*(k_last - 1)) * Accum(unit) - hi;
  bool constant = k == std::distance(first, last);
  if (!constant) {
    const Accum max_L = static_cast<Accum>(*std::max_element(k_last, last)) * Accum(unit);
    const Accum eps = Accum(detail::topk_cone_tolerance(Result(std::max(top, std::abs(max_L)))));
    constant = t >= max_L - eps;
  }
  if (constant) {
    proj.projection = projection::constant;
    proj.thresholds = thresholds<Result, Iterator>(Result(t / Accum(unit)), 0,
                                                   Result(hi / Accum(unit)), k_last, k_last);
    return proj;
  }

  proj.projection = projection::general;
  return proj;
}

namespace detail {

// The search for rho >= 0 (zero: the plain cone) over scores sorted here in
// decreasing order, for U = [first, first + u) at hi + t, M = [first + u,
// first + u + m) free and L at zero (lo = 0 by definition):
//    t = ((u + rho k^2) S_M - (k - u) S_U) / D
//   hi = (m S_U + (k - u) S_M) / D
//    D = (k - u)^2 + (u + rho k^2) m
// such that
//  (1)  t      >= max_L = (first + u + m) or (-Inf)
//  (2)  t      <= min_M = (first + u + m - 1) or (+Inf)
//  (3)  hi + t >= max_M = (first + u) or (-Inf)
//  (4)  hi + t <= min_U = (first + u - 1) or (+Inf)
// For a fixed u, t(m) is a weighted mean of t(m - 1) and the m-th score b_m
// of M (D grows by u + rho k^2), so b_m > t(m) holds on a prefix of m whose
// last m satisfies (1) and (2), as in the sort-based simplex projection: it
// is bisected on compensated prefix sums, O(k log n) after the sort. Each u
// tests it and its neighbours (rounding at ties) within eps; the first set
// that passes, or else the least violating one, is the solution. The sums
// are in double at least: float loses t to cancellation at a large rho.
template <typename Result, typename Iterator>
inline thresholds<Result, Iterator> topk_cone_search(
    Iterator first, Iterator last, const typename std::iterator_traits<Iterator>::difference_type k,
    const Result rho, const Result unit) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  using Accum = typename std::common_type<Result, double>::type;
  std::sort(first, last, std::greater<Data>());
  const std::size_t n = static_cast<std::size_t>(std::distance(first, last));
  const std::size_t uk = static_cast<std::size_t>(k);
  auto value = [=](const std::size_t i) { return static_cast<Accum>(first[i]) * Accum(unit); };
  const prefix_sums<Accum> sum(n, value);

  const Accum K = Accum(k), rho_k2 = Accum(rho) * K * K, top = std::abs(value(0));
  const Accum inf = std::numeric_limits<Accum>::infinity();
  topk_cone_best<Accum> best;
  for (std::size_t u = 0; u <= uk && best.violation > 0; ++u) {
    const Accum c = K - Accum(u), w = rho_k2 + Accum(u), sum_U = sum(0, u);
    const Accum min_U = u > 0 ? value(u - 1) : inf;
    auto solve = [&](const std::size_t m, Accum& hi) {
      const Accum sum_M = sum(u, u + m), D = c * c + Accum(m) * w;
      hi = (Accum(m) * sum_U + c * sum_M) / D;
      return (w * sum_M - c * sum_U) / D;
    };
    // The largest m with b_m > t(m), or zero.
    std::size_t m_lo = 0, m_hi = n - u;
    while (m_lo < m_hi) {
      const std::size_t m = m_hi - (m_hi - m_lo) / 2;
      Accum hi;
      if (value(u + m - 1) > solve(m, hi))
        m_lo = m;
      else
        m_hi = m - 1;
    }
    for (std::size_t m = m_lo > 0 ? m_lo - 1 : 0; m <= std::min(m_lo + 1, n - u); ++m) {
      if (c == 0 && m == 0) continue;  // D = 0
      Accum hi;
      const Accum t = solve(m, hi);
      const Accum max_M = m > 0 ? value(u) : -inf, min_M = m > 0 ? value(u + m - 1) : inf;
      const Accum max_L = u + m < n ? value(u + m) : -inf;
      // Within the tolerance of the magnitude of U and M, [first, first + u + m).
      const Accum scale = u + m > 0 ? std::max(top, std::abs(value(u + m - 1))) : top;
      const Accum eps = Accum(topk_cone_tolerance(Result(scale)));
      best.update(t, hi, max_M - eps, min_M + eps, min_U + eps, max_L - eps, u, m);
      if (!(best.violation > 0)) break;
    }
  }
  // The first active set that passed within eps, or else (rounding at a tie)
  // the least violating one.
  return thresholds<Result, Iterator>(Result(best.t / Accum(unit)), 0,
                                      Result(best.hi / Accum(unit)), first + best.u,
                                      first + (best.u + best.m));
}

}  // namespace detail

template <typename Result = double, typename Iterator>
inline thresholds<Result, Iterator> thresholds_topk_cone_search(
    Iterator first, Iterator last, const typename std::iterator_traits<Iterator>::difference_type k,
    const Result unit = 1) {
  return detail::topk_cone_search(first, last, k, Result(0), unit);
}

/**
 * Solve
 *    min_x 0.5 * <x, x> - <a, x>
 *          0 <= x_i <= <1, x> / k
 *
 * The solution is
 *    x = max(0, min(a - t, hi))
 **/
template <typename Result = double, typename Iterator>
inline thresholds<Result, Iterator> thresholds_topk_cone(
    Iterator first, Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1) {
  if (k < 1 || k > std::distance(first, last))
    throw std::invalid_argument("Invalid top-k projection dimensions");

  auto proj = topk_cone_special_cases(first, last, k, static_cast<Result>(k));
  if (proj.projection == projection::general) {
    return thresholds_topk_cone_search<Result, Iterator>(first, last, k, proj.unit);
  }
  return proj.thresholds;
}

// [first, last) in place, through a temporary copy.
template <typename Result = double, typename Iterator>
inline void prox_topk_cone(Iterator first, Iterator last,
                           const typename std::iterator_traits<Iterator>::difference_type k = 1) {
  prox(first, last,
       thresholds_topk_cone<
           Result,
           typename std::vector<typename std::iterator_traits<Iterator>::value_type>::iterator>,
       k);
}

// [first, last) in place; aux (as long) is scratch.
template <typename Result = double, typename Iterator>
inline void prox_topk_cone(Iterator first, Iterator last, Iterator aux,
                           const typename std::iterator_traits<Iterator>::difference_type k = 1) {
  prox(first, last, aux, thresholds_topk_cone<Result, Iterator>, k);
}

// Each block of dim consecutive entries of [first, last) in place; aux holds dim
// entries.
template <typename Result = double, typename Iterator>
inline void prox_topk_cone(const typename std::iterator_traits<Iterator>::difference_type dim,
                           Iterator first, Iterator last, Iterator aux,
                           const typename std::iterator_traits<Iterator>::difference_type k = 1) {
  prox(dim, first, last, aux, thresholds_topk_cone<Result, Iterator>, k);
}

}  // namespace sdca

#endif
