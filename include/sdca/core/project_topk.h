#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_CORE_PROJECT_TOPK_H)
#if SDCA_ISA_ID == 1
#define SDCA_CORE_PROJECT_TOPK_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_CORE_PROJECT_TOPK_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_CORE_PROJECT_TOPK_H3 1
#else
#define SDCA_CORE_PROJECT_TOPK_H4 1
#endif

#include <functional>

#include "sdca/core/project_knapsack.h"
#include "sdca/core/lanes.h"

SDCA_ISA_BEGIN
namespace core {
namespace topk {

// The candidate active set (|U|, |M|) = (u, m) of a cone search that violates its conditions
// least, for when rounding at a tie rejects all.
template <typename Result>
struct best_set {
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

enum class shape { zero, constant, general };

template <typename Result, typename Iterator>
struct special_case {
  shape kind;
  core::thresholds<Result, Iterator> thresholds;
  Result unit;      // a power of two that keeps the search's sums finite (usually one)
  Result hi_error;  // a bound on the error of the constant hi, in units of 1 / unit
};

// The zero and constant solutions of a top-k cone or simplex, with the k largest scores moved
// to the front. The sums are in Accum (double at least) with u its unit roundoff: the sum of
// the k largest errs by (k - 1) u (sum of magnitudes), at most k u for k^2 u <= 1; each test
// below compares a computed value with exact scores, within that value's error bound.
template <typename Result, typename Iterator>
inline special_case<Result, Iterator> special_cases(
    Iterator first, Iterator last, const typename std::iterator_traits<Iterator>::difference_type k,
    const Result rho, const Result rhs) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  bool finite = true;
  Result magnitude = std::abs(rhs);
  for (Iterator it = first; it != last; ++it) {
    finite &= *it - *it == 0;
    magnitude = std::max(magnitude, std::abs(Result(*it)));
  }
  if (!finite) throw std::invalid_argument("Top-k scores must be finite");
  // Near overflow the solve runs in units of a power of two (exact): the searches' sums and
  // their products with counts and rho k^2 stay finite.
  special_case<Result, Iterator> sc;
  const Result num_X = Result(std::distance(first, last)), K = Result(k);
  sc.unit = overflow_unit(magnitude, num_X * (num_X + K * (K + rho * K * K)));
  sc.hi_error = 0;
  const Result unit = sc.unit;
  Iterator k_last = std::next(first, k);
  std::nth_element(first, k_last - 1, last, std::greater<Data>());
  // In double at least: float cancels at a large rho.
  typedef typename std::common_type<Result, double>::type Accum;
  const Accum u = limits::roundoff<Accum>(), ka = Accum(k);
  Accum scale = 0, sum_k_largest = 0;
  for (Iterator it = first; it != k_last; ++it) {
    const Accum x = Accum(*it) * Accum(unit);
    scale += std::abs(x);
    sum_k_largest += x;
  }
  const Accum sum_error = ka * u * scale;
  if (sum_k_largest <= sum_error) {
    sc.kind = shape::zero;
    sc.thresholds = thresholds<Result, Iterator>(0, 0, 0, first, first);
    return sc;
  }
  // The k largest at hi + t and the rest at zero. div = k + rho k^2 rounds twice (k^2 is
  // exact), hi = sum / div once more, t = x_k - hi once.
  const Accum div = ka + Accum(rho) * (ka * ka), hi = sum_k_largest / div;
  const Accum t = Accum(*(k_last - 1)) * Accum(unit) - hi;
  const Accum hi_error = sum_error / div + 3 * u * hi, t_error = hi_error + u * std::abs(t);
  sc.hi_error = Result(hi_error);
  bool constant = k == std::distance(first, last);
  if (!constant) {
    const Accum max_L = Accum(*std::max_element(k_last, last)) * Accum(unit);
    constant = t >= max_L - t_error;
  }
  if (constant) {
    sc.kind = shape::constant;
    sc.thresholds = thresholds<Result, Iterator>(Result(t / Accum(unit)), 0,
                                                 Result(hi / Accum(unit)), k_last, k_last);
    return sc;
  }
  sc.kind = shape::general;
  return sc;
}

// The search for rho >= 0 over scores sorted here in decreasing order, for
// U = [first, first + u) at hi + t, M = [first + u, first + u + m) free and L at zero:
//    t = ((u + rho k^2) S_M - (k - u) S_U) / D
//   hi = (m S_U + (k - u) S_M) / D
//    D = (k - u)^2 + (u + rho k^2) m
// such that
//  (1)  t >= max_L,  (2)  t <= min_M,  (3)  hi + t >= max_M,  (4)  hi + t <= min_U.
// For a fixed u, t(m) is a weighted mean of t(m - 1) and the m-th score b_m of M (D grows by
// u + rho k^2), so b_m > t(m) holds on a prefix of m whose last m satisfies (1) and (2): it
// is bisected on compensated prefix sums, O(k log n) after the sort. Each u tests it and its
// neighbours (rounding at ties): the first set that passes outright is the solution; where none
// does, the first that passes within the error bounds of t and hi + t, or else the least
// violating one. The sums are in double at least: float loses t to cancellation at a large rho.
//
// The bounds, with u the unit roundoff of Accum (to first order, and u^2 terms of the prefix):
// a range of the compensated prefix sums errs by 2 u (sum of its magnitudes) and u^2 j (those
// of its prefix of j entries), the magnitudes from the same sums split at the first negative
// score (to first order); w = rho k^2 + u by 2 u w (k^2 is exact); D = c^2 + m w by 4 u D
// (c^2 and m are exact); a numerator p S + q S' by p e(S) + q e(S') + u (p |S| + q |S'|) and
// u of itself (2 u w |S_M| more for p = w); its quotient by D by 5 u of the quotient more, and
// hi + t by u of itself more.
template <typename Result, typename Iterator>
inline thresholds<Result, Iterator> cone_search(
    Iterator first, Iterator last, const typename std::iterator_traits<Iterator>::difference_type k,
    const Result rho, const Result unit) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  typedef typename std::common_type<Result, double>::type Accum;
  std::sort(first, last, std::greater<Data>());
  const std::size_t n = std::size_t(std::distance(first, last)), uk = std::size_t(k);
  auto value = [=](const std::size_t i) { return Accum(first[i]) * Accum(unit); };
  const prefix_sums<Accum> sum(n, value);
  // The sum of |b| over [i, j): the scores before the first negative one, less the others.
  const std::size_t zero = std::size_t(
      std::partition_point(first, last, [](const Data x) { return !(x < Data(0)); }) - first);
  auto size = [&](const std::size_t i, const std::size_t j) {
    const std::size_t mid = std::max(i, std::min(j, zero));
    return sum(i, mid) - sum(mid, j);
  };
  const Accum u_A = limits::roundoff<Accum>();
  auto range_error = [&](const std::size_t i, const std::size_t j) {
    return 2 * u_A * size(i, j) + u_A * u_A * Accum(j) * size(0, j);
  };
  const Accum K = Accum(k), rho_k2 = Accum(rho) * K * K;
  const Accum inf = std::numeric_limits<Accum>::infinity();
  best_set<Accum> best;
  // A set that passes its tests outright needs no tolerance: the first such is the solution,
  // and only where none is (rounding at a tie) does a second search allow for the errors.
  for (int tolerant = 0; tolerant < 2 && best.violation > 0; ++tolerant) {
    best = best_set<Accum>();
    for (std::size_t u = 0; u <= uk && best.violation > 0; ++u) {
      const Accum c = K - Accum(u), w = rho_k2 + Accum(u), sum_U = sum(0, u);
      const Accum min_U = u > 0 ? value(u - 1) : inf;
      auto solve_t = [&](const std::size_t m) {
        return (w * sum(u, u + m) - c * sum_U) / (c * c + Accum(m) * w);
      };
      // The largest m with b_m > t(m), or zero.
      std::size_t m_lo = 0, m_hi = n - u;
      while (m_lo < m_hi) {
        const std::size_t m = m_hi - (m_hi - m_lo) / 2;
        if (value(u + m - 1) > solve_t(m))
          m_lo = m;
        else
          m_hi = m - 1;
      }
      const Accum e_U = tolerant ? range_error(0, u) : Accum(0);
      for (std::size_t m = m_lo > 0 ? m_lo - 1 : 0; m <= std::min(m_lo + 1, n - u); ++m) {
        if (c == 0 && m == 0) continue;  // D = 0
        const Accum mm = Accum(m), sum_M = sum(u, u + m), D = c * c + mm * w;
        const Accum n_t = w * sum_M - c * sum_U, n_h = mm * sum_U + c * sum_M;
        const Accum t = n_t / D, hi = n_h / D;
        const Accum max_M = m > 0 ? value(u) : -inf, min_M = m > 0 ? value(u + m - 1) : inf;
        const Accum max_L = u + m < n ? value(u + m) : -inf;
        Accum t_error(0), tt_error(0);
        if (tolerant) {
          const Accum e_M = range_error(u, u + m), inv_D = 1 / D;
          t_error = (w * e_M + c * e_U +
                     u_A * (3 * w * std::abs(sum_M) + c * std::abs(sum_U) + std::abs(n_t))) *
                        inv_D +
                    5 * u_A * std::abs(t);
          const Accum hi_error =
              (mm * e_U + c * e_M +
               u_A * (mm * std::abs(sum_U) + c * std::abs(sum_M) + std::abs(n_h))) *
                  inv_D +
              5 * u_A * std::abs(hi);
          tt_error = t_error + hi_error + u_A * std::abs(hi + t);
        }
        best.update(t, hi, max_M - tt_error, min_M + t_error, min_U + tt_error, max_L - t_error, u,
                    m);
        if (!(best.violation > 0)) break;
      }
    }
  }
  return thresholds<Result, Iterator>(Result(best.t / Accum(unit)), 0,
                                      Result(best.hi / Accum(unit)), first + best.u,
                                      first + (best.u + best.m));
}

template <typename Iterator>
inline void check_k(Iterator first, Iterator last,
                    const typename std::iterator_traits<Iterator>::difference_type k) {
  if (k < 1 || k > std::distance(first, last))
    throw std::invalid_argument("Invalid top-k projection dimensions");
}

}  // namespace topk

/**
 * x = max(0, min(a - t, hi)) minimizing 0.5 (<x, x> + rho <1, x>^2) - <a, x>
 * subject to 0 <= x_i <= <1, x> / k.
 **/
template <typename Result = double, typename Iterator>
inline thresholds<Result, Iterator> thresholds_topk_cone_biased(
    Iterator first, Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1, const Result rho = 1) {
  topk::check_k(first, last, k);
  if (!(rho >= 0)) throw std::invalid_argument("rho must be nonnegative");
  topk::special_case<Result, Iterator> sc = topk::special_cases(first, last, k, rho, Result(0));
  if (sc.kind == topk::shape::general) return topk::cone_search(first, last, k, rho, sc.unit);
  return sc.thresholds;
}

// thresholds_topk_cone_biased at rho = 0.
template <typename Result = double, typename Iterator>
inline thresholds<Result, Iterator> thresholds_topk_cone(
    Iterator first, Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1) {
  return thresholds_topk_cone_biased<Result>(first, last, k, Result(0));
}

/**
 * x = max(0, min(a - t, hi)) minimizing 0.5 (<x, x> + rho <1, x>^2) - <a, x>
 * subject to <1, x> <= rhs, 0 <= x_i <= <1, x> / k.
 **/
template <typename Result = double, typename Iterator>
inline thresholds<Result, Iterator> thresholds_topk_simplex_biased(
    Iterator first, Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1, const Result rhs = 1,
    const Result rho = 1) {
  topk::check_k(first, last, k);
  if (!(rho >= 0)) throw std::invalid_argument("rho must be nonnegative");
  if (!(rhs >= 0) || !std::isfinite(rhs))
    throw std::invalid_argument("Infeasible top-k projection");
  const Result K = Result(k);
  // The tests and the cone search are in units of 1 / unit (exact). Each test compares two
  // computed sides within their roundings in Result (unit roundoff u): the constant's hi errs
  // by sc.hi_error and its rounding, the threshold tt by its rounding, a sum of j entries by
  // (j - 1) u of its magnitudes, and each product and sum by u of itself.
  topk::special_case<Result, Iterator> sc = topk::special_cases(first, last, k, rho, rhs);
  const Result unit = sc.unit, rhs_u = rhs * unit, u = limits::roundoff<Result>();
  switch (sc.kind) {
    case topk::shape::zero:
      break;
    case topk::shape::constant: {
      const Result mass = K * (sc.thresholds.hi * unit);
      if (mass > rhs_u + (K * sc.hi_error + 2 * u * mass))
        return thresholds_knapsack_eq(first, last, Result(0), rhs / K, rhs);
      break;
    }
    case topk::shape::general: {
      thresholds<Result, Iterator> t = thresholds_knapsack_eq(first, last, Result(0), rhs / K, rhs);
      // Whether the equality's solution leaves the top-k cone's optimality conditions.
      const Result tt = t.t * unit;
      bool below;
      if (first == t.first) {
        const Result bound = rho * rhs_u;
        below = tt < bound - u * (std::abs(tt) + bound);
      } else {
        const Result num_U = Result(std::distance(first, t.first));
        Result sum_U = 0, size_U = 0;
        for (Iterator it = first; it != t.first; ++it) {
          sum_U += Result(*it) * unit;
          size_U += std::abs(Result(*it) * unit);
        }
        const Result rest = (K - num_U) * tt, inner = sum_U + rest, lhs = K * inner;
        const Result rhs_side = rhs_u * (num_U + rho * K * K);
        const Result error =
            K * ((num_U - 1) * u * size_U + 2 * u * std::abs(rest) + u * std::abs(inner)) +
            u * std::abs(lhs) + 4 * u * rhs_side;
        below = lhs < rhs_side - error;
      }
      if (below) return topk::cone_search(first, last, k, rho, unit);
      return t;
    }
  }
  return sc.thresholds;
}

// thresholds_topk_simplex_biased at rho = 0.
template <typename Result = double, typename Iterator>
inline thresholds<Result, Iterator> thresholds_topk_simplex(
    Iterator first, Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1, const Result rhs = 1) {
  return thresholds_topk_simplex_biased<Result>(first, last, k, rhs, Result(0));
}

}  // namespace core
SDCA_ISA_END

#endif  // SDCA_CORE_PROJECT_TOPK_H
