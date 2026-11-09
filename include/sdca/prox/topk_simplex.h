#ifndef SDCA_PROX_TOPK_SIMPLEX_H
#define SDCA_PROX_TOPK_SIMPLEX_H

#include "sdca/prox/knapsack_eq.h"
#include "sdca/prox/topk_cone.h"

namespace sdca {

template <typename Result, typename Iterator>
inline bool is_topk_simplex_lt(const Iterator u_first, const Iterator u_last, const Result t,
                               const Result k, const Result rhs, const Result eps,
                               const Result unit = 1) {
  if (u_first == u_last) {
    return t < -eps;
  } else {
    Result num_U = static_cast<Result>(std::distance(u_first, u_last));
    Result sum_U = 0;
    for (auto it = u_first; it != u_last; ++it) sum_U += static_cast<Result>(*it) * unit;
    return k * (sum_U + (k - num_U) * t) < rhs * num_U - eps;
  }
}

/**
 * Solve
 *    min_x 0.5 * <x, x> - <a, x>
 *          <1, x> <= rhs
 *          0 <= x_i <= <1, x> / k
 *
 * The solution is
 *    x = max(0, min(a - t, hi))
 **/
template <typename Result = double, typename Iterator>
inline thresholds<Result, Iterator> thresholds_topk_simplex(
    Iterator first, Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1, const Result rhs = 1) {
  if (k < 1 || k > std::distance(first, last))
    throw std::invalid_argument("Invalid top-k projection dimensions");

  detail::validate_topk_simplex(rhs);
  Result K = static_cast<Result>(k), lo(0);
  // The tests and the cone search are in units of 1/unit (see the cone).
  auto proj = topk_cone_special_cases(first, last, k, K, rhs);
  const Result unit = proj.unit, rhs_u = rhs * unit;
  Result eps = detail::topk_simplex_tolerance(proj.scale, rhs_u);
  switch (proj.projection) {
    case projection::zero:
      break;
    case projection::constant:
      if (K * (proj.thresholds.hi * unit) > rhs_u + eps) {
        return thresholds_knapsack_eq(first, last, lo, rhs / K, rhs);
      }
      break;
    case projection::general:
      auto t = thresholds_knapsack_eq(first, last, lo, rhs / K, rhs);
      if (is_topk_simplex_lt(first, t.first, t.t * unit, K, rhs_u, eps, unit)) {
        return thresholds_topk_cone_search<Result, Iterator>(first, last, k, unit);
      }
      return t;
  }
  return proj.thresholds;
}

// [first, last) in place, through a temporary copy.
template <typename Result = double, typename Iterator>
inline void prox_topk_simplex(Iterator first, Iterator last,
                              const typename std::iterator_traits<Iterator>::difference_type k = 1,
                              const Result rhs = 1) {
  prox(first, last,
       thresholds_topk_simplex<
           Result,
           typename std::vector<typename std::iterator_traits<Iterator>::value_type>::iterator>,
       k, rhs);
}

// [first, last) in place; aux (as long) is scratch.
template <typename Result = double, typename Iterator>
inline void prox_topk_simplex(Iterator first, Iterator last, Iterator aux,
                              const typename std::iterator_traits<Iterator>::difference_type k = 1,
                              const Result rhs = 1) {
  prox(first, last, aux, thresholds_topk_simplex<Result, Iterator>, k, rhs);
}

// Each block of dim consecutive entries of [first, last) in place; aux holds dim
// entries.
template <typename Result = double, typename Iterator>
inline void prox_topk_simplex(const typename std::iterator_traits<Iterator>::difference_type dim,
                              Iterator first, Iterator last, Iterator aux,
                              const typename std::iterator_traits<Iterator>::difference_type k = 1,
                              const Result rhs = 1) {
  prox(dim, first, last, aux, thresholds_topk_simplex<Result, Iterator>, k, rhs);
}

}  // namespace sdca

#endif
