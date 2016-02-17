#ifndef SDCA_PROX_TOPK_SIMPLEX_BIASED_H
#define SDCA_PROX_TOPK_SIMPLEX_BIASED_H

#include "sdca/prox/knapsack_eq.h"
#include "sdca/prox/topk_cone_biased.h"

namespace sdca {

template <typename Result,
          typename Iterator>
inline bool
is_topk_simplex_biased_lt(
    const Iterator u_first,
    const Iterator u_last,
    const Result t,
    const Result k,
    const Result rhs,
    const Result rho,
    const Result eps
    ) {
  if (u_first == u_last) {
    return t < rho * rhs - eps;
  } else {
    Result num_U = static_cast<Result>(std::distance(u_first, u_last));
    Result sum_U = std::accumulate(u_first, u_last, static_cast<Result>(0));
    return k * ( sum_U + (k - num_U) * t) < rhs * (num_U + rho * k * k) - eps;
  }
}

/**
 * Solve
 *    min_x 0.5 * (<x, x> + rho * <1, x>^2) - <a, x>
 *          <1, x> <= rhs
 *          0 <= x_i <= <1, x> / k
 *
 * The solution is
 *    x = max(0, min(a - t, hi))
 **/
template <typename Result = double,
          typename Iterator>
inline thresholds<Result, Iterator>
thresholds_topk_simplex_biased(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rhs = 1,
    const Result rho = 1
    ) {
  assert(rho >= 0);
  Result K = static_cast<Result>(k), lo(0);
  Result eps = std::numeric_limits<Result>::epsilon()
    * std::max(static_cast<Result>(1), std::abs(rhs));
  auto proj = topk_cone_special_cases(first, last, k, K + rho * K * K);
  switch (proj.projection) {
    case projection::zero:
      break;
    case projection::constant:
      if (K * proj.thresholds.hi > rhs + eps) {
        return thresholds_knapsack_eq(first, last, lo, rhs / K, rhs);
      }
      break;
    case projection::general:
      auto t = thresholds_knapsack_eq(first, last, lo, rhs / K, rhs);
      if (is_topk_simplex_biased_lt(first, t.first, t.t, K, rhs, rho, eps)) {
        return thresholds_topk_cone_biased_search(first, last, k, rho);
      }
      return t;
  }
  return proj.thresholds;
}

template <typename Result = double,
          typename Iterator>
inline void
prox_topk_simplex_biased(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rhs = 1,
    const Result rho = 1
    ) {
  prox(first, last,
    thresholds_topk_simplex_biased<Result, Iterator>, k, rhs, rho);
}

template <typename Result = double,
          typename Iterator>
inline void
prox_topk_simplex_biased(
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rhs = 1,
    const Result rho = 1
    ) {
  prox(first, last, aux_first, aux_last,
    thresholds_topk_simplex_biased<Result, Iterator>, k, rhs, rho);
}

template <typename Result = double,
          typename Iterator>
inline void
prox_topk_simplex_biased(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rhs = 1,
    const Result rho = 1
    ) {
  prox(dim, first, last, aux_first, aux_last,
    thresholds_topk_simplex_biased<Result, Iterator>, k, rhs, rho);
}

}

#endif
