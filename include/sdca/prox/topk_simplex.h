#ifndef SDCA_PROX_TOPK_SIMPLEX_H
#define SDCA_PROX_TOPK_SIMPLEX_H

#include "sdca/prox/knapsack_eq.h"
#include "sdca/prox/topk_cone.h"

namespace sdca {

template <typename Result,
          typename Iterator>
inline bool
is_topk_simplex_lt(
    const Iterator u_first,
    const Iterator u_last,
    const Result t,
    const Result k,
    const Result rhs,
    const Result eps
    ) {
  if (u_first == u_last) {
    return t < -eps;
  } else {
    Result num_U = static_cast<Result>(std::distance(u_first, u_last));
    Result sum_U = std::accumulate(u_first, u_last, static_cast<Result>(0));
    return k * ( sum_U + (k - num_U) * t) < rhs * num_U - eps;
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
template <typename Result = double,
          typename Iterator>
inline thresholds<Result, Iterator>
thresholds_topk_simplex(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rhs = 1
    ) {
  Result K = static_cast<Result>(k), lo(0);
  Result eps = std::numeric_limits<Result>::epsilon()
    * std::max(static_cast<Result>(1), std::abs(rhs));
  auto proj = topk_cone_special_cases(first, last, k, K);
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
      if (is_topk_simplex_lt(first, t.first, t.t, K, rhs, eps)) {
        return thresholds_topk_cone_search<Result, Iterator>(first, last, k);
      }
      return t;
  }
  return proj.thresholds;
}


template <typename Result = double,
          typename Iterator>
inline void
prox_topk_simplex(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rhs = 1
    ) {
  prox(first, last,
       thresholds_topk_simplex<Result, Iterator>, k, rhs);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_topk_simplex(
    Iterator first,
    Iterator last,
    Iterator aux,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rhs = 1
    ) {
  prox(first, last, aux,
       thresholds_topk_simplex<Result, Iterator>, k, rhs);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_topk_simplex(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rhs = 1
    ) {
  prox(dim, first, last, aux,
       thresholds_topk_simplex<Result, Iterator>, k, rhs);
}

}

#endif
