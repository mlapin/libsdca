#ifndef SDCA_PROX_TOPK_SIMPLEX_BIASED_H
#define SDCA_PROX_TOPK_SIMPLEX_BIASED_H

#include <algorithm>
#include <iterator>
#include <numeric>

#include "knapsack_eq.h"
#include "topk_cone_biased.h"

namespace sdca {

template <class ForwardIterator>
thresholds<ForwardIterator>
thresholds_topk_simplex_biased(
    ForwardIterator first,
    ForwardIterator last,
    const typename std::iterator_traits<ForwardIterator>::difference_type k,
    const typename std::iterator_traits<ForwardIterator>::value_type rhs,
    const typename std::iterator_traits<ForwardIterator>::value_type rho
    ) {
  using Type = typename std::iterator_traits<ForwardIterator>::value_type;
  const Type K = static_cast<Type>(k);
  auto proj = topk_cone_special_cases(first, last, k, K);
  switch (proj.projection) {
    case projection_case::constant:
      if (K * proj.result.hi > rhs) {
        return thresholds_knapsack_eq(first, last, 0, rhs / K, rhs);
      }
      break;
    case projection_case::general:
      auto t = thresholds_knapsack_eq(first, last, 0, rhs / K, rhs);
      if (is_topk_simplex_biased_lt(first, t.first, t.t, K, rhs, rho)) {
        return thresholds_topk_cone_biased_search(first, last, k, rho);
      }
      return t;
  }
  return proj.result;
}

template <class ForwardIterator>
inline
bool
is_topk_simplex_biased_lt(
    const ForwardIterator u_first,
    const ForwardIterator u_last,
    const typename std::iterator_traits<ForwardIterator>::value_type t,
    const typename std::iterator_traits<ForwardIterator>::value_type k,
    const typename std::iterator_traits<ForwardIterator>::value_type rhs,
    const typename std::iterator_traits<ForwardIterator>::value_type rho
    ) {
  using Type = typename std::iterator_traits<ForwardIterator>::value_type;
  auto size = std::distance(u_first, u_last);
  if (size) {
    const Type num_U = static_cast<Type>(size);
    const Type sum_U = std::accumulate(u_first, u_last, static_cast<Type>(0));
    return k * ( sum_U + (k - num_U) * t) < rhs * (num_U + rho * k * k);
  } else {
    return t < rho * rhs;
  }
}

}

#endif
