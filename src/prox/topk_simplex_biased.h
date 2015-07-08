#ifndef SDCA_PROX_TOPK_SIMPLEX_BIASED_H
#define SDCA_PROX_TOPK_SIMPLEX_BIASED_H

#include <algorithm>
#include <iterator>
#include <numeric>

#include "knapsack_eq.h"
#include "topk_cone_biased.h"

namespace sdca {

template <typename ForwardIterator>
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

template <typename ForwardIterator>
thresholds<ForwardIterator>
thresholds_topk_simplex_biased(
    ForwardIterator first,
    ForwardIterator last,
    const typename std::iterator_traits<ForwardIterator>::difference_type k = 1,
    const typename std::iterator_traits<ForwardIterator>::value_type rhs = 1,
    const typename std::iterator_traits<ForwardIterator>::value_type rho = 1
    ) {
  using Type = typename std::iterator_traits<ForwardIterator>::value_type;
  const Type K = static_cast<Type>(k);
  auto proj = topk_cone_special_cases(first, last, k, K + rho * K * K);
  switch (proj.type) {
    case projection::zero:
      break;
    case projection::constant:
      if (K * proj.result.hi > rhs) {
        return thresholds_knapsack_eq(first, last, 0, rhs / K, rhs);
      }
      break;
    case projection::general:
      auto t = thresholds_knapsack_eq(first, last, 0, rhs / K, rhs);
      if (is_topk_simplex_biased_lt(first, t.first, t.t, K, rhs, rho)) {
        return thresholds_topk_cone_biased_search(first, last, k, rho);
      }
      return t;
  }
  return proj.result;
}

template <typename ForwardIterator>
inline
void
project_topk_simplex_biased(
    ForwardIterator first,
    ForwardIterator last,
    const typename std::iterator_traits<ForwardIterator>::difference_type k = 1,
    const typename std::iterator_traits<ForwardIterator>::value_type rhs = 1,
    const typename std::iterator_traits<ForwardIterator>::value_type rho = 1
    ) {
  project(first, last,
          thresholds_topk_simplex_biased<ForwardIterator>, k, rhs, rho);
}

template <typename ForwardIterator>
inline
void
project_topk_simplex_biased(
    ForwardIterator first,
    ForwardIterator last,
    ForwardIterator aux_first,
    ForwardIterator aux_last,
    const typename std::iterator_traits<ForwardIterator>::difference_type k = 1,
    const typename std::iterator_traits<ForwardIterator>::value_type rhs = 1,
    const typename std::iterator_traits<ForwardIterator>::value_type rho = 1
    ) {
  project(first, last, aux_first, aux_last,
          thresholds_topk_simplex_biased<ForwardIterator>, k, rhs, rho);
}

template <typename ForwardIterator>
inline
void
project_topk_simplex_biased(
    const typename std::iterator_traits<ForwardIterator>::difference_type dim,
    ForwardIterator first,
    ForwardIterator last,
    ForwardIterator aux_first,
    ForwardIterator aux_last,
    const typename std::iterator_traits<ForwardIterator>::difference_type k = 1,
    const typename std::iterator_traits<ForwardIterator>::value_type rhs = 1,
    const typename std::iterator_traits<ForwardIterator>::value_type rho = 1
    ) {
  project(dim, first, last, aux_first, aux_last,
          thresholds_topk_simplex_biased<ForwardIterator>, k, rhs, rho);
}

}

#endif
