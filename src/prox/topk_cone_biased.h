#ifndef SDCA_PROX_TOPK_CONE_BIASED_H
#define SDCA_PROX_TOPK_CONE_BIASED_H

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>

#include "topk_cone.h"

namespace sdca {

template <typename ForwardIterator>
thresholds<ForwardIterator>
thresholds_topk_cone_biased_search(
    ForwardIterator first,
    ForwardIterator last,
    const typename std::iterator_traits<ForwardIterator>::difference_type k,
    const typename std::iterator_traits<ForwardIterator>::value_type rho
    ) {
  using Type = typename std::iterator_traits<ForwardIterator>::value_type;

  // Sort data to search efficiently
  std::sort(first, last, std::greater<Type>());

  // Precompute some constants
  auto k_last = std::next(first, k);
  Type k_minus_num_U = static_cast<Type>(k);
  Type num_U_plus_rho_k_2 = rho * static_cast<Type>(k) * static_cast<Type>(k);
  Type min_U = +std::numeric_limits<Type>::infinity();
  Type sum_U = 0;

  // Grow U starting with empty
  for (auto m_first = first;;) {

    Type min_M = +std::numeric_limits<Type>::infinity();
    Type max_M = -std::numeric_limits<Type>::infinity();
    Type sum_M = 0, num_M_sum_U = 0;
    Type D = k_minus_num_U * k_minus_num_U;
    Type k_minus_num_U_sum_U = k_minus_num_U * sum_U;

    // Grow M starting with empty
    for (auto m_last = m_first;;) {
      // Compute t and hi as follows (lo = 0 by definition):
      //    t = ((num_U + rho * k^2) * sum_M - (k - num_U) * sum_U) / D
      //   hi = (num_M * sum_U + (k - num_U) * sum_M) / D
      //    D = (k - num_U)^2 + (num_U + rho * k^2) * num_M
      // and check that
      //  (1)  lo + t  >= max_L = (m_last) or (-Inf)
      //  (2)  lo + t  <= min_M = (m_last - 1) or (+Inf)
      //  (3)  hi + t  >= max_M = (m_first) or (-Inf)
      //  (4)  hi + t  <= min_U = (m_first - 1) or (+Inf)

      Type t  = (num_U_plus_rho_k_2 * sum_M - k_minus_num_U_sum_U) / D;
      Type hi = (num_M_sum_U + k_minus_num_U * sum_M) / D;
      Type tt = hi + t;
      if (max_M <= tt && tt <= min_U) {
        if (t <= min_M && ((m_last == last) || *m_last <= t)) {
          return make_thresholds(t, 0, hi, m_first, m_last);
        }
      }

      // Increment the set M
      if (m_last == last) {
        break;
      }
      min_M = *m_last;
      max_M = *m_first;
      sum_M += min_M;
      ++m_last;
      D += num_U_plus_rho_k_2;
      num_M_sum_U += sum_U;
    }

    // Increment the set U
    if (m_first == k_last) {
      break;
    }
    min_U = *m_first;
    sum_U += min_U;
    ++m_first;
    --k_minus_num_U;
    ++num_U_plus_rho_k_2;
  }

  // Default to 0
  return make_thresholds(0, 0, 0, first, first);
}

template <typename ForwardIterator>
thresholds<ForwardIterator>
thresholds_topk_cone_biased(
    ForwardIterator first,
    ForwardIterator last,
    const typename std::iterator_traits<ForwardIterator>::difference_type k = 1,
    const typename std::iterator_traits<ForwardIterator>::value_type rho = 1
    ) {
  using Type = typename std::iterator_traits<ForwardIterator>::value_type;
  const Type K = static_cast<Type>(k);
  auto proj = topk_cone_special_cases(first, last, k, K + rho * K * K);
  if (proj.projection == projection::general) {
    return thresholds_topk_cone_biased_search(first, last, k, rho);
  } else {
    return proj.result;
  }
}

template <typename ForwardIterator>
inline
void
project_topk_cone_biased(
    ForwardIterator first,
    ForwardIterator last,
    const typename std::iterator_traits<ForwardIterator>::difference_type k = 1,
    const typename std::iterator_traits<ForwardIterator>::value_type rho = 1
    ) {
  project(first, last,
          thresholds_topk_cone_biased<ForwardIterator>, k, rho);
}

template <typename ForwardIterator>
inline
void
project_topk_cone_biased(
    ForwardIterator first,
    ForwardIterator last,
    ForwardIterator aux_first,
    ForwardIterator aux_last,
    const typename std::iterator_traits<ForwardIterator>::difference_type k = 1,
    const typename std::iterator_traits<ForwardIterator>::value_type rho = 1
    ) {
  project(first, last, aux_first, aux_last,
          thresholds_topk_cone_biased<ForwardIterator>, k, rho);
}

}

#endif
