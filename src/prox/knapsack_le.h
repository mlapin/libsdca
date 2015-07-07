#ifndef SDCA_PROX_KNAPSACK_LE_H
#define SDCA_PROX_KNAPSACK_LE_H

#include "knapsack_eq.h"

namespace sdca {

template <typename ForwardIterator>
thresholds<ForwardIterator>
thresholds_knapsack_le(
    ForwardIterator first,
    ForwardIterator last,
    const typename std::iterator_traits<ForwardIterator>::value_type lo = 0,
    const typename std::iterator_traits<ForwardIterator>::value_type hi = 1,
    const typename std::iterator_traits<ForwardIterator>::value_type rhs = 1
    ) {
  using Type = typename std::iterator_traits<ForwardIterator>::value_type;

  // First, check if the inequality constraint is active (sum > rhs)
  auto m_first = std::partition(first, last,
    [=](const Type &x){ return x >= hi; });
  auto m_last = std::partition(m_first, last,
    [=](const Type &x){ return x > lo; });

  auto sum = std::accumulate(m_first, m_last, static_cast<Type>(0));
  sum += hi * static_cast<Type>(std::distance(first, m_first));
  sum += lo * static_cast<Type>(std::distance(m_last, last));

  // If (sum > rhs), we have an equality constraint; otherwise t = 0
  if (sum > rhs) {
    return thresholds_knapsack_eq(first, last, lo, hi, rhs);
  } else {
    return make_thresholds(0, lo, hi, m_first, m_last);
  }
}

template <typename ForwardIterator>
inline
void
project_knapsack_le(
    ForwardIterator first,
    ForwardIterator last,
    const typename std::iterator_traits<ForwardIterator>::value_type lo = 0,
    const typename std::iterator_traits<ForwardIterator>::value_type hi = 1,
    const typename std::iterator_traits<ForwardIterator>::value_type rhs = 1
    ) {
  project(first, last,
          thresholds_knapsack_le<ForwardIterator>, lo, hi, rhs);
}

template <typename ForwardIterator>
inline
void
project_knapsack_le(
    ForwardIterator first,
    ForwardIterator last,
    ForwardIterator aux_first,
    ForwardIterator aux_last,
    const typename std::iterator_traits<ForwardIterator>::value_type lo = 0,
    const typename std::iterator_traits<ForwardIterator>::value_type hi = 1,
    const typename std::iterator_traits<ForwardIterator>::value_type rhs = 1
    ) {
  project(first, last, aux_first, aux_last,
          thresholds_knapsack_le<ForwardIterator>, lo, hi, rhs);
}

template <typename ForwardIterator>
inline
void
project_knapsack_le(
    const typename std::iterator_traits<ForwardIterator>::difference_type dim,
    ForwardIterator first,
    ForwardIterator last,
    ForwardIterator aux_first,
    ForwardIterator aux_last,
    const typename std::iterator_traits<ForwardIterator>::value_type lo = 0,
    const typename std::iterator_traits<ForwardIterator>::value_type hi = 1,
    const typename std::iterator_traits<ForwardIterator>::value_type rhs = 1
    ) {
  project(dim, first, last, aux_first, aux_last,
          thresholds_knapsack_le<ForwardIterator>, lo, hi, rhs);
}

}

#endif
