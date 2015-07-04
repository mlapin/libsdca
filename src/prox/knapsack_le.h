#ifndef SDCA_PROX_KNAPSACK_LE_H
#define SDCA_PROX_KNAPSACK_LE_H

#include "knapsack_eq.h"

namespace sdca {

template <class ForwardIterator>
thresholds<ForwardIterator>
thresholds_knapsack_le(
    ForwardIterator first,
    ForwardIterator last,
    const typename std::iterator_traits<ForwardIterator>::value_type lo,
    const typename std::iterator_traits<ForwardIterator>::value_type hi,
    const typename std::iterator_traits<ForwardIterator>::value_type rhs
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
    return make_thresholds(static_cast<Type>(0), lo, hi, m_first, m_last);
  }
}

}

#endif
