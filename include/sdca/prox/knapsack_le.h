#ifndef SDCA_PROX_KNAPSACK_LE_H
#define SDCA_PROX_KNAPSACK_LE_H

#include "sdca/prox/knapsack_eq.h"

namespace sdca {

/**
 * Solve
 *    min_x 0.5 * <x, x> - <a, x>
 *          <1, x> <= rhs
 *          lo <= x_i <= hi
 *
 * The solution is
 *    x = max(lo, min(a - t, hi))
 **/
template <typename Result = double,
          typename Iterator>
inline thresholds<Result, Iterator>
thresholds_knapsack_le(
    Iterator first,
    Iterator last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1
    ) {
  Result eps = std::numeric_limits<Result>::epsilon()
    * std::max(static_cast<Result>(1), std::abs(rhs));

  // First, check if the inequality constraint is active (sum > rhs)
  auto m_first = std::partition(first, last,
    [=](const Result x){ return x >= hi; });
  auto m_last = std::partition(m_first, last,
    [=](const Result x){ return x > lo; });
  Result s = std::accumulate(m_first, m_last, static_cast<Result>(0))
    + hi * static_cast<Result>(std::distance(first, m_first))
    + lo * static_cast<Result>(std::distance(m_last, last));

  if (s > rhs + eps) {
    return thresholds_knapsack_eq(first, last, lo, hi, rhs);
  } else {
    return thresholds<Result, Iterator>(0, lo, hi, m_first, m_last);
  }
}


template <typename Result = double,
          typename Iterator>
inline void
prox_knapsack_le(
    Iterator first,
    Iterator last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1
    ) {
  prox(first, last,
       thresholds_knapsack_le<Result, Iterator>, lo, hi, rhs);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_knapsack_le(
    Iterator first,
    Iterator last,
    Iterator aux,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1
    ) {
  prox(first, last, aux,
       thresholds_knapsack_le<Result, Iterator>, lo, hi, rhs);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_knapsack_le(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1
    ) {
  prox(dim, first, last, aux,
       thresholds_knapsack_le<Result, Iterator>, lo, hi, rhs);
}

}

#endif
