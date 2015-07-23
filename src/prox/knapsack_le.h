#ifndef SDCA_PROX_KNAPSACK_LE_H
#define SDCA_PROX_KNAPSACK_LE_H

#include "knapsack_eq.h"

namespace sdca {

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
thresholds<Iterator, Result>
thresholds_knapsack_le(
    Iterator first,
    Iterator last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1,
    Summation sum = Summation()
    ) {
  Result eps = std::numeric_limits<Result>::epsilon()
    * std::max(static_cast<Result>(1), std::abs(rhs));

  // First, check if the inequality constraint is active (sum > rhs)
  auto m_first = std::partition(first, last,
    [=](const Result x){ return x >= hi; });
  auto m_last = std::partition(m_first, last,
    [=](const Result x){ return x > lo; });
  Result s = sum(m_first, m_last, static_cast<Result>(0))
    + hi * static_cast<Result>(std::distance(first, m_first))
    + lo * static_cast<Result>(std::distance(m_last, last));

  if (s > rhs + eps) {
    return thresholds_knapsack_eq(first, last, lo, hi, rhs, sum);
  } else {
    return thresholds<Iterator, Result>(0, lo, hi, m_first, m_last);
  }
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
project_knapsack_le(
    Iterator first,
    Iterator last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1,
    Summation sum = Summation()
    ) {
  project(first, last,
    thresholds_knapsack_le<Iterator, Result, Summation>, lo, hi, rhs, sum);
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
project_knapsack_le(
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1,
    Summation sum = Summation()
    ) {
  project(first, last, aux_first, aux_last,
    thresholds_knapsack_le<Iterator, Result, Summation>, lo, hi, rhs, sum);
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
project_knapsack_le(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1,
    Summation sum = Summation()
    ) {
  project(dim, first, last, aux_first, aux_last,
    thresholds_knapsack_le<Iterator, Result, Summation>, lo, hi, rhs, sum);
}

}

#endif
