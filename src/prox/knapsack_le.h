#ifndef SDCA_PROX_KNAPSACK_LE_H
#define SDCA_PROX_KNAPSACK_LE_H

#include "knapsack_eq.h"

namespace sdca {

template <typename Iterator,
          typename Result,
          typename Summator = std_sum<Iterator, Result>>
thresholds<Iterator, Result>
thresholds_knapsack_le(
    Iterator first,
    Iterator last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1,
    Summator sum = Summator()
    ) {
  typedef typename std::iterator_traits<Iterator>::value_type data_type;
  typedef Result result_type;

  // First, check if the inequality constraint is active (sum > rhs)
  auto m_first = std::partition(first, last,
    [=](const result_type x){ return x >= hi; });
  auto m_last = std::partition(m_first, last,
    [=](const result_type x){ return x > lo; });

  result_type s = sum(m_first, m_last, static_cast<result_type>(0))
    + hi * static_cast<result_type>(std::distance(first, m_first))
    + lo * static_cast<result_type>(std::distance(m_last, last));

  if (s > rhs) {
    return thresholds_knapsack_eq(first, last, lo, hi, rhs, sum);
  } else {
    return thresholds<Iterator, Result>(0, lo, hi, m_first, m_last);
  }
}

template <typename Iterator,
          typename Result,
          typename Summator = std_sum<Iterator, Result>>
inline
void
project_knapsack_le(
    Iterator first,
    Iterator last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1,
    Summator sum = Summator()
    ) {
  project(first, last,
    thresholds_knapsack_le<Iterator, Result, Summator>, lo, hi, rhs, sum);
}

template <typename Iterator,
          typename Result,
          typename Summator = std_sum<Iterator, Result>>
inline
void
project_knapsack_le(
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1,
    Summator sum = Summator()
    ) {
  project(first, last, aux_first, aux_last,
    thresholds_knapsack_le<Iterator, Result, Summator>, lo, hi, rhs, sum);
}

template <typename Iterator,
          typename Result,
          typename Summator = std_sum<Iterator, Result>>
inline
void
project_knapsack_le(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1,
    Summator sum = Summator()
    ) {
  project(dim, first, last, aux_first, aux_last,
    thresholds_knapsack_le<Iterator, Result, Summator>, lo, hi, rhs, sum);
}

}

#endif
