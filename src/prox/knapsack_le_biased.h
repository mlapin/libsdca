#ifndef SDCA_PROX_KNAPSACK_LE_BIASED_H
#define SDCA_PROX_KNAPSACK_LE_BIASED_H

#include <functional>

#include "knapsack_eq.h"

namespace sdca {

template <typename Iterator,
          typename Result>
thresholds<Iterator, Result>
thresholds_knapsack_le_biased_search(
    Iterator first,
    Iterator last,
    const Result lo,
    const Result hi,
    const Result rhs,
    const Result rho
    ) {
  typedef typename std::iterator_traits<Iterator>::value_type data_type;
  typedef Result result_type;

  // At this point, rho must be positive
  assert(rho > 0);

  // Sort data to search efficiently
  std::sort(first, last, std::greater<data_type>());

  // Precompute some constants
  result_type rho_rhs = rho * rhs;
  result_type rho_inverse = static_cast<result_type>(1) / rho;
  result_type num_X = static_cast<result_type>(std::distance(first, last));
  result_type num_U = 0;
  result_type min_U = +std::numeric_limits<result_type>::infinity();

  // Grow U starting with empty
  for (auto m_first = first;;) {

    result_type min_M = +std::numeric_limits<result_type>::infinity();
    result_type max_M = -std::numeric_limits<result_type>::infinity();
    result_type num_M = 0, sum_M = 0;
    result_type num_L = num_X - num_U;

    // Grow M starting with empty
    for (auto m_last = m_first;;) {
      // Compute t as follows:
      //   t = (lo * num_L + hi * num_U + sum_M) / (1/rho + num_M)
      // and check that
      //  (1)  lo + t  >= max_L = (m_last) or (-Inf)
      //  (2)  lo + t  <= min_M = (m_last - 1) or (+Inf)
      //  (3)  hi + t  >= max_M = (m_first) or (-Inf)
      //  (4)  hi + t  <= min_U = (m_first - 1) or (+Inf)
      //  (5)       t  <= rho * rhs

      result_type t = (lo * num_L + hi * num_U + sum_M) / (rho_inverse + num_M);
      if (t <= rho_rhs) {
        result_type tt = hi + t;
        if (max_M <= tt && tt <= min_U) {
          tt = lo + t;
          if (tt <= min_M &&
              ((m_last == last) || static_cast<result_type>(*m_last) <= tt)) {
            return make_thresholds(t, lo, hi, m_first, m_last);
          }
        }
      }

      // Increment the set M
      if (m_last == last) {
        break;
      }
      min_M = *m_last;
      max_M = *m_first;
      sum_M += min_M; // TODO: kahan_add
      ++num_M;
      --num_L;
      ++m_last;
    }

    // Increment the set U
    if (m_first == last) {
      break;
    }
    min_U = *m_first;
    ++num_U;
    ++m_first;
  }

  // Should never reach here
  assert(false);
  return thresholds<Iterator, Result>(0, lo, hi, first, first);
}

template <typename Iterator,
          typename Result,
          typename Summator = std_sum<Iterator, Result>>
thresholds<Iterator, Result>
thresholds_knapsack_le_biased(
    Iterator first,
    Iterator last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1,
    const Result rho = 1,
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

  // Special cases: 1) equality constraint; 2) t = 0
  if (s > rhs) {
    auto t = thresholds_knapsack_eq(first, last, lo, hi, rhs, sum);
    if (t.t >= rho * rhs) {
      return t;
    }
  } else if (rho * s == static_cast<result_type>(0)) {
    return thresholds<Iterator, Result>(0, lo, hi, m_first, m_last);
  }

  // General case
  return thresholds_knapsack_le_biased_search(first, last, lo, hi, rhs, rho);
}

template <typename Iterator,
          typename Result,
          typename Summator = std_sum<Iterator, Result>>
inline
void
project_knapsack_le_biased(
    Iterator first,
    Iterator last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1,
    const Result rho = 1,
    Summator sum = Summator()
    ) {
  project(first, last,
    thresholds_knapsack_le_biased<Iterator, Result, Summator>,
    lo, hi, rhs, rho, sum);
}

template <typename Iterator,
          typename Result,
          typename Summator = std_sum<Iterator, Result>>
inline
void
project_knapsack_le_biased(
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1,
    const Result rho = 1,
    Summator sum = Summator()
    ) {
  project(first, last, aux_first, aux_last,
    thresholds_knapsack_le_biased<Iterator, Result, Summator>,
    lo, hi, rhs, rho, sum);
}

template <typename Iterator,
          typename Result,
          typename Summator = std_sum<Iterator, Result>>
inline
void
project_knapsack_le_biased(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1,
    const Result rho = 1,
    Summator sum = Summator()
    ) {
  project(dim, first, last, aux_first, aux_last,
    thresholds_knapsack_le_biased<Iterator, Result, Summator>,
    lo, hi, rhs, rho, sum);
}

}

#endif
