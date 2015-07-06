#ifndef SDCA_PROX_KNAPSACK_LE_BIASED_H
#define SDCA_PROX_KNAPSACK_LE_BIASED_H

#include <functional>

#include "knapsack_eq.h"

namespace sdca {

template <typename ForwardIterator>
thresholds<ForwardIterator>
thresholds_knapsack_le_biased_search(
    ForwardIterator first,
    ForwardIterator last,
    const typename std::iterator_traits<ForwardIterator>::value_type lo,
    const typename std::iterator_traits<ForwardIterator>::value_type hi,
    const typename std::iterator_traits<ForwardIterator>::value_type rhs,
    const typename std::iterator_traits<ForwardIterator>::value_type rho
    ) {
  using Type = typename std::iterator_traits<ForwardIterator>::value_type;

  // At this point, rho must be positive
  assert(rho > 0);

  // Sort data to search efficiently
  std::sort(first, last, std::greater<Type>());

  // Precompute some constants
  Type rho_rhs = rho * rhs;
  Type rho_inverse = static_cast<Type>(1) / rho;
  Type num_X = static_cast<Type>(std::distance(first, last));
  Type num_U = 0;
  Type min_U = +std::numeric_limits<Type>::infinity();

  // Grow U starting with empty
  for (auto m_first = first;;) {

    Type min_M = +std::numeric_limits<Type>::infinity();
    Type max_M = -std::numeric_limits<Type>::infinity();
    Type num_M = 0, sum_M = 0;
    Type num_L = num_X - num_U;

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

      Type t = (lo * num_L + hi * num_U + sum_M) / (rho_inverse + num_M);
      if (t <= rho_rhs) {
        Type tt = hi + t;
        if (max_M <= tt && tt <= min_U) {
          tt = lo + t;
          if (tt <= min_M && ((m_last == last) || *m_last <= tt)) {
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
      sum_M += min_M;
      ++num_M;
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
  return make_thresholds(lo, lo, hi, first, first);
}

template <typename ForwardIterator>
thresholds<ForwardIterator>
thresholds_knapsack_le_biased(
    ForwardIterator first,
    ForwardIterator last,
    const typename std::iterator_traits<ForwardIterator>::value_type lo = 0,
    const typename std::iterator_traits<ForwardIterator>::value_type hi = 1,
    const typename std::iterator_traits<ForwardIterator>::value_type rhs = 1,
    const typename std::iterator_traits<ForwardIterator>::value_type rho = 1
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

  // Special cases: 1) equality constraint; 2) t = 0
  if (sum > rhs) {
    auto t = thresholds_knapsack_eq(first, last, lo, hi, rhs);
    if (t.t >= rho * rhs) {
      return t;
    }
  } else if (rho * sum == static_cast<Type>(0)) {
    return make_thresholds(0, lo, hi, m_first, m_last);
  }

  // General case
  return thresholds_knapsack_le_biased_search(first, last, lo, hi, rhs, rho);
}

template <typename ForwardIterator>
inline
void
project_knapsack_le_biased(
    ForwardIterator first,
    ForwardIterator last,
    const typename std::iterator_traits<ForwardIterator>::value_type lo = 0,
    const typename std::iterator_traits<ForwardIterator>::value_type hi = 1,
    const typename std::iterator_traits<ForwardIterator>::value_type rhs = 1,
    const typename std::iterator_traits<ForwardIterator>::value_type rho = 1
    ) {
  project(first, last,
          thresholds_knapsack_le_biased<ForwardIterator>, lo, hi, rhs, rho);
}

template <typename ForwardIterator>
inline
void
project_knapsack_le_biased(
    ForwardIterator first,
    ForwardIterator last,
    ForwardIterator aux_first,
    ForwardIterator aux_last,
    const typename std::iterator_traits<ForwardIterator>::value_type lo = 0,
    const typename std::iterator_traits<ForwardIterator>::value_type hi = 1,
    const typename std::iterator_traits<ForwardIterator>::value_type rhs = 1,
    const typename std::iterator_traits<ForwardIterator>::value_type rho = 1
    ) {
  project(first, last, aux_first, aux_last,
          thresholds_knapsack_le_biased<ForwardIterator>, lo, hi, rhs, rho);
}

}

#endif
