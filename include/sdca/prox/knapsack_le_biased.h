#ifndef SDCA_PROX_KNAPSACK_LE_BIASED_H
#define SDCA_PROX_KNAPSACK_LE_BIASED_H

#include <functional>

#include "sdca/prox/knapsack_le.h"

namespace sdca {

template <typename Result,
          typename Iterator>
inline thresholds<Result, Iterator>
thresholds_knapsack_le_biased_search(
    Iterator first,
    Iterator last,
    const Result lo,
    const Result hi,
    const Result rhs,
    const Result rho
    ) {
  // At this point, rho must be positive
  assert(rho > 0);
  Result eps = std::numeric_limits<Result>::epsilon()
    * std::max(static_cast<Result>(1), std::abs(rhs));

  // Sort data to search efficiently
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  std::sort(first, last, std::greater<Data>());

  // Precompute some constants
  Result rho_rhs = rho * rhs;
  Result rho_inverse = static_cast<Result>(1) / rho;
  Result num_X = static_cast<Result>(std::distance(first, last));
  Result num_U = 0, min_U = +std::numeric_limits<Result>::infinity();

  // Grow U starting with empty
  for (auto m_first = first;;) {

    Result min_M = +std::numeric_limits<Result>::infinity();
    Result max_M = -std::numeric_limits<Result>::infinity();
    Result num_M = 0, sum_M = 0;
    Result num_L = num_X - num_U;

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

      Result t = (lo * num_L + hi * num_U + sum_M) / (rho_inverse + num_M);
      if (t <= rho_rhs + eps) {
        Result tt = hi + t;
        if (max_M - eps <= tt && tt <= min_U + eps) {
          tt = lo + t;
          if (tt <= min_M + eps &&
              ((m_last == last) || static_cast<Result>(*m_last) - eps <= tt)) {
            return make_thresholds(t, lo, hi, m_first, m_last);
          }
        }
      }

      // Increment the set M
      if (m_last == last) {
        break;
      }
      min_M = static_cast<Result>(*m_last);
      max_M = static_cast<Result>(*m_first);
      sum_M += min_M;
      --num_L;
      ++num_M;
      ++m_last;
    }

    // Increment the set U
    if (m_first == last) {
      break;
    }
    min_U = static_cast<Result>(*m_first);
    ++num_U;
    ++m_first;
  }

  // Default to 0
  return thresholds<Result, Iterator>(0, 0, 0, first, first);
}


/**
 * Solve
 *    min_x 0.5 * (<x, x> + rho * <1, x>^2) - <a, x>
 *          <1, x> <= rhs
 *          lo <= x_i <= hi
 *
 * The solution is
 *    x = max(lo, min(a - t, hi))
 **/
template <typename Result = double,
          typename Iterator>
inline thresholds<Result, Iterator>
thresholds_knapsack_le_biased(
    Iterator first,
    Iterator last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1,
    const Result rho = 1
    ) {
  assert(rho >= 0);
  if (rho == 0) {
    return thresholds_knapsack_le(first, last, lo, hi, rhs);
  }

  Result eps = std::numeric_limits<Result>::epsilon()
    * std::max(static_cast<Result>(1), std::abs(rhs));

  // Equality constraint
  auto t = thresholds_knapsack_eq(first, last, lo, hi, rhs);
  if (t.t >= rho * rhs - eps) {
    return t;
  }

  // General case
  return thresholds_knapsack_le_biased_search(first, last, lo, hi, rhs, rho);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_knapsack_le_biased(
    Iterator first,
    Iterator last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1,
    const Result rho = 1
    ) {
  prox(first, last,
       thresholds_knapsack_le_biased<Result, Iterator>, lo, hi, rhs, rho);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_knapsack_le_biased(
    Iterator first,
    Iterator last,
    Iterator aux,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1,
    const Result rho = 1
    ) {
  prox(first, last, aux,
       thresholds_knapsack_le_biased<Result, Iterator>, lo, hi, rhs, rho);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_knapsack_le_biased(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1,
    const Result rho = 1
    ) {
  prox(dim, first, last, aux,
       thresholds_knapsack_le_biased<Result, Iterator>, lo, hi, rhs, rho);
}

}

#endif
