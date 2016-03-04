#ifndef SDCA_PROX_KNAPSACK_EQ_H
#define SDCA_PROX_KNAPSACK_EQ_H

#include <cmath>

#include "sdca/prox/proxdef.h"

/*
 * Based on the Algorithm 3.1 in Kiwiel, K. C.,
 * "Variable fixing algorithms for the continuous quadratic knapsack problem.",
 * Journal of Optimization Theory and Applications 136.3 (2008): 445-458.
 */

namespace sdca {

/**
 * Solve
 *    min_x 0.5 * <x, x> - <a, x>
 *          <1, x> = rhs
 *          lo <= x_i <= hi
 *
 * The solution is
 *    x = max(lo, min(a - t, hi))
 **/
template <typename Result = double,
          typename Iterator>
inline thresholds<Result, Iterator>
thresholds_knapsack_eq(
    Iterator first,
    Iterator last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1
    ) {
  assert(std::distance(first, last) > 0);

  // Initialization
  Result eps = std::numeric_limits<Result>::epsilon()
             * std::max(static_cast<Result>(1), std::abs(rhs));

  auto m = std::distance(first, last);
  Result t = (std::accumulate(first, last, static_cast<Result>(0)) - rhs)
             / static_cast<Result>(m);

  Iterator m_first(first), m_last(last);
  for (;;) {
    // Re-partition and compute sums
    Result tt = lo + t;
    auto it_lo = std::partition(m_first, m_last,
                                [=](const Result x){ return x > tt; });
    auto sum_lo = std::accumulate(it_lo, m_last, static_cast<Result>(0));
    auto n_lo = std::distance(it_lo, m_last);

    tt = hi + t;
    auto it_hi = std::partition(m_first, it_lo,
                                [=](const Result x){ return x > tt; });
    auto sum_hi = std::accumulate(m_first, it_hi, static_cast<Result>(0));
    auto n_hi = std::distance(m_first, it_hi);

    // Check feasibility and fix variables
    Result s_hi = static_cast<Result>(n_hi) * hi;
    Result s_lo = static_cast<Result>(n_lo) * lo;
    Result infeas = sum_hi + sum_lo - (s_hi + s_lo)
                  - static_cast<Result>(n_hi + n_lo) * t;
    if (infeas > eps) {
      m_first = it_hi;
      tt = static_cast<Result>(m) * t - sum_hi + s_hi;
      m -= n_hi;
    } else if (infeas < -eps) {
      m_last = it_lo;
      tt = static_cast<Result>(m) * t - sum_lo + s_lo;
      m -= n_lo;
    } else {
      m_first = it_hi;
      m_last = it_lo;
      break;
    }

    // Update t or stop if degenerated
    if (m > 0) {
      t = tt / static_cast<Result>(m);
    } else {
      break;
    }
  }

#ifdef SDCA_ACCURATE_MATH
  // (Optional) Recompute t to increase numerical accuracy (see Lemma 5.3)
  Result t_lo(std::numeric_limits<Result>::lowest());
  Result t_hi(std::numeric_limits<Result>::max());
  if (m_last != last) {
    t_lo = static_cast<Result>(*std::max_element(m_last, last)) - lo;
  }
  if (first != m_first) {
    t_hi = static_cast<Result>(*std::min_element(first, m_first)) - hi;
  }
  if (m_first != m_last) {
    t = (std::accumulate(m_first, m_last, static_cast<Result>(0)) - rhs
        + hi * static_cast<Result>(std::distance(first, m_first))
        + lo * static_cast<Result>(std::distance(m_last, last))
        ) / static_cast<Result>(std::distance(m_first, m_last));
    t = std::max(t_lo, std::min(t, t_hi));
  } else {
    t = static_cast<Result>(0.5) * (t_lo + t_hi);
  }
#endif

  return make_thresholds(t, lo, hi, m_first, m_last);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_knapsack_eq(
    Iterator first,
    Iterator last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1
    ) {
  prox(first, last,
       thresholds_knapsack_eq<Result, Iterator>, lo, hi, rhs);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_knapsack_eq(
    Iterator first,
    Iterator last,
    Iterator aux,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1
    ) {
  prox(first, last, aux,
       thresholds_knapsack_eq<Result, Iterator>, lo, hi, rhs);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_knapsack_eq(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1
    ) {
  prox(dim, first, last, aux,
       thresholds_knapsack_eq<Result, Iterator>, lo, hi, rhs);
}

}

#endif
