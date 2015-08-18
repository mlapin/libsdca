#ifndef SDCA_PROX_KNAPSACK_EQ_H
#define SDCA_PROX_KNAPSACK_EQ_H

#include <cmath>

#include "proxdef.h"

/*
 * Based on the Algorithm 3.1 in Kiwiel, K. C.,
 * "Variable fixing algorithms for the continuous quadratic knapsack problem.",
 * Journal of Optimization Theory and Applications 136.3 (2008): 445-458.
 */

namespace sdca {

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
thresholds<Iterator, Result>
thresholds_knapsack_eq(
    Iterator first,
    Iterator last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1,
    const Summation sum = Summation()
    ) {
  // Initialization
  Result eps = std::numeric_limits<Result>::epsilon()
    * std::max(static_cast<Result>(1), std::abs(rhs));
  Result t = (sum(first, last, static_cast<Result>(0)) - rhs) /
    static_cast<Result>(std::distance(first, last));

  Iterator m_first = first, m_last = last;
  for (;;) {
    // Feasibility check
    Result tt = lo + t;
    auto lo_it = std::partition(m_first, m_last,
      [=](const Result x){ return x > tt; });
    Result infeas_lo =
      + tt * static_cast<Result>(std::distance(lo_it, m_last))
      - sum(lo_it, m_last, static_cast<Result>(0));

    tt = hi + t;
    auto hi_it = std::partition(m_first, lo_it,
      [=](const Result x){ return x > tt; });
    Result infeas_hi =
      - tt * static_cast<Result>(std::distance(m_first, hi_it))
      + sum(m_first, hi_it, static_cast<Result>(0));

    // Variable fixing (using the incremental multiplier formula (23))
    if (std::abs(infeas_hi - infeas_lo) <= eps) {
      m_first = hi_it;
      m_last = lo_it;
      break;
    } else if (infeas_lo < infeas_hi) {
      m_first = hi_it;
      tt = -infeas_hi;
    } else { //infeas_lo > infeas_hi
      m_last = lo_it;
      tt = +infeas_lo;
    }
    if (m_first == m_last) {
      break;
    } else {
      t += tt / static_cast<Result>(std::distance(m_first, m_last));
    }
  }

  // (Optional) Recompute t to increase numerical accuracy (see Lemma 5.3)
  if (m_first == m_last) {
    if (m_last != last) {
      t = static_cast<Result>(*std::max_element(m_last, last)) - lo;
    } else if (first != m_first) {
      t = static_cast<Result>(*std::min_element(first, m_first)) - hi;
    }
  } else {
    t = rhs - hi * static_cast<Result>(std::distance(first, m_first))
            - lo * static_cast<Result>(std::distance(m_last, last));
    t = sum(m_first, m_last, -t)
      / static_cast<Result>(std::distance(m_first, m_last));
  }

  return make_thresholds(t, lo, hi, m_first, m_last);
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
project_knapsack_eq(
    Iterator first,
    Iterator last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1,
    const Summation sum = Summation()
    ) {
  project(first, last,
    thresholds_knapsack_eq<Iterator, Result, Summation>, lo, hi, rhs, sum);
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
project_knapsack_eq(
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1,
    const Summation sum = Summation()
    ) {
  project(first, last, aux_first, aux_last,
    thresholds_knapsack_eq<Iterator, Result, Summation>, lo, hi, rhs, sum);
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
project_knapsack_eq(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1,
    const Summation sum = Summation()
    ) {
  project(dim, first, last, aux_first, aux_last,
    thresholds_knapsack_eq<Iterator, Result, Summation>, lo, hi, rhs, sum);
}

}

#endif
