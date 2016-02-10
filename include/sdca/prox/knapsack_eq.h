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
  // Initialization
  Result eps = std::numeric_limits<Result>::epsilon()
    * std::max(static_cast<Result>(1), std::abs(rhs));
  Result t = (std::accumulate(first, last, static_cast<Result>(0)) - rhs) /
    static_cast<Result>(std::distance(first, last));

  Iterator m_first = first, m_last = last;
  for (;;) {
    // Feasibility check
    Result tt = lo + t;
    auto lo_it = std::partition(m_first, m_last,
      [=](const Result x){ return x > tt; });
    Result infeas_lo = std::max(static_cast<Result>(0),
      + tt * static_cast<Result>(std::distance(lo_it, m_last))
      - std::accumulate(lo_it, m_last, static_cast<Result>(0)));

    tt = hi + t;
    auto hi_it = std::partition(m_first, lo_it,
      [=](const Result x){ return x > tt; });
    Result infeas_hi = std::max(static_cast<Result>(0),
      - tt * static_cast<Result>(std::distance(m_first, hi_it))
      + std::accumulate(m_first, hi_it, static_cast<Result>(0)));

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
    Iterator aux_first,
    Iterator aux_last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1
    ) {
  prox(first, last, aux_first, aux_last,
    thresholds_knapsack_eq<Result, Iterator>, lo, hi, rhs);
}

template <typename Result = double,
          typename Iterator>
inline void
prox_knapsack_eq(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const Result lo = 0,
    const Result hi = 1,
    const Result rhs = 1
    ) {
  prox(dim, first, last, aux_first, aux_last,
    thresholds_knapsack_eq<Result, Iterator>, lo, hi, rhs);
}

}

#endif
