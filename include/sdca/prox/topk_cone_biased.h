#ifndef SDCA_PROX_TOPK_CONE_BIASED_H
#define SDCA_PROX_TOPK_CONE_BIASED_H

#include "sdca/prox/topk_cone.h"

namespace sdca {

template <typename Result,
          typename Iterator>
inline thresholds<Result, Iterator>
thresholds_topk_cone_biased_search(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k,
    const Result rho
    ) {
  // Sort data to search efficiently
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  std::sort(first, last, std::greater<Data>());

  // Precompute some constants
  auto k_last = std::next(first, k);
  Result k_minus_num_U = static_cast<Result>(k);
  Result num_U_plus_rho_k_2 =
    rho * static_cast<Result>(k) * static_cast<Result>(k);
  Result min_U = +std::numeric_limits<Result>::infinity();
  Result sum_U = 0;
  Result eps = 16 * std::numeric_limits<Result>::epsilon();

  // Grow U starting with empty
  for (auto m_first = first;;) {

    Result min_M = +std::numeric_limits<Result>::infinity();
    Result max_M = -std::numeric_limits<Result>::infinity();
    Result sum_M = 0, num_M_sum_U = 0;
    Result D = k_minus_num_U * k_minus_num_U;
    Result k_minus_num_U_sum_U = k_minus_num_U * sum_U;

    // Grow M starting with empty
    for (auto m_last = m_first;;) {
      // Compute t and hi as follows (lo = 0 by definition):
      //    t = ((num_U + rho * k^2) * sum_M - (k - num_U) * sum_U) / D
      //   hi = (num_M * sum_U + (k - num_U) * sum_M) / D
      //    D = (k - num_U)^2 + (num_U + rho * k^2) * num_M
      // and check that
      //  (1)  lo + t  >= max_L = (m_last) or (-Inf)
      //  (2)  lo + t  <= min_M = (m_last - 1) or (+Inf)
      //  (3)  hi + t  >= max_M = (m_first) or (-Inf)
      //  (4)  hi + t  <= min_U = (m_first - 1) or (+Inf)

      Result t  = (num_U_plus_rho_k_2 * sum_M - k_minus_num_U_sum_U) / D;
      Result hi = (num_M_sum_U + k_minus_num_U * sum_M) / D;
      Result tt = hi + t;
      if (max_M - eps <= tt && tt <= min_U + eps) {
        if (t <= min_M + eps &&
            ((m_last == last) || static_cast<Result>(*m_last) - eps <= t)) {
          return thresholds<Result, Iterator>(t, 0, hi, m_first, m_last);
        }
      }

      // Increment the set M
      if (m_last == last) {
        break;
      }
      min_M = static_cast<Result>(*m_last);
      max_M = static_cast<Result>(*m_first);
      sum_M += min_M;
      num_M_sum_U += sum_U;
      D += num_U_plus_rho_k_2;
      ++m_last;
    }

    // Increment the set U
    if (m_first == k_last) {
      break;
    }
    min_U = static_cast<Result>(*m_first);
    sum_U += min_U;
    --k_minus_num_U;
    ++num_U_plus_rho_k_2;
    ++m_first;
  }

  // Default to 0
  return thresholds<Result, Iterator>(0, 0, 0, first, first);
}

/**
 * Solve
 *    min_x 0.5 * (<x, x> + rho * <1, x>^2) - <a, x>
 *          0 <= x_i <= <1, x> / k
 *
 * The solution is
 *    x = max(0, min(a - t, hi))
 **/
template <typename Result = double,
          typename Iterator>
inline thresholds<Result, Iterator>
thresholds_topk_cone_biased(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rho = 1
    ) {
  assert(rho >= 0);
  Result K = static_cast<Result>(k);
  auto proj = topk_cone_special_cases(first, last, k, K + rho * K * K);
  if (proj.projection == projection::general) {
    return thresholds_topk_cone_biased_search(first, last, k, rho);
  }
  return proj.thresholds;
}

template <typename Result = double,
          typename Iterator>
inline void
prox_topk_cone_biased(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rho = 1
    ) {
  prox(first, last,
    thresholds_topk_cone_biased<Result, Iterator>, k, rho);
}

template <typename Result = double,
          typename Iterator>
inline void
prox_topk_cone_biased(
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rho = 1
    ) {
  prox(first, last, aux_first, aux_last,
    thresholds_topk_cone_biased<Result, Iterator>, k, rho);
}

template <typename Result = double,
          typename Iterator>
inline void
prox_topk_cone_biased(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rho = 1
    ) {
  prox(dim, first, last, aux_first, aux_last,
    thresholds_topk_cone_biased<Result, Iterator>, k, rho);
}

}

#endif
