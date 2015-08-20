#ifndef SDCA_PROX_TOPK_CONE_BIASED_H
#define SDCA_PROX_TOPK_CONE_BIASED_H

#include "topk_cone.h"

namespace sdca {

template <typename Iterator,
          typename Result,
          typename Summation = std_sum<Iterator, Result>>
thresholds<Iterator, Result>
thresholds_topk_cone_biased_search(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k,
    const Result rho,
    const Summation sum = Summation()
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
  Result sum_U = 0, sum_U_comp = 0;
  Result eps = static_cast<Result>(k) * std::numeric_limits<Result>::epsilon();

  // Grow U starting with empty
  for (auto m_first = first;;) {

    Result min_M = +std::numeric_limits<Result>::infinity();
    Result max_M = -std::numeric_limits<Result>::infinity();
    Result sum_M = 0, sum_M_comp = 0;
    Result num_M_sum_U = 0, num_M_sum_U_comp = 0;
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
          return thresholds<Iterator, Result>(t, 0, hi, m_first, m_last);
        }
      }

      // Increment the set M
      if (m_last == last) {
        break;
      }
      min_M = static_cast<Result>(*m_last);
      max_M = static_cast<Result>(*m_first);
      sum.add(min_M, sum_M, sum_M_comp); // sum_M += min_M;
      sum.add(sum_U, num_M_sum_U, num_M_sum_U_comp); // num_M_sum_U += sum_U;
      D += num_U_plus_rho_k_2;
      ++m_last;
    }

    // Increment the set U
    if (m_first == k_last) {
      break;
    }
    min_U = static_cast<Result>(*m_first);
    sum.add(min_U, sum_U, sum_U_comp); // sum_U += min_U;
    --k_minus_num_U;
    ++num_U_plus_rho_k_2;
    ++m_first;
  }

  // Default to 0
  return thresholds<Iterator, Result>(0, 0, 0, first, first);
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
thresholds<Iterator, Result>
thresholds_topk_cone_biased(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rho = 1,
    const Summation sum = Summation()
    ) {
  assert(rho >= 0);
  Result K = static_cast<Result>(k);
  auto proj = topk_cone_special_cases(first, last, k, K + rho * K * K, sum);
  if (proj.projection == projection::general) {
    return thresholds_topk_cone_biased_search(first, last, k, rho, sum);
  } else {
    return proj.thresholds;
  }
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
prox_topk_cone_biased(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rho = 1,
    const Summation sum = Summation()
    ) {
  prox(first, last,
    thresholds_topk_cone_biased<Iterator, Result, Summation>, k, rho, sum);
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
prox_topk_cone_biased(
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rho = 1,
    const Summation sum = Summation()
    ) {
  prox(first, last, aux_first, aux_last,
    thresholds_topk_cone_biased<Iterator, Result, Summation>, k, rho, sum);
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
prox_topk_cone_biased(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rho = 1,
    const Summation sum = Summation()
    ) {
  prox(dim, first, last, aux_first, aux_last,
    thresholds_topk_cone_biased<Iterator, Result, Summation>, k, rho, sum);
}

}

#endif
