#ifndef SDCA_PROX_TOPK_CONE_BIASED_H
#define SDCA_PROX_TOPK_CONE_BIASED_H

#include "topk_cone.h"

namespace sdca {

template <typename Iterator,
          typename Result>
thresholds<Iterator, Result>
thresholds_topk_cone_biased_search(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k,
    const Result rho
    ) {
  typedef typename std::iterator_traits<Iterator>::value_type data_type;
  typedef Result result_type;

  // Sort data to search efficiently
  std::sort(first, last, std::greater<data_type>());

  // Precompute some constants
  auto k_last = std::next(first, k);
  result_type k_minus_num_U = static_cast<result_type>(k);
  result_type num_U_plus_rho_k_2 =
    rho * static_cast<result_type>(k) * static_cast<result_type>(k);
  result_type min_U = +std::numeric_limits<result_type>::infinity();
  result_type sum_U = 0;

  // Grow U starting with empty
  for (auto m_first = first;;) {

    result_type min_M = +std::numeric_limits<result_type>::infinity();
    result_type max_M = -std::numeric_limits<result_type>::infinity();
    result_type sum_M = 0, num_M_sum_U = 0;
    result_type D = k_minus_num_U * k_minus_num_U;
    result_type k_minus_num_U_sum_U = k_minus_num_U * sum_U;

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

      result_type t  = (num_U_plus_rho_k_2 * sum_M - k_minus_num_U_sum_U) / D;
      result_type hi = (num_M_sum_U + k_minus_num_U * sum_M) / D;
      result_type tt = hi + t;
      if (max_M <= tt && tt <= min_U) {
        if (t <= min_M &&
            ((m_last == last) || static_cast<result_type>(*m_last) <= t)) {
          return thresholds<Iterator, Result>(t, 0, hi, m_first, m_last);
        }
      }

      // Increment the set M
      if (m_last == last) {
        break;
      }
      min_M = *m_last;
      max_M = *m_first;
      sum_M += min_M; // TODO: kahan_add
      ++m_last;
      D += num_U_plus_rho_k_2;
      num_M_sum_U += sum_U;
    }

    // Increment the set U
    if (m_first == k_last) {
      break;
    }
    min_U = *m_first;
    sum_U += min_U; // TODO: kahan_add
    ++m_first;
    --k_minus_num_U;
    ++num_U_plus_rho_k_2;
  }

  // Default to 0
  return thresholds<Iterator, Result>(0, 0, 0, first, first);
}

template <typename Iterator,
          typename Result,
          typename Summator = std_sum<Iterator, Result>>
thresholds<Iterator, Result>
thresholds_topk_cone_biased(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rho = 1,
    Summator sum = Summator()
    ) {
  Result K = static_cast<Result>(k);
  auto proj = topk_cone_special_cases(first, last, k, K + rho * K * K, sum);
  if (proj.projection == projection::general) {
    return thresholds_topk_cone_biased_search(first, last, k, rho);
  } else {
    return proj.result;
  }
}

template <typename Iterator,
          typename Result,
          typename Summator = std_sum<Iterator, Result>>
inline
void
project_topk_cone_biased(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rho = 1,
    Summator sum = Summator()
    ) {
  project(first, last,
    thresholds_topk_cone_biased<Iterator, Result, Summator>, k, rho, sum);
}

template <typename Iterator,
          typename Result,
          typename Summator = std_sum<Iterator, Result>>
inline
void
project_topk_cone_biased(
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rho = 1,
    Summator sum = Summator()
    ) {
  project(first, last, aux_first, aux_last,
    thresholds_topk_cone_biased<Iterator, Result, Summator>, k, rho, sum);
}

template <typename Iterator,
          typename Result,
          typename Summator = std_sum<Iterator, Result>>
inline
void
project_topk_cone_biased(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rho = 1,
    Summator sum = Summator()
    ) {
  project(dim, first, last, aux_first, aux_last,
    thresholds_topk_cone_biased<Iterator, Result, Summator>, k, rho, sum);
}

}

#endif
