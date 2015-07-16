#ifndef SDCA_PROX_TOPK_CONE_H
#define SDCA_PROX_TOPK_CONE_H

#include "proxdef.h"

namespace sdca {

template <typename Iterator,
          typename Result>
struct topk_cone_projection {
  sdca::projection projection;
  thresholds<Iterator, Result> result;
};

template <typename Iterator,
          typename Result,
          typename Summator = std_sum<Iterator, Result>>
inline
topk_cone_projection<Iterator, Result>
topk_cone_special_cases(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k,
    const Result div_const,
    Summator sum = Summator()
    ) {
  typedef typename std::iterator_traits<Iterator>::value_type data_type;
  typedef Result result_type;

  // Partially sort data around the kth element
  auto k_last = std::next(first, k);
  std::nth_element(first, k_last - 1, last, std::greater<data_type>());

  // Sum k largest elements
  result_type sum_k_largest = sum(first, k_last, static_cast<result_type>(0));

  topk_cone_projection<Iterator, Result> proj;

  // Case 1: U empty, M empty, proj = 0
  if (sum_k_largest <= 0) {
    proj.projection = projection::zero;
    proj.result = thresholds<Iterator, Result>(0, 0, 0, first, first);
    return proj;
  }

  // Case 2: U not empty, M empty, proj = const * sum_k_largest for k largest
  result_type hi = sum_k_largest / div_const;
  result_type t = static_cast<result_type>(*(k_last - 1)) - hi;
  if ((k == std::distance(first, last)) ||
      (t >= static_cast<result_type>(*std::max_element(k_last, last)))) {
    proj.projection = projection::constant;
    proj.result = thresholds<Iterator, Result>(t, 0, hi, k_last, k_last);
    return proj;
  }

  proj.projection = projection::general;
  return proj;
}

template <typename Iterator,
          typename Result = double>
thresholds<Iterator, Result>
thresholds_topk_cone_search(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k
    ) {
  typedef typename std::iterator_traits<Iterator>::value_type data_type;
  typedef Result result_type;

  // Sort data to search efficiently
  std::sort(first, last, std::greater<data_type>());

  // Precompute some constants
  auto k_last = std::next(first, k);
  result_type k_minus_num_U = static_cast<result_type>(k);
  result_type min_U = +std::numeric_limits<result_type>::infinity();
  result_type num_U = 0, sum_U = 0;

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
      //    t = (num_U * sum_M - (k - num_U) * sum_U) / D
      //   hi = (num_M * sum_U + (k - num_U) * sum_M) / D
      //    D = (k - num_U)^2 + num_M * num_U
      // and check that
      //  (1)  lo + t  >= max_L = (m_last) or (-Inf)
      //  (2)  lo + t  <= min_M = (m_last - 1) or (+Inf)
      //  (3)  hi + t  >= max_M = (m_first) or (-Inf)
      //  (4)  hi + t  <= min_U = (m_first - 1) or (+Inf)

      result_type t  = (num_U * sum_M - k_minus_num_U_sum_U) / D;
      result_type hi = (num_M_sum_U   + k_minus_num_U * sum_M) / D;
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
      D += num_U;
      num_M_sum_U += sum_U; // TODO: kahan_add
    }

    // Increment the set U
    if (m_first == k_last) {
      break;
    }
    min_U = *m_first;
    sum_U += min_U; // TODO: kahan_add
    ++num_U;
    ++m_first;
    --k_minus_num_U;
  }

  // Default to 0
  return thresholds<Iterator, Result>(0, 0, 0, first, first);
}

template <typename Iterator,
          typename Result = double,
          typename Summator = std_sum<Iterator, Result>>
thresholds<Iterator, Result>
thresholds_topk_cone(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    Summator sum = Summator()
    ) {
  auto proj = topk_cone_special_cases(
    first, last, k, static_cast<Result>(k), sum);
  if (proj.projection == projection::general) {
    return thresholds_topk_cone_search<Iterator, Result>(first, last, k);
  } else {
    return proj.result;
  }
}

template <typename Iterator,
          typename Result = double,
          typename Summator = std_sum<Iterator, Result>>
inline
void
project_topk_cone(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    Summator sum = Summator()
    ) {
  project(first, last,
    thresholds_topk_cone<Iterator, Result, Summator>, k, sum);
}

template <typename Iterator,
          typename Result = double,
          typename Summator = std_sum<Iterator, Result>>
inline
void
project_topk_cone(
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    Summator sum = Summator()
    ) {
  project(first, last, aux_first, aux_last,
    thresholds_topk_cone<Iterator, Result, Summator>, k, sum);
}

template <typename Iterator,
          typename Result = double,
          typename Summator = std_sum<Iterator, Result>>
inline
void
project_topk_cone(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    Summator sum = Summator()
    ) {
  project(dim, first, last, aux_first, aux_last,
    thresholds_topk_cone<Iterator, Result, Summator>, k, sum);
}

}

#endif
