#ifndef SDCA_PROX_TOPK_CONE_H
#define SDCA_PROX_TOPK_CONE_H

#include "proxdef.h"

namespace sdca {

template <typename Iterator,
          typename Result>
struct topk_cone_projection {
  sdca::projection projection;
  sdca::thresholds<Iterator, Result> thresholds;
};

template <typename Iterator,
          typename Result,
          typename Summation = std_sum<Iterator, Result>>
topk_cone_projection<Iterator, Result>
topk_cone_special_cases(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k,
    const Result div_const,
    const Summation sum = Summation()
    ) {
  // Partially sort data around the kth element
  auto k_last = std::next(first, k);
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  std::nth_element(first, k_last - 1, last, std::greater<Data>());

  topk_cone_projection<Iterator, Result> proj;

  // Case 1: U empty, M empty, proj = 0
  Result eps = 16 * std::numeric_limits<Result>::epsilon();
  Result sum_k_largest = sum(first, k_last, static_cast<Result>(0));
  if (sum_k_largest <= eps) {
    proj.projection = projection::zero;
    proj.thresholds = thresholds<Iterator, Result>(0, 0, 0, first, first);
    return proj;
  }

  // Case 2: U not empty, M empty, proj = const * sum_k_largest for k largest
  Result hi = sum_k_largest / div_const;
  Result t = static_cast<Result>(*(k_last - 1)) - hi;
  if ((k == std::distance(first, last)) ||
      (t >= static_cast<Result>(*std::max_element(k_last, last)) - eps )) {
    proj.projection = projection::constant;
    proj.thresholds = thresholds<Iterator, Result>(t, 0, hi, k_last, k_last);
    return proj;
  }

  proj.projection = projection::general;
  return proj;
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
thresholds<Iterator, Result>
thresholds_topk_cone_search(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k,
    const Summation sum = Summation()
    ) {
  // Sort data to search efficiently
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  std::sort(first, last, std::greater<Data>());

  // Precompute some constants
  auto k_last = std::next(first, k);
  Result k_minus_num_U = static_cast<Result>(k);
  Result min_U = +std::numeric_limits<Result>::infinity();
  Result num_U = 0, sum_U = 0, sum_U_comp = 0;
  Result eps = 16 * std::numeric_limits<Result>::epsilon();

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
      //    t = (num_U * sum_M - (k - num_U) * sum_U) / D
      //   hi = (num_M * sum_U + (k - num_U) * sum_M) / D
      //    D = (k - num_U)^2 + num_M * num_U
      // and check that
      //  (1)  lo + t  >= max_L = (m_last) or (-Inf)
      //  (2)  lo + t  <= min_M = (m_last - 1) or (+Inf)
      //  (3)  hi + t  >= max_M = (m_first) or (-Inf)
      //  (4)  hi + t  <= min_U = (m_first - 1) or (+Inf)

      Result t  = (num_U * sum_M - k_minus_num_U_sum_U) / D;
      Result hi = (num_M_sum_U   + k_minus_num_U * sum_M) / D;
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
      D += num_U;
      ++m_last;
    }

    // Increment the set U
    if (m_first == k_last) {
      break;
    }
    min_U = static_cast<Result>(*m_first);
    sum.add(min_U, sum_U, sum_U_comp); // sum_U += min_U;
    --k_minus_num_U;
    ++num_U;
    ++m_first;
  }

  // Default to 0
  return thresholds<Iterator, Result>(0, 0, 0, first, first);
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
thresholds<Iterator, Result>
thresholds_topk_cone(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Summation sum = Summation()
    ) {
  auto proj = topk_cone_special_cases(
    first, last, k, static_cast<Result>(k), sum);
  if (proj.projection == projection::general) {
    return thresholds_topk_cone_search<Iterator, Result, Summation>(
      first, last, k, sum);
  } else {
    return proj.thresholds;
  }
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
prox_topk_cone(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Summation sum = Summation()
    ) {
  prox(first, last,
    thresholds_topk_cone<Iterator, Result, Summation>, k, sum);
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
prox_topk_cone(
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Summation sum = Summation()
    ) {
  prox(first, last, aux_first, aux_last,
    thresholds_topk_cone<Iterator, Result, Summation>, k, sum);
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
prox_topk_cone(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Summation sum = Summation()
    ) {
  prox(dim, first, last, aux_first, aux_last,
    thresholds_topk_cone<Iterator, Result, Summation>, k, sum);
}

}

#endif
