#ifndef SDCA_PROX_TOPK_SIMPLEX_BIASED_H
#define SDCA_PROX_TOPK_SIMPLEX_BIASED_H

#include "knapsack_eq.h"
#include "topk_cone_biased.h"

namespace sdca {

template <typename Iterator,
          typename Result,
          typename Summation = std_sum<Iterator, Result>>
inline
bool
is_topk_simplex_biased_lt(
    const Iterator u_first,
    const Iterator u_last,
    const Result t,
    const Result k,
    const Result rhs,
    const Result rho,
    Summation sum = Summation()
    ) {
  Result eps = std::numeric_limits<Result>::epsilon()
    * std::max(static_cast<Result>(1), std::abs(rhs));
  if (u_first == u_last) {
    return t < rho * rhs - eps;
  } else {
    Result num_U = static_cast<Result>(std::distance(u_first, u_last));
    Result sum_U = sum(u_first, u_last, static_cast<Result>(0));
    return k * ( sum_U + (k - num_U) * t) < rhs * (num_U + rho * k * k) - eps;
  }
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
thresholds<Iterator, Result>
thresholds_topk_simplex_biased(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rhs = 1,
    const Result rho = 1,
    Summation sum = Summation()
    ) {
  Result K = static_cast<Result>(k), lo(0);
  auto proj = topk_cone_special_cases(first, last, k, K + rho * K * K, sum);
  switch (proj.projection) {
    case projection::zero:
      break;
    case projection::constant:
      if (K * proj.thresholds.hi > rhs) {
        return thresholds_knapsack_eq(first, last, lo, rhs / K, rhs, sum);
      }
      break;
    case projection::general:
      auto t = thresholds_knapsack_eq(first, last, lo, rhs / K, rhs, sum);
      if (is_topk_simplex_biased_lt(first, t.first, t.t, K, rhs, rho, sum)) {
        return thresholds_topk_cone_biased_search(first, last, k, rho, sum);
      }
      return t;
  }
  return proj.thresholds;
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline
void
project_topk_simplex_biased(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rhs = 1,
    const Result rho = 1,
    Summation sum = Summation()
    ) {
  project(first, last,
    thresholds_topk_simplex_biased<Iterator, Result, Summation>,
    k, rhs, rho, sum);
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline
void
project_topk_simplex_biased(
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rhs = 1,
    const Result rho = 1,
    Summation sum = Summation()
    ) {
  project(first, last, aux_first, aux_last,
    thresholds_topk_simplex_biased<Iterator, Result, Summation>,
    k, rhs, rho, sum);
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline
void
project_topk_simplex_biased(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result rhs = 1,
    const Result rho = 1,
    Summation sum = Summation()
    ) {
  project(dim, first, last, aux_first, aux_last,
    thresholds_topk_simplex_biased<Iterator, Result, Summation>,
    k, rhs, rho, sum);
}

}

#endif
