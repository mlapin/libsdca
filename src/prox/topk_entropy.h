#ifndef SDCA_PROX_TOPK_ENTROPY_H
#define SDCA_PROX_TOPK_ENTROPY_H

#include "proxdef.h"
#include "util/lambert.h"
#include "util/numeric.h"

namespace sdca {

/**
 * Partition 'a' and compute the thresholds 't', 'hi'
 * such that the solution to the optimization problem
 *    min_{x,s} <x, log(x)> + (1 - s) * log(1 - s) - <a, x>
 *    s.t.      <1, x> = s, s <= 1, 0 <= x_i <= s / k,
 * can be computed as
 *    x_i = hi, if i in U;
 *    x_i = exp(a_i - t), otherwise.
 **/
template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
exp_thresholds<Iterator, Result>
thresholds_topk_entropy(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Summation sum = Summation()
    ) {
  Result eps = 16 * std::numeric_limits<Result>::epsilon();
  assert(k <= std::distance(first, last));
  Result K = static_cast<Result>(k);
  Iterator max_el = std::max_element(first, last);
  Result log_z, log_z_1;
  log_sum_exp_1(first, last, max_el, log_z, log_z_1, sum);

  // Check if t = log(1 + \sum_i exp a_i) is feasible
  Result t(log_z_1), lo(0), hi(1);
  if (k <= 1 || static_cast<Result>(*max_el) - eps <= log_z - std::log(K)) {
    return make_exp_thresholds(t, lo, hi, first, last);
  }

  // k > 1 and U is not empty
  Result min_U(0), sum_U(0), sum_U_comp(0), k_u(K), z(0);
  Iterator m_first = first;
  typedef typename std::iterator_traits<Iterator>::difference_type diff_t;
  for (diff_t num_U = 1; num_U < k; ++num_U) {
    min_U = static_cast<Result>(*max_el);
    sum.add(min_U, sum_U, sum_U_comp);
    std::swap(*m_first, *max_el);
    ++m_first; --k_u;
    max_el = std::max_element(m_first, last);

    z = log_sum_exp(m_first, last, max_el, log_z, sum);

    // Check feasibility
    Result tt = log_z - std::log(k_u);
    if (static_cast<Result>(*max_el) - eps <= tt && tt <= min_U + eps) {
      break;
    }
  }

  Result tmp = ((K - k_u) * log_z + k_u * std::log(k_u) - sum_U) / K;
  Result B = std::exp(tmp - static_cast<Result>(*max_el)) / K;
  t = static_cast<Result>(*max_el) + std::log1p(z + B) - std::log(k_u / K);
  hi = (1 + z) / ((1 + z + B) * K);

  return make_exp_thresholds(t, lo, hi, m_first, last);
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
prox_topk_entropy(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Summation sum = Summation()
    ) {
  prox(first, last,
    thresholds_topk_entropy<Iterator, Result, Summation>, k, sum);
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
prox_topk_entropy(
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Summation sum = Summation()
    ) {
  prox(first, last, aux_first, aux_last,
    thresholds_topk_entropy<Iterator, Result, Summation>, k, sum);
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
prox_topk_entropy(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Summation sum = Summation()
    ) {
  prox(dim, first, last, aux_first, aux_last,
    thresholds_topk_entropy<Iterator, Result, Summation>, k, sum);
}

}

#endif
