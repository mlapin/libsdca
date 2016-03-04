#ifndef SDCA_PROX_TOPK_ENTROPY_H
#define SDCA_PROX_TOPK_ENTROPY_H

#include "sdca/math/functor.h"
#include "sdca/math/log_exp.h"
#include "sdca/prox/proxdef.h"

namespace sdca {

/**
 * Solve
 *    min_{x,s} <x, log(x)> + (1 - s) * log(1 - s) - <a, x>
 *              <1, x> = s
 *              s <= 1
 *              0 <= x_i <= s / k
 *
 * The solution is
 *    x = max(0, min(exp(a - t), hi))
 **/
template <typename Result = double,
          typename Iterator>
inline generalized_thresholds<Result, Iterator,
    exp_map<typename std::iterator_traits<Iterator>::value_type>>
thresholds_topk_entropy(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1
    ) {
  assert(k <= std::distance(first, last));
  exp_map<typename std::iterator_traits<Iterator>::value_type> map;
  Result eps = 16 * std::numeric_limits<Result>::epsilon();
  Result K = static_cast<Result>(k);
  Iterator max_el = std::max_element(first, last);
  Result log_z, log_z_1;
  log_sum_exp(first, last, max_el, log_z, log_z_1);

  // Check if t = log(1 + \sum_i exp a_i) is feasible
  Result t(log_z_1), lo(0), hi(1);
  if (k <= 1 || static_cast<Result>(*max_el) - eps <= log_z - std::log(K)) {
    return make_thresholds(t, lo, hi, first, last, map);
  }

  // k > 1 and U is not empty
  Result min_U(0), sum_U(0), k_u(K), z(0);
  Iterator m_first = first;
  typedef typename std::iterator_traits<Iterator>::difference_type diff_t;
  for (diff_t num_U = 1; num_U < k; ++num_U) {
    min_U = static_cast<Result>(*max_el);
    sum_U += min_U;
    std::swap(*m_first, *max_el);
    ++m_first; --k_u;
    max_el = std::max_element(m_first, last); // pre-sorting might be faster

    log_z = log_sum_exp(m_first, last, max_el, z);

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

  return make_thresholds(t, lo, hi, m_first, last, map);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_topk_entropy(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1
    ) {
  prox(first, last,
       thresholds_topk_entropy<Result, Iterator>, k);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_topk_entropy(
    Iterator first,
    Iterator last,
    Iterator aux,
    const typename std::iterator_traits<Iterator>::difference_type k = 1
    ) {
  prox(first, last, aux,
       thresholds_topk_entropy<Result, Iterator>, k);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_topk_entropy(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux,
    const typename std::iterator_traits<Iterator>::difference_type k = 1
    ) {
  prox(dim, first, last, aux,
       thresholds_topk_entropy<Result, Iterator>, k);
}

}

#endif
