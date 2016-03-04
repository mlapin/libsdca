#ifndef SDCA_PROX_ENTROPY_H
#define SDCA_PROX_ENTROPY_H

#include "sdca/math/functor.h"
#include "sdca/math/log_exp.h"
#include "sdca/prox/proxdef.h"

namespace sdca {

/**
 * Solve
 *    min_x <x, log(x)> - <a, x>
 *          <1, x> = rhs
 *          0 <= x_i <= hi
 *
 * The solution is
 *    x = max(0, min(exp(a - t), hi))
 **/
template <typename Result = double,
          typename Iterator>
inline generalized_thresholds<Result, Iterator,
    exp_map<typename std::iterator_traits<Iterator>::value_type>>
thresholds_entropy(
    Iterator first,
    Iterator last,
    const Result hi = 1,
    const Result rhs = 1
    ) {
  // Initialization
  Result eps = 16 * std::numeric_limits<Result>::epsilon()
                  * std::max(static_cast<Result>(1), rhs);;
  Result lo(0), r(rhs), u = std::log(hi) + eps;
  Result t = log_sum_exp<Result>(first, last) - std::log(rhs);

  Iterator m_first = first;
  for (;;) {
    Result tt = t + u;
    auto it = std::partition(m_first, last, [=](Result a){ return a > tt; });
    if (it == m_first) break;
    r -= hi * static_cast<Result>(std::distance(m_first, it));
    m_first = it;
    if (it == last) break;
    if (r <= eps) {
      t = static_cast<Result>(*std::max_element(m_first, last))
        - exp_traits<Result>::min_arg() + 1;
      break;
    }
    t = log_sum_exp<Result>(m_first, last) - std::log(r);
  }

  exp_map<typename std::iterator_traits<Iterator>::value_type> map;
  return make_thresholds(t, lo, hi, m_first, last, map);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_entropy(
    Iterator first,
    Iterator last,
    const Result hi = 1,
    const Result rhs = 1
    ) {
  prox(first, last,
       thresholds_entropy<Result, Iterator>, hi, rhs);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_entropy(
    Iterator first,
    Iterator last,
    Iterator aux,
    const Result hi = 1,
    const Result rhs = 1
    ) {
  prox(first, last, aux,
       thresholds_entropy<Result, Iterator>, hi, rhs);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_entropy(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux,
    const Result hi = 1,
    const Result rhs = 1
    ) {
  prox(dim, first, last, aux,
       thresholds_entropy<Result, Iterator>, hi, rhs);
}

}

#endif
