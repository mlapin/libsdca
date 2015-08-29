#ifndef SDCA_PROX_TOPK_ENTROPY_BIASED_H
#define SDCA_PROX_TOPK_ENTROPY_BIASED_H

#include "entropy.h"

namespace sdca {

/**
 * Newton's iteration for the nonlinear system described in
 *    topk_entropy_biased_kkt_iterate
 * below.
 **/
template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
topk_entropy_biased_kkt_iter_2(
    const Iterator m_first,
    const Iterator last,
    const Result su,
    const Result k,
    const Result alpha,
    const Result rho,
    Result &t,
    Result &s,
    const Summation sum = Summation()
    ) {
  // Compute the sum of x_i(t) and their derivatives over the set M
  Result sm0(0), sm1(0), cm0(0), cm1(0);
  for (auto a = m_first; a != last; ++a) {
    Result x = lambert_w_exp(static_cast<Result>(*a) - t);
    sum.add(x, sm0, cm0);
    sum.add(x / (1 + x), sm1, cm1);
  }

  Result f1 = (1 - rho) * t + lambert_w_exp_inverse(alpha * (1 - s))
              - rho * lambert_w_exp_inverse(alpha * s / k) - alpha + su / k;
  Result f2 = alpha * (1 - rho) * s - sm0;
  Result A = 1 - rho;
  Result B = - alpha / k * (k + rho) - (s + rho * (1 - s)) / (s * (1 - s));
  Result C = sm1;
  Result D = alpha * (1 - rho);

  Result DD = A * D - B * C;
  Result d1 = (B * f2 - D * f1) / DD;
  Result d2 = (C * f1 - A * f2) / DD;

  if (std::isfinite(d1) && std::isfinite(d2)) {
    t += d1;
    s += d2;
  }
}

/**
 * The KKT conditions for the optimization problem in
 *    thresholds_topk_entropy_biased
 * lead to the following system of nonlinear equations:
 *    f1: (1 - rho) * t + V^{-1}(alpha * (1 - s)) - rho * V^{-1}(alpha * s / k)
 *                     - alpha + 1 / k * \sum_U a_i = 0,
 *    f2: alpha * (1 - rho) * s - \sum_M V(a_i - t) = 0,
 * where
 *    V(t) = W(exp(t)),
 *    V'(t) = V(t) / (1 + V(t)),
 *    V^{-1}(v) = v + log(v),
 *    V^{-1}'(v) = 1 + 1 / v.
 * The Jacobian J is given as
 *    / A B \ _ / df1/dt  df1/ds \
 *    \ C D / - \ df2/dt  df2/ds /,
 * and the Newton's step d = (d1, d2) is computed from
 *    J * d = - F.
 **/
template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
topk_entropy_biased_kkt_iterate(
    const Iterator m_first,
    const Iterator last,
    const Result su,
    const Result k,
    const Result alpha,
    const Result rho,
    Result &t,
    Result &s,
    const Summation sum = Summation(),
    const std::size_t max_num_iter = 64
    ) {
  Result eps = 8 * std::numeric_limits<Result>::epsilon()
      * std::max(static_cast<Result>(1), std::abs(t));

  // Guard bounds on s
  Result lb = 8 * std::numeric_limits<Result>::epsilon();
  Result ub = 1 - 8 * std::numeric_limits<Result>::epsilon();

  for (std::size_t iter = 0; iter < max_num_iter; ++iter) {
    Result t1(t), s1(s);
    s = std::min(std::max(lb, s), ub);
    topk_entropy_biased_kkt_iter_2(m_first, last, su, k, alpha, rho, t, s, sum);

    if (std::abs(t1 - t) + std::abs(s1 - s) <= eps) break;
  }
  s = std::min(std::max(static_cast<Result>(0), s), static_cast<Result>(1));
}

/**
 * Partition 'a' and compute the thresholds 't', 'hi'
 * such that the solution to the optimization problem
 *    min_{x,s} 0.5 * alpha * (<x, x> + s * s) - <a, x>
 *              + <x, log(x)> + (1 - s) * log(1 - s)
 *    s.t.      <1, x> = s, s <= 1, 0 <= x_i <= s / k,
 * can be computed as
 *    x_i = hi, if i in U;
 *    x_i = 1 / alpha * W_0(exp(a_i - t)), otherwise.
 **/
template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
lambert_a_thresholds<Iterator, Result>
thresholds_topk_entropy_biased(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result alpha = 1,
    const Summation sum = Summation()
    ) {
  assert(alpha > 0);
  assert(k <= std::distance(first, last));
  Result K = static_cast<Result>(k);
  Result k_inv = 1 / K, a_inv = 1 / alpha;

  Iterator max_el = std::max_element(first, last);
  Result eps = 8 * std::numeric_limits<Result>::epsilon()
    * std::max(static_cast<Result>(1), static_cast<Result>(*max_el));

  // Initial guess and thresholds
  Result t(static_cast<Result>(*max_el)), s(static_cast<Result>(0.999));
  Result lo(0), hi(0);

  // Grow U starting with empty
  Result min_U = +std::numeric_limits<Result>::infinity(), sum_U(0), rho(0);
  Iterator m_first = first;
  typedef typename std::iterator_traits<Iterator>::difference_type diff_t;
  for (diff_t num_U = 0; num_U < k; ++num_U) {
    std::swap(*m_first, *max_el);

    t = static_cast<Result>(*m_first); s = static_cast<Result>(0.9);
    topk_entropy_biased_kkt_iterate(m_first, last,
      sum_U, K, alpha, rho, t, s, sum);

    Result tt = lambert_w_exp_inverse(alpha * s / K) + t;
    if (static_cast<Result>(*m_first) - eps <= tt && tt <= min_U + eps) {
      hi = s / K;
      break;
    }

    // Increment U
    min_U = static_cast<Result>(*m_first);
    sum_U += static_cast<Result>(*m_first);
    ++m_first;
    max_el = std::max_element(m_first, last);
    rho += k_inv;
  }

  return make_lambert_a_thresholds(a_inv, t, lo, hi, m_first, last);
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
prox_topk_entropy_biased(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result alpha = 1,
    const Summation sum = Summation()
    ) {
  prox(first, last,
    thresholds_topk_entropy_biased<Iterator, Result, Summation>, k, alpha, sum);
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
prox_topk_entropy_biased(
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result alpha = 1,
    const Summation sum = Summation()
    ) {
  prox(first, last, aux_first, aux_last,
    thresholds_topk_entropy_biased<Iterator, Result, Summation>, k, alpha, sum);
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
prox_topk_entropy_biased(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result alpha = 1,
    const Summation sum = Summation()
    ) {
  prox(dim, first, last, aux_first, aux_last,
    thresholds_topk_entropy_biased<Iterator, Result, Summation>, k, alpha, sum);
}

}

#endif
