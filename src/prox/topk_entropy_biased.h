#ifndef SDCA_PROX_TOPK_ENTROPY_BIASED_H
#define SDCA_PROX_TOPK_ENTROPY_BIASED_H

#include "proxdef.h"
#include "util/lambert.h"

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
    const Result sum_U_k_alpha, // sum_U / k - alpha
    const Result alpha,
    const Result alpha_k, // alpha / k
    const Result rho,
    const Result rho_1, // 1 - rho
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

  Result f1 = + rho_1 * t - rho * lambert_w_exp_inverse(alpha_k * s)
              + lambert_w_exp_inverse(alpha - alpha * s) + sum_U_k_alpha;
  Result f2 = alpha * rho_1 * s - sm0;
  Result A = rho_1;
  Result B = - (rho / s + 1 / (1 - s) + alpha_k * rho + alpha);
  Result C = sm1;
  Result D = alpha * rho_1;

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
 * lead to the following system of nonlinear equations in two variables (s,t):
 *    f1: V^{-1}(alpha * (1 - s)) - rho * V^{-1}(alpha * s / k)
 *        + (1 - rho) * t - alpha + 1 / k * \sum_U a_i = 0,
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
    const Result sum_U_k_alpha,
    const Result alpha,
    const Result alpha_k,
    const Result rho,
    const Result rho_1,
    Result &t,
    Result &s,
    const Summation sum = Summation(),
    const std::size_t max_num_iter = 16
    ) {
  Result eps = 16 * std::numeric_limits<Result>::epsilon()
      * std::max(static_cast<Result>(1), std::abs(t));

  // Guard bounds on s
  Result lb = 16 * std::numeric_limits<Result>::epsilon();
  Result ub = 1 - 16 * std::numeric_limits<Result>::epsilon();

  for (std::size_t iter = 0; iter < max_num_iter; ++iter) {
    Result t1(t), s1(s);
    s = std::min(std::max(lb, s), ub);
    topk_entropy_biased_kkt_iter_2(m_first, last,
      sum_U_k_alpha, alpha, alpha_k, rho, rho_1, t, s, sum);

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
  Result alpha_k(alpha / K);

  Iterator max_el = std::max_element(first, last);
  Result eps = 16 * std::numeric_limits<Result>::epsilon()
    * std::max(static_cast<Result>(1), static_cast<Result>(*max_el));

  // Grow U starting with empty
  Result t(0), s(0);
  Result min_U = +std::numeric_limits<Result>::infinity();
  Result sum_U_k_alpha(-alpha), sum_U_comp(0), rho(0);
  Iterator m_first = first;
  typedef typename std::iterator_traits<Iterator>::difference_type diff_t;
  for (diff_t num_U = 0; num_U < k;) {
    std::swap(*m_first, *max_el);

    // Compute t and s
    t = static_cast<Result>(*m_first); s = static_cast<Result>(0.9);
    topk_entropy_biased_kkt_iterate(m_first, last,
      sum_U_k_alpha, alpha, alpha_k, rho, 1 - rho, t, s, sum);

    // Check feasibility
    Result tt = lambert_w_exp_inverse(alpha_k * s) + t;
    if (static_cast<Result>(*m_first) - eps <= tt && tt <= min_U + eps) {
      break;
    }

    // Increment U
    min_U = static_cast<Result>(*m_first);
    sum.add(static_cast<Result>(*m_first) / K, sum_U_k_alpha, sum_U_comp);
    ++m_first; ++num_U;
    rho = static_cast<Result>(num_U) / K;
    max_el = std::max_element(m_first, last); // pre-sorting might be faster
  }

  Result a(1 / alpha), lo(0), hi(s / K);
  return make_lambert_a_thresholds(a, t, lo, hi, m_first, last);
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
