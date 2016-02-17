#ifndef SDCA_PROX_TOPK_ENTROPY_BIASED_H
#define SDCA_PROX_TOPK_ENTROPY_BIASED_H

#include "sdca/math/functor.h"
#include "sdca/math/log_exp.h"
#include "sdca/prox/proxdef.h"

namespace sdca {

/**
 * Newton's iteration for the nonlinear system described in
 *    topk_entropy_biased_kkt_iterate
 * below.
 **/
template <typename Result,
          typename Iterator>
inline void
topk_entropy_biased_kkt_iter_2(
    const Iterator m_first,
    const Iterator last,
    const Result K,
    const Result alpha,
    const Result num_U,
    const Result beta,  // sum_U + num_U * (1 + log(alpha / K))
    Result &s,
    Result &t
    ) {
  // Compute the sum of x_i(t) and their derivatives over the set M
  Result sum0(0), sum1(0);
  for (auto a = m_first; a != last; ++a) {
    Result x = lambert_w_exp(static_cast<Result>(*a) - t);
    sum0 += x;
    sum1 += x / (1 + x);
  }

  // It is numerically more stable to consider s -> 0 and s -> 1 separately
  Result A, B, C;
  if (s < static_cast<Result>(0.5)) {
    // s -> 0
    Result k_1_s = K / (1 - s);
    A = s * (K * k_1_s + (K * K + num_U) * alpha) + K * num_U;
    B = s * (s * k_1_s + K * std::log1p(-s) + beta) - num_U * x_log_x(s);
    C = s;
  } else {
    // s -> 1
    Result z = 1 - s;
    A = K * (K - num_U + num_U / (1 - z)) + (K * K + num_U) * alpha * z;
    B = K * ((1 - z) + x_log_x(z)) + z * (beta - num_U * std::log1p(-z));
    C = z;
  }

  // Intermediate computations
  Result sum0_t_sum1 = sum0 + t * sum1;
  Result k_u = K - num_U;
  Result denom = A * sum1 + alpha * (k_u * k_u) * C;

  // Updated variables (the update step is absorbed)
  s = K * (sum0_t_sum1 * k_u * C + B * sum1) / denom;
  t = (A * sum0_t_sum1 - alpha * k_u * B) / denom;
}

/**
 * The KKT conditions for the optimization problem in
 *    thresholds_topk_entropy_biased
 * lead to the following system of nonlinear equations in two variables (s,t):
 *    F(s,t) = 0,
 * where
 *    F = (f1, f2),
 *    f1(s,t) = (1 + rho / k) * alpha * s + rho * log(s) - log(1 - s)
 *            - (1 - rho) * t + rho * log(alpha / k) - sum_U a_i / k,
 *    f2(s,t) = (1 - rho) * alpha * s - sum_M V(a_i - t),
 * moreover,
 *    rho = num_U / k,
 *    V(x) = W(exp(x)),
 *    V'(x) = V(x) / (1 + V(x)).
 *
 * The Newton's step d = (d1, d2) is computed from
 *    J * d = - F,
 * where J is the Jacobian matrix.
 **/
template <typename Result = double,
          typename Iterator>
inline void
topk_entropy_biased_kkt_iterate(
    const Iterator m_first,
    const Iterator last,
    const Result K,
    const Result alpha,
    const Result log_alpha_k,
    const Result num_U,
    const Result sum_U,
    Result &s,
    Result &t,
    const std::size_t max_num_iter = 32
    ) {
  Result lb(0), ub(1), eps = 16 * std::numeric_limits<Result>::epsilon();
  Result beta = sum_U + num_U + num_U * log_alpha_k;
  for (std::size_t iter = 0; iter < max_num_iter; ++iter) {
    Result s1(s), t1(t);
    s = std::min(std::max(lb, s), ub);
    topk_entropy_biased_kkt_iter_2(m_first, last, K, alpha, num_U, beta, s, t);

    if (std::abs(s1 - s) + std::abs(t1 - t)<= eps) break;
  }
  s = std::min(std::max(lb, s), ub);
}

/**
 * Solve
 *    min_{x,s} 0.5 * alpha * (<x, x> + s * s)
 *              + <x, log(x)> + (1 - s) * log(1 - s) - <a, x>
 *    s.t.      <1, x> = s
 *              s <= 1
 *              0 <= x_i <= s / k
 *
 * The solution is
 *    x = max(0, min(lambert_w_exp(a - t) / alpha, hi))
 **/
template <typename Result = double,
          typename Iterator>
inline generalized_thresholds<Result, Iterator,
    a_lambert_w_exp_map<typename std::iterator_traits<Iterator>::value_type>>
thresholds_topk_entropy_biased(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result alpha = 1
    ) {
  assert(alpha > 0);
  assert(k <= std::distance(first, last));
  Result K = static_cast<Result>(k);
  Result alpha_k = alpha / K;
  Result log_alpha_k = std::log(alpha_k);

  Iterator max_el = std::max_element(first, last);
  Result eps = 16 * std::numeric_limits<Result>::epsilon()
    * std::max(static_cast<Result>(1), static_cast<Result>(*max_el));

  // Grow U starting with empty
  Result s(0), t(0), sum_U(0);
  Result min_U = +std::numeric_limits<Result>::infinity();
  Iterator m_first = first;
  typedef typename std::iterator_traits<Iterator>::difference_type diff_t;
  for (diff_t num_U = 0; ;) {
    std::swap(*m_first, *max_el);

    // Compute s and t starting from an initial guess
    s = 1; t = static_cast<Result>(*m_first);
    topk_entropy_biased_kkt_iterate(
      m_first, last, K, alpha, log_alpha_k,
      static_cast<Result>(num_U), sum_U, s, t);

    // No need to check feasibility if that was the last case
    if (++num_U >= k) break;

    // Check feasibility
    Result tt = lambert_w_exp_inverse(alpha_k * s) + t;
    if (static_cast<Result>(*m_first) - eps <= tt && tt <= min_U + eps) {
      break;
    }

    // Increment U
    min_U = static_cast<Result>(*m_first);
    sum_U += static_cast<Result>(*m_first);
    max_el = std::max_element(++m_first, last); // pre-sorting might be faster
  }

  typedef typename std::iterator_traits<Iterator>::value_type Data;
  a_lambert_w_exp_map<Data> map(1 / static_cast<Data>(alpha));
  Result lo(0), hi(s / K);
  return make_thresholds(t, lo, hi, m_first, last, map);
}

template <typename Result = double,
          typename Iterator>
inline void
prox_topk_entropy_biased(
    Iterator first,
    Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result alpha = 1
    ) {
  prox(first, last,
    thresholds_topk_entropy_biased<Result, Iterator>, k, alpha);
}

template <typename Result = double,
          typename Iterator>
inline void
prox_topk_entropy_biased(
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result alpha = 1
    ) {
  prox(first, last, aux_first, aux_last,
    thresholds_topk_entropy_biased<Result, Iterator>, k, alpha);
}

template <typename Result = double,
          typename Iterator>
inline void
prox_topk_entropy_biased(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result alpha = 1
    ) {
  prox(dim, first, last, aux_first, aux_last,
    thresholds_topk_entropy_biased<Result, Iterator>, k, alpha);
}

}

#endif
