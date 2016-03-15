#ifndef SDCA_PROX_TOPK_ENTROPY_BIASED_H
#define SDCA_PROX_TOPK_ENTROPY_BIASED_H

#include "sdca/math/functor.h"
#include "sdca/math/log_exp.h"
#include "sdca/prox/proxdef.h"

namespace sdca {

/**
 * Householder's iteration for the equation
 *    W_0(exp(alpha - t)) + sum_i W_0(exp(a_i - t)) - alpha = 0
 * with convergence rate of order 4.
 **/
template <typename Result,
          typename Iterator>
inline Result
alpha_sum_w_exp_iter_4(
    const Iterator first,
    const Iterator last,
    const Result alpha,
    const Result t,
    Result& s
    ) {
  // alpha term
  Result v = lambert_w_exp(alpha - t);
  Result d = 1 / (1 + v);
  Result d2 = d * d;
  Result v1 = v * d;
  Result v2 = v1 * d2;

  Result f0(v), f1(v1), f2(v2), f3(v2 * (1 - 2 * v) * d2);

  // the sum
  s = 0;
  for (auto a = first; a != last; ++a) {
    v = lambert_w_exp(static_cast<Result>(*a) - t);
    d = 1 / (1 + v);
    d2 = d * d;
    v1 = v * d;
    v2 = v1 * d2;
    s += v;
    f0 += v;
    f1 += v1; // minus is absorbed in the update step
    f2 += v2;
    f3 += v2 * (1 - 2 * v) * d2;
  }
  s /= alpha;
  f0 -= alpha;
  Result f02 = f0 * f2, f11 = f1 * f1;
  return t - 3 * f0 * (2 * f11 - f02) / (6 * f1 * (f02 - f11) - f0 * (f0 * f3));
}


/**
 * Use Householder's method to find an approximate solution to
 *    W_0(exp(alpha - t)) + sum_i W_0(exp(a_i - t)) = alpha.
 * Compute s as
 *    s = sum_i W_0(exp(a_i - t)) / alpha.
 **/
template <typename Result,
          typename Iterator>
inline void
solve_alpha_sum_w_exp_iterate(
    const Iterator first,
    const Iterator last,
    const Result max,   // max in [first, last)
    const Result alpha,
    Result& s,
    Result& t,
    const std::size_t max_num_iter = 32
    ) {
  // A guard to prevent exp underflow which results in division by zero
  Result ub = max - exp_traits<Result>::min_arg();

  Result eps = 1 * std::numeric_limits<Result>::epsilon();
  for (std::size_t iter = 0; iter < max_num_iter; ++iter) {
    Result t1(t);
    t = alpha_sum_w_exp_iter_4(first, last, alpha, std::min(t, ub), s);
    if (std::abs(t1 - t) <= eps) break;
  }
}


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
    const Result beta,  // sum_U + num_U * (1 - log(alpha / k)) + k * log(alpha)
    Result &s,
    Result &t
    ) {
  // Compute the sum of x_i(t) and their derivatives over the set M
  Result sum0(0), sum1(0);
  for (auto a = m_first; a != last; ++a) {
    Result v = lambert_w_exp(static_cast<Result>(*a) - t);
    sum0 += v;
    sum1 += v / (1 + v);
  }

  // It is numerically more stable to consider s -> 0 and s -> 1 separately
  Result A, B, C;
  if (s < static_cast<Result>(0.5)) {
    // s -> 0
    Result a_s = alpha * s, s_1_s = s / (1 - s);
    A = K * K * (a_s + s_1_s) + num_U * (a_s + K);
    B = s * (beta + K * (s_1_s + std::log1p(-s))) - num_U * x_log_x(s);
    C = s;
  } else {
    // s -> 1
    Result z = 1 - s, uz = num_U * (1 - s);
    A = K * K * (1 + alpha * z) + uz * (alpha + K);
    B = z * beta + K * (1 - z + x_log_x(z)) - uz * std::log1p(-z);
    C = z;
  }

  Result sum0_t_sum1 = sum0 + t * sum1;
  Result k_u = K - num_U;
  Result denom = A * sum1 + (k_u * k_u) * alpha * C;

  // Updated variables (the update step is absorbed)
  assert(denom != 0);
  if (denom != 0) {
    s = K * (sum0_t_sum1 * k_u * C + B * sum1) / denom;
    t = (A * sum0_t_sum1 - alpha * k_u * B) / denom;
  }
}


/**
 * The KKT conditions for the optimization problem in
 *    thresholds_topk_entropy_biased
 * lead to the following system of nonlinear equations in two variables (s,t):
 *    F(s,t) = 0,
 * where
 *    F = (f1, f2),
 *    f1(s,t) = V^{-1}(alpha * (1 - s)) - rho * V^{-1}(alpha * s / k)
 *            + (1 - rho) * t - alpha + sum_U a_i / k,
 *    f2(s,t) = (1 - rho) * alpha * s - sum_M V(a_i - t),
 * and,
 *    rho = num_U / k,
 *    V(x) = W(exp(x)),
 *    V'(x) = V(x) / (1 + V(x)),
 *    V^{-1}(x) = x + log(x).
 *
 * The Newton's step d = (d1, d2) is computed from
 *    J * d = - F,
 * where J is the Jacobian matrix:
 *    J11 = - alpha * (1 + rho / k) - rho / s - 1 / (1 - s),
 *    J12 = 1 - rho,
 *    J21 = (1 - rho) * alpha,
 *    J22 = sum_M V(a_i - t) / (1 + V(a_i - t)).
 * We have
 *    d1 = (J12 * f2 - J22 * f1) / (J11 * J22 - J12 * J21),
 *    d2 = (J21 * f1 - J11 * f2) / (J11 * J22 - J12 * J21),
 * and
 *    s <- s + d1,
 *    t <- t + d2.
 **/
template <typename Result = double,
          typename Iterator>
inline void
topk_entropy_biased_kkt_iterate(
    const Iterator m_first,
    const Iterator last,
    const Result K,
    const Result alpha,
    const Result log_alpha,
    const Result log_alpha_k,
    const Result num_U,
    const Result sum_U,
    Result &s,
    Result &t,
    const std::size_t max_num_iter = 32
    ) {
  Result lb(0), ub(1), eps = 16 * std::numeric_limits<Result>::epsilon();
  Result beta = sum_U + num_U * (1 - log_alpha_k) + K * log_alpha;
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
  assert((0 < k) && (k <= std::distance(first, last)));

  // Auxiliary variables
  Iterator max_el = std::max_element(first, last);
  Result max = static_cast<Result>(*max_el);
  const Result eps = 16 * std::numeric_limits<Result>::epsilon()
                   * std::max(static_cast<Result>(1), max);

  // Initial guess for t
  Result s(0), t(max);

  // Case 1: U is empty. Find a t such that:
  //    V(alpha - t) + sum_i V(a_i - t) = alpha
  // also, compute the corresponding s as
  //    s = sum_i V(a_i - t) / alpha
  solve_alpha_sum_w_exp_iterate(first, last, max, alpha, s, t);

  // Auxiliary variables
  const Result lo(0), K(static_cast<Result>(k));
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  a_lambert_w_exp_map<Data> map(1 / static_cast<Data>(alpha));

  // If k = 1, done.
  if (k == 1) {
    return make_thresholds(t, lo, s / K, first, last, map);
  }

  // If k > 1, check feasibility.
  const Result alpha_k(alpha / K);
  Result tt = lambert_w_exp_inverse(alpha_k * s) + t;
  if (max - eps <= tt) {
    return make_thresholds(t, lo, s / K, first, last, map);
  }

  // Case 2: U is not empty (some x_i's are at the upper bound).
  // Grow U by adding the largest elements in [first, last)
  std::swap(*first, *max_el);
  const Result log_alpha = std::log(alpha);
  const Result log_alpha_k = std::log(alpha_k);
  Result min_U(max), sum_U(max);
  Iterator m_first = first + 1;
  typedef typename std::iterator_traits<Iterator>::difference_type diff_t;
  for (diff_t num_U = 1; m_first != last;) {
    // Recompute s and t
    topk_entropy_biased_kkt_iterate(
      m_first, last, K, alpha, log_alpha, log_alpha_k,
      static_cast<Result>(num_U), sum_U, s, t);

    // No need to check feasibility if that was the last possibility
    if (++num_U >= k) break;

    // Check feasibility
    max_el = std::max_element(m_first, last);
    max = static_cast<Result>(*max_el);
    tt = lambert_w_exp_inverse(alpha_k * s) + t;
    if (max - eps <= tt && tt <= min_U + eps) {
      break;
    }

    // Increment U
    min_U = max;
    sum_U += max;
    std::swap(*m_first, *max_el);
    ++m_first;
  }

  return make_thresholds(t, lo, s / K, first, last, map);
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
    Iterator aux,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result alpha = 1
    ) {
  prox(first, last, aux,
       thresholds_topk_entropy_biased<Result, Iterator>, k, alpha);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_topk_entropy_biased(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux,
    const typename std::iterator_traits<Iterator>::difference_type k = 1,
    const Result alpha = 1
    ) {
  prox(dim, first, last, aux,
       thresholds_topk_entropy_biased<Result, Iterator>, k, alpha);
}

}

#endif
