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
inline void
alpha_sum_w_exp_iter_4(
    const Iterator first,
    const Iterator last,
    const Result alpha,
    Result& s,
    Result& t
    ) {
  Result f0(0), f1(0), f2(0), f3(0);
  sum_lambert_w_exp_derivatives(first, last, -t, f0, f1, f2, f3);
  s = f0 / alpha;

  sum_lambert_w_exp_derivatives(&alpha, &alpha + 1, -t, f0, f1, f2, f3);
  f0 -= alpha;

  Result f02 = f0 * f2, f11 = f1 * f1;
  Result D = 6 * f1 * (f02 - f11) - f0 * (f0 * f3);
  Result eps = 64 * std::numeric_limits<Result>::min();
  if (std::abs(D) > eps) {
    t -= 3 * f0 * (2 * f11 - f02) / D;
  }
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
    const Result alpha,
    Result& s,
    Result& t,
    const std::size_t max_num_iter = numeric_defaults<Result>::max_num_iter()
    ) {
  Result eps = std::numeric_limits<Result>::epsilon();
  for (std::size_t iter = 0; iter < max_num_iter; ++iter) {
    Result t1(t);
    alpha_sum_w_exp_iter_4(first, last, alpha, s, t);
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
    const Iterator first,
    const Iterator last,
    const Result K,
    const Result U,
    const Result C0, // K * (U - 1) * log(alpha) - U * K * log(K) - sum_U
    const Result C1, // U * K;
    const Result C2, // (K + U) * alpha;
    const Result C3, // (K - U) * (U - 1) * alpha;
    const Result C4, // K * (U - 1);
    const Result C5, // (1 - U / K) * alpha;
    Result &s,
    Result &t
    ) {
  Result S0(0), S1(0);
  sum_lambert_w_exp_derivatives(first, last, -t, S0, S1);
  Result E = S0 + t * S1;

  // It is numerically more stable to consider s -> 0 and s -> 1 separately
  Result z, Az, Bz;
  if (s < static_cast<Result>(0.5)) {
    // s -> 0
    z = s;
    Az = K * (z * (U + std::log1p(-z) + z / (1 - z)) - U * x_log_x(z)) - C0 * z;
    Bz = z * (C2 + K / (1 - z)) + C1;
  } else {
    // s -> 1
    z = 1 - s;
    Az = K * (z * (U - U * std::log1p(-z) - 1) + x_log_x(z) + 1) - C0 * z;
    Bz = z * (C2 + C1 / (1 - z)) + K;
  }

  // Updated variables (the update step is absorbed)
  Result Dz = C3 * z - Bz * S1;
  Result eps = 64 * std::numeric_limits<Result>::min();
  if (std::abs(Dz) > eps) {
    s = (C4 * E * z - Az * S1) / Dz;
    t = (C5 * Az - E * Bz) / Dz;
  }
}


/**
 * The KKT conditions for the optimization problem in
 *    thresholds_topk_entropy_biased
 * lead to the following system of nonlinear equations in two variables (s,t):
 *    F(s,t) = 0,
 * where
 *    F = (f1, f2),
 *    f1 = (k - u) * alpha * s - k * sum_M V(a_i - t),
 *    f2 = (k + u) * alpha * s + k * (u * log(s) - log(1 - s) + (u - 1) * t)
 *       + c,
 *    c = k * ((u - 1) * log(alpha) - u * log(k)) - sum_U a_i.
 *
 * Note that:
 *    V(x) = W(exp(x)),
 *    V'(x) = V(x) / (1 + V(x)),
 *    V^{-1}(x) = x + log(x).
 *
 * The Newton's step d = (d1, d2) is computed from
 *    J * d = - F,
 * and the update is given by
 *    s <- s + d1,
 *    t <- t + d2.
 **/
template <typename Result = double,
          typename Iterator>
inline void
topk_entropy_biased_kkt_iterate(
    const Iterator m_first,
    const Iterator last,
    const Result ,
    const Result K,
    const Result alpha,
    const Result log_k,
    const Result log_alpha,
    const Result U,
    const Result sum_U,
    Result &s,
    Result &t,
    const std::size_t max_num_iter = numeric_defaults<Result>::max_num_iter()
    ) {
  Result lb(0), ub(1), eps = 16 * std::numeric_limits<Result>::epsilon();

  Result C1 = U * K;
  Result C2 = (K + U) * alpha;
  Result C3 = (K - U) * (U - 1) * alpha;
  Result C4 = K * (U - 1);
  Result C5 = (1 - U / K) * alpha;
  Result C0 = C4 * log_alpha - C1 * log_k - sum_U;

  for (std::size_t iter = 0; iter < max_num_iter; ++iter) {
    Result s1(s), t1(t);
    s = std::min(std::max(lb, s), ub);
    topk_entropy_biased_kkt_iter_2(m_first, last, K, U, C0, C1, C2, C3, C4, C5,
                                   s, t);

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
  solve_alpha_sum_w_exp_iterate(first, last, alpha, s, t);

  // Auxiliary variables
  const Result lo(0), K(static_cast<Result>(k));
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  a_lambert_w_exp_map<Data> map(1 / static_cast<Data>(alpha));

  // If k = 1, done.
  if (k == 1) {
    return make_thresholds(t, lo, s, first, last, map);
  }

  // If k > 1, check feasibility.
  const Result alpha_k(alpha / K);
  Result tt = t + std::max(lambert_w_exp_inverse(alpha_k * s),
                           exp_traits<Result>::min_arg());
  if (max - eps <= tt) {
    return make_thresholds(t, lo, s / K, first, last, map);
  }

  // Case 2: U is not empty (some x_i's are at the upper bound).
  // Grow U by adding the largest elements in [first, last)
  std::swap(*first, *max_el);
  const Result log_k = std::log(K);
  const Result log_alpha = std::log(alpha);
  Result min_U(max), sum_U(max);
  Iterator m_first = first + 1;
  typedef typename std::iterator_traits<Iterator>::difference_type diff_t;
  for (diff_t num_U = 1; m_first != last;) {
    max_el = std::max_element(m_first, last);
    max = static_cast<Result>(*max_el);

    // Compute s and t starting from some initial guess
    s = static_cast<Result>(0.999);
    t = max;
    topk_entropy_biased_kkt_iterate(
      m_first, last, max, K, alpha, log_k, log_alpha,
      static_cast<Result>(num_U), sum_U, s, t);

    // Check feasibility
    tt = t + std::max(lambert_w_exp_inverse(alpha_k * s),
                      exp_traits<Result>::min_arg());
    if (max - eps <= tt && tt <= min_U + eps) {
      break;
    }

    // Increment U
    if (++num_U > k) break;
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
