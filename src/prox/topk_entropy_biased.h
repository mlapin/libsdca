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
    const Iterator first,
    const Iterator last,
    const Iterator m_first,
    const Result k_inv, // 1 / k
    const Result alpha,
    const Result a_inv, // 1 / alpha
    const Result log_a, // log(alpha)
    const Result rho,
    const Result rho_1, // 1 / (1 - rho)
    Result &t,
    Result &s,
    Result &p,
    const Summation sum = Summation()
    ) {
  // Compute sums of x_i(t) and their derivatives over the sets U and M
  Result su0(0), su1(0), sm0(0), sm1(0);
  Result cu0(0), cu1(0), cm0(0), cm1(0);
  for (auto a = first; a != m_first; ++a) {
    Result x = lambert_w_exp(static_cast<Result>(*a) - t);
    sum.add(x, su0, cu0);
    sum.add(- x / (1 + x), su1, cu1);
  }
  for (auto a = m_first; a != last; ++a) {
    Result x = lambert_w_exp(static_cast<Result>(*a) - t);
    sum.add(x, sm0, cm0);
    sum.add(- x / (1 + x), sm1, cm1);
  }
  su0 *= a_inv; su1 *= a_inv;
  sm0 *= a_inv; sm1 *= a_inv;

  // Solve the linear system J(x) * d = -F(x), x = (t, s, p)
  Result f1 = t - alpha * s + std::log(1 - s) + log_a + k_inv * p;
  Result f2 = rho_1 * sm0;
  Result f3 = su0 + rho * f2 - p;
  f2 -= s;
  Result A = -(alpha + 1 / (1 - s));
  Result B = rho_1 * sm1;
  Result C = su1 + rho * B;
  Result d1 = - (f1 + A * f2 + k_inv * f3) / (1 + A * B + k_inv * C);
  Result d2 = B * d1 + f2;
  Result d3 = C * d1 + f3;

  // Newton's step
  if (std::isfinite(d1) && std::isfinite(d2) && std::isfinite(d3)) {
    t += d1;
    s += d2;
    p += d3;
  }
}

/**
 * The KKT conditions for the optimization problem in
 *    thresholds_topk_entropy_biased
 * lead to the following system of nonlinear equations:
 *    t - alpha * s + log(alpha * (1 - s)) + p / k        = 0,
 *    1 / (1 - rho) * \sum_M x_i(t) - s                   = 0,
 *    \sum_U x_i(t) + rho / (1 - rho) * \sum_M x_i(t) - p = 0,
 * where
 *    x_i(t) = 1 / alpha * W_0(exp(a_i - t)) = V(a_i - t) / alpha
 * and
 *    x_i'(t) = -V'(a_i - t) / alpha,
 *    V'(t) = V(t) / (1 + V(t)).
 * Let x = (t, s, p), the Jacobian is given as
 *        / 1   A  1/k \
 *    J = | B  -1   0  |.
 *        \ C   0  -1  /
 * The system is solved using Newton's method.
 **/
template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
topk_entropy_biased_kkt_iterate(
    const Iterator first,
    const Iterator last,
    const Iterator m_first,
    const Result k_inv,
    const Result alpha,
    const Result a_inv,
    const Result log_a,
    const Result rho_0,
    Result &t,
    Result &s,
    Result &p,
    const Summation sum = Summation(),
    const std::size_t max_num_iter = 64
    ) {
  Result eps = 8 * std::numeric_limits<Result>::epsilon()
      * std::max(static_cast<Result>(1), std::max(std::abs(t), p));

  // Guard bounds on s
  Result lb = 8 * std::numeric_limits<Result>::epsilon();
  Result ub = 1 - 8 * std::numeric_limits<Result>::epsilon();

  // rho cannot be 1
  Result rho = std::min(rho_0, ub);
  Result rho_1 = 1 / (1 - rho);
  for (std::size_t iter = 0; iter < max_num_iter; ++iter) {
    Result t1(t), s1(s), p1(p);
    s = std::min(std::max(lb, s), ub);

    // Newton's step to update (t, s, p)
    topk_entropy_biased_kkt_iter_2(first, last, m_first,
      k_inv, alpha, a_inv, log_a, rho, rho_1, t, s, p, sum);

    if (std::abs(t1 - t) + std::abs(s1 - s) + std::abs(p1 - p) <= eps) break;
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
  Result k_inv = 1 / K, a_inv = 1 / alpha, log_a = std::log(alpha);

  Iterator max_el = std::max_element(first, last);
  Result eps = 8 * std::numeric_limits<Result>::epsilon()
    * std::max(static_cast<Result>(1), static_cast<Result>(*max_el));

  // Initial guess and thresholds
  Result t(static_cast<Result>(*max_el)), s(static_cast<Result>(0.999)), p(0);
  Result lo(0), hi(0);

  // Grow U starting with empty
  Result min_U = +std::numeric_limits<Result>::infinity(), rho(0);
  Iterator m_first = first;
  typedef typename std::iterator_traits<Iterator>::difference_type diff_t;
  for (diff_t num_U = 0; num_U < k; ++num_U) {
    std::swap(*m_first, *max_el);

    topk_entropy_biased_kkt_iterate(first, last, m_first,
      k_inv, alpha, a_inv, log_a, rho, t, s, p, sum);

    hi = s / K;
    Result tt = alpha * hi;
    tt += std::log(tt) + t;
    if (static_cast<Result>(*m_first) - eps <= tt && tt <= min_U + eps) {
      break;
    }

    // Increment U
    min_U = static_cast<Result>(*m_first);
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
