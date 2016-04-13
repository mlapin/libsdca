#ifndef SDCA_PROX_ENTROPY_NORM_H
#define SDCA_PROX_ENTROPY_NORM_H

#include "sdca/math/functor.h"
#include "sdca/math/log_exp.h"
#include "sdca/prox/proxdef.h"

namespace sdca {

/**
 * Householder's iteration for the equation
 *    sum_i W_0(exp(a_i - t)) - rhs = 0
 * with convergence rate of order 2 (Newton-Raphson method).
 **/
template <typename Result,
          typename Iterator>
inline Result
sum_w_exp_iter_2(
    const Iterator first,
    const Iterator last,
    const Result rhs,
    const Result t
    ) {
  Result f0(0), f1(0);
  // Minus is absorbed in the update step
  sum_lambert_w_exp_derivatives(first, last, -t, f0, f1);
  f0 -= rhs;
  return t + f0 / f1;
}


/**
 * Householder's iteration for the equation
 *    sum_i W_0(exp(a_i - t)) - rhs = 0
 * with convergence rate of order 3 (Halley's method).
 **/
template <typename Result,
          typename Iterator>
inline Result
sum_w_exp_iter_3(
    const Iterator first,
    const Iterator last,
    const Result rhs,
    const Result t
    ) {
  Result f0(0), f1(0), f2(0);
  // Minus is absorbed in the update step
  sum_lambert_w_exp_derivatives(first, last, -t, f0, f1, f2);
  f0 -= rhs;
  return t - 2 * f0 * f1 / (f0 * f2 - 2 * f1 * f1);
}


/**
 * Householder's iteration for the equation
 *    sum_i W_0(exp(a_i - t)) - rhs = 0
 * with convergence rate of order 4.
 **/
template <typename Result,
          typename Iterator>
inline Result
sum_w_exp_iter_4(
    const Iterator first,
    const Iterator last,
    const Result rhs,
    const Result t
    ) {
  Result f0(0), f1(0), f2(0), f3(0);
  // Minus is absorbed in the update step
  sum_lambert_w_exp_derivatives(first, last, -t, f0, f1, f2, f3);
  f0 -= rhs;
  Result f02 = f0 * f2, f11 = f1 * f1;
  return t - 3 * f0 * (2 * f11 - f02) / (6 * f1 * (f02 - f11) - f0 * (f0 * f3));
}


/**
 * Use Householder's method to find an approximate solution to
 *    sum_i W_0(exp(a_i - t)) = rhs.
 **/
template <typename Result,
          typename Iterator>
inline Result
solve_sum_w_exp_iterate(
    const Iterator first,
    const Iterator last,
    const Result rhs,
    const Result t0,
    const std::size_t max_num_iter = numeric_defaults<Result>::max_num_iter()
    ) {
  // A guard to prevent exp underflow which results in division by zero
  Result ub = static_cast<Result>(*std::max_element(first, last))
    - exp_traits<Result>::min_arg();

  Result t(t0), eps = 16 * std::numeric_limits<Result>::epsilon();
  for (std::size_t iter = 0; iter < max_num_iter; ++iter) {
    Result t1(t);
    t = sum_w_exp_iter_4(first, last, rhs, std::min(t, ub));
    if (std::abs(t1 - t) <= eps) break;
  }
  return t;
}


/**
 * Find the root t of the nonlinear equation
 *    sum_i W_0(exp(a_i - t)) = rhs,
 * where W_0 is the Lambert function.
 **/
template <typename Result = double,
          typename Iterator>
inline Result
solve_sum_w_exp(
    Iterator first,
    Iterator last,
    const Result rhs = 1
    ) {
  // Initial guess based on W_0(exp(a_i - t)) \approx a_i - t (for a_i - t > 0)
  Result t = (std::accumulate(first, last, static_cast<Result>(0)) - rhs) /
    static_cast<Result>(std::distance(first, last));

  // Use a heuristic to improve the initial guess
  Iterator u_last = last;
  for (;;) {
    // Partition: a_i - t > 0 for all i in U and t = (sum_U a_i - rhs) / |U|
    auto it = std::partition(first, u_last, [=](Result a){ return a > t; });
    if (it == u_last || it == first) break;
    u_last = it;
    t = (std::accumulate(first, u_last, static_cast<Result>(0)) - rhs) /
      static_cast<Result>(std::distance(first, u_last));
  }

  return solve_sum_w_exp_iterate(first, last, rhs, t);
}


/**
 * Solve
 *    min_x 0.5 * <x, x> + <x, log(x)> - <a, x>
 *          <1, x> = rhs
 *          0 <= x_i <= hi
 *
 * The solution is
 *    x = max(0, min(lambert_w_exp(a - t), hi))
 **/
template <typename Result = double,
          typename Iterator>
inline generalized_thresholds<Result, Iterator,
    lambert_w_exp_map<typename std::iterator_traits<Iterator>::value_type>>
thresholds_entropy_norm(
    Iterator first,
    Iterator last,
    const Result hi = 1,
    const Result rhs = 1
    ) {
  // Initialization
  Result eps = 16 * std::numeric_limits<Result>::epsilon()
                  * std::max(static_cast<Result>(1), rhs);
  Result lo(0), r(rhs), u = hi + std::log(hi) + eps;

  Result t = solve_sum_w_exp(first, last, rhs);

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
    t = solve_sum_w_exp(m_first, last, r);
  }

  lambert_w_exp_map<typename std::iterator_traits<Iterator>::value_type> map;
  return make_thresholds(t, lo, hi, m_first, last, map);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_entropy_norm(
    Iterator first,
    Iterator last,
    const Result hi = 1,
    const Result rhs = 1
    ) {
  prox(first, last,
       thresholds_entropy_norm<Result, Iterator>, hi, rhs);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_entropy_norm(
    Iterator first,
    Iterator last,
    Iterator aux,
    const Result hi = 1,
    const Result rhs = 1
    ) {
  prox(first, last, aux,
       thresholds_entropy_norm<Result, Iterator>, hi, rhs);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_entropy_norm(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux,
    const Result hi = 1,
    const Result rhs = 1
    ) {
  prox(dim, first, last, aux,
       thresholds_entropy_norm<Result, Iterator>, hi, rhs);
}

}

#endif
