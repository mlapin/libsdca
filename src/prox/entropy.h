#ifndef SDCA_PROX_ENTROPY_H
#define SDCA_PROX_ENTROPY_H

#include <cmath>

#include "proxdef.h"
#include "util/lambert.h"

namespace sdca {

/**
 * Householder's iteration for the equation
 *    \sum_i W_0(exp(a_i - t)) - rhs = 0
 * with convergence rate of order 2 (Newton-Raphson method).
 **/
template <typename Iterator,
          typename Result,
          typename Summation = std_sum<Iterator, Result>>
inline Result
sum_w_exp_iter_2(
    const Iterator first,
    const Iterator last,
    const Result rhs,
    const Result t,
    const Summation sum = Summation()
    ) {
  Result f0(0), f1(0), c0(0), c1(0);
  for (auto a = first; a != last; ++a) {
    Result w = lambert_w_exp(static_cast<Result>(*a) - t);
    sum.add(w, f0, c0);
    sum.add(w / (w + 1), f1, c1);
  }
  sum.add(-rhs, f0, c0);
  return t + f0 / f1;
}

/**
 * Householder's iteration for the equation
 *    \sum_i W_0(exp(a_i - t)) - rhs = 0
 * with convergence rate of order 3 (Halley's method).
 **/
template <typename Iterator,
          typename Result,
          typename Summation = std_sum<Iterator, Result>>
inline Result
sum_w_exp_iter_3(
    const Iterator first,
    const Iterator last,
    const Result rhs,
    const Result t,
    const Summation sum = Summation()
    ) {
  Result f0(0), f1(0), f2(0), c0(0), c1(0), c2(0);
  for (auto a = first; a != last; ++a) {
    Result w = lambert_w_exp(static_cast<Result>(*a) - t);
    Result v = 1 / (w + 1);
    Result v2 = v * v;
    Result wv = w * v;
    Result wv3 = wv * v2;
    sum.add(w, f0, c0);
    sum.add(wv, f1, c1);
    sum.add(wv3, f2, c2);
  }
  sum.add(-rhs, f0, c0);
  return t - 2 * f0 * f1 / (f0 * f2 - 2 * f1 * f1);
}

/**
 * Householder's iteration for the equation
 *    \sum_i W_0(exp(a_i - t)) - rhs = 0
 * with convergence rate of order 4.
 **/
template <typename Iterator,
          typename Result,
          typename Summation = std_sum<Iterator, Result>>
inline Result
sum_w_exp_iter_4(
    const Iterator first,
    const Iterator last,
    const Result rhs,
    const Result t,
    const Summation sum = Summation()
    ) {
  Result f0(0), f1(0), f2(0), f3(0), c0(0), c1(0), c2(0), c3(0);
  for (auto a = first; a != last; ++a) {
    Result w = lambert_w_exp(static_cast<Result>(*a) - t);
    Result v = 1 / (w + 1);
    Result v2 = v * v;
    Result wv = w * v;
    Result wv3 = wv * v2;
    sum.add(w, f0, c0);
    sum.add(wv, f1, c1);
    sum.add(wv3, f2, c2);
    sum.add(wv3 * ((1 - 2 * w) * v2), f3, c3);
  }
  sum.add(-rhs, f0, c0);
  Result f02 = f0 * f2, f11 = f1 * f1;
  return t - 3 * f0 * (2 * f11 - f02) / (6 * f1 * (f02 - f11) - f0 * (f0 * f3));
}

/**
 * Use Householder's method to find an approximate solution to
 *    \sum_i W_0(exp(a_i - t)) = rhs.
 **/
template <typename Iterator,
          typename Result,
          typename Summation = std_sum<Iterator, Result>>
inline Result
solve_sum_w_exp_iterate(
    const Iterator first,
    const Iterator last,
    const Result rhs,
    const Result t0,
    const Summation sum = Summation(),
    const std::size_t max_num_iter = 32
    ) {
  // A guard to prevent exp underflow which results in division by zero
  Result guard = static_cast<Result>(*std::max_element(first, last))
    - type_traits<Result>::min_exp_arg();

  Result t(t0), eps = std::numeric_limits<Result>::epsilon()
    * std::max(static_cast<Result>(1), rhs);
  for (std::size_t iter = 0; iter < max_num_iter; ++iter) {
    Result t1(t);
    t = sum_w_exp_iter_4(first, last, rhs, std::min(t, guard), sum);
    if (std::abs(t1 - t) <= eps) break;
  }
  return t;
}

/**
 * Find the root t of the nonlinear equation
 *    \sum_i W_0(exp(a_i - t)) = rhs,
 * where W_0 is the Lambert function.
 **/
template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
Result
solve_sum_w_exp(
    Iterator first,
    Iterator last,
    const Result rhs = 1,
    const Summation sum = Summation()
    ) {
  // Initial guess based on W_0(exp(a_i - t)) \approx a_i - t (for a_i - t > 0)
  Result t = (sum(first, last, static_cast<Result>(0)) - rhs) /
    static_cast<Result>(std::distance(first, last));

  // Use a heuristic to improve the initial guess
  Iterator u_last = last;
  for (;;) {
    // Partition: a_i - t > 0 for all i in U and t = (sum_U a_i - rhs) / |U|
    auto it = std::partition(first, u_last, [=](Result a){ return a > t; });
    if (it == u_last || it == first) break;
    u_last = it;
    t = (sum(first, u_last, static_cast<Result>(0)) - rhs) /
      static_cast<Result>(std::distance(first, u_last));
  }

  return solve_sum_w_exp_iterate(first, last, rhs, t, sum);
}

/**
 * Partition 'a' and compute the threshold 't'
 * such that the solution to the optimization problem
 *    min_x 0.5 * <x, x> - <a, x> + <x, ln(x)>
 *    s.t.  <1, x> = rhs, 0 <= x_i <= hi,
 * can be computed as
 *    x_i = hi, if i in U;
 *    x_i = W_0(exp(a_i - t)), otherwise.
 **/
template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
lambert_thresholds<Iterator, Result>
thresholds_entropy(
    Iterator first,
    Iterator last,
    const Result hi = 1,
    const Result rhs = 1,
    const Summation sum = Summation()
    ) {
  // Initialization
  Result eps = 16 * std::numeric_limits<Result>::epsilon();
  Result lo(0), r(rhs), u = hi + std::log(hi) + eps;

  Result t = solve_sum_w_exp(first, last, rhs, sum);

  Iterator u_last = first;
  for (;;) {
    Result tt = t + u;
    auto it = std::partition(u_last, last, [=](Result a){ return a > tt; });
    if (it == u_last) break;
    r -= hi * static_cast<Result>(std::distance(u_last, it));
    u_last = it;
    if (it == last) break;
    if (r <= eps) {
      t = static_cast<Result>(*std::max_element(u_last, last))
        - type_traits<Result>::min_exp_arg() + 1;
      break;
    }
    t = solve_sum_w_exp(u_last, last, r, sum);
  }

  return make_lambert_thresholds(t, lo, hi, u_last, u_last);
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
prox_entropy(
    Iterator first,
    Iterator last,
    const Result hi = 1,
    const Result rhs = 1,
    const Summation sum = Summation()
    ) {
  prox(first, last,
    thresholds_entropy<Iterator, Result, Summation>, hi, rhs, sum);
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
prox_entropy(
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const Result hi = 1,
    const Result rhs = 1,
    const Summation sum = Summation()
    ) {
  prox(first, last, aux_first, aux_last,
    thresholds_entropy<Iterator, Result, Summation>, hi, rhs, sum);
}

template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline void
prox_entropy(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    const Result hi = 1,
    const Result rhs = 1,
    const Summation sum = Summation()
    ) {
  prox(dim, first, last, aux_first, aux_last,
    thresholds_entropy<Iterator, Result, Summation>, hi, rhs, sum);
}

}

#endif
