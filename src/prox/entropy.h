#ifndef SDCA_PROX_ENTROPY_H
#define SDCA_PROX_ENTROPY_H

#include <cmath>

#include "proxdef.h"
#include "util/lambert.h"

namespace sdca {

/**
 * Householder's iteration for the equation
 *    \sum_i W_0(exp(x_i + t)) - rhs = 0
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
    Result t,
    Summation sum = Summation()
    ) {
  Result f0(0), f1(0), c0(0), c1(0);
  for (auto x = first; x != last; ++x) {
    Result w = lambert_w_exp(*x + t);
    sum.add(w, f0, c0);
    sum.add(w / (w + 1), f1, c1);
  }
  sum.add(-rhs, f0, c0);
  return t - f0 / f1;
}

/**
 * Householder's iteration for the equation
 *    \sum_i W_0(exp(x_i + t)) - rhs = 0
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
    Result t,
    Summation sum = Summation()
    ) {
  Result f0(0), f1(0), f2(0), c0(0), c1(0), c2(0);
  for (auto x = first; x != last; ++x) {
    Result w = lambert_w_exp(*x + t);
    Result v = 1 / (w + 1);
    Result v2 = v * v;
    Result wv = w * v;
    Result wv3 = wv * v2;
    sum.add(w, f0, c0);
    sum.add(wv, f1, c1);
    sum.add(wv3, f2, c2);
  }
  sum.add(-rhs, f0, c0);
  return t + 2 * f0 * f1 / (f0 * f2 - 2 * f1 * f1);
}

/**
 * Householder's iteration for the equation
 *    \sum_i W_0(exp(x_i + t)) - rhs = 0
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
    Result t,
    Summation sum = Summation()
    ) {
  Result f0(0), f1(0), f2(0), f3(0), c0(0), c1(0), c2(0), c3(0);
  for (auto x = first; x != last; ++x) {
    Result w = lambert_w_exp(*x + t);
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
  return t + 3 * f0 * (2 * f11 - f02) / (6 * f1 * (f02 - f11) - f0 * (f0 * f3));
}

/**
 * Find the root t of the nonlinear equation
 *    \sum_i W_0(exp(x_i + t)) = rhs,
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
    Summation sum = Summation()
    ) {
  // Initial guess based on W_0(exp(x_i + t)) \approx x_i + t (for x_i + t > 0)
  Result t = (rhs - sum(first, last, static_cast<Result>(0))) /
    static_cast<Result>(std::distance(first, last));

  // Use a heuristic to improve the initial guess
  Iterator u_last = last;
  for (;;) {
    // Partition: x_i + t > 0 for all i in U and t = (rhs - sum_U x_i) / |U|
    auto it = std::partition(first, u_last, [=](Result x){ return x > -t; });
    if (it == u_last || it == first) break;
    u_last = it;
    t = (rhs - sum(first, u_last, static_cast<Result>(0))) /
      static_cast<Result>(std::distance(first, u_last));
  }

  // A guard to prevent exp underflow which results in division by 0
  Result guard = type_traits<Result>::min_exp_arg()
    - *std::max_element(first, u_last);

  // Use Householder's method to find an approximate solution
  Result eps = std::numeric_limits<Result>::epsilon()
    * std::max(static_cast<Result>(1), std::max(rhs, std::abs(t)));
  for (std::size_t iter = 0; iter < 32; ++iter) {
    Result t1(t);
    t = sum_w_exp_iter_4(first, last, rhs, std::max(t, guard), sum);
    if (std::abs(t1 - t) <= eps) break;
  }
  return t;
}

}

#endif
