#ifndef SDCA_PROX_TWO_ENTROPY_H
#define SDCA_PROX_TWO_ENTROPY_H

#include "sdca/math/functor.h"
#include "sdca/math/log_exp.h"
#include "sdca/prox/proxdef.h"

namespace sdca {

/**
 * Householder's iteration for the equation
 *    sum_i W_0(exp(a_i - t)) + sum_i W_0(exp(b_i - t - c)) - alpha = 0
 * with convergence rate of order 4.
 **/
template <typename Result,
          typename Iterator>
inline Result
two_sum_w_exp_iter_4(
    const Iterator a_first,
    const Iterator a_last,
    const Iterator b_first,
    const Iterator b_last,
    const Result alpha,
    const Result c,
    Result& t
    ) {
  Result f0(0), f1(0), f2(0), f3(0);
  sum_lambert_w_exp_derivatives(a_first, a_last, -t, f0, f1, f2, f3);
  sum_lambert_w_exp_derivatives(b_first, b_last, -t - c, f0, f1, f2, f3);
  f0 -= alpha;

  Result f02 = f0 * f2, f11 = f1 * f1;
  Result D = 6 * f1 * (f02 - f11) - f0 * (f0 * f3);
  Result eps = 64 * std::numeric_limits<Result>::min();
  if (std::abs(D) > eps) {
    t -= 3 * f0 * (2 * f11 - f02) / D;
  }
  return f0;
}


/**
 * Compute
 *    sum_i W_0(exp(a_i - t)) + sum_i W_0(exp(b_i - t - c)) - alpha.
 **/
template <typename Result,
          typename Iterator>
inline Result
two_sum_w_exp(
    const Iterator a_first,
    const Iterator a_last,
    const Iterator b_first,
    const Iterator b_last,
    const Result alpha,
    const Result c,
    const Result t
    ) {
  Result f0(0);
  sum_lambert_w_exp(a_first, a_last, -t, f0);
  sum_lambert_w_exp(b_first, b_last, -t - c, f0);
  return f0 - alpha;
}


/**
 * Find the root of
 *    sum_i W_0(exp(a_i - t)) + sum_i W_0(exp(b_i - t - c)) - alpha = 0
 * using bisection.
 **/
template <typename Result,
          typename Iterator>
inline void
two_sum_w_exp_bisection(
    const Iterator a_first,
    const Iterator a_last,
    const Iterator b_first,
    const Iterator b_last,
    const Result alpha,
    const Result c,
    const Result eps,
    const std::size_t max_num_iter,
    Result& t,
    Result& lb,
    Result& ub,
    Result& lf,
    Result& uf
    ) {
  assert(uf <= 0 && 0 <= lf);
  for (std::size_t iter = 0; iter < max_num_iter; ++iter) {
    t = (lb + ub) / 2;
    Result f1 = two_sum_w_exp(a_first, a_last, b_first, b_last, alpha, c, t);
    if (eps < f1 && f1 < lf) {
      lf = f1;
      lb = t;
    } else if (uf < f1 && f1 < -eps) {
      uf = f1;
      ub = t;
    } else {
      break;
    }
  }
  t = (lb + ub) / 2;
}


/**
 * Solve
 *    min_{x,y} 0.5 * alpha * ||x - a / alpha - 1 / p||^2 + <x, log(x)> +
 *              0.5 * alpha * ||y - b / alpha||^2 + <y, log(y)>
 *    s.t.      <1, x> = <1, y> = 1
 *              0 <= x_i,  0 <= y_j
 *
 * where p = dim(a) is the dimensionality (i.e. size) of vector a.
 *
 * The solution is
 *    x = lambert_w_exp(a - t) / alpha
 *    y = lambert_w_exp(b - s) / alpha,
 *
 * where s = t + alpha / p.
 *
 * Note that the above is equivalent to
 *    min_{x,y} 0.5 * alpha * ||x - u||^2 + <x, log(x)> +
 *              0.5 * alpha * ||y - v||^2 + <y, log(y)>
 *    s.t.      <1, x> = <1, y> = 1
 *              0 <= x_i,  0 <= y_j
 *
 * whith
 *    u = a / alpha + 1 / p,
 *    v = b / alpha.
 **/
template <typename Result = double,
          typename Iterator>
inline std::pair<
  generalized_thresholds<Result, Iterator,
    a_lambert_w_exp_map<typename std::iterator_traits<Iterator>::value_type>>,
  generalized_thresholds<Result, Iterator,
    a_lambert_w_exp_map<typename std::iterator_traits<Iterator>::value_type>>>
thresholds_two_entropy(
    Iterator a_first,
    Iterator a_last,
    Iterator b_first,
    Iterator b_last,
    const Result alpha = 1
    ) {
  assert(std::distance(a_first, a_last) > 0);
  assert(std::distance(b_first, b_last) > 0);

  // Initialization
  std::size_t max_num_iter = numeric_defaults<Result>::max_num_iter();
  Result eps = std::numeric_limits<Result>::epsilon();
  Result num_a = static_cast<Result>(std::distance(a_first, a_last));
  Result num_b = static_cast<Result>(std::distance(b_first, b_last));
  Result max_a = static_cast<Result>(*std::max_element(a_first, a_last));
  Result max_b = static_cast<Result>(*std::max_element(b_first, b_last));
  Result sum_a = std::accumulate(a_first, a_last, static_cast<Result>(0));
  Result sum_b = std::accumulate(b_first, b_last, static_cast<Result>(0));
  Result c = alpha / num_a;
  Result max = std::max(max_a, max_b - c);

  // Guards for bracketing the solution (and preventing under/overflow)
  Result lb = std::numeric_limits<Result>::lowest();
  Result ub = max - 64 * exp_traits<Result>::min_arg();
  Result lf = std::numeric_limits<Result>::max(); // approx. f(lb)
  Result uf = std::numeric_limits<Result>::lowest(); // approx. f(ub)

  // Guess1: assume the Lambert function behaves like exp
  Result t01 = max - std::log(alpha);
  Result t1(t01);
  Result f1 = two_sum_w_exp_iter_4(
              a_first, a_last, b_first, b_last, alpha, c, t1);

  // Guess2: assume the Lambert function behaves like linear
  Result t02 = (sum_a + sum_b - num_b * c - alpha) / (num_a + num_b);
  Result t2(t02);
  Result f2 = two_sum_w_exp_iter_4(
              a_first, a_last, b_first, b_last, alpha, c, t2);

  // Ensure f1 <= f2
  if (f1 > f2) {
    std::swap(t01, t02);
    std::swap(t1, t2);
    std::swap(f1, f2);
  }

  // Update the guard bounds and choose an initial point
  Result t;
  if (f1 > 0) {
    lb = t01;
    lf = f1;
    t = t1;
  } else if (f2 < 0) {
    ub = t02;
    uf = f2;
    t = t2;
  } else {
    lb = t02;
    ub = t01;
    lf = f2;
    uf = f1;
    t = std::max(lb, std::min(std::abs(f1) < std::abs(f2) ? t1 : t2, ub));

    // Do bisection for a few iterations to improve the guess
    std::size_t num_iter = std::min(max_num_iter,
      static_cast<std::size_t>(std::log(ub - lb)));
    two_sum_w_exp_bisection(a_first, a_last, b_first, b_last, alpha, c,
                            eps, num_iter, t, lb, ub, lf, uf);
  }

  // Householder's method
  for (std::size_t iter = 0; iter < max_num_iter; ++iter) {
    t1 = t;
    f1 = two_sum_w_exp_iter_4(a_first, a_last, b_first, b_last, alpha, c, t);
    if (0 <= f1 && f1 < lf) {
      lb = t1;
      lf = f1;
    } else if (uf < f1 && f1 <= 0) {
      ub = t1;
      uf = f1;
    }
    t = std::max(lb, std::min(t, ub));
    if (std::abs(t1 - t) <= eps) break;
  }

  Result lo(0), hi(1);
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  a_lambert_w_exp_map<Data> map(1 / static_cast<Data>(alpha));
  return std::make_pair(
    make_thresholds(t, lo, hi, a_first, a_last, map),
    make_thresholds(t + c, lo, hi, b_first, b_last, map));
}


template <typename Result = double,
          typename Iterator>
inline void
prox_two_entropy(
    Iterator a_first,
    Iterator a_last,
    Iterator b_first,
    Iterator b_last,
    const Result alpha = 1
    ) {
  prox(a_first, a_last, b_first, b_last,
       thresholds_two_entropy<Result, Iterator>, alpha);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_two_entropy(
    Iterator a_first,
    Iterator a_last,
    Iterator b_first,
    Iterator b_last,
    Iterator a_aux,
    Iterator b_aux,
    const Result alpha = 1
    ) {
  prox(a_first, a_last, b_first, b_last, a_aux, b_aux,
       thresholds_two_entropy<Result, Iterator>, alpha);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_two_entropy(
    Iterator first,
    Iterator middle,
    Iterator last,
    const Result alpha = 1
    ) {
  prox(first, middle, middle, last,
       thresholds_two_entropy<Result, Iterator>, alpha);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_two_entropy(
    std::ptrdiff_t p,
    Iterator first,
    Iterator last,
    Iterator aux,
    const Result alpha = 1
    ) {
  prox(first, first + p, first + p, last, aux, aux + p,
       thresholds_two_entropy<Result, Iterator>, alpha);
}

}

#endif
