#ifndef SDCA_PROX_TWO_ENTROPY_H
#define SDCA_PROX_TWO_ENTROPY_H

#include "sdca/math/functor.h"
#include "sdca/prox/proxdef.h"

namespace sdca {

/**
 * Householder's iteration for the equation
 *    sum_i W_0(exp(a_i - t)) + sum_i W_0(exp(b_i - t - c)) - alpha = 0
 * with convergence rate of order 4.
 **/
template <typename Result,
          typename Iterator>
inline void
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

  // Initialize
  const std::size_t max_num_iter = 32;
  Result eps = std::numeric_limits<Result>::epsilon();
  Result c = alpha / static_cast<Result>(std::distance(a_first, a_last));
  Result t = std::max(static_cast<Result>(*std::max_element(a_first, a_last)),
               static_cast<Result>(*std::max_element(b_first, b_last)) - c);

  // Householder's method
  for (std::size_t iter = 0; iter < max_num_iter; ++iter) {
    Result t1(t);
    two_sum_w_exp_iter_4(a_first, a_last, b_first, b_last, alpha, c, t);
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
