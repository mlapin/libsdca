#ifndef SDCA_PROX_TWO_SIMPLEX_H
#define SDCA_PROX_TWO_SIMPLEX_H

#include "sdca/prox/proxdef.h"

namespace sdca {

/**
 * Solve
 *    min_{x,y} ||x - a||^2 + ||y - b||^2
 *              <1, x> = <1, y> <= rhs
 *              0 <= x_i,  0 <= y_j
 *
 * The solution is
 *    x = max(0, a - t)
 *    y = max(0, b - s)
 **/
template <typename Result = double,
          typename Iterator>
inline std::pair<thresholds<Result, Iterator>, thresholds<Result, Iterator>>
thresholds_two_simplex(
    Iterator a_first,
    Iterator a_last,
    Iterator b_first,
    Iterator b_last,
    const Result rhs = 1
    ) {
  assert(rhs > 0); // otherwise just set x = y = 0
  assert(std::distance(a_first, a_last) > 0);
  assert(std::distance(b_first, b_last) > 0);

  // Initialize
  Result t, s, lo(0), hi(rhs), eps = std::numeric_limits<Result>::epsilon()
    * std::max(static_cast<Result>(1), rhs);

  // Phase 1: <1, x> = <1, y> = rhs
  Iterator x_last(a_last);
  for (;;) {
    // Solve the restricted subproblem
    t = (std::accumulate(a_first, x_last, static_cast<Result>(0)) - rhs)
        / static_cast<Result>(std::distance(a_first, x_last));
    auto x_it = std::partition(a_first, x_last,
                               [=](const Result x){ return x > t; });

    // Feasibility check
    if (x_it == x_last) break;
    x_last = x_it;
  }

  Iterator y_last(b_last);
  for (;;) {
    // Solve the restricted subproblem
    s = (std::accumulate(b_first, y_last, static_cast<Result>(0)) - rhs)
        / static_cast<Result>(std::distance(b_first, y_last));
    auto y_it = std::partition(b_first, y_last,
                               [=](const Result y){ return y > s; });

    // Feasibility check
    if (y_it == y_last) break;
    y_last = y_it;
  }

  // Check if (t,s) is a feasible solution
  if (t + s >= - eps) {
    return std::make_pair(
      make_thresholds(t, lo, hi, a_first, x_last),
      make_thresholds(s, lo, hi, b_first, y_last));
  }

  // Should never degenerate at this point if rhs > 0
  assert(std::distance(a_first, x_last) > 0);
  assert(std::distance(b_first, y_last) > 0);

  // Phase 2: <1, x> = <1, y> = r < rhs
  auto m = std::distance(a_first, x_last) + std::distance(b_first, y_last);
  t = (+ std::accumulate(a_first, x_last, static_cast<Result>(0))
       - std::accumulate(b_first, y_last, static_cast<Result>(0))
       ) / static_cast<Result>(m);
  for (;;) {
    // Re-partition and compute the sums
    auto x_it = std::partition(a_first, x_last,
                               [=](const Result x){ return x > t; });
    auto sum_x = std::accumulate(x_it, x_last, static_cast<Result>(0));
    auto n_x = std::distance(x_it, x_last);

    auto y_it = std::partition(b_first, y_last,
                               [=](const Result y){ return y > -t; });
    auto sum_y = std::accumulate(y_it, y_last, static_cast<Result>(0));
    auto n_y = std::distance(y_it, y_last);

    // Check feasibility and fix variables
    Result tt, infeas = sum_x - sum_y - static_cast<Result>(n_x + n_y) * t;
    if (n_y > 0 && infeas > eps) {
      y_last = y_it;
      tt = static_cast<Result>(m) * t + sum_y;
      m -= n_y;
    } else if (n_x > 0 && infeas < -eps) {
      x_last = x_it;
      tt = static_cast<Result>(m) * t - sum_x;
      m -= n_x;
    } else {
      x_last = x_it;
      y_last = y_it;
      break;
    }

    // Update t or stop if degenerated
    if (m > 0) {
      t = tt / static_cast<Result>(m);
    } else {
      break;
    }
  }

  // s = -t in Phase 2
  return std::make_pair(
    make_thresholds(t, lo, hi, a_first, x_last),
    make_thresholds(-t, lo, hi, b_first, y_last));
}


template <typename Result = double,
          typename Iterator>
inline void
prox_two_simplex(
    Iterator a_first,
    Iterator a_last,
    Iterator b_first,
    Iterator b_last,
    const Result rhs = 1
    ) {
  prox(a_first, a_last, b_first, b_last,
       thresholds_two_simplex<Result, Iterator>, rhs);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_two_simplex(
    Iterator a_first,
    Iterator a_last,
    Iterator b_first,
    Iterator b_last,
    Iterator a_aux,
    Iterator b_aux,
    const Result rhs = 1
    ) {
  prox(a_first, a_last, b_first, b_last, a_aux, b_aux,
       thresholds_two_simplex<Result, Iterator>, rhs);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_two_simplex(
    Iterator first,
    Iterator middle,
    Iterator last,
    const Result rhs = 1
    ) {
  prox(first, middle, middle, last,
       thresholds_two_simplex<Result, Iterator>, rhs);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_two_simplex(
    const typename std::iterator_traits<Iterator>::difference_type p,
    Iterator first,
    Iterator last,
    Iterator aux,
    const Result rhs = 1
    ) {
  prox(first, first + p, first + p, last, aux, aux + p,
       thresholds_two_simplex<Result, Iterator>, rhs);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_two_simplex(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    const typename std::iterator_traits<Iterator>::difference_type p,
    Iterator first,
    Iterator last,
    Iterator aux,
    const Result rhs = 1
    ) {
  prox(dim, p, first, last, aux,
       thresholds_two_simplex<Result, Iterator>, rhs);
}

}

#endif
