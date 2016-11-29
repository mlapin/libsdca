#ifndef SDCA_PROX_TWO_SIMPLEX_SORT_H
#define SDCA_PROX_TWO_SIMPLEX_SORT_H

#include <functional>

#include "sdca/prox/proxdef.h"

namespace sdca {

// The code below is based on the bipartite_solver.cc
// in the Sopopo solver implemented by Shai Shalev-Shwartz.
// http://www.cs.huji.ac.il/~shais/code/index.html

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
thresholds_two_simplex_sort(
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
  Result lo(0), hi(rhs), eps = std::numeric_limits<Result>::epsilon()
    * std::max(static_cast<Result>(1), rhs);

  // Sort the margins in descending order
  std::sort(a_first, a_last, std::greater<Result>());
  std::sort(b_first, b_last, std::greater<Result>());

  Result c = 0, C = rhs, best_c = rhs, sum_mu = *a_first, sum_nu = *b_first;

  // The proposed number of "active" margins
  typename std::iterator_traits<Iterator>::difference_type r = 1, s = 1;

  // The pseudo points beyond the end are for the termination conditions
  Result a_r = a_first + r != a_last
             ? static_cast<Result>(a_first[r])
             : static_cast<Result>(*std::prev(a_last)) - C;
  Result b_s = b_first + s != b_last
             ? static_cast<Result>(b_first[s])
             : static_cast<Result>(*std::prev(b_last)) - C;

  while (c < C) {
    // Compute the optimal value of c under the assumption that the number
    // of active margins is r & s
    Result copt = (s * sum_mu + r * sum_nu) / static_cast<Result>(r + s);

    // Compute the potential next grid point for c by increasing r
    Result next_cr = (sum_mu + a_r) - (r + 1) * a_r;

    // Compute the potential next grid point for c by increasing s
    Result next_cs = (sum_nu + b_s) - (s + 1) * b_s;

    // The right point of the interval cannot exceed C
    Result next_c = std::min(std::min(next_cr, next_cs), C);

    // If the optimal value indeed falls in [c,next_c) we are done
    if (c <= copt && copt < next_c) {
      best_c = copt;
      break;
    }

    // If we hit C we must stop and set the optimum to be C
    if (next_c >= C - eps) {
      best_c = C;
      break;
    }

    // Update the candidates for next_c
    if (next_cr < next_cs) {
      sum_mu += a_r;
      r++;
      a_r = a_first + r != a_last
          ? static_cast<Result>(a_first[r])
          : static_cast<Result>(*std::prev(a_last)) - C;
    } else {
      sum_nu += b_s;
      s++;
      b_s = b_first + s != b_last
          ? static_cast<Result>(b_first[s])
          : static_cast<Result>(*std::prev(b_last)) - C;
    }

    // Switch from the interval [c,next_c) to the interval [next_c,?)
    c = next_c;
  }

  // Calculate the thresholds
  Result theta_a = (sum_mu - best_c) / static_cast<Result>(r);
  Result theta_b = (sum_nu - best_c) / static_cast<Result>(s);

  return std::make_pair(
    make_thresholds(theta_a, lo, hi, a_first, a_first + r),
    make_thresholds(theta_b, lo, hi, b_first, b_first + s));
}


template <typename Result = double,
          typename Iterator>
inline void
prox_two_simplex_sort(
    Iterator a_first,
    Iterator a_last,
    Iterator b_first,
    Iterator b_last,
    const Result rhs = 1
    ) {
  prox(a_first, a_last, b_first, b_last,
       thresholds_two_simplex_sort<Result, Iterator>, rhs);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_two_simplex_sort(
    Iterator a_first,
    Iterator a_last,
    Iterator b_first,
    Iterator b_last,
    Iterator a_aux,
    Iterator b_aux,
    const Result rhs = 1
    ) {
  prox(a_first, a_last, b_first, b_last, a_aux, b_aux,
       thresholds_two_simplex_sort<Result, Iterator>, rhs);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_two_simplex_sort(
    Iterator first,
    Iterator middle,
    Iterator last,
    const Result rhs = 1
    ) {
  prox(first, middle, middle, last,
       thresholds_two_simplex_sort<Result, Iterator>, rhs);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_two_simplex_sort(
    const typename std::iterator_traits<Iterator>::difference_type p,
    Iterator first,
    Iterator last,
    Iterator aux,
    const Result rhs = 1
    ) {
  prox(first, first + p, first + p, last, aux, aux + p,
       thresholds_two_simplex_sort<Result, Iterator>, rhs);
}


template <typename Result = double,
          typename Iterator>
inline void
prox_two_simplex_sort(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    const typename std::iterator_traits<Iterator>::difference_type p,
    Iterator first,
    Iterator last,
    Iterator aux,
    const Result rhs = 1
    ) {
  prox(dim, p, first, last, aux,
       thresholds_two_simplex_sort<Result, Iterator>, rhs);
}

}

#endif
