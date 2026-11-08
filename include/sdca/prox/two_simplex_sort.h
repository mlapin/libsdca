#ifndef SDCA_PROX_TWO_SIMPLEX_SORT_H
#define SDCA_PROX_TWO_SIMPLEX_SORT_H

#include <functional>

#include "sdca/prox/two_simplex.h"

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
template <typename Result = double, typename Iterator>
inline std::pair<thresholds<Result, Iterator>, thresholds<Result, Iterator>>
thresholds_two_simplex_sort(Iterator a_first, Iterator a_last, Iterator b_first, Iterator b_last,
                            const Result rhs = 1) {
  return thresholds_two_simplex<Result>(a_first, a_last, b_first, b_last, rhs);
}

// [a_first, a_last) and [b_first, b_last) in place, through temporary copies.
template <typename Result = double, typename Iterator>
inline void prox_two_simplex_sort(Iterator a_first, Iterator a_last, Iterator b_first,
                                  Iterator b_last, const Result rhs = 1) {
  prox(a_first, a_last, b_first, b_last,
       thresholds_two_simplex_sort<
           Result,
           typename std::vector<typename std::iterator_traits<Iterator>::value_type>::iterator>,
       rhs);
}

// [a_first, a_last) and [b_first, b_last) in place; a_aux and b_aux are as long.
template <typename Result = double, typename Iterator>
inline void prox_two_simplex_sort(Iterator a_first, Iterator a_last, Iterator b_first,
                                  Iterator b_last, Iterator a_aux, Iterator b_aux,
                                  const Result rhs = 1) {
  prox(a_first, a_last, b_first, b_last, a_aux, b_aux,
       thresholds_two_simplex_sort<Result, Iterator>, rhs);
}

// [first, middle) and [middle, last) in place, through a temporary copy.
template <typename Result = double, typename Iterator>
inline void prox_two_simplex_sort(Iterator first, Iterator middle, Iterator last,
                                  const Result rhs = 1) {
  prox(first, middle, middle, last,
       thresholds_two_simplex_sort<
           Result,
           typename std::vector<typename std::iterator_traits<Iterator>::value_type>::iterator>,
       rhs);
}

// [first, first + p) and [first + p, last) in place; aux is as long.
template <typename Result = double, typename Iterator>
inline void prox_two_simplex_sort(const typename std::iterator_traits<Iterator>::difference_type p,
                                  Iterator first, Iterator last, Iterator aux,
                                  const Result rhs = 1) {
  prox(first, first + p, first + p, last, aux, aux + p,
       thresholds_two_simplex_sort<Result, Iterator>, rhs);
}

// Each block of dim consecutive entries of [first, last), its first p entries
// the first set; aux holds dim entries.
template <typename Result = double, typename Iterator>
inline void prox_two_simplex_sort(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    const typename std::iterator_traits<Iterator>::difference_type p, Iterator first, Iterator last,
    Iterator aux, const Result rhs = 1) {
  prox(dim, p, first, last, aux, thresholds_two_simplex_sort<Result, Iterator>, rhs);
}

}  // namespace sdca

#endif
