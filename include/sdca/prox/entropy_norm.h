#ifndef SDCA_PROX_ENTROPY_NORM_H
#define SDCA_PROX_ENTROPY_NORM_H

#include "sdca/math/functor.h"
#include "sdca/math/log_exp.h"
#include "sdca/prox/proxdef.h"
#include "sdca/prox/entropy.h"

namespace sdca {

/**
 * Solve
 *    min_x 0.5 * <x, x> + <x, log(x)> - <a, x>
 *          <1, x> = rhs
 *          0 <= x_i <= hi
 *
 * The solution is
 *    x = max(0, min(lambert_w_exp(a - t), hi))
 * Entries are relatively accurate down to tiny ones where long double is
 * wider than double (docs/numerics.md).
 **/
template <typename Result = double, typename Iterator>
inline auto thresholds_entropy_norm(Iterator first, Iterator last, const Result hi = 1,
                                    const Result rhs = 1)
    -> decltype(thresholds_capped_entropy(first, last, hi, rhs, true)) {
  return thresholds_capped_entropy(first, last, hi, rhs, true);
}

// [first, last) in place, through a temporary copy.
template <typename Result = double, typename Iterator>
inline void prox_entropy_norm(Iterator first, Iterator last, const Result hi = 1,
                              const Result rhs = 1) {
  prox(first, last,
       thresholds_entropy_norm<
           Result,
           typename std::vector<typename std::iterator_traits<Iterator>::value_type>::iterator>,
       hi, rhs);
}

// [first, last) in place; aux (as long) is scratch.
template <typename Result = double, typename Iterator>
inline void prox_entropy_norm(Iterator first, Iterator last, Iterator aux, const Result hi = 1,
                              const Result rhs = 1) {
  prox(first, last, aux, thresholds_entropy_norm<Result, Iterator>, hi, rhs);
}

// Each block of dim consecutive entries of [first, last) in place; aux holds dim
// entries.
template <typename Result = double, typename Iterator>
inline void prox_entropy_norm(const typename std::iterator_traits<Iterator>::difference_type dim,
                              Iterator first, Iterator last, Iterator aux, const Result hi = 1,
                              const Result rhs = 1) {
  prox(dim, first, last, aux, thresholds_entropy_norm<Result, Iterator>, hi, rhs);
}

}  // namespace sdca

#endif
