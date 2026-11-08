#ifndef SDCA_PROX_TOPK_CONE_BIASED_H
#define SDCA_PROX_TOPK_CONE_BIASED_H

#include "sdca/prox/topk_cone.h"

namespace sdca {

template <typename Result, typename Iterator>
inline thresholds<Result, Iterator> thresholds_topk_cone_biased_search(
    Iterator first, Iterator last, const typename std::iterator_traits<Iterator>::difference_type k,
    const Result rho, const Result unit = 1) {
  return detail::topk_cone_search(first, last, k, rho, unit);
}

/**
 * Solve
 *    min_x 0.5 * (<x, x> + rho * <1, x>^2) - <a, x>
 *          0 <= x_i <= <1, x> / k
 *
 * The solution is
 *    x = max(0, min(a - t, hi))
 **/
template <typename Result = double, typename Iterator>
inline thresholds<Result, Iterator> thresholds_topk_cone_biased(
    Iterator first, Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1, const Result rho = 1) {
  if (k < 1 || k > std::distance(first, last))
    throw std::invalid_argument("Invalid top-k projection dimensions");

  if (!(rho >= 0)) throw std::invalid_argument("rho must be nonnegative");
  Result K = static_cast<Result>(k);
  auto proj = topk_cone_special_cases(first, last, k, K + rho * K * K);
  if (proj.projection == projection::general) {
    return thresholds_topk_cone_biased_search(first, last, k, rho, proj.unit);
  }
  return proj.thresholds;
}

// [first, last) in place, through a temporary copy.
template <typename Result = double, typename Iterator>
inline void prox_topk_cone_biased(
    Iterator first, Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1, const Result rho = 1) {
  prox(first, last,
       thresholds_topk_cone_biased<
           Result,
           typename std::vector<typename std::iterator_traits<Iterator>::value_type>::iterator>,
       k, rho);
}

// [first, last) in place; aux (as long) is scratch.
template <typename Result = double, typename Iterator>
inline void prox_topk_cone_biased(
    Iterator first, Iterator last, Iterator aux,
    const typename std::iterator_traits<Iterator>::difference_type k = 1, const Result rho = 1) {
  prox(first, last, aux, thresholds_topk_cone_biased<Result, Iterator>, k, rho);
}

// Each block of dim consecutive entries of [first, last) in place; aux holds dim
// entries.
template <typename Result = double, typename Iterator>
inline void prox_topk_cone_biased(
    const typename std::iterator_traits<Iterator>::difference_type dim, Iterator first,
    Iterator last, Iterator aux,
    const typename std::iterator_traits<Iterator>::difference_type k = 1, const Result rho = 1) {
  prox(dim, first, last, aux, thresholds_topk_cone_biased<Result, Iterator>, k, rho);
}

}  // namespace sdca

#endif
