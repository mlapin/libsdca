#ifndef SDCA_PROX_KNAPSACK_LE_H
#define SDCA_PROX_KNAPSACK_LE_H

#include "sdca/prox/knapsack_eq.h"

namespace sdca {

namespace detail {

// Throws unless lo <= x_i <= hi, <1, x> <= rhs has a solution.
template <typename Work, typename Result, typename Count>
inline void validate_knapsack_le(const Count count, const Result lo, const Result hi,
                                 const Result rhs) {
  const Work slack_lo =
      Work(4) * static_cast<Work>(std::numeric_limits<Result>::epsilon()) * std::abs(Work(lo));
  if (count <= 0 || !std::isfinite(lo) || !std::isfinite(hi) || !std::isfinite(rhs) || lo > hi ||
      Work(rhs) / Work(count) < Work(lo) - slack_lo)
    throw std::invalid_argument("Infeasible knapsack projection");
}

}  // namespace detail

/**
 * Solve
 *    min_x 0.5 * <x, x> - <a, x>
 *          <1, x> <= rhs
 *          lo <= x_i <= hi
 *
 * The solution is
 *    x = max(lo, min(a - t, hi))
 **/
template <typename Result = double, typename Iterator>
inline thresholds<Result, Iterator> thresholds_knapsack_le(Iterator first, Iterator last,
                                                           const Result lo = 0, const Result hi = 1,
                                                           const Result rhs = 1) {
  using Data = typename std::iterator_traits<Iterator>::value_type;
  using Compare = typename std::common_type<Result, Data>::type;
  using Work = typename detail::wider<Compare>::type;
  detail::validate_knapsack_le<Work>(std::distance(first, last), lo, hi, rhs);
  // Test the actual box-projected mass. An absolute epsilon floor can
  // discard the entire constraint when rhs is small (e.g. smooth hinge's
  // gamma). Accumulate in extended precision, also for signed bounds.
  // Fold validation into the first partition and sum only free entries;
  // the inactive all-at-bound cases need no per-entry extended arithmetic.
  auto m_first = std::partition(first, last, [=](const Data x) {
    if (!std::isfinite(x)) throw std::invalid_argument("Knapsack scores must be finite");
    return Compare(x) >= Compare(hi);
  });
  auto m_last =
      std::partition(m_first, last, [=](const Data x) { return Compare(x) > Compare(lo); });
  // Opposing saturated masses cancel exactly. The free entries are summed
  // plainly, and again with compensation when that rounding decides the test.
  Work error;
  const Work bounds = detail::bound_mass(Work(lo), Work(std::distance(m_last, last)), Work(hi),
                                         Work(std::distance(first, m_first)), error);
  Work mass = bounds;
  double size = std::abs(double(bounds)) + std::abs(double(rhs));
  for (auto it = m_first; it != m_last; ++it) {
    mass += Work(*it);
    size += std::abs(double(*it));
  }
  Work over = (mass - Work(rhs)) + error;
  const Work n_free = static_cast<Work>(std::distance(m_first, m_last));
  if (std::abs(over) <= 2 * (n_free + 2) * std::numeric_limits<Work>::epsilon() * Work(size)) {
    mass = bounds;
    for (auto it = m_first; it != m_last; ++it) {
      Work e;
      mass = detail::two_sum(mass, Work(*it), e);
      error += e;
    }
    Work f;
    over = detail::two_sum(mass, -Work(rhs), f);
    over += f + error;
  }
  if (over > 0) return thresholds_knapsack_eq(first, last, lo, hi, rhs);
  return thresholds<Result, Iterator>(0, lo, hi, m_first, m_last);
}

// [first, last) in place, through a temporary copy.
template <typename Result = double, typename Iterator>
inline void prox_knapsack_le(Iterator first, Iterator last, const Result lo = 0,
                             const Result hi = 1, const Result rhs = 1) {
  prox(first, last,
       thresholds_knapsack_le<
           Result,
           typename std::vector<typename std::iterator_traits<Iterator>::value_type>::iterator>,
       lo, hi, rhs);
}

// [first, last) in place; aux (as long) is scratch.
template <typename Result = double, typename Iterator>
inline void prox_knapsack_le(Iterator first, Iterator last, Iterator aux, const Result lo = 0,
                             const Result hi = 1, const Result rhs = 1) {
  prox(first, last, aux, thresholds_knapsack_le<Result, Iterator>, lo, hi, rhs);
}

// Each block of dim consecutive entries of [first, last) in place; aux holds dim
// entries.
template <typename Result = double, typename Iterator>
inline void prox_knapsack_le(const typename std::iterator_traits<Iterator>::difference_type dim,
                             Iterator first, Iterator last, Iterator aux, const Result lo = 0,
                             const Result hi = 1, const Result rhs = 1) {
  prox(dim, first, last, aux, thresholds_knapsack_le<Result, Iterator>, lo, hi, rhs);
}

}  // namespace sdca

#endif
