#include <algorithm>
#include <cassert>
#include <vector>

#include "knapsackprojector.hpp"

namespace sdca {

template <typename RealType>
void KnapsackProjector<RealType>::ComputeThresholds(
    std::vector<RealType> x,
    RealType &t,
    RealType &lo,
    RealType &hi) {

  typename std::vector<RealType>::const_iterator first, last;
  ComputeThresholdsAndMidBoundary(x, t, lo, hi, first, last);
}

template <typename RealType>
void KnapsackProjector<RealType>::ComputeThresholdsAndMidBoundary(
    std::vector<RealType> x,
    RealType &t,
    RealType &lo,
    RealType &hi,
    typename std::vector<RealType>::const_iterator &first,
    typename std::vector<RealType>::const_iterator &last) {
  /*
   * Based on the Algorithm 3.1 in
   * Kiwiel, K. C. "Variable fixing algorithms for the continuous
   * quadratic knapsack problem."
   * Journal of Optimization Theory and Applications 136.3 (2008): 445-458.
   */

  // Initialization (note: t here is -t in Algorithm 3.1)
  first = x.cbegin();
  last = x.cend();
  assert(std::distance(first, last));
  t = (rhs_ - std::accumulate(first, last, static_cast<RealType>(0))) /
    static_cast<RealType>(std::distance(first, last));
  lo = lo_;
  hi = hi_;

  for (std::size_t i = 0; i < x.size(); ++i) {
    // Feasibility check
    RealType tt = lo_ - t;
    auto lo_it = std::partition(first, last, [tt](const RealType &a){
      return a > tt; });
    RealType infeas_lo = static_cast<RealType>(std::distance(lo_it, last)) * tt
      - std::accumulate(lo_it, last, static_cast<RealType>(0));

    tt = hi_ - t;
    auto hi_it = std::partition(first, lo_it, [tt](const RealType &a){
      return a > tt; });
    RealType infeas_hi = std::accumulate(first, hi_it, static_cast<RealType>(0))
      - static_cast<RealType>(std::distance(first, hi_it)) * tt;

    // Variable fixing (using the incremental multiplier formula (23))
    if (infeas_lo > infeas_hi) {
      last = lo_it;
      assert(std::distance(first, last));
      t -= infeas_lo / static_cast<RealType>(std::distance(first, last));
    } else if (infeas_lo < infeas_hi) {
      first = hi_it;
      assert(std::distance(first, last));
      t += infeas_hi / static_cast<RealType>(std::distance(first, last));
    } else {
      break;
    }
  }
}

template class KnapsackProjector<float>;
template class KnapsackProjector<double>;

}
