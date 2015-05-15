#include <algorithm>
#include <cassert>
#include <vector>

#include "knapsack_projector.hpp"

namespace sdca {

template <typename RealType>
void KnapsackProjector<RealType>::ComputeThresholds(
    std::vector<RealType> &x,
    RealType &t,
    RealType &lo,
    RealType &hi) {

  typename std::vector<RealType>::iterator m_begin, l_begin;
  PartitionAndComputeThresholds(x, t, lo, hi, m_begin, l_begin);
}

template <typename RealType>
void KnapsackProjector<RealType>::PartitionAndComputeThresholds(
    std::vector<RealType> &x,
    RealType &t,
    RealType &lo,
    RealType &hi,
    typename std::vector<RealType>::iterator &m_begin,
    typename std::vector<RealType>::iterator &l_begin) {
  /*
   * Based on the Algorithm 3.1 in
   * Kiwiel, K. C. "Variable fixing algorithms for the continuous
   * quadratic knapsack problem."
   * Journal of Optimization Theory and Applications 136.3 (2008): 445-458.
   */

  // Initialization
  m_begin = x.begin();
  l_begin = x.end();
  assert(std::distance(m_begin, l_begin));
  t = (std::accumulate(m_begin, l_begin, static_cast<RealType>(0)) - rhs_) /
    static_cast<RealType>(std::distance(m_begin, l_begin));
  lo = lo_;
  hi = hi_;

  for (std::size_t i = 0; i < x.size(); ++i) {
    // Feasibility check
    RealType tt = lo_ + t;
    auto lo_it = std::partition(m_begin, l_begin, [tt](const RealType &a){
      return a > tt; });
    RealType infeas_lo = + static_cast<RealType>(std::distance(lo_it, l_begin))
      * tt - std::accumulate(lo_it, l_begin, static_cast<RealType>(0));

    tt = hi_ + t;
    auto hi_it = std::partition(m_begin, lo_it, [tt](const RealType &a){
      return a > tt; });
    RealType infeas_hi = - static_cast<RealType>(std::distance(m_begin, hi_it))
      * tt + std::accumulate(m_begin, hi_it, static_cast<RealType>(0));

    // Variable fixing (using the incremental multiplier formula (23))
    if (infeas_lo > infeas_hi) {
      l_begin = lo_it;
      tt = infeas_lo;
    } else if (infeas_lo < infeas_hi) {
      m_begin = hi_it;
      tt = -infeas_hi;
    } else {
      m_begin = hi_it;
      l_begin = lo_it;
      break;
    }
    auto size = std::distance(m_begin, l_begin);
    if (size) {
      t += tt / static_cast<RealType>(size);
    } else {
      break;
    }
  }
}

template class KnapsackProjector<float>;
template class KnapsackProjector<double>;

}
