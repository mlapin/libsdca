#include <algorithm>
#include <cassert>
#include <vector>

#include "knapsack_le_projector.hpp"

namespace sdca {

template <typename RealType>
void KnapsackLEProjector<RealType>::ComputeThresholds(
    std::vector<RealType> &x,
    RealType &t,
    RealType &lo,
    RealType &hi) const {

  // First, check if the inequality constraint can be removed (sum <= rhs)
  t = 0;
  lo = lo_;
  hi = hi_;
  auto l_begin = std::partition(x.begin(), x.end(),
    [=](const RealType &a){ return a > lo; });
  auto m_begin = std::partition(x.begin(), l_begin,
    [=](const RealType &a){ return a >= hi; });

  RealType sum = std::accumulate(m_begin, l_begin, static_cast<RealType>(0));
  sum += lo * static_cast<RealType>(std::distance(l_begin, x.end()));
  sum += hi * static_cast<RealType>(std::distance(x.begin(), m_begin));

  // Otherwise (sum > rhs), we have an equality constraint
  if (sum > rhs_) {
    KnapsackProjector<RealType>::PartitionAndComputeThresholds(
      x, t, lo, hi, m_begin, l_begin);
  }
}

template class KnapsackLEProjector<float>;
template class KnapsackLEProjector<double>;

}
