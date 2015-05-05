#include <algorithm>
#include <cassert>
#include <numeric>
#include <utility>
#include <vector>

#include "topksimplexprojector.hpp"

namespace sdca {

template <typename RealType>
void TopKSimplexProjector<RealType>::ComputeThresholds(
    std::vector<RealType> x,
    RealType &t,
    RealType &lo,
    RealType &hi) {

  RealType sum_k, sum_pos;
  typename std::vector<RealType>::iterator first, last;
  switch (cone_.CheckSpecialCases(x, t, lo, hi, sum_k, sum_pos)) {
    case TopKConeCase::NoneUpperNoneMiddle: // projection is 0
      break;
    case TopKConeCase::NoneUpperSomeMiddle: // proj = max(0,x)
      if (sum_pos > knap_.rhs()) {
        knap_.ComputeThresholdsAndMidBoundary(x, t, lo, hi, first, last);
      }
      break;
    case TopKConeCase::SomeUpperNoneMiddle: // proj = 1/k sum_k_largest
      if (sum_k > knap_.rhs()) {
        knap_.ComputeThresholdsAndMidBoundary(x, t, lo, hi, first, last);
      }
      break;
    case TopKConeCase::SomeUpperSomeMiddle:
      knap_.ComputeThresholdsAndMidBoundary(x, t, lo, hi, first, last);
      if (CheckNeedFallback(x, t, first)) {
        cone_.FallBackCase(x, t, lo, hi);
      }
      break;
  }

}

template <typename RealType>
bool TopKSimplexProjector<RealType>::CheckNeedFallback(
    const std::vector<RealType> x,
    const RealType t,
    const typename std::vector<RealType>::const_iterator first) {

  RealType u = static_cast<RealType>(std::distance(x.begin(), first));
  RealType sum_u = std::accumulate(x.begin(), first, static_cast<RealType>(0));
  RealType k = cone_.kk();
  assert(u <= k);

  // Check if the corresponding lambda is negative
  return k * ( sum_u + (u - k) * t) < u;
}

template class TopKSimplexProjector<float>;
template class TopKSimplexProjector<double>;

}
