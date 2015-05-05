#include <algorithm>
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
      cone_.FallBackCase(x, t, lo, hi);
      break;
  }

}

template <typename RealType>
bool TopKSimplexProjector<RealType>::CheckKnapsackSolution(
    const std::vector<RealType> x,
    const typename std::vector<RealType>::iterator first,
    const typename std::vector<RealType>::iterator last) {

  RealType u = static_cast<RealType>(std::distance(x.begin(), first));
  RealType sum_u = std::accumulate(x.begin(), first, static_cast<RealType>(0));

  RealType m = static_cast<RealType>(std::distance(first, last));
  RealType sum_m = std::accumulate(first, last, static_cast<RealType>(0));

  RealType k_minus_u = cone_.kk() - u;
  RealType D = k_minus_u * k_minus_u + u * m;

  RealType tD = u * sum_m - k_minus_u * sum_u;
  RealType hiD = k_minus_u * sum_m + m * sum_u;
  RealType s_minus_p_D = hiD + tD;

  bool u_empty = first == x.begin();
  bool m_empty = first == last;
  bool l_empty = last == x.end();

  bool t_lo = l_empty || (*last * D <= tD);
  bool t_hi = l_empty || (*last * D <= tD);

  return false;
}

template class TopKSimplexProjector<float>;
template class TopKSimplexProjector<double>;

}
