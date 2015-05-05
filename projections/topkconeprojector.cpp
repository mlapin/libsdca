#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "topkconeprojector.hpp"

namespace sdca {

template <typename RealType>
void TopKConeProjector<RealType>::ComputeThresholds(
    std::vector<RealType> x,
    RealType &t,
    RealType &lo,
    RealType &hi) {

  RealType sum_k, sum_pos;
  switch (CheckSpecialCases(x, t, lo, hi, sum_k, sum_pos)) {
    case TopKConeCase::NoneUpperNoneMiddle: // projection is 0
      break;
    case TopKConeCase::NoneUpperSomeMiddle: // proj = max(0,x)
      break;
    case TopKConeCase::SomeUpperNoneMiddle: // proj = 1/k sum_k_largest
      break;
    case TopKConeCase::SomeUpperSomeMiddle:
      FallBackCase(x, t, lo, hi);
      break;
  }

}

template <typename RealType>
TopKConeCase TopKConeProjector<RealType>::CheckSpecialCases(
    std::vector<RealType> x,
    RealType &t,
    RealType &lo,
    RealType &hi,
    RealType &sum_k_largest,
    RealType &sum_positive) {

  // Partially sort x around the kth element
  std::nth_element(x.begin(), x.begin() + k_ - 1, x.end(),
    std::greater<RealType>());

  // Sum k largest elements
  sum_k_largest = std::accumulate(x.begin(), x.begin() + k_,
    static_cast<RealType>(0));

  // Case 1: U empty, M empty, proj = 0
  t = lo = hi = 0;
  if (sum_k_largest <= 0) {
    return TopKConeCase::NoneUpperNoneMiddle;
  }

  // Sum all positive, find the 1st (max) and the (k+1)st elements
  sum_positive = 0;
  RealType max_elem = -std::numeric_limits<RealType>::infinity();
  RealType kp1_elem = -std::numeric_limits<RealType>::infinity();
  auto it = x.begin();
  for (; it != x.begin() + k_; ++it) {
    if (*it > 0) sum_positive += *it;
    if (*it > max_elem) max_elem = *it;
  }
  for (; it != x.end(); ++it) {
    if (*it > 0) sum_positive += *it;
    if (*it > kp1_elem) kp1_elem = *it;
  }

  // Case 2: U empty, M not empty, proj = max(0,x)
  if (sum_positive >= kk_ * max_elem) {
    hi = std::numeric_limits<RealType>::infinity();
    return TopKConeCase::NoneUpperSomeMiddle;
  }

  // Case 3: U not empty, M empty, proj = 1/k sum_k_largest for k largest
  if (sum_k_largest <= kk_ * (x[k_-1] - kp1_elem)) {
    hi = sum_k_largest / kk_;
    t = hi - x[k_-1];
    return TopKConeCase::SomeUpperNoneMiddle;
  }

  // Case 4: U not empty, M not empty, no closed-form solution
  return TopKConeCase::SomeUpperSomeMiddle;
}

template <typename RealType>
void TopKConeProjector<RealType>::FallBackCase(
    std::vector<RealType> x,
    RealType &t,
    RealType &lo,
    RealType &hi) {

  // Sort x for a more efficient search
  std::sort(x.begin(), x.end(), std::greater<RealType>());

  // Case 4: U not empty, M not empty, exhaustive search
  RealType sum_u = 0, u = 0, k_minus_u = kk_;
  for (auto min_u = x.begin(); min_u != x.begin() + k_ - 1; ++min_u) {
    sum_u += *min_u;
    ++u;
    --k_minus_u;

    RealType sum_m = 0, m_sum_u = 0, D = k_minus_u * k_minus_u;
    RealType u_minus_k_sum_u = -k_minus_u * sum_u;
    auto min_m = min_u + 1;
    auto max_m = min_m;
    for (; min_m != x.end(); ++min_m) {
      sum_m += *min_m;
      m_sum_u += sum_u;
      D += u; // D = (k-u)^2 + m*u

      RealType tD = u * sum_m + u_minus_k_sum_u;
      RealType hiD = k_minus_u * sum_m + m_sum_u;
      RealType s_minus_p_D = hiD + tD;
      auto max_l = min_m + 1;
      if ( (max_l == x.end()) || (*max_l * D <= tD) ) {
        if (*max_m * D <= s_minus_p_D) {
          if ( (s_minus_p_D <= *min_u * D) && (tD <= *min_m * D) ) {
            t = -tD / D;
            hi = hiD / D;
            lo = 0;
            return;
          }
        } else {
          break; // stop early since s/k - p/k will decrease from now on
        }
      }
    }
  }

}

template class TopKConeProjector<float>;
template class TopKConeProjector<double>;

}
