#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "topk_cone_projector.hpp"

namespace sdca {

template <typename RealType>
void TopKConeProjector<RealType>::ComputeThresholds(
    std::vector<RealType> &x,
    RealType &t,
    RealType &lo,
    RealType &hi) {

  switch (CheckSpecialCases(x, t, lo, hi)) {
    case Projection::Zero:
    case Projection::Constant:
      break;
    case Projection::General:
      ComputeGeneralCase(x, t, lo, hi);
      break;
  }

}

template <typename RealType>
Projection TopKConeProjector<RealType>::CheckSpecialCases(
    std::vector<RealType> &x,
    RealType &t,
    RealType &lo,
    RealType &hi) {

  // Partially sort x around the kth element
  std::nth_element(x.begin(), x.begin() + k_ - 1, x.end(),
    std::greater<RealType>());

  // Sum k largest elements
  RealType sum_k_largest = std::accumulate(x.begin(), x.begin() + k_,
    static_cast<RealType>(0));

  // Case 1: U empty, M empty, proj = 0
  t = lo = hi = 0;
  if (sum_k_largest <= 0) {
    return Projection::Zero;
  }

  // Case 2: U not empty, M empty, proj = const * sum_k_largest for k largest
  hi = projection_const_ * sum_k_largest;
  t = x[k_-1] - hi;
  if ((k_ == x.size()) || (t >= *std::max_element(x.begin() + k_, x.end()))) {
    return Projection::Constant;
  }

  // Case 3: U undefined, M not empty, no closed-form solution
  return Projection::General;
}

template <typename RealType>
void TopKConeProjector<RealType>::ComputeGeneralCase(
    std::vector<RealType> &x,
    RealType &t,
    RealType &lo,
    RealType &hi) {

  // Lower bound is always zero
  lo = 0;

  // Sort x to search efficiently
  std::sort(x.begin(), x.end(), std::greater<RealType>());

  RealType min_U = std::numeric_limits<RealType>::infinity();
  RealType sum_u = 0, u = 0, k_minus_u = kk_, k_minus_u_sum_u = 0;

  // U is empty in the beginning (m_begin = x.begin)
  for (auto m_begin = x.begin(); m_begin != x.begin() + k_; ++m_begin) {

    RealType min_M = *m_begin;
    RealType sum_m = min_M, m_sum_u = sum_u, D = k_minus_u * k_minus_u + u;

    // Start with (m_begin + 1) so that M is not empty
    auto l_begin = m_begin + 1;
    for (; l_begin != x.end(); ++l_begin) {
      // Set the thresholds for the current U and M as follows:
      //    s/k = [(k-u)*sum_m + m*sum_u] / D
      //   -p/k = [u*sum_m - (k-u)*sum_u] / D
      //      D = (k - u)^2 + u*m
      // where u = |U|, m = |M|.
      // Then, check that the thresholds are consistent with the partitioning:
      //  (1)  - p/k        >= max_L = l_begin
      //  (2)  - p/k        <= min_M = l_begin - 1
      //  (3)    s/k - p/k  >= max_M = m_begin
      //  (4)    s/k - p/k  <= min_U = m_begin - 1 or +Inf

      RealType pkD = u * sum_m - k_minus_u_sum_u;
      if (pkD >= *l_begin * D) {
        RealType skD = k_minus_u * sum_m + m_sum_u;
        RealType tt = pkD + skD;
        if (tt >= *m_begin * D) {
          if ((pkD <= min_M * D) && (tt <= min_U * D)) {
            t = pkD / D;
            hi = skD / D;
            return;
          }
        } else {
          // (1) holds, but (3) does not => exit the inner loop
          break;
        }
      }

      // Increment the set M
      min_M = *l_begin;
      sum_m += min_M;
      m_sum_u += sum_u;
      D += u;
    }

    // L is emtpy
    if (l_begin == x.end()) {
      RealType pkD = u * sum_m - k_minus_u_sum_u;
      if (pkD <= min_M * D) {
        RealType skD = k_minus_u * sum_m + m_sum_u;
        RealType tt = pkD + skD;
        if ((tt >= *m_begin * D) && (tt <= min_U * D)) {
          t = pkD / D;
          hi = skD / D;
          return;
        }
      }
    }

    // Increment the set U
    min_U = *m_begin;
    sum_u += min_U;
    ++u;
    --k_minus_u;
    k_minus_u_sum_u = k_minus_u * sum_u;
  }

}

template class TopKConeProjector<float>;
template class TopKConeProjector<double>;

}
