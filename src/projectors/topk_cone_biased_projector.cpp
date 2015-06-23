#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

#include "topk_cone_biased_projector.hpp"

namespace sdca {

template <typename RealType>
void TopKConeBiasedProjector<RealType>::ComputeGeneralCase(
    std::vector<RealType> &x,
    RealType &t,
    RealType &lo,
    RealType &hi) const {

  // Lower bound is always zero
  lo = 0;

  // Sort x to search efficiently
  std::sort(x.begin(), x.end(), std::greater<RealType>());

  RealType min_U = std::numeric_limits<RealType>::infinity();
  RealType sum_u = 0, k_minus_u = kk_;
  RealType u_plus_rho_k_2 = rho_k_k_, u_rho_k_plus_1 = 0;

  // U is empty in the beginning (m_begin = x.begin)
  auto k_end = x.begin()
    + static_cast<typename std::vector<RealType>::difference_type>(k_);
  for (auto m_begin = x.begin(); m_begin != k_end; ++m_begin) {

    RealType min_M = *m_begin;
    RealType sum_m = min_M, m_sum_u = sum_u;
    RealType D = k_minus_u * k_minus_u + u_plus_rho_k_2;
    RealType k_minus_u_plus_rho_k_m = k_minus_u + rho_k_;

    // Start with (m_begin + 1) so that M is not empty
    auto l_begin = m_begin + 1;
    for (; l_begin != x.end(); ++l_begin) {
      // Set the thresholds for the current U and M as follows:
      //    s/k = [(k-u)*sum_m + m*sum_u] / D
      //   -p/k = [u*(1+rho*k)*sum_m - (k-u+rho*k*m)*sum_u] / D
      //      D = (k - u)^2 + (u + rho*k^2)*m
      // where u = |U|, m = |M|.
      // Then, check that the thresholds are consistent with the partitioning:
      //  (1)  rho*k*s/k - p/k     >= max_L = l_begin
      //  (2)  rho*k*s/k - p/k     <= min_M = l_begin - 1
      //  (3)  (1+rho*k)s/k - p/k  >= max_M = m_begin
      //  (4)  (1+rho*k)s/k - p/k  <= min_U = m_begin - 1 or +Inf
/*
      RealType toleps = static_cast<RealType>(1e-12);
      RealType u = static_cast<RealType>(std::distance(x.begin(), m_begin));
      RealType m = static_cast<RealType>(std::distance(m_begin, l_begin));
      assert(k_minus_u == kk_ - u);
      assert(sum_u == std::accumulate(x.begin(), m_begin,
        static_cast<RealType>(0)));
      assert(sum_m == std::accumulate(m_begin, l_begin,
        static_cast<RealType>(0)));
      assert(std::abs(m_sum_u - m * sum_u) <= toleps);
      assert(std::abs(u_rho_k_plus_1
        - u * (rho_ * kk_ + static_cast<RealType>(1))) <= toleps);
      assert(std::abs(k_minus_u_plus_rho_k_m - (kk_ - u + rho_ * kk_ * m))
        <= toleps);
*/
      RealType skD = k_minus_u * sum_m + m_sum_u;
      RealType pkD = u_rho_k_plus_1 * sum_m - k_minus_u_plus_rho_k_m * sum_u;
      RealType t1 = rho_k_ * skD + pkD;
      if (t1 >= *l_begin * D) {
        RealType t2 = t1 + skD;
        if (t2 >= *m_begin * D) {
          if ((t1 <= min_M * D) && (t2 <= min_U * D)) {
            t = t1 / D;
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
      D += u_plus_rho_k_2;
      k_minus_u_plus_rho_k_m += rho_k_;
    }

    // L is emtpy
    if (l_begin == x.end()) {
      RealType skD = k_minus_u * sum_m + m_sum_u;
      RealType pkD = u_rho_k_plus_1 * sum_m - k_minus_u_plus_rho_k_m * sum_u;
      RealType t1 = rho_k_ * skD + pkD;
      if (t1 <= min_M * D) {
        RealType t2 = t1 + skD;
        if ((t2 >= *m_begin * D) && (t2 <= min_U * D)) {
          t = t1 / D;
          hi = skD / D;
          return;
        }
      }
    }

    // Increment the set U
    min_U = *m_begin;
    sum_u += min_U;
    --k_minus_u;
    ++u_plus_rho_k_2;
    u_rho_k_plus_1 += rho_k_plus_1_;
  }

}

template class TopKConeBiasedProjector<float>;
template class TopKConeBiasedProjector<double>;

}
