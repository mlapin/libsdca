#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "topkconeprojector.hpp"

namespace sdca {

template <typename RealType>
void TopKConeProjector<RealType>::Project(RealType *x, const std::size_t m,
    const std::size_t n) {
  RealType t, hi;
  std::vector<RealType> aux(m);
  for (std::size_t i = 0; i < n; ++i) {
    std::copy(x + m*i, x + m*(i+1), &aux[0]);
    ComputeThresholds(aux, t, hi);
    TopKProjector<RealType>::Clamp(x + m*i, m, t, hi);
  }
}

template <typename RealType>
void TopKConeProjector<RealType>::ComputeThresholds(std::vector<RealType> x,
    RealType &t, RealType &hi) {
  // Partially sort x around the kth element
  std::nth_element(x.begin(), x.begin() + k_ - 1, x.end(),
    std::greater<RealType>());

  RealType sum_k_largest = std::accumulate(x.begin(), x.begin() + k_,
    static_cast<RealType>(0));

  // Case 1: U empty, M empty, proj = 0
  t = 0;
  hi = 0;
  if (sum_k_largest <= 0) {
    return;
  }

  // Sum all positive, compute the 1st and the (k+1)st elements
  RealType sum_pos = 0;
  RealType max_elem = -std::numeric_limits<RealType>::infinity();
  RealType kp1_elem = -std::numeric_limits<RealType>::infinity();
  typename std::vector<RealType>::iterator it = x.begin();
  for (; it != x.begin() + k_; ++it) {
    if (*it > 0) sum_pos += *it;
    if (*it > max_elem) max_elem = *it;
  }
  for (; it != x.end(); ++it) {
    if (*it > 0) sum_pos += *it;
    if (*it > kp1_elem) kp1_elem = *it;
  }

  // Case 2: U empty, M not empty, proj = max(0,x)
  if (sum_pos >= kk_ * max_elem) {
    t = 0;
    hi = std::numeric_limits<RealType>::infinity();
    return;
  }

  // Case 3: U not empty, M empty, proj = 1/k sum_k_largest for k largest
  if (sum_k_largest <= kk_ * (x[k_-1] - kp1_elem)) {
    hi = sum_k_largest / kk_;
    t = hi - x[k_-1];
    return;
  }

  std::sort(x.begin(), x.end(), std::greater<RealType>());

  // Case 4: U not empty, M not empty, exhaustive search
  RealType su = 0;
  typedef typename std::vector<RealType>::size_type VectorSize;
  for (VectorSize u = 1; u < k_; ++u) {
    su += x[u-1]; // sum over U
    RealType uu = static_cast<RealType>(u);
    RealType ku = kk_ - uu;
    RealType kusu = ku * su; // (k-u) * sum_U x_i
    RealType sm = 0;
    RealType D = ku * ku;

    for (VectorSize m = 1; m < x.size() - u; ++m) {
      sm += x[m+u-1]; // sum over M
      D += uu; // (k-u)^2 + m*u
      RealType pkD = kusu - uu*sm; // (p/k)*D
      if (0 <= D*x[m+u-1] + pkD && D*x[m+u] + pkD <= 0) {
        RealType skD = ku*sm + static_cast<RealType>(m)*su; // (s/k)*D
        if (0 <= D*x[u-1] + pkD - skD && D*x[u] + pkD - skD <= 0) {
          t = pkD / D;
          hi = skD / D;
          return;
        }
      }
    }

    // L empty (u + m = x.size)
    sm += x[x.size()-1];
    D += uu;
    RealType pkD = kusu - uu*sm;
    if (0 <= D*x[x.size()-1] + pkD) {
      RealType skD = ku*sm + static_cast<RealType>(x.size()-u)*su;
      if (0 <= D*x[u-1] + pkD - skD && D*x[u] + pkD - skD <= 0) {
        t = pkD / D;
        hi = skD / D;
        return;
      }
    }
  }
}

template class TopKConeProjector<float>;
template class TopKConeProjector<double>;

}
