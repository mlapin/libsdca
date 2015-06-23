#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include <vector>

#include "knapsack_le_biased_projector.hpp"

namespace sdca {

template <typename RealType>
void KnapsackLEBiasedProjector<RealType>::ComputeThresholds(
    std::vector<RealType> &x,
    RealType &t,
    RealType &lo,
    RealType &hi) const {

  // First, consider the case (t = 0)
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

  // Second, consider the CQKP if (sum > rhs)
  if (sum > rhs_) {
    KnapsackProjector<RealType>::PartitionAndComputeThresholds(
      x, t, lo, hi, m_begin, l_begin);
    if (t >= rho_ * rhs_) {
      return;
    }
  } else if (t == rho_ * sum) {
    return;
  }

  // Finally, consider the general case
  ComputeGeneralCase(x, t, lo, hi);
}

template <typename RealType>
void KnapsackLEBiasedProjector<RealType>::ComputeGeneralCase(
    std::vector<RealType> &x,
    RealType &t,
    RealType &lo,
    RealType &hi) const {

  // Lower and upper bounds do not change
  lo = lo_;
  hi = hi_;

  // Sort x to search efficiently
  std::sort(x.begin(), x.end(), std::greater<RealType>());

  // At this point, rho must be positive
  assert(rho_ > 0);
  RealType rho_rhs = rho_ * rhs_;
  RealType rho_inverse = static_cast<RealType>(1) / rho_;

  RealType num_U = 0;
  RealType num_X = static_cast<RealType>(x.size());

  RealType min_U = +std::numeric_limits<RealType>::infinity();

  // Grow U starting with empty
  for (auto m_begin = x.begin();;) {

    RealType min_M = +std::numeric_limits<RealType>::infinity();
    RealType max_M = -std::numeric_limits<RealType>::infinity();

    RealType num_M = 0, sum_M = 0;
    RealType num_L = num_X - num_U;

    // Grow M starting with empty
    for (auto l_begin = m_begin;;) {
      // Compute t as follows:
      //    t = (lo*num_L + hi*num_U + sum_M) / (1/rho + num_M)
      // and check that
      //  (1)  lo + t  >= max_L = (l_begin) or (-Inf)
      //  (2)  lo + t  <= min_M = (l_begin - 1) or (+Inf)
      //  (3)  hi + t  >= max_M = (m_begin) or (-Inf)
      //  (4)  hi + t  <= min_U = (m_begin - 1) or (+Inf)
      //  (5)       t  <= rho * rhs

      t = (lo * num_L + hi * num_U + sum_M) / (rho_inverse + num_M);
      if (t <= rho_rhs) {
        RealType tt = hi + t;
        if (max_M <= tt && tt <= min_U) {
          tt = lo + t;
          if (tt <= min_M && ((l_begin == x.end()) || *l_begin <= tt)) {
            return;
          }
        }
      }

      // Increment the set M
      if (l_begin == x.end()) {
        break;
      }
      min_M = *l_begin;
      max_M = *m_begin;
      sum_M += min_M;
      ++num_M;
      ++l_begin;
    }

    // Increment the set U
    if (m_begin == x.end()) {
      break;
    }
    min_U = *m_begin;
    ++num_U;
    ++m_begin;
  }

  // Should never reach here
  t = lo;
  assert(false);
}

template class KnapsackLEBiasedProjector<float>;
template class KnapsackLEBiasedProjector<double>;

}
