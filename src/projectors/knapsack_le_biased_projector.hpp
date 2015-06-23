#ifndef SDCA_KNAPSACK_LE_BIASED_PROJECTOR_HPP
#define SDCA_KNAPSACK_LE_BIASED_PROJECTOR_HPP

#include "knapsack_projector.hpp"

namespace sdca {

template <typename RealType = double>
class KnapsackLEBiasedProjector : public KnapsackProjector<RealType> {

using KnapsackProjector<RealType>::lo_;
using KnapsackProjector<RealType>::hi_;
using KnapsackProjector<RealType>::rhs_;

public:
  KnapsackLEBiasedProjector(
      const RealType lo = 0,
      const RealType hi = 1,
      const RealType rhs = 1,
      const RealType rho = 1
    ) :
      KnapsackProjector<RealType>::KnapsackProjector(lo, hi, rhs),
      rho_(rho)
  {}

  void ComputeThresholds(
      std::vector<RealType> &x,
      RealType &t,
      RealType &lo,
      RealType &hi
    ) const override;

  void ComputeGeneralCase(
      std::vector<RealType> &x,
      RealType &t,
      RealType &lo,
      RealType &hi
    ) const;

  RealType get_rho() const { return rho_; }
  void set_rho(const RealType rho) { rho_ = rho; }

private:
  RealType rho_;

};

}

#endif // SDCA_KNAPSACK_LE_BIASED_PROJECTOR_HPP
