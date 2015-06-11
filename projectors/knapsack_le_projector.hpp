#ifndef SDCA_KNAPSACK_LE_PROJECTOR_HPP
#define SDCA_KNAPSACK_LE_PROJECTOR_HPP

#include "knapsack_projector.hpp"

namespace sdca {

template <typename RealType = double>
class KnapsackLEProjector : public KnapsackProjector<RealType> {

using KnapsackProjector<RealType>::lo_;
using KnapsackProjector<RealType>::hi_;
using KnapsackProjector<RealType>::rhs_;

public:
  KnapsackLEProjector(
      const RealType lo = 0,
      const RealType hi = 1,
      const RealType rhs = 1
    ) :
      KnapsackProjector<RealType>::KnapsackProjector(lo, hi, rhs)
  {}

  void ComputeThresholds(
      std::vector<RealType> &x,
      RealType &t,
      RealType &lo,
      RealType &hi
    ) const override;

private:

};

}

#endif // SDCA_KNAPSACK_LE_PROJECTOR_HPP
