#ifndef SDCA_KNAPSACK_PROJECTOR_HPP
#define SDCA_KNAPSACK_PROJECTOR_HPP

#include "projector.hpp"

namespace sdca {

template <typename RealType = double>
class KnapsackProjector : public Projector<RealType> {

public:
  KnapsackProjector(
      const RealType lo = 0,
      const RealType hi = 1,
      const RealType rhs = 1
    ) :
      lo_(lo),
      hi_(hi),
      rhs_(rhs)
  {}

  void ComputeThresholds(
      std::vector<RealType> &x,
      RealType &t,
      RealType &lo,
      RealType &hi
    ) const override;

  void PartitionAndComputeThresholds(
      std::vector<RealType> &x,
      RealType &t,
      RealType &lo,
      RealType &hi,
      typename std::vector<RealType>::iterator &first,
      typename std::vector<RealType>::iterator &last
    ) const;

  RealType get_lo() const { return lo_; }

  RealType get_hi() const { return hi_; }

  RealType get_rhs() const { return rhs_; }

protected:
  const RealType lo_;
  const RealType hi_;
  const RealType rhs_;

};

}

#endif // SDCA_KNAPSACK_PROJECTOR_HPP
