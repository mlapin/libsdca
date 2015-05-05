#ifndef SDCA_TOPKSIMPLEXPROJECTOR_HPP
#define SDCA_TOPKSIMPLEXPROJECTOR_HPP

#include "knapsackprojector.hpp"
#include "topkconeprojector.hpp"

namespace sdca {

template <typename RealType = double>
class TopKSimplexProjector : public Projector<RealType> {

public:
  TopKSimplexProjector(
      const std::size_t k,
      const RealType rhs = 1
    ) :
      cone_(k),
      knap_(0, rhs/static_cast<RealType>(k), rhs)
    {}

  void ComputeThresholds(
      std::vector<RealType> x,
      RealType &t,
      RealType &lo,
      RealType &hi
    );

  bool CheckNeedFallback(
      const std::vector<RealType> x,
      const RealType t,
      const typename std::vector<RealType>::const_iterator first
    );

  TopKConeProjector<RealType> cone() const { return cone_; }

  KnapsackProjector<RealType> knap() const { return knap_; }

private:
  TopKConeProjector<RealType> cone_;
  KnapsackProjector<RealType> knap_;
};

}

#endif // SDCA_TOPKSIMPLEXPROJECTOR_HPP
