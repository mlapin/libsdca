#ifndef SDCA_TOPKSIMPLEXPROJECTOR_HPP
#define SDCA_TOPKSIMPLEXPROJECTOR_HPP

#include "knapsackprojector.hpp"
#include "topkconeprojector.hpp"

namespace sdca {

template <typename RealType = double>
class TopKSimplexProjector : public Projector<RealType> {

public:
  TopKSimplexProjector(
      const std::size_t k = 1,
      const RealType rhs = 1
    ) :
      top_k_cone_(k),
      knapsack_(0, rhs/static_cast<RealType>(k), rhs)
  {}

  void ComputeThresholds(
      std::vector<RealType> &x,
      RealType &t,
      RealType &lo,
      RealType &hi
    ) override;

  bool CheckOnTopKCone(
      std::vector<RealType> &x,
      RealType &t,
      typename std::vector<RealType>::iterator &first
    );

  TopKConeProjector<RealType> get_top_k_cone() const { return top_k_cone_; }

  KnapsackProjector<RealType> get_knapsack() const { return knapsack_; }

private:
  TopKConeProjector<RealType> top_k_cone_;
  KnapsackProjector<RealType> knapsack_;
};

}

#endif // SDCA_TOPKSIMPLEXPROJECTOR_HPP
