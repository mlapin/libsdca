#ifndef SDCA_TOPK_SIMPLEX_PROJECTOR_HPP
#define SDCA_TOPK_SIMPLEX_PROJECTOR_HPP

#include "knapsack_projector.hpp"
#include "topk_cone_projector.hpp"

namespace sdca {

template <typename RealType = double>
class TopKSimplexProjector : public Projector<RealType> {

public:
  TopKSimplexProjector(
      const std::size_t k = 1,
      const RealType rhs = 1
    ) :
      cone_(k),
      knapsack_(0, rhs/static_cast<RealType>(k), rhs)
  {}

  void ComputeThresholds(
      std::vector<RealType> &x,
      RealType &t,
      RealType &lo,
      RealType &hi
    ) const override;

  bool CheckProjectOntoCone(
      std::vector<RealType> &x,
      RealType &t,
      typename std::vector<RealType>::iterator &m_begin
    ) const;

  TopKConeProjector<RealType> get_cone() const { return cone_; }

  KnapsackProjector<RealType> get_knapsack() const { return knapsack_; }

private:
  TopKConeProjector<RealType> cone_;
  KnapsackProjector<RealType> knapsack_;
};

}

#endif // SDCA_TOPK_SIMPLEX_PROJECTOR_HPP
