#ifndef SDCA_BIASEDTOPKSIMPLEXPROJECTOR_HPP
#define SDCA_BIASEDTOPKSIMPLEXPROJECTOR_HPP

#include "knapsackprojector.hpp"
#include "biasedtopkconeprojector.hpp"

namespace sdca {

template <typename RealType = double>
class BiasedTopKSimplexProjector : public Projector<RealType> {

public:
  BiasedTopKSimplexProjector(
      const std::size_t k = 1,
      const RealType rho = 1,
      const RealType rhs = 1
    ) :
      cone_(k, rho),
      knapsack_(0, rhs/static_cast<RealType>(k), rhs)
  {}

  void ComputeThresholds(
      std::vector<RealType> &x,
      RealType &t,
      RealType &lo,
      RealType &hi
    ) override;

  bool CheckProjectOntoCone(
      std::vector<RealType> &x,
      RealType &t,
      typename std::vector<RealType>::iterator &m_begin
    );

  BiasedTopKConeProjector<RealType> get_cone() const { return cone_; }

  KnapsackProjector<RealType> get_knapsack() const { return knapsack_; }

private:
  BiasedTopKConeProjector<RealType> cone_;
  KnapsackProjector<RealType> knapsack_;
};

}

#endif // SDCA_BIASEDTOPKSIMPLEXPROJECTOR_HPP
