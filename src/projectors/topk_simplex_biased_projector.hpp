#ifndef SDCA_TOPK_SIMPLEX_BIASED_PROJECTOR_HPP
#define SDCA_TOPK_SIMPLEX_BIASED_PROJECTOR_HPP

#include "knapsack_projector.hpp"
#include "topk_cone_biased_projector.hpp"

namespace sdca {

template <typename RealType = double>
class TopKSimplexBiasedProjector : public Projector<RealType> {

public:
  TopKSimplexBiasedProjector(
      const std::size_t k = 1,
      const RealType rho = 1,
      const RealType rhs = 1
    ) :
      knapsack_(0, rhs/static_cast<RealType>(k), rhs),
      cone_(k, rho),
      rho_rhs_(rho * rhs)
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

  void set_rho(const RealType rho) {
    rho_rhs_ = rho * knapsack_.get_rhs();
    cone_.set_rho(rho);
  }

private:
  const KnapsackProjector<RealType> knapsack_;
  TopKConeBiasedProjector<RealType> cone_;
  RealType rho_rhs_;
};

}

#endif // SDCA_TOPK_SIMPLEX_BIASED_PROJECTOR_HPP
