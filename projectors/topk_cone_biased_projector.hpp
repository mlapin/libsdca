#ifndef SDCA_TOPK_CONE_BIASED_PROJECTOR_HPP
#define SDCA_TOPK_CONE_BIASED_PROJECTOR_HPP

#include "topk_cone_projector.hpp"

namespace sdca {

template <typename RealType = double>
class TopKConeBiasedProjector : public TopKConeProjector<RealType> {

using TopKConeProjector<RealType>::k_;
using TopKConeProjector<RealType>::kk_;
using TopKConeProjector<RealType>::projection_const_;

public:
  TopKConeBiasedProjector(
      const std::size_t k = 1,
      const RealType rho = 1
    ) :
      TopKConeProjector<RealType>::TopKConeProjector(k),
      rho_(rho)
  { precompute_common(); }

  void ComputeGeneralCase(
      std::vector<RealType> &x,
      RealType &t,
      RealType &lo,
      RealType &hi
    ) const override;

  void set_k(const std::size_t k) override {
    k_ = k;
    kk_ = static_cast<RealType>(k);
    precompute_common();
  }

  RealType get_rho() const { return rho_; }
  void set_rho(const RealType rho) {
    rho_ = rho;
    precompute_common();
  }

private:
  RealType rho_;
  RealType rho_k_;
  RealType rho_k_2_;
  RealType rho_k_plus_1_;

  void precompute_common() {
    rho_k_ = rho_ * kk_;
    rho_k_2_ = rho_k_ * kk_;
    rho_k_plus_1_ = rho_k_ + static_cast<RealType>(1);
    projection_const_ = static_cast<RealType>(1) / (kk_ + rho_k_2_);
  }
};

}

#endif // SDCA_TOPK_CONE_BIASED_PROJECTOR_HPP
