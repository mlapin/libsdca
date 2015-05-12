#ifndef SDCA_TOPKCONEPROJECTOR_HPP
#define SDCA_TOPKCONEPROJECTOR_HPP

#include "projector.hpp"

namespace sdca {

template <typename RealType = double>
class TopKConeProjector : public Projector<RealType> {

public:
  TopKConeProjector(
      const std::size_t k = 1
    ) :
      k_(k),
      kk_(static_cast<RealType>(k)),
      projection_const_(static_cast<RealType>(1) / kk_)
  {}

  void ComputeThresholds(
      std::vector<RealType> &x,
      RealType &t,
      RealType &lo,
      RealType &hi
    ) override;

  Projection CheckSpecialCases(
      std::vector<RealType> &x,
      RealType &t,
      RealType &lo,
      RealType &hi
    );

  virtual void ComputeGeneralCase(
      std::vector<RealType> &x,
      RealType &t,
      RealType &lo,
      RealType &hi
    );

  std::size_t get_k() const { return k_; }
  virtual void set_k(const std::size_t k) {
    k_ = k;
    kk_ = static_cast<RealType>(k);
    projection_const_ = static_cast<RealType>(1) / kk_;
  }

  RealType get_k_real() const { return kk_; }

protected:
  std::size_t k_;
  RealType kk_;
  RealType projection_const_;

};

}

#endif // SDCA_TOPKCONEPROJECTOR_HPP
