#ifndef SDCA_TOPKCONEPROJECTOR_HPP
#define SDCA_TOPKCONEPROJECTOR_HPP

#include <vector>

#include "projector.hpp"

namespace sdca {

template <typename RealType = double>
class TopKConeProjector : public Projector<RealType> {

public:
  TopKConeProjector(
      std::size_t k) :
    k_(k),
    kk_(k) {}

  void ComputeThresholds(
      std::vector<RealType> x,
      RealType &t,
      RealType &lo,
      RealType &hi);

private:
  std::size_t k_;
  RealType kk_;
};

}

#endif // SDCA_TOPKCONEPROJECTOR_HPP
