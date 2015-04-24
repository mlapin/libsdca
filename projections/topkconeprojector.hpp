#ifndef SDCA_TOPKCONEPROJECTOR_HH
#define SDCA_TOPKCONEPROJECTOR_HH

#include <vector>

#include "topkprojector.hpp"

namespace sdca {

template <typename RealType = double>
class TopKConeProjector : public TopKProjector<RealType> {

public:
  TopKConeProjector(std::size_t k) : k_(k), kk_(k) {}

  void Project(RealType *x, const std::size_t m,
    const std::size_t n = 1);

protected:
  void ComputeThresholds(std::vector<RealType> x,
    RealType &t, RealType &hi);

  std::size_t k_;
  RealType kk_;
};

}

#endif // SDCA_TOPKCONEPROJECTOR_HH
