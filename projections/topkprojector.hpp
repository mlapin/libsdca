#ifndef SDCA_TOPKPROJECTOR_HPP
#define SDCA_TOPKPROJECTOR_HPP

#include <limits>
#include <vector>

namespace sdca {

template <typename RealType = double>
class TopKProjector {

public:
  virtual void Project(RealType *x, const std::size_t m,
    const std::size_t n = 1) = 0;

  void Clamp(RealType *x, const std::size_t m,
    const RealType t,
    const RealType hi = std::numeric_limits<RealType>::infinity(),
    const RealType lo = static_cast<RealType>(0));

};

}

#endif // SDCA_TOPKPROJECTOR_HPP
