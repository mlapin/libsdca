#ifndef SDCA_PROJECTOR_HPP
#define SDCA_PROJECTOR_HPP

#include <limits>
#include <vector>

namespace sdca {

enum class Projection {
  Zero,
  Constant,
  General
};


template <typename RealType = double>
class Projector {

public:
  void Project(
      RealType *x,
      const std::size_t n
    );
  void Project(
      RealType *x,
      const std::size_t n,
      const std::size_t num_col = 1
    );

  void Clamp(
      RealType *first,
      RealType *last,
      const RealType t,
      const RealType lo = -std::numeric_limits<RealType>::infinity(),
      const RealType hi = +std::numeric_limits<RealType>::infinity()
    );

  virtual void ComputeThresholds(
      std::vector<RealType> &x,
      RealType &t,
      RealType &lo,
      RealType &hi
    ) = 0;

};

}

#endif // SDCA_PROJECTOR_HPP
