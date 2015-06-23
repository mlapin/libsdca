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
    const std::size_t n,
    RealType *first) const;

  void Project(
    RealType *first,
    RealType *last) const;

  void Project(
    const std::size_t n,
    RealType *first,
    std::vector<RealType> &aux) const;

  void Project(
    RealType *first,
    RealType *last,
    std::vector<RealType> &aux) const;

  void Project(
    const std::size_t num_row,
    const std::size_t num_col,
    RealType *x) const;

  void Clamp(
      RealType *first,
      RealType *last,
      const RealType t,
      const RealType lo = -std::numeric_limits<RealType>::infinity(),
      const RealType hi = +std::numeric_limits<RealType>::infinity()
    ) const;

  virtual void ComputeThresholds(
      std::vector<RealType> &x,
      RealType &t,
      RealType &lo,
      RealType &hi
    ) const = 0;

};

}

#endif // SDCA_PROJECTOR_HPP
