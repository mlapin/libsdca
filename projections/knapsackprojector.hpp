#ifndef SDCA_KNAPSACKPROJECTOR_HPP
#define SDCA_KNAPSACKPROJECTOR_HPP

#include "projector.hpp"

namespace sdca {

template <typename RealType = double>
class KnapsackProjector : public Projector<RealType> {

public:
  KnapsackProjector(
      const RealType lo = 0,
      const RealType hi = 1,
      const RealType rhs = 1
    ) :
      lo_(lo),
      hi_(hi),
      rhs_(rhs)
    {}

  void ComputeThresholds(
      std::vector<RealType> x,
      RealType &t,
      RealType &lo,
      RealType &hi
    );

  void ComputeThresholdsAndMidBoundary(
      std::vector<RealType> x,
      RealType &t,
      RealType &lo,
      RealType &hi,
      typename std::vector<RealType>::const_iterator &first,
      typename std::vector<RealType>::const_iterator &last
    );

  RealType lo() const { return lo_; }
  void lo(const RealType lo) { lo_ = lo; }

  RealType hi() const { return hi_; }
  void hi(const RealType hi) { hi_ = hi; }

  RealType rhs() const { return rhs_; }
  void rhs(const RealType rhs) { rhs_ = rhs; }

private:
  RealType lo_;
  RealType hi_;
  RealType rhs_;

};

}

#endif // SDCA_KNAPSACKPROJECTOR_HPP
