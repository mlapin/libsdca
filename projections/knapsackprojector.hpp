#ifndef SDCA_KNAPSACKPROJECTOR_HPP
#define SDCA_KNAPSACKPROJECTOR_HPP

#include "projector.hpp"

namespace sdca {

template <typename RealType = double>
class KnapsackProjector : public Projector<RealType> {

public:
  KnapsackProjector(RealType lo = 0, RealType hi = 1, RealType rhs = 1) :
    lo_(lo), hi_(hi), rhs_(rhs) {}

  void ComputeThresholds(std::vector<RealType> x,
    RealType &t, RealType &lo, RealType &hi);

private:
  RealType lo_;
  RealType hi_;
  RealType rhs_;

};

}

#endif // SDCA_KNAPSACKPROJECTOR_HPP
