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
      std::vector<RealType> &x,
      RealType &t,
      RealType &get_lo,
      RealType &get_hi
    ) override;

  void PartitionAndComputeThresholds(
      std::vector<RealType> &x,
      RealType &t,
      RealType &get_lo,
      RealType &get_hi,
      typename std::vector<RealType>::iterator &first,
      typename std::vector<RealType>::iterator &last
    );

  RealType get_lo() const { return lo_; }
  void set_lo(const RealType lo) { lo_ = lo; }

  RealType get_hi() const { return hi_; }
  void set_hi(const RealType hi) { hi_ = hi; }

  RealType get_rhs() const { return rhs_; }
  void set_rhs(const RealType rhs) { rhs_ = rhs; }

private:
  RealType lo_;
  RealType hi_;
  RealType rhs_;

};

}

#endif // SDCA_KNAPSACKPROJECTOR_HPP
