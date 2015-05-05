#ifndef SDCA_TOPKCONEPROJECTOR_HPP
#define SDCA_TOPKCONEPROJECTOR_HPP

#include "projector.hpp"

namespace sdca {

enum class TopKConeCase {
  NoneUpperNoneMiddle,
  NoneUpperSomeMiddle,
  SomeUpperNoneMiddle,
  SomeUpperSomeMiddle
};

template <typename RealType = double>
class TopKConeProjector : public Projector<RealType> {

public:
  TopKConeProjector(
      const std::size_t k
    ) :
      k_(k),
      kk_(k)
    {}

  void ComputeThresholds(
      std::vector<RealType> x,
      RealType &t,
      RealType &lo,
      RealType &hi
    );

  TopKConeCase CheckSpecialCases(
      std::vector<RealType> x,
      RealType &t,
      RealType &lo,
      RealType &hi,
      RealType &sum_k_largest,
      RealType &sum_positive
    );

  void FallBackCase(
      std::vector<RealType> x,
      RealType &t,
      RealType &lo,
      RealType &hi
    );

  std::size_t k() const { return k_; }
  void k(const std::size_t k) { k_ = k; kk_ = k; }

  RealType kk() const { return kk_; }

private:
  std::size_t k_;
  RealType kk_;
};

}

#endif // SDCA_TOPKCONEPROJECTOR_HPP
