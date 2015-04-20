#ifndef PROJECTOR_HH
#define PROJECTOR_HH

#include <cstddef>
#include <vector>

namespace sdca {

template <typename RealType = double>
class Projector {
public:
  const RealType kObjectiveChangeEpsilon = 1e-9;
  const std::size_t kMaxNumIterations = 10000;

  Projector();

  void VectorToKSimplex(const std::size_t k, const std::size_t n, RealType *x);
  void MatrixToKSimplex(const std::size_t k, const std::size_t n,
    const std::size_t m, RealType *x);

  const RealType ObjectiveValue() { return obj_val_; }
  const RealType ObjectiveValueOld()  { return obj_old_; }
  const std::size_t Iteration() { return iter_; }

private:
  RealType obj_val_;
  RealType obj_old_;
  std::size_t iter_;

};

}

#endif // PROJECTOR_HH
