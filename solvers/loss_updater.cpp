#include <algorithm>
#include <vector>

#include "math_util.hpp"
#include "loss_updater.hpp"

namespace sdca {

template <typename RealType>
LossDualUpdater<RealType, TopKLoss, L2Regularizer<RealType>>
::LossDualUpdater(
    const TopKLoss &loss,
    const L2Regularizer<RealType> &regularizer
  ) :
    Loss(loss),
    Regularizer(regularizer),
    KInverse(static_cast<RealType>(1) / static_cast<RealType>(loss.K))
{}

template <typename RealType>
void LossDualUpdater<RealType, TopKLoss, L2Regularizer<RealType>>
::operator()(
    const SizeType num_tasks,
    const SizeType label,
    const RealType *__restrict__ variables,
    std::vector<RealType> &scores,
    RealType &regularizer,
    RealType &primal_loss,
    RealType &dual_loss) const {

  regularizer += sdca_blas_dot(static_cast<IndexType>(num_tasks),
    &scores[0], variables);

  dual_loss += variables[label];

  RealType a = static_cast<RealType>(1) - scores[label];
  std::for_each(scores.begin(), scores.end(), [&](RealType &x){ x += a; });
  scores[label] = static_cast<RealType>(0);

  // Sum k largest elements
  std::nth_element(scores.begin(), scores.begin() + Loss.K - 1, scores.end(),
    std::greater<RealType>());
  a = std::accumulate(scores.begin(), scores.begin() + Loss.K,
    static_cast<RealType>(0));

  if (a > static_cast<RealType>(0)) {
    primal_loss += a * KInverse;
  }
}

template class LossDualUpdater<float, TopKLoss, L2Regularizer<float>>;
template class LossDualUpdater<double, TopKLoss, L2Regularizer<double>>;

}
