#include <algorithm>
#include <vector>

#include "math_util.hpp"
#include "objective_computer.hpp"

namespace sdca {

template <typename RealType>
ObjectiveDualComputer<RealType, TopKLoss, L2Regularizer<RealType>>
::ObjectiveDualComputer(
    const TopKLoss &loss,
    const L2Regularizer<RealType> &regularizer,
    const SizeType num_examples
  ) :
    Loss(loss),
    Regularizer(regularizer),
    NInverse(static_cast<RealType>(1) / static_cast<RealType>(num_examples)),
    LambdaHalf(static_cast<RealType>(0.5) * regularizer.Lambda)
{}

template <typename RealType>
void ObjectiveDualComputer<RealType, TopKLoss, L2Regularizer<RealType>>
::operator()(
    const RealType regularizer,
    const RealType primal_loss,
    const RealType dual_loss,
    RealType &primal_objective,
    RealType &dual_objective) const {
  primal_objective = LambdaHalf * regularizer;
  dual_objective = Regularizer.Lambda * dual_loss - primal_objective;
  primal_objective += NInverse * primal_loss;
}

template class ObjectiveDualComputer<float, TopKLoss, L2Regularizer<float>>;
template class ObjectiveDualComputer<double, TopKLoss, L2Regularizer<double>>;

}
