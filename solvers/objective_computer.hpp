#ifndef SDCA_OBJECTIVE_COMPUTER_HPP
#define SDCA_OBJECTIVE_COMPUTER_HPP

#include "loss_functions.hpp"
#include "regularizers.hpp"

namespace sdca {

template <typename RealType, typename LossType, typename RegularizerType>
class ObjectiveDualComputer {
public:
  const LossType &Loss;
  const RegularizerType &Regularizer;

  ObjectiveDualComputer(
      const TopKLoss &loss,
      const L2Regularizer<RealType> &regularizer,
      const SizeType num_examples
    );

  void operator()(
      const RealType regularizer,
      const RealType primal_loss,
      const RealType dual_loss,
      RealType &primal_objective,
      RealType &dual_objective
    ) const;
};

/*
 *  Specializations:
 */

template <typename RealType>
class ObjectiveDualComputer<RealType, TopKLoss, L2Regularizer<RealType>> {
public:
  const TopKLoss &Loss;
  const L2Regularizer<RealType> &Regularizer;
  const RealType NInverse;
  const RealType LambdaHalf;

  ObjectiveDualComputer(
      const TopKLoss &loss,
      const L2Regularizer<RealType> &regularizer,
      const SizeType num_examples
    );

  void operator()(
      const RealType regularizer,
      const RealType primal_loss,
      const RealType dual_loss,
      RealType &primal_objective,
      RealType &dual_objective) const;
};

}

#endif // SDCA_OBJECTIVE_COMPUTER_HPP
