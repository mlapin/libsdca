#ifndef SDCA_REGULARIZED_LOSS_HPP
#define SDCA_REGULARIZED_LOSS_HPP

#include "common.hpp"

namespace sdca {

class TopKLoss {
public:
  const SizeType K;
  TopKLoss(SizeType k) : K(k) {}
};

template <typename RealType = double>
class L2Regularizer {
public:
  const RealType Lambda;
  const RealType LambdaInverse;
  L2Regularizer(RealType lambda) :
    Lambda(lambda),
    LambdaInverse(static_cast<RealType>(1)/lambda)
  {}
};


template <typename RealType, typename LossType, typename RegularizerType>
class VariableDualUpdater {
public:
  const TopKLoss &Loss;
  const L2Regularizer<RealType> &Regularizer;

  VariableDualUpdater(
      const TopKLoss &loss,
      const L2Regularizer<RealType> &regularizer,
      const SizeType num_examples
    );
  void operator()(
      const SizeType num_tasks,
      const SizeType label,
      const RealType norm_squared,
      RealType *__restrict__ variables,
      std::vector<RealType> &scores
    ) const;
};

template <typename RealType, typename LossType, typename RegularizerType>
class LossDualUpdater {
public:
  const LossType &Loss;
  const RegularizerType &Regularizer;

  LossDualUpdater(
      const TopKLoss &loss,
      const L2Regularizer<RealType> &regularizer
    );
  void operator()(
      const SizeType num_tasks,
      const SizeType label,
      const RealType *__restrict__ variables,
      std::vector<RealType> &scores,
      RealType &regularizer,
      RealType &primal_loss,
      RealType &dual_loss
    ) const;
};

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

}

#endif // SDCA_REGULARIZED_LOSS_HPP
