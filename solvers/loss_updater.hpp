#ifndef SDCA_LOSS_UPDATER_HPP
#define SDCA_LOSS_UPDATER_HPP

#include "loss_functions.hpp"
#include "regularizers.hpp"

namespace sdca {

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

/*
 *  Specializations:
 */

template <typename RealType>
class LossDualUpdater<RealType, TopKLoss, L2Regularizer<RealType>> {
public:
  const TopKLoss &Loss;
  const L2Regularizer<RealType> &Regularizer;
  const RealType KInverse;

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
      RealType &dual_loss) const;
};

}

#endif // SDCA_LOSS_UPDATER_HPP
