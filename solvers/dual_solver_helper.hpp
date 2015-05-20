#ifndef SDCA_DUAL_SOLVER_HELPER_HPP
#define SDCA_DUAL_SOLVER_HELPER_HPP

#include "common.hpp"
#include "projectors.hpp"

namespace sdca {

template <typename RealType = double>
class TopKLossL2RegularizerDualSolverHelper {

public:
  TopKLossL2RegularizerDualSolverHelper(
      const SizeType k,
      const RealType lambda,
      const SizeType num_examples
    ) :
      k_(k),
      k_minus_1_(k-1),
      k_inverse_(static_cast<RealType>(1) / static_cast<RealType>(k)),
      lambda_(lambda),
      lambda_half_(static_cast<RealType>(0.5) * lambda),
      num_examples_inverse_(static_cast<RealType>(1) /
        static_cast<RealType>(num_examples)),
      projector_(k, 1, static_cast<RealType>(1) /
        (static_cast<RealType>(num_examples) * lambda))
  {}

  void UpdateVariables(
      const SizeType num_tasks,
      const SizeType label,
      const RealType norm_squared,
      RealType *variables,
      std::vector<RealType> &scores) const;

  void UpdateLosses(
      const SizeType num_tasks,
      const SizeType label,
      const RealType *variables,
      std::vector<RealType> &scores,
      RealType &regularizer,
      RealType &primal_loss,
      RealType &dual_loss
    ) const;

  void ComputeObjectives(
      const RealType regularizer,
      const RealType primal_loss,
      const RealType dual_loss,
      RealType &primal_objective,
      RealType &dual_objective
    ) const;

private:
  const SizeType k_;
  const SizeType k_minus_1_;
  const RealType k_inverse_;
  const RealType lambda_;
  const RealType lambda_half_;
  const RealType num_examples_inverse_;
  const TopKSimplexBiasedProjector<RealType> projector_;
};

}

#endif // SDCA_DUAL_SOLVER_HELPER_HPP
