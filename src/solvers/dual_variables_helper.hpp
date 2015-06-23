#ifndef SDCA_DUAL_SOLVER_HELPER_HPP
#define SDCA_DUAL_SOLVER_HELPER_HPP

#include <string>

#include "common.hpp"
#include "projectors.hpp"

namespace sdca {

template <typename RealType = double>
class TopKLossL2RegularizerDualVariablesHelper {

public:
  TopKLossL2RegularizerDualVariablesHelper(
      const SizeType num_examples,
      const SizeType k,
      const RealType lambda
    ) :
      k_(k),
      k_minus_1_(k-1),
      lambda_(lambda),
      lambda_half_(static_cast<RealType>(0.5) * lambda),
      num_examples_k_inverse_(static_cast<RealType>(1) /
        (static_cast<RealType>(num_examples) * static_cast<RealType>(k))),
      projector_(k, 1, static_cast<RealType>(1) /
        (static_cast<RealType>(num_examples) * lambda))
  {}

  std::string get_name() const { return "TopKLossL2Regularizer"; }

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
  const RealType lambda_;
  const RealType lambda_half_;
  const RealType num_examples_k_inverse_;
  const TopKSimplexBiasedProjector<RealType> projector_;
};



template <typename RealType = double>
class SmoothTopKLossL2RegularizerDualVariablesHelper {

public:
  SmoothTopKLossL2RegularizerDualVariablesHelper(
      const SizeType num_examples,
      const SizeType num_tasks,
      const SizeType k,
      const RealType lambda,
      const RealType gamma
    ) :
      k_(k),
      lambda_(lambda),
      gamma_(gamma),
      gamma_n_lambda_(gamma * static_cast<RealType>(num_examples) * lambda),
      minus_half_(static_cast<RealType>(-0.5)),
      dual_loss_coeff_(minus_half_ * gamma_n_lambda_),
      lambda_half_(static_cast<RealType>(0.5) * lambda),
      n_gamma_inverse_(static_cast<RealType>(1) /
        (static_cast<RealType>(num_examples) * gamma)),
      projector_(k, gamma),
      biased_projector_(k, 1, static_cast<RealType>(1) /
        (static_cast<RealType>(num_examples) * lambda)),
      scores_proj_(num_tasks),
      scratch_(num_tasks)
  {}

  std::string get_name() const { return "SmoothTopKLossL2Regularizer"; }

  void UpdateVariables(
      const SizeType num_tasks,
      const SizeType label,
      const RealType norm_squared,
      RealType *variables,
      std::vector<RealType> &scores);

  void UpdateLosses(
      const SizeType num_tasks,
      const SizeType label,
      const RealType *variables,
      std::vector<RealType> &scores,
      RealType &regularizer,
      RealType &primal_loss,
      RealType &dual_loss
    );

  void ComputeObjectives(
      const RealType regularizer,
      const RealType primal_loss,
      const RealType dual_loss,
      RealType &primal_objective,
      RealType &dual_objective
    ) const;

private:
  const SizeType k_;
  const RealType lambda_;
  const RealType gamma_;
  const RealType gamma_n_lambda_;
  const RealType minus_half_;
  const RealType dual_loss_coeff_;
  const RealType lambda_half_;
  const RealType n_gamma_inverse_;
  const TopKSimplexProjector<RealType> projector_;
  TopKSimplexBiasedProjector<RealType> biased_projector_;
  std::vector<RealType> scores_proj_;
  std::vector<RealType> scratch_;
};


template <typename RealType = double>
class HingeTopKLossL2RegularizerDualVariablesHelper {

public:
  HingeTopKLossL2RegularizerDualVariablesHelper(
      const SizeType num_examples,
      const SizeType k,
      const RealType lambda
    ) :
      k_(k),
      k_minus_1_(k-1),
      lambda_(lambda),
      lambda_half_(static_cast<RealType>(0.5) * lambda),
      svm_c_(static_cast<RealType>(1) /
        (static_cast<RealType>(num_examples) * lambda)),
      num_examples_k_inverse_(static_cast<RealType>(1) /
        (static_cast<RealType>(num_examples) * static_cast<RealType>(k))),
      projector_(0, svm_c_ / static_cast<RealType>(k), svm_c_, 1)
  {}

  std::string get_name() const { return "HingeTopKLossL2Regularizer"; }

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
  const RealType lambda_;
  const RealType lambda_half_;
  const RealType svm_c_;
  const RealType num_examples_k_inverse_;
  const KnapsackLEBiasedProjector<RealType> projector_;
};

}

#endif // SDCA_DUAL_SOLVER_HELPER_HPP
