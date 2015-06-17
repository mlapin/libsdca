#include <algorithm>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <vector>

#include "math_util.hpp"
#include "dual_variables_helper.hpp"

namespace sdca {

template <typename RealType>
void TopKLossL2RegularizerDualVariablesHelper<RealType>::
UpdateVariables(
      const SizeType num_tasks,
      const SizeType label,
      const RealType norm_squared,
      RealType *variables,
      std::vector<RealType> &scores) const {

  RealType a = static_cast<RealType>(1) / norm_squared;
  sdca_blas_axpby(static_cast<IndexType>(num_tasks), a, &scores[0],
    static_cast<RealType>(-1), variables);

  // Place ground truth at the back
  RealType *variables_back = variables + num_tasks - 1;
  std::swap(*variables_back, variables[label]);
  a -= *variables_back;
  std::swap(scores.back(), scores[label]);
  scores.pop_back();

  // Project all but last one
  std::for_each(variables, variables_back, [&](RealType &x){ x += a; });
  projector_.Project(variables, variables_back, scores);

  // The last one is the sum
  *variables_back = std::accumulate(
    variables, variables_back, static_cast<RealType>(0));

  // Change sign of all but last one
  std::for_each(variables, variables_back, [](RealType &x){ x = -x; });

  // Revert ground truth index
  std::swap(*variables_back, variables[label]);
  scores.push_back(RealType());
}

template <typename RealType>
void TopKLossL2RegularizerDualVariablesHelper<RealType>::
UpdateLosses(
      const SizeType num_tasks,
      const SizeType label,
      const RealType *variables,
      std::vector<RealType> &scores,
      RealType &regularizer,
      RealType &primal_loss,
      RealType &dual_loss
    ) const {

  regularizer += sdca_blas_dot(static_cast<IndexType>(num_tasks),
    variables, &scores[0]);

  dual_loss += variables[label];

  RealType a = static_cast<RealType>(1) - scores[label];
  std::for_each(scores.begin(), scores.end(), [&](RealType &x){ x += a; });
  scores[label] = static_cast<RealType>(0);

  // Sum k largest elements
  auto k_end = scores.begin()
    + static_cast<typename std::vector<RealType>::difference_type>(k_);
  std::nth_element(scores.begin(), k_end - 1, scores.end(),
    std::greater<RealType>());
  a = std::accumulate(scores.begin(), k_end, static_cast<RealType>(0));

  // max{0, sum_k_largest} (division by k happens later)
  if (a > static_cast<RealType>(0)) {
    primal_loss += a;
  }
}

template <typename RealType>
void TopKLossL2RegularizerDualVariablesHelper<RealType>::
ComputeObjectives(
      const RealType regularizer,
      const RealType primal_loss,
      const RealType dual_loss,
      RealType &primal_objective,
      RealType &dual_objective
    ) const {
  primal_objective = num_examples_k_inverse_ * primal_loss
    + lambda_half_ * regularizer;
  dual_objective = lambda_ * dual_loss
    - lambda_half_ * regularizer;
}

template class TopKLossL2RegularizerDualVariablesHelper<float>;
template class TopKLossL2RegularizerDualVariablesHelper<double>;



template <typename RealType>
void SmoothTopKLossL2RegularizerDualVariablesHelper<RealType>::
UpdateVariables(
      const SizeType num_tasks,
      const SizeType label,
      const RealType norm_squared,
      RealType *variables,
      std::vector<RealType> &scores) {

  RealType a = static_cast<RealType>(1) / (norm_squared + gamma_n_lambda_);
  RealType rho = a * norm_squared;
  sdca_blas_axpby(static_cast<IndexType>(num_tasks), a, &scores[0],
    -rho, variables);

  // Place ground truth at the back
  RealType *variables_back = variables + num_tasks - 1;
  std::swap(*variables_back, variables[label]);
  a -= *variables_back;
  std::swap(scores.back(), scores[label]);
  scores.pop_back();

  // Project all but last one
  std::for_each(variables, variables_back, [&](RealType &x){ x += a; });
  biased_projector_.set_rho(rho);
  biased_projector_.Project(variables, variables_back, scores);

  // The last one is the sum
  *variables_back = std::accumulate(
    variables, variables_back, static_cast<RealType>(0));

  // Change sign of all but last one
  std::for_each(variables, variables_back, [](RealType &x){ x = -x; });

  // Revert ground truth index
  std::swap(*variables_back, variables[label]);
  scores.push_back(RealType());
}

template <typename RealType>
void SmoothTopKLossL2RegularizerDualVariablesHelper<RealType>::
UpdateLosses(
      const SizeType num_tasks,
      const SizeType label,
      const RealType *variables,
      std::vector<RealType> &scores,
      RealType &regularizer,
      RealType &primal_loss,
      RealType &dual_loss
    ) {

  regularizer += sdca_blas_dot(static_cast<IndexType>(num_tasks),
    variables, &scores[0]);

  RealType a = variables[label];
  dual_loss += a + dual_loss_coeff_ * (
    sdca_blas_dot(static_cast<IndexType>(num_tasks), variables, variables)
    - a * a);

  a = static_cast<RealType>(1) - scores[label];
  std::for_each(scores.begin(), scores.end(), [&](RealType &x){ x += a; });
  scores[label] = static_cast<RealType>(0);

  // Project scores
  scores_proj_ = scores;
  projector_.Project(num_tasks, &scores_proj_[0], scratch_);

  // Compute primal loss
  a = sdca_blas_dot(static_cast<IndexType>(num_tasks),
    &scores_proj_[0], &scores_proj_[0]);
  primal_loss += sdca_blas_dot(static_cast<IndexType>(num_tasks),
    &scores_proj_[0], &scores[0]) + minus_half_ * a;
}

template <typename RealType>
void SmoothTopKLossL2RegularizerDualVariablesHelper<RealType>::
ComputeObjectives(
      const RealType regularizer,
      const RealType primal_loss,
      const RealType dual_loss,
      RealType &primal_objective,
      RealType &dual_objective
    ) const {
  primal_objective = n_gamma_inverse_ * primal_loss
    + lambda_half_ * regularizer;
  dual_objective = lambda_ * dual_loss
    - lambda_half_ * regularizer;
}

template class SmoothTopKLossL2RegularizerDualVariablesHelper<float>;
template class SmoothTopKLossL2RegularizerDualVariablesHelper<double>;



template <typename RealType>
void HingeTopKLossL2RegularizerDualVariablesHelper<RealType>::
UpdateVariables(
      const SizeType num_tasks,
      const SizeType label,
      const RealType norm_squared,
      RealType *variables,
      std::vector<RealType> &scores) const {

  RealType a = static_cast<RealType>(1) / norm_squared;
  sdca_blas_axpby(static_cast<IndexType>(num_tasks), a, &scores[0],
    static_cast<RealType>(-1), variables);

  // Place ground truth at the back
  RealType *variables_back = variables + num_tasks - 1;
  std::swap(*variables_back, variables[label]);
  a -= *variables_back;
  std::swap(scores.back(), scores[label]);
  scores.pop_back();

  // Project all but last one
  std::for_each(variables, variables_back, [&](RealType &x){ x += a; });
  projector_.Project(variables, variables_back, scores);

  // The last one is the sum
  *variables_back = std::accumulate(
    variables, variables_back, static_cast<RealType>(0));

  // Change sign of all but last one
  std::for_each(variables, variables_back, [](RealType &x){ x = -x; });

  // Revert ground truth index
  std::swap(*variables_back, variables[label]);
  scores.push_back(RealType());
}

template <typename RealType>
void HingeTopKLossL2RegularizerDualVariablesHelper<RealType>::
UpdateLosses(
      const SizeType num_tasks,
      const SizeType label,
      const RealType *variables,
      std::vector<RealType> &scores,
      RealType &regularizer,
      RealType &primal_loss,
      RealType &dual_loss
    ) const {

  regularizer += sdca_blas_dot(static_cast<IndexType>(num_tasks),
    variables, &scores[0]);

  dual_loss += variables[label];

  RealType a = static_cast<RealType>(1) - scores[label];
  std::for_each(scores.begin(), scores.end(), [&](RealType &x){ x += a; });
  scores[label] = static_cast<RealType>(0);

  // Sum k largest positive elements
  auto k_end = scores.begin()
    + static_cast<typename std::vector<RealType>::difference_type>(k_);
  std::nth_element(scores.begin(), k_end - 1, scores.end(),
    std::greater<RealType>());
  for (auto it = scores.begin(); it != k_end; ++it) {
    if (*it > static_cast<RealType>(0)) {
      primal_loss += *it;
    }
  }
}

template <typename RealType>
void HingeTopKLossL2RegularizerDualVariablesHelper<RealType>::
ComputeObjectives(
      const RealType regularizer,
      const RealType primal_loss,
      const RealType dual_loss,
      RealType &primal_objective,
      RealType &dual_objective
    ) const {
  primal_objective = num_examples_k_inverse_ * primal_loss
    + lambda_half_ * regularizer;
  dual_objective = lambda_ * dual_loss
    - lambda_half_ * regularizer;
}

template class HingeTopKLossL2RegularizerDualVariablesHelper<float>;
template class HingeTopKLossL2RegularizerDualVariablesHelper<double>;

}
