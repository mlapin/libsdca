#include <algorithm>
#include <vector>

#include "math_util.hpp"
#include "dual_solver_helper.hpp"

namespace sdca {

template <typename RealType>
void TopKLossL2RegularizerDualSolverHelper<RealType>::
UpdateVariables(
      const SizeType num_tasks,
      const SizeType label,
      const RealType norm_squared,
      RealType *variables,
      std::vector<RealType> &scores) const {

  RealType a = lambda_ / norm_squared;
  sdca_blas_axpby(static_cast<IndexType>(num_tasks), a, &scores[0],
    -lambda_, variables);

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
void TopKLossL2RegularizerDualSolverHelper<RealType>::
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
    &scores[0], variables);

  dual_loss += variables[label];

  RealType a = static_cast<RealType>(1) - scores[label];
  std::for_each(scores.begin(), scores.end(), [&](RealType &x){ x += a; });
  scores[label] = static_cast<RealType>(0);

  // Sum k largest elements
  std::nth_element(scores.begin(), scores.begin() + k_minus_1_, scores.end(),
    std::greater<RealType>());
  a = std::accumulate(scores.begin(), scores.begin() + k_,
    static_cast<RealType>(0));

  if (a > static_cast<RealType>(0)) {
    primal_loss += a * k_inverse_;
  }
}

template <typename RealType>
void TopKLossL2RegularizerDualSolverHelper<RealType>::
ComputeObjectives(
      const RealType regularizer,
      const RealType primal_loss,
      const RealType dual_loss,
      RealType &primal_objective,
      RealType &dual_objective
    ) const {
  primal_objective = lambda_half_ * regularizer;
  dual_objective = lambda_ * dual_loss - primal_objective;
  primal_objective += num_examples_inverse_ * primal_loss;
}

template class TopKLossL2RegularizerDualSolverHelper<float>;
template class TopKLossL2RegularizerDualSolverHelper<double>;

}
