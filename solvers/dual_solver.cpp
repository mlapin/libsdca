#include <algorithm>

#include "dual_solver.hpp"
#include "math_util.hpp"

namespace sdca {

template <typename RealType, typename SolverHelperType>
DualSolver<RealType, SolverHelperType>::DualSolver(
    const SolverHelperType &solver_helper,
    const SizeType num_examples,
    const SizeType num_tasks,
    const RealType *gram_matrix,
    const SizeType *labels,
    RealType *dual_variables
  ) :
    Solver<RealType>::Solver(num_examples, num_tasks,
      solver_helper.get_name() + "DualSolver"),
    solver_helper_(solver_helper),
    gram_matrix_(gram_matrix),
    labels_(labels),
    dual_variables_(dual_variables),
    scores_(num_tasks)
{}

template <typename RealType, typename SolverHelperType>
void DualSolver<RealType, SolverHelperType>
::BeginSolve() {
  Solver<RealType>::BeginSolve();
  std::fill_n(dual_variables_, num_tasks_ * num_examples_, 0);
}

template <typename RealType, typename SolverHelperType>
void DualSolver<RealType, SolverHelperType>
::SolveExample(SizeType example) {

  // Let K_i = i'th column of the gram matrix
  const RealType *K_i = gram_matrix_ + num_examples_ * example;
  if (K_i[example] <= static_cast<RealType>(0)) {
    return;
  }

  // Let scores = A * K_i = W' * x_i
  sdca_blas_gemv<RealType>(
    static_cast<IndexType>(num_tasks_),
    static_cast<IndexType>(num_examples_),
    dual_variables_, K_i, &scores_[0]);

  // Update variables
  RealType *variables = dual_variables_ + num_tasks_ * example;
  solver_helper_.UpdateVariables(num_tasks_, labels_[example], K_i[example],
    variables, scores_);
}

template <typename RealType, typename SolverHelperType>
void DualSolver<RealType, SolverHelperType>
::ComputePrimalDualObjectives() {

  RealType regularizer = 0;
  RealType primal_loss = 0;
  RealType dual_loss = 0;

  for (SizeType example = 0; example < num_examples_; ++example) {

    // Let K_i = i'th column of the gram matrix
    const RealType *K_i = gram_matrix_ + num_examples_ * example;
    if (K_i[example] <= static_cast<RealType>(0)) {
      continue;
    }

    // Let scores = A * K_i = W' * x_i
    sdca_blas_gemv<RealType>(
      static_cast<IndexType>(num_tasks_),
      static_cast<IndexType>(num_examples_),
      dual_variables_, K_i, &scores_[0]);

    // Update losses and regularizer
    RealType *variables = dual_variables_ + num_tasks_ * example;
    solver_helper_.UpdateLosses(num_tasks_, labels_[example],
      variables, scores_, regularizer, primal_loss, dual_loss);
  }

  // Compute final objectives
  solver_helper_.ComputeObjectives(regularizer, primal_loss, dual_loss,
    this->primal_objective_, this->dual_objective_);
}

template class DualSolver<float,
  TopKLossL2RegularizerDualVariablesHelper<float>>;
template class DualSolver<double,
  TopKLossL2RegularizerDualVariablesHelper<double>>;

}
