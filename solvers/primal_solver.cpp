#include <algorithm>

#include "primal_solver.hpp"
#include "math_util.hpp"

namespace sdca {

template <typename RealType, typename SolverHelperType>
PrimalSolver<RealType, SolverHelperType>::PrimalSolver(
    const SolverHelperType &solver_helper,
    const SizeType num_dimensions,
    const SizeType num_examples,
    const SizeType num_tasks,
    const RealType *features,
    const SizeType *labels,
    RealType *primal_variables,
    RealType *dual_variables
  ) :
    Solver<RealType>::Solver(num_examples, num_tasks,
      solver_helper.get_name() + "PrimalSolver"),
    solver_helper_(solver_helper),
    num_dimensions_(num_dimensions),
    features_(features),
    labels_(labels),
    primal_variables_(primal_variables),
    dual_variables_(dual_variables),
    norms_(num_examples),
    scores_(num_tasks),
    dual_old_(num_tasks),
    diff_tolerance_(static_cast<RealType>(num_tasks) *
      std::numeric_limits<RealType>::epsilon())
{}

template <typename RealType, typename SolverHelperType>
void PrimalSolver<RealType, SolverHelperType>
::BeginSolve() {
  Solver<RealType>::BeginSolve();
  std::fill_n(primal_variables_, num_dimensions_ * num_tasks_, 0);
  std::fill_n(dual_variables_, num_tasks_ * num_examples_, 0);
  const IndexType num_dim = static_cast<IndexType>(num_dimensions_);
  for (SizeType i = 0; i < num_examples_; ++i) {
    const RealType *x_i = features_ + num_dimensions_ * i;
    norms_[i] = sdca_blas_dot(num_dim, x_i, x_i);
  }
}


template <typename RealType, typename SolverHelperType>
void PrimalSolver<RealType, SolverHelperType>
::SolveExample(SizeType example) {

  // Let x_i = i'th feature vector
  if (norms_[example] <= static_cast<RealType>(0)) {
    return;
  }
  const RealType *x_i = features_ + num_dimensions_ * example;

  // Let scores = A * K_i = W' * x_i
  sdca_blas_gemv<RealType>(
    static_cast<IndexType>(num_dimensions_),
    static_cast<IndexType>(num_tasks_),
    primal_variables_, x_i, &scores_[0], CblasTrans);

  // Update dual variables
  RealType *variables = dual_variables_ + num_tasks_ * example;
  std::copy_n(variables, num_tasks_, &dual_old_[0]);
  solver_helper_.UpdateVariables(num_tasks_, labels_[example], norms_[example],
    variables, scores_);

  // Update primal variables
  sdca_blas_axpy<RealType>(static_cast<IndexType>(num_tasks_), -1, variables,
    &dual_old_[0]);
  const RealType diff = sdca_blas_asum(static_cast<IndexType>(num_tasks_),
    &dual_old_[0]);
  if (diff > diff_tolerance_) {
    sdca_blas_ger<RealType>(
      static_cast<IndexType>(num_dimensions_),
      static_cast<IndexType>(num_tasks_),
      -1, x_i, &dual_old_[0], primal_variables_);
  }
}

template <typename RealType, typename SolverHelperType>
void PrimalSolver<RealType, SolverHelperType>
::ComputePrimalDualObjectives() {

  RealType regularizer = 0;
  RealType primal_loss = 0;
  RealType dual_loss = 0;

  for (SizeType example = 0; example < num_examples_; ++example) {

    // Let x_i = i'th feature vector
    if (norms_[example] <= static_cast<RealType>(0)) {
      continue;
    }
    const RealType *x_i = features_ + num_dimensions_ * example;

    // Let scores = A * K_i = W' * x_i
    sdca_blas_gemv<RealType>(
      static_cast<IndexType>(num_dimensions_),
      static_cast<IndexType>(num_tasks_),
      primal_variables_, x_i, &scores_[0], CblasTrans);

    // Update losses and regularizer
    RealType *variables = dual_variables_ + num_tasks_ * example;
    solver_helper_.UpdateLosses(num_tasks_, labels_[example],
      variables, scores_, regularizer, primal_loss, dual_loss);
  }

  // Compute final objectives
  solver_helper_.ComputeObjectives(regularizer, primal_loss, dual_loss,
    this->primal_objective_, this->dual_objective_);
}

template class PrimalSolver<float,
  TopKLossL2RegularizerDualVariablesHelper<float>>;
template class PrimalSolver<double,
  TopKLossL2RegularizerDualVariablesHelper<double>>;

}
