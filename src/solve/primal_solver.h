#ifndef SDCA_SOLVE_PRIMAL_SOLVER_H
#define SDCA_SOLVE_PRIMAL_SOLVER_H

#include "util/util.h"
#include "solver.h"

namespace sdca {

template <typename Objective,
          typename Data,
          typename Result>
class primal_solver : public solver_base<Result> {
public:
  typedef solver_base<Result> base;
  typedef problem_data<Data> problem_type;
  typedef Objective objective_type;
  typedef Data data_type;
  typedef Result result_type;

  primal_solver(
      const problem_type& __problem,
      const stopping_criteria& __criteria,
      const objective_type& __objective
    ) :
      base::solver_base(__criteria, __problem.num_examples),
      objective_(__objective),
      num_dimensions_(__problem.num_dimensions),
      num_tasks_(__problem.num_tasks),
      labels_(__problem.labels),
      features_(__problem.data),
      primal_variables_(__problem.primal_variables),
      dual_variables_(__problem.dual_variables),
      norm_inv_(__problem.num_examples),
      scores_(__problem.num_tasks),
      vars_before_(__problem.num_tasks),
      diff_tolerance_(std::numeric_limits<data_type>::epsilon()),
      D(static_cast<blas_int>(__problem.num_dimensions)),
      N(static_cast<blas_int>(__problem.num_examples)),
      T(static_cast<blas_int>(__problem.num_tasks))
  {
    LOG_INFO << "solver: " << base::name() << " (primal)" << std::endl
      << "problem: " << __problem.to_string() << std::endl
      << "objective: " << __objective.to_string() << std::endl
      << "stopping criteria: " << __criteria.to_string() << std::endl;
    LOG_DEBUG << __objective.precision_string()  << std::endl;
  }

protected:
  // Protected members of the base class
  using base::num_examples_;
  using base::primal_;
  using base::dual_;
  using base::gap_;

  // Main variables
  const objective_type objective_;
  const size_type num_dimensions_;
  const size_type num_tasks_;
  const size_type* labels_;
  const data_type* features_;
  data_type* primal_variables_;
  data_type* dual_variables_;

  // Other
  std::vector<data_type> norm_inv_;
  std::vector<data_type> scores_;
  std::vector<data_type> vars_before_;
  const data_type diff_tolerance_;

  // BLAS (avoid static casts)
  const blas_int D;
  const blas_int N;
  const blas_int T;

  // Initialization
  void initialize() override {
    base::initialize();
    for (size_type i = 0; i < num_examples_; ++i) {
      const data_type* x_i = features_ + num_dimensions_ * i;
      data_type a = sdca_blas_dot(D, x_i, x_i);
      norm_inv_[i] = (a > 0) ? static_cast<data_type>(1) / a : 0;
    }
  }

  void solve_example(const size_type i) override {
    if (norm_inv_[i] <= 0) return;

    // Let x_i = i'th feature vector
    const data_type* x_i = features_ + num_dimensions_ * i;

    // Let scores = A * K_i = W' * x_i
    sdca_blas_gemv(D, T, primal_variables_, x_i, &scores_[0], CblasTrans);

    // Put ground truth at 0
    data_type* variables = dual_variables_ + num_tasks_ * i;
    std::copy_n(variables, num_tasks_, &vars_before_[0]);
    std::swap(variables[0], variables[labels_[i]]);
    std::swap(scores_[0], scores_[labels_[i]]);

    // Update dual variables
    objective_.update_variables(T, norm_inv_[i], variables, &scores_[0]);

    // Put back the ground truth variable
    std::swap(variables[0], variables[labels_[i]]);

    // Update primal variables
    sdca_blas_axpy(T, -1, variables, &vars_before_[0]);
    data_type diff = sdca_blas_asum(T, &vars_before_[0]);
    if (diff > diff_tolerance_) {
      sdca_blas_ger(D, T, -1, x_i, &vars_before_[0], primal_variables_);
    }
  }

  void compute_objectives() override {
    // Let W = X * A'
    // (recompute W from scratch to minimize the accumulated numerical error)
    sdca_blas_gemm(D, T, N, features_, D, dual_variables_, T,
      primal_variables_, CblasNoTrans, CblasTrans);

    // Compute the three sums independently over all training examples
    result_type regul_sum = 0;
    result_type p_loss_sum = 0;
    result_type d_loss_sum = 0;

    // Compensation variables for the Kahan summation
    result_type regul_comp = 0;
    result_type p_loss_comp = 0;
    result_type d_loss_comp = 0;

    for (size_type i = 0; i < num_examples_; ++i) {
      if (norm_inv_[i] <= 0) continue;

      // Let x_i = i'th feature vector
      const data_type* x_i = features_ + num_dimensions_ * i;

      // Let scores = A * K_i = W' * x_i
      sdca_blas_gemv(D, T, primal_variables_, x_i, &scores_[0], CblasTrans);

      // Put ground truth at 0
      data_type* variables = dual_variables_ + num_tasks_ * i;
      std::swap(variables[0], variables[labels_[i]]);
      std::swap(scores_[0], scores_[labels_[i]]);

      // Compute the regularization term and primal/dual losses
      result_type regul(0), p_loss(0), d_loss(0);
      objective_.regularized_loss(T, variables, &scores_[0],
        regul, p_loss, d_loss);

      // Increment the sums
      kahan_add(regul, regul_sum, regul_comp);
      kahan_add(p_loss, p_loss_sum, p_loss_comp);
      kahan_add(d_loss, d_loss_sum, d_loss_comp);

      // Put back the ground truth variable
      std::swap(variables[0], variables[labels_[i]]);
    }

    // Compute the overall primal/dual objectives and the duality gap
    objective_.primal_dual_gap(regul_sum, p_loss_sum, d_loss_sum,
      primal_, dual_, gap_);
  }

};

}

#endif
