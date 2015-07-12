#ifndef SDCA_SOLVE_PRIMAL_SOLVER_H
#define SDCA_SOLVE_PRIMAL_SOLVER_H

#include "linalg/linalg.h"
#include "solver.h"

namespace sdca {

template <typename Objective, typename Data, typename Result = long double>
class primal_solver : public solver_base<Result> {
public:
  typedef solver_base<Result> base;
  typedef problem_data<data_type> problem_type;
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
      norms_(__problem.num_examples),
      scores_(__problem.num_tasks),
      vars_before_(__problem.num_tasks),
      diff_tolerance_(static_cast<data_type>(__problem.num_tasks) *
        2 * std::numeric_limits<data_type>::epsilon()),
      D(static_cast<blas_int>(__problem.num_dimensions)),
      N(static_cast<blas_int>(__problem.num_examples)),
      T(static_cast<blas_int>(__problem.num_tasks))
  {}

protected:
  // Protected members of the base class
  using base::num_examples_;
  using base::primal_;
  using base::dual_;
  using base::gap_;

  // Main variables
  objective_type objective_;
  const size_type num_dimensions_;
  const size_type num_tasks_;
  const size_type* labels_;
  const data_type* features_;
  data_type* primal_variables_;
  data_type* dual_variables_;

  // Other
  std::vector<data_type> norms_;
  std::vector<data_type> scores_;
  std::vector<data_type> vars_before_;
  const data_type diff_tolerance_;

  // BLAS
  const blas_int D;
  const blas_int N;
  const blas_int T;

  // Initialization
  void begin_solve() override {
    base::begin_solve();
    for (size_type i = 0; i < num_examples_; ++i) {
      const data_type* x_i = features_ + num_dimensions_ * i;
      norms_[i] = sdca_blas_dot(D, x_i, x_i);
    }
  }

  void solve_example(const size_type i) override {
    if (norms_[i] <= 0) return;

    // Let x_i = i'th feature vector
    const data_type* x_i = features_ + num_dimensions_ * i;

    // Let scores = A * K_i = W' * x_i
    sdca_blas_gemv(D, T, primal_variables_, x_i, &scores_[0], CblasTrans);

    // Update dual variables
    data_type* vars = dual_variables_ + num_tasks_ * i;
    std::copy_n(vars, num_tasks_, &vars_before_[0]);
    objective_.update_dual(T, labels_[i], norms_[i], vars, &scores_[0]);

    // Update primal variables
    sdca_blas_axpy(T, -1, vars, &vars_before_[0]);
    data_type diff = sdca_blas_asum(T, &vars_before_[0]);
    if (diff > diff_tolerance_) {
      sdca_blas_ger(D, T, -1, x_i, &vars_before_[0], primal_variables_);
    }
  }

  void compute_objectives() override {
    // Let W = X * A'
    sdca_blas_gemm(D, T, N, features_, D, dual_variables_, T,
      primal_variables_, CblasNoTrans, CblasTrans);

    // Compute the sums over all examples
    result_type regul_sum = 0;
    result_type p_loss_sum = 0;
    result_type d_loss_sum = 0;

    // Use Kahan summation
    result_type regul_comp = 0;
    result_type p_loss_comp = 0;
    result_type d_loss_comp = 0;

    for (size_type i = 0; i < num_examples_; ++i) {
      if (norms_[i] <= 0) continue;

      // Let x_i = i'th feature vector
      const data_type* x_i = features_ + num_dimensions_ * i;

      // Let scores = A * K_i = W' * x_i
      sdca_blas_gemv(D, T, primal_variables_, x_i, &scores_[0], CblasTrans);

      // Compute regularizer and losses
      data_type regul, p_loss, d_loss;
      data_type* vars = dual_variables_ + num_tasks_ * i;
      objective_.regularized_loss(T, labels_[i], vars, &scores_[0],
        regul, p_loss, d_loss);

      // Increment the sums
      kahan_sum(regul, regul_sum, regul_comp);
      kahan_sum(p_loss, p_loss_sum, p_loss_comp);
      kahan_sum(d_loss, d_loss_sum, d_loss_comp);
    }

    // Compute primal/dual objectives and the duality gap
    objective_.primal_dual_gap(regul_sum, p_loss_sum, d_loss_sum,
      primal_, dual_, gap_);
  }

};

template <typename Objective, typename Data, typename Result = long double>
inline
primal_solver<Objective, Data, Result>
make_primal_solver(
    const problem_data<Data>& problem,
    const Objective& objective,
    const stopping_criteria& criteria
  ) {
  return primal_solver<Objective, Data, Result>(
    problem, objective, criteria);
}

}

#endif
