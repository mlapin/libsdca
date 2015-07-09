#ifndef SDCA_SOLVE_PRIMAL_SOLVER_H
#define SDCA_SOLVE_PRIMAL_SOLVER_H

#include "solver.h"

namespace sdca {

template <typename Computer, typename Data, typename Result = long double>
class primal_solver : public solver_base<Result> {
public:
  typedef solver_base<Result> base;
  typedef typename base::index_type index_type;
  typedef typename base::result_type result_type;
  typedef Computer computer_type;
  typedef Data data_type;

  primal_solver(
      const sdca::stopping_criteria& __criteria,
      const computer_type& __computer,
      const std::size_t __num_dimensions,
      const std::size_t __num_examples,
      const std::size_t __num_tasks,
      const std::size_t* __labels,
      const data_type* __features,
      data_type* __primal_variables,
      data_type* __dual_variables,
      const bool __is_warm_restart = false
    ) :
      base::solver_base(__criteria, __num_examples),
      computer_(__computer),
      num_dimensions_(__num_dimensions),
      num_tasks_(__num_tasks),
      labels_(__labels),
      features_(__features),
      primal_variables_(__primal_variables),
      dual_variables_(__dual_variables),
      is_warm_restart_(__is_warm_restart),
      norms_(__num_examples),
      scores_(__num_tasks),
      vars_before_(__num_tasks),
      diff_tolerance_(static_cast<data_type>(__num_tasks) *
        2 * std::numeric_limits<data_type>::epsilon())
  {}

  bool is_warm_restart() const { return is_warm_restart_; }
  void is_warm_restart(const bool value) { is_warm_restart_ = value; }

protected:
  // Base class
  using base::num_examples_;
  using base::primal_;
  using base::dual_;
  using base::gap_;

  // Main variables
  computer_type computer_;
  const index_type num_dimensions_;
  const index_type num_tasks_;
  const std::size_t* labels_;
  const data_type* features_;
  data_type* primal_variables_;
  data_type* dual_variables_;

  // Other
  bool is_warm_restart_;
  std::vector<data_type> norms_;
  std::vector<data_type> scores_;
  std::vector<data_type> vars_before_;
  const data_type diff_tolerance_;

  // Initialization
  void begin_solve() override {
    base::begin_solve();
    if (is_warm_restart_) {
      // Let W = X * A'
      sdca_blas_gemm(num_dimensions_, num_tasks_, num_examples_,
        features_, num_dimensions_, dual_variables_, num_tasks_,
        primal_variables_, CblasNoTrans, CblasTrans);
    } else {
      std::fill_n(primal_variables_, num_dimensions_ * num_tasks_, 0);
      std::fill_n(dual_variables_, num_tasks_ * num_examples_, 0);
    }
    for (index_type i = 0; i < num_examples_; ++i) {
      const data_type* x_i = features_ + num_dimensions_ * i;
      norms_[i] = sdca_blas_dot(num_dimensions_, x_i, x_i);
    }
  }

  void solve_example(const index_type i) override {
    if (norms_[i] <= 0) return;

    // Let x_i = i'th feature vector
    const data_type* x_i = features_ + num_dimensions_ * i;

    // Let scores = A * K_i = W' * x_i
    sdca_blas_gemv(num_dimensions_, num_tasks_,
      primal_variables_, x_i, &scores_[0], CblasTrans);

    // Update dual variables
    data_type* vars = dual_variables_ + num_tasks_ * i;
    std::copy_n(vars, num_tasks_, &vars_before_[0]);
    computer_.update(num_tasks_, labels_[i], norms_[i], vars, &scores_[0]);

    // Update primal variables
    sdca_blas_axpy(num_tasks_, -1, vars, &vars_before_[0]);
    data_type diff = sdca_blas_asum(num_tasks_, &vars_before_[0]);
    if (diff > diff_tolerance_) {
      sdca_blas_ger(num_dimensions_, num_tasks_,
        -1, x_i, &vars_before_[0], primal_variables_);
    }
  }

  void compute_objectives() override {
    // Let W = X * A'
    sdca_blas_gemm(num_dimensions_, num_tasks_, num_examples_,
      features_, num_dimensions_, dual_variables_, num_tasks_,
      primal_variables_, CblasNoTrans, CblasTrans);

    // Compute the sums over all examples
    result_type regul_sum = 0;
    result_type p_loss_sum = 0;
    result_type d_loss_sum = 0;

    // Use Kahan summation
    result_type regul_comp = 0;
    result_type p_loss_comp = 0;
    result_type d_loss_comp = 0;

    for (index_type i = 0; i < num_examples_; ++i) {
      if (norms_[i] <= 0) continue;

      // Let x_i = i'th feature vector
      const data_type* x_i = features_ + num_dimensions_ * i;

      // Let scores = A * K_i = W' * x_i
      sdca_blas_gemv(num_dimensions_, num_tasks_,
        primal_variables_, x_i, &scores_[0], CblasTrans);

      // Compute regularizer and losses
      data_type regul, p_loss, d_loss;
      data_type* vars = dual_variables_ + num_tasks_ * i;
      computer_.loss(num_tasks_, labels_[i], vars, &scores_[0],
        regul, p_loss, d_loss);

      // Increment the sums
      kahan_sum(regul, regul_sum, regul_comp);
      kahan_sum(p_loss, p_loss_sum, p_loss_comp);
      kahan_sum(d_loss, d_loss_sum, d_loss_comp);
    }

    // Compute final objectives
    computer_.objective(regul_sum, p_loss_sum, d_loss_sum,
      primal_, dual_, gap_);
  }

};

template <typename Computer, typename Data, typename Result = long double>
inline
primal_solver<Computer, Data, Result>
make_primal_solver(
    const sdca::stopping_criteria& criteria,
      const Computer& computer,
      const std::size_t num_dimensions,
      const std::size_t num_examples,
      const std::size_t num_tasks,
      const std::size_t* labels,
      const Data* features,
      Data* primal_variables,
      Data* dual_variables,
      const bool is_warm_restart = false
  ) {
  return primal_solver<Computer, Data, Result>(
    criteria, computer, num_dimensions, num_examples, num_tasks,
    labels, features, primal_variables, dual_variables, is_warm_restart);
}

}

#endif
