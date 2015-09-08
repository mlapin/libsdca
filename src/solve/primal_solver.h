#ifndef SDCA_SOLVE_PRIMAL_SOLVER_H
#define SDCA_SOLVE_PRIMAL_SOLVER_H

#include "util/util.h"
#include "solver.h"

namespace sdca {

template <typename Objective,
          typename Data,
          typename Result>
class primal_solver : public multiset_solver<Data, Result> {
public:
  typedef multiset_solver<Data, Result> base;
  typedef typename base::context_type context_type;
  typedef typename base::dataset_type dataset_type;
  typedef typename base::evaluation_type evaluation_type;
  typedef Objective objective_type;
  typedef Data data_type;
  typedef Result result_type;

  primal_solver(
      const objective_type& __obj,
      const context_type& __ctx
    ) :
      base::multiset_solver(__ctx),
      objective_(__obj),
      num_dimensions_(__ctx.datasets[0].num_dimensions),
      num_tasks_(__ctx.datasets[0].num_tasks),
      labels_(&(__ctx.datasets[0].labels[0])),
      features_(__ctx.datasets[0].data),
      primal_variables_(__ctx.primal_variables),
      dual_variables_(__ctx.dual_variables),
      norm2_(__ctx.datasets[0].num_examples),
      scores_(__ctx.datasets[0].num_tasks),
      vars_before_(__ctx.datasets[0].num_tasks),
      diff_tolerance_(std::numeric_limits<data_type>::epsilon()),
      D(static_cast<blas_int>(__ctx.datasets[0].num_dimensions)),
      N(static_cast<blas_int>(__ctx.datasets[0].num_examples)),
      T(static_cast<blas_int>(__ctx.datasets[0].num_tasks))
  {
    LOG_INFO << "solver: " << base::name() << " (primal)" << std::endl
      << "objective: " << __obj.to_string() << std::endl
      << "stopping criteria: " << __ctx.criteria.to_string() << std::endl;
    LOG_DEBUG << "precision options: " << __obj.precision_string() << std::endl;
    for (auto d : __ctx.datasets) {
      LOG_INFO << "dataset: " << d.to_string() << std::endl;
    }
  }

protected:
  // Protected members of the base class
  using base::num_examples_;

  // Main variables
  const objective_type& objective_;
  const size_type num_dimensions_;
  const size_type num_tasks_;
  const size_type* labels_;
  const data_type* features_;
  data_type* primal_variables_;
  data_type* dual_variables_;

  // Other
  std::vector<data_type> norm2_;
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
      norm2_[i] = sdca_blas_dot(D, x_i, x_i);
    }
  }

  void solve_example(const size_type i) override {
    // Let x_i = i'th feature vector
    if (norm2_[i] <= 0) return;
    const data_type* x_i = features_ + num_dimensions_ * i;

    // Update dual variables
    data_type* variables = dual_variables_ + num_tasks_ * i;
    std::copy_n(variables, num_tasks_, &vars_before_[0]);
    compute_scores_swap_gt(labels_[i], x_i, variables);
    objective_.update_variables(T, norm2_[i], variables, &scores_[0]);
    std::swap(variables[0], variables[labels_[i]]);

    // Update primal variables
    sdca_blas_axpy(T, -1, variables, &vars_before_[0]);
    data_type diff = sdca_blas_asum(T, &vars_before_[0]);
    if (diff > diff_tolerance_) {
      sdca_blas_ger(D, T, -1, x_i, &vars_before_[0], primal_variables_);
    }
  }

  void evaluate_solution() override {
    // Let W = X * A'
    // (recompute W from scratch to minimize the accumulated numerical error)
    sdca_blas_gemm(D, T, N, features_, D, dual_variables_, T,
      primal_variables_, CblasNoTrans, CblasTrans);
    base::evaluate_solution();
  }

  inline evaluation_type
  evaluate_dataset(const dataset_type& set) override {
    evaluation_type stats;
    stats.accuracy.resize(num_tasks_);
    auto acc_first = stats.accuracy.begin();
    auto acc_last = stats.accuracy.end();

    // Compute the three sums independently over all training examples
    result_type regul_sum = 0;
    result_type p_loss_sum = 0;
    result_type d_loss_sum = 0;

    // Compensation variables for the Kahan summation
    result_type regul_comp = 0;
    result_type p_loss_comp = 0;
    result_type d_loss_comp = 0;

    size_type num_examples = set.num_examples;
    for (size_type i = 0; i < num_examples; ++i) {
      // Let x_i = i'th feature vector
      const data_type* x_i = set.data + num_dimensions_ * i;

      data_type* variables = dual_variables_ + num_tasks_ * i;
      compute_scores_swap_gt(set.labels[i], x_i, variables);

      // Count correct predictions
      auto it = std::partition(scores_.begin() + 1, scores_.end(),
        [=](const data_type x){ return x > scores_[0]; });
      acc_first[std::distance(scores_.begin() + 1, it)] += 1;

      // Compute the regularization term and primal/dual losses
      result_type regul, p_loss, d_loss;
      objective_.regularized_loss(T, variables, &scores_[0],
        regul, p_loss, d_loss);

      // Increment the sums
      kahan_add(regul, regul_sum, regul_comp);
      kahan_add(p_loss, p_loss_sum, p_loss_comp);
      kahan_add(d_loss, d_loss_sum, d_loss_comp);

      // Put back the ground truth variable
      std::swap(variables[0], variables[set.labels[i]]);
    }

    // Compute the overall primal/dual objectives and the duality gap
    objective_.primal_dual_gap(regul_sum, p_loss_sum, d_loss_sum,
      stats.primal, stats.dual, stats.gap);

    // Top-k accuracies for all k
    std::partial_sum(acc_first, acc_last, acc_first);
    std::for_each(acc_first, acc_last,
      [=](result_type &x){ x /= static_cast<result_type>(num_examples); });

    return stats;
  }

  inline void
  compute_scores_swap_gt(
      const size_type label,
      const data_type* x_i,
      data_type* variables
    ) {
    // Let scores = A * K_i = W' * x_i
    sdca_blas_gemv(D, T, primal_variables_, x_i, &scores_[0], CblasTrans);

    // Put ground truth at 0
    std::swap(variables[0], variables[label]);
    std::swap(scores_[0], scores_[label]);
  }

};

}

#endif
