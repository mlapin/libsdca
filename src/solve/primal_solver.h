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
      num_classes_(__ctx.datasets[0].num_classes),
      labels_(&(__ctx.datasets[0].labels[0])),
      features_(__ctx.datasets[0].data),
      primal_variables_(__ctx.primal_variables),
      dual_variables_(__ctx.dual_variables),
      norm2_(__ctx.datasets[0].num_examples),
      scores_(__ctx.datasets[0].num_classes),
      vars_before_(__ctx.datasets[0].num_classes),
      diff_tolerance_(std::numeric_limits<data_type>::epsilon()),
      D(static_cast<blas_int>(__ctx.datasets[0].num_dimensions)),
      N(static_cast<blas_int>(__ctx.datasets[0].num_examples)),
      T(static_cast<blas_int>(__ctx.datasets[0].num_classes))
  {
    LOG_INFO << "solver: " << base::name() << " (primal)" << std::endl
      << "objective: " << __obj.to_string() << std::endl
      << "stopping criteria: " << __ctx.criteria.to_string() << std::endl;
    LOG_DEBUG << "precision options: " << __obj.precision_string() << std::endl;
    size_type i = 0;
    for (auto d : __ctx.datasets) {
      LOG_VERBOSE << "dataset #" << ++i << ": " << d.to_string() << std::endl;
    }
  }

protected:
  // Protected members of the base class
  using base::num_examples_;
  using base::primal_loss_;
  using base::dual_loss_;
  using base::regularizer_;
  using base::primal_;
  using base::dual_;
  using base::gap_;

  // Main variables
  const objective_type& objective_;
  const size_type num_dimensions_;
  const size_type num_classes_;
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
    data_type* variables = dual_variables_ + num_classes_ * i;
    std::copy_n(variables, num_classes_, &vars_before_[0]);
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
  evaluate_train() override {
    evaluation_type stats;
    stats.accuracy.resize(num_classes_);
    auto acc_first = stats.accuracy.begin();
    auto acc_last = stats.accuracy.end();

    // Compute the three sums independently using Kahan summation
    primal_loss_ = 0; dual_loss_ = 0; regularizer_ = 0;
    result_type p_loss_comp(0), d_loss_comp(0), regul_comp(0);

    for (size_type i = 0; i < num_examples_; ++i) {
      // Let x_i = i'th feature vector
      const data_type* x_i = features_ + num_dimensions_ * i;

      // Compute prediction scores on example i
      data_type* variables = dual_variables_ + num_classes_ * i;
      compute_scores_swap_gt(labels_[i], x_i, variables);

      // Increment the regularization term (before re-ordering scores)
      kahan_add(objective_.regularizer(T, variables, &scores_[0]),
        regularizer_, regul_comp);

      // Count correct predictions - re-orders scores!
      auto it = std::partition(scores_.begin() + 1, scores_.end(),
        [=](const data_type x){ return x >= scores_[0]; });
      acc_first[std::distance(scores_.begin() + 1, it)] += 1;

      // Increment the primal-dual losses
      kahan_add(objective_.primal_loss(T, &scores_[0]),
        primal_loss_, p_loss_comp); // may re-order scores!
      kahan_add(objective_.dual_loss(T, variables),
        dual_loss_, d_loss_comp);

      // Put back the ground truth variable
      std::swap(variables[0], variables[labels_[i]]);
    }

    // Compute the overall primal/dual objectives and the duality gap
    objective_.update_all(
      primal_loss_, dual_loss_, regularizer_, primal_, dual_, gap_);
    stats.loss = primal_loss_;

    // Top-k accuracies for all k
    std::partial_sum(acc_first, acc_last, acc_first);
    result_type coeff(1 / static_cast<result_type>(num_examples_));
    std::for_each(acc_first, acc_last, [=](result_type &x){ x *= coeff; });

    return stats;
  }

  inline evaluation_type
  evaluate_test(const dataset_type& set) override {
    evaluation_type stats;
    stats.accuracy.resize(num_classes_);
    auto acc_first = stats.accuracy.begin();
    auto acc_last = stats.accuracy.end();

    // Compute the primal loss using Kahan summation
    result_type p_loss(0), p_loss_comp(0);

    size_type num_examples = set.num_examples;
    for (size_type i = 0; i < num_examples; ++i) {
      // Let x_i = i'th feature vector
      const data_type* x_i = set.data + num_dimensions_ * i;

      // Let scores = A * K_i = W' * x_i
      sdca_blas_gemv(D, T, primal_variables_, x_i, &scores_[0], CblasTrans);
      std::swap(scores_[0], scores_[set.labels[i]]);

      // Count correct predictions - re-orders scores!
      auto it = std::partition(scores_.begin() + 1, scores_.end(),
        [=](const data_type x){ return x >= scores_[0]; });
      acc_first[std::distance(scores_.begin() + 1, it)] += 1;

      // Compute the primal loss
      kahan_add(objective_.primal_loss(T, &scores_[0]),
        p_loss, p_loss_comp); // may re-order scores
    }

    // The loss term may need an update (e.g. rescaling with C)
    objective_.update_loss(p_loss);
    stats.loss = p_loss;

    // Top-k accuracies for all k
    std::partial_sum(acc_first, acc_last, acc_first);
    result_type coeff(1 / static_cast<result_type>(num_examples));
    std::for_each(acc_first, acc_last, [=](result_type &x){ x *= coeff; });

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
