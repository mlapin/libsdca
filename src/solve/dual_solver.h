#ifndef SDCA_SOLVE_DUAL_SOLVER_H
#define SDCA_SOLVE_DUAL_SOLVER_H

#include "util/util.h"
#include "solver.h"

namespace sdca {

template <typename Objective,
          typename Data,
          typename Result>
class dual_solver : public multiset_solver<Data, Result> {
public:
  typedef multiset_solver<Data, Result> base;
  typedef typename base::context_type context_type;
  typedef typename base::dataset_type dataset_type;
  typedef typename base::evaluation_type evaluation_type;
  typedef Objective objective_type;
  typedef Data data_type;
  typedef Result result_type;

  dual_solver(
      const objective_type& __obj,
      const context_type& __ctx
    ) :
      base::multiset_solver(__ctx),
      objective_(__obj),
      num_tasks_(__ctx.datasets[0].num_tasks),
      labels_(&(__ctx.datasets[0].labels[0])),
      gram_matrix_(__ctx.datasets[0].data),
      dual_variables_(__ctx.dual_variables),
      scores_(__ctx.datasets[0].num_tasks),
      N(static_cast<blas_int>(__ctx.datasets[0].num_examples)),
      T(static_cast<blas_int>(__ctx.datasets[0].num_tasks))
  {
    LOG_INFO << "solver: " << base::name() << " (dual)" << std::endl
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
  using base::primal_loss_;
  using base::dual_loss_;
  using base::regularizer_;
  using base::primal_;
  using base::dual_;
  using base::gap_;

  // Main variables
  const objective_type& objective_;
  const size_type num_tasks_;
  const size_type* labels_;
  const data_type* gram_matrix_;
  data_type* dual_variables_;

  // Other
  std::vector<data_type> scores_;

  // BLAS (avoid static casts)
  const blas_int N;
  const blas_int T;

  void solve_example(const size_type i) override {
    // Let K_i = i'th column of the Gram matrix
    const data_type* K_i = gram_matrix_ + num_examples_ * i;
    if (K_i[i] <= 0) return;

    // Update dual variables
    data_type* variables = dual_variables_ + num_tasks_ * i;
    compute_scores_swap_gt(labels_[i], K_i, variables);
    objective_.update_variables(T, K_i[i], variables, &scores_[0]);
    std::swap(variables[0], variables[labels_[i]]);
  }

  inline evaluation_type
  evaluate_train() override {
    evaluation_type stats;
    stats.accuracy.resize(num_tasks_);
    auto acc_first = stats.accuracy.begin();
    auto acc_last = stats.accuracy.end();

    // Compute the three sums independently using Kahan summation
    primal_loss_ = 0; dual_loss_ = 0; regularizer_ = 0;
    result_type p_loss_comp(0), d_loss_comp(0), regul_comp(0);

    for (size_type i = 0; i < num_examples_; ++i) {
      // Let K_i = i'th column of the Gram matrix
      const data_type* K_i = gram_matrix_ + num_examples_ * i;

      // Compute prediction scores for example i
      data_type* variables = dual_variables_ + num_tasks_ * i;
      compute_scores_swap_gt(labels_[i], K_i, variables);

      // Count correct predictions
      auto it = std::partition(scores_.begin() + 1, scores_.end(),
        [=](const data_type x){ return x > scores_[0]; });
      acc_first[std::distance(scores_.begin() + 1, it)] += 1;

      // Increment the loss and regularization terms
      kahan_add(objective_.regularizer(T, variables, &scores_[0]),
        regularizer_, regul_comp);
      kahan_add(objective_.dual_loss(T, variables),
        dual_loss_, d_loss_comp);
      kahan_add(objective_.primal_loss(T, &scores_[0]),
        primal_loss_, p_loss_comp); // may re-order scores

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
    stats.accuracy.resize(num_tasks_);
    auto acc_first = stats.accuracy.begin();
    auto acc_last = stats.accuracy.end();

    // Compute the primal loss using Kahan summation
    result_type p_loss(0), p_loss_comp(0);

    size_type num_examples = set.num_examples;
    for (size_type i = 0; i < num_examples; ++i) {
      // Let K_i = i'th column of the Gram matrix
      const data_type* K_i = set.data + num_examples_ * i;

      // Let scores = A * K_i = W' * x_i
      sdca_blas_gemv(T, N, dual_variables_, K_i, &scores_[0]);
      std::swap(scores_[0], scores_[set.labels[i]]);

      // Count correct predictions
      auto it = std::partition(scores_.begin() + 1, scores_.end(),
        [=](const data_type x){ return x > scores_[0]; });
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
      const data_type* K_i,
      data_type* variables
    ) {
    // Let scores = A * K_i = W' * x_i
    sdca_blas_gemv(T, N, dual_variables_, K_i, &scores_[0]);

    // Put ground truth at 0
    std::swap(variables[0], variables[label]);
    std::swap(scores_[0], scores_[label]);
  }
};

}

#endif
