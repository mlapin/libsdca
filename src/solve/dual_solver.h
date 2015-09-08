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
      // Let K_i = i'th column of the Gram matrix
      const data_type* K_i = set.data + num_examples * i;

      data_type* variables = dual_variables_ + num_tasks_ * i;
      compute_scores_swap_gt(set.labels[i], K_i, variables);

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
      const data_type* K_i,
      data_type* variables
    ) {
    // Let scores = A * K_i = W' * x_i
    sdca_blas_gemv(T, N,
      dual_variables_, K_i, &scores_[0]);

    // Put ground truth at 0
    std::swap(variables[0], variables[label]);
    std::swap(scores_[0], scores_[label]);
  }
};

}

#endif
