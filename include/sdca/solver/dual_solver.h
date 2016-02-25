#ifndef SDCA_SOLVER_DUAL_SOLVER_H
#define SDCA_SOLVER_DUAL_SOLVER_H

#include "sdca/math/blas.h"
#include "sdca/solver/multiset_solver.h"

namespace sdca {

template <typename Data,
          typename Result,
          typename Dataset,
          template <typename, typename> class Objective>
class dual_solver
    : public multiset_solver<Data, Result, Dataset> {
public:
  typedef multiset_solver<Data, Result, Dataset> base;

  typedef Data data_type;
  typedef Result result_type;
  typedef Dataset dataset_type;
  typedef Objective<Data, Result> objective_type;

  typedef typename base::context_type context_type;
  typedef typename base::evaluation_type evaluation_type;


  dual_solver(
      context_type&& __ctx,
      objective_type&& __obj
    ) :
      base::multiset_solver(std::move(__ctx)),
      objective_(std::move(__obj)),
      num_classes_(__ctx.test[0].out.num_classes),
      labels_(&(__ctx.test[0].out.labels[0])),
      gram_matrix_(__ctx.test[0].in.kernel),
      dual_variables_(__ctx.dual_variables),
      N(static_cast<blas_int>(__ctx.test[0].in.num_examples)),
      T(static_cast<blas_int>(__ctx.test[0].out.num_classes))
  {
    LOG_INFO << "solver: " << base::name() << " (dual)" << std::endl
      << "objective: " << __obj.to_string() << std::endl
      << "stopping criteria: " << __ctx.criteria.to_string() << std::endl;

    LOG_DEBUG << "precision options: " << __obj.precision_string() << std::endl;

    size_type i = 0;
    for (auto& d : __ctx.test) {
      LOG_VERBOSE << "dataset " << ++i << ": " << d.to_string() << std::endl;
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
  using base::scores_;

  // Main variables
  objective_type objective_;
  const size_type num_classes_;
  const size_type* labels_;
  const data_type* gram_matrix_;
  data_type* dual_variables_;

  // BLAS (avoid static casts)
  const blas_int N;
  const blas_int T;


  void solve_example(const size_type i) override {
    // Let K_i = i'th column of the Gram matrix
    const data_type* K_i = gram_matrix_ + num_examples_ * i;
    if (K_i[i] <= 0) return;

    // Update dual variables
    data_type* variables = dual_variables_ + num_classes_ * i;
    compute_scores(K_i);
    base::swap_ground_truth(labels_[i], variables);
    objective_.update_dual_variables(T, K_i[i], variables, &scores_[0]);
    base::swap_ground_truth(labels_[i], variables);
  }


  inline evaluation_type
  evaluate_train() override {
    evaluation_type stats;
    stats.accuracy.resize(num_classes_);
    auto acc_first = stats.accuracy.begin();
    auto acc_last = stats.accuracy.end();

    // Compute the three terms independently
    regularizer_ = 0;
    primal_loss_ = 0;
    dual_loss_ = 0;

    for (size_type i = 0; i < num_examples_; ++i) {
      // Let K_i = i'th column of the Gram matrix
      const data_type* K_i = gram_matrix_ + num_examples_ * i;

      // Compute prediction scores on example i
      data_type* variables = dual_variables_ + num_classes_ * i;
      compute_scores(K_i);
      base::swap_ground_truth(labels_[i], variables);

      // Increment the regularization term (before re-ordering the scores)
      regularizer_ += objective_.regularizer_dual(T, variables, &scores_[0]);

      // Count correct predictions - re-orders the scores!
      auto it = std::partition(scores_.begin() + 1, scores_.end(),
        [=](const data_type x){ return x >= scores_[0]; });
      acc_first[std::distance(scores_.begin() + 1, it)] += 1;

      // Increment the primal-dual losses (may re-order the scores)
      primal_loss_ += objective_.primal_loss(T, &scores_[0]);
      dual_loss_ += objective_.dual_loss(T, variables);

      // Put back the ground truth variable
      base::swap_ground_truth(labels_[i], variables);
    }

    // Compute the overall primal/dual objectives and the duality gap
    objective_.update_all(
      primal_loss_, dual_loss_, regularizer_, primal_, dual_, gap_);
    stats.loss = primal_loss_;

    // Top-k accuracies for all k
    std::partial_sum(acc_first, acc_last, acc_first);
    result_type coeff(1 / static_cast<result_type>(num_examples_));
    sdca_blas_scal(T, coeff, &stats.accuracy[0]);

    return stats;
  }


  inline evaluation_type
  evaluate_test(const dataset_type& set) override {
    evaluation_type stats;
    stats.accuracy.resize(num_classes_);
    auto acc_first = stats.accuracy.begin();
    auto acc_last = stats.accuracy.end();

    // Compute the primal loss
    result_type p_loss(0);

    size_type num_examples = set.in.num_examples;
    for (size_type i = 0; i < num_examples; ++i) {
      // Let K_i = i'th column of the Gram matrix
      const data_type* K_i = set.in.kernel + num_examples_ * i;

      compute_scores(K_i);
      base::swap_ground_truth(set.out.labels[i]);

      // Count correct predictions - re-orders the scores!
      auto it = std::partition(scores_.begin() + 1, scores_.end(),
        [=](const data_type x){ return x >= scores_[0]; });
      acc_first[std::distance(scores_.begin() + 1, it)] += 1;

      // Compute the primal loss (may re-order the scores)
      p_loss += objective_.primal_loss(T, &scores_[0]);
    }

    // The loss term may need an update (e.g. rescaling with C)
    objective_.update_primal_loss(p_loss);
    stats.loss = p_loss;

    // Top-k accuracies for all k
    std::partial_sum(acc_first, acc_last, acc_first);
    result_type coeff(1 / static_cast<result_type>(num_examples));
    sdca_blas_scal(T, coeff, &stats.accuracy[0]);

    return stats;
  }


  inline void
  compute_scores(
      const data_type* K_i
    ) {
    // Let scores = A * K_i = W' * x_i
    sdca_blas_gemv(T, N, dual_variables_, K_i, &scores_[0]);
  }

};


template <typename Data,
          typename Result,
          typename Dataset,
          template <typename, typename> class Objective>
inline dual_solver<Data, Result, Dataset, Objective>
make_dual_solver(
    solver_context<Data, Dataset>&& context,
    Objective<Data, Result>&& objective
  ) {
  return dual_solver<Data, Result, Dataset, Objective>(
    std::move(context), std::move(objective));
}


}

#endif
