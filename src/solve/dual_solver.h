#ifndef SDCA_SOLVE_DUAL_SOLVER_H
#define SDCA_SOLVE_DUAL_SOLVER_H
#include <iterator>
#include "util/util.h"
#include "solver.h"

namespace sdca {

template <typename Objective,
          typename Data,
          typename Result>
class dual_solver : public solver_base<Result> {
public:
  typedef solver_base<Result> base;
  typedef Objective objective_type;
  typedef solver_context<Data, Result> context_type;
  typedef dataset<Data> dataset_type;
  typedef Data data_type;
  typedef Result result_type;

  dual_solver(
      const objective_type& __objective,
      const context_type& __ctx
    ) :
      base::solver_base(__ctx.criteria, __ctx.datasets[0].num_examples),
      objective_(__objective),
      num_tasks_(__ctx.datasets[0].num_tasks),
      labels_(&(__ctx.datasets[0].labels[0])),
      gram_matrix_(__ctx.datasets[0].data),
      dual_variables_(__ctx.dual_variables),
      scores_(__ctx.datasets[0].num_tasks),
      N(static_cast<blas_int>(__ctx.datasets[0].num_examples)),
      T(static_cast<blas_int>(__ctx.datasets[0].num_tasks))
  {
    LOG_INFO << "solver: " << base::name() << " (dual)" << std::endl
      << "objective: " << __objective.to_string() << std::endl
      << "stopping criteria: " << __ctx.criteria.to_string() << std::endl;
    LOG_DEBUG << __objective.precision_string()  << std::endl;
    for (auto d : __ctx.datasets) {
      LOG_INFO << "dataset: " << d.to_string() << std::endl;
    }
  }

protected:
  // Protected members of the base class
  using base::num_examples_;
  using base::primal_;
  using base::dual_;
  using base::gap_;

  // Main variables
  const objective_type objective_;
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

    // Let scores = A * K_i = W' * x_i
    sdca_blas_gemv(T, N, dual_variables_, K_i, &scores_[0]);

    // Put ground truth at 0
    data_type* variables = dual_variables_ + num_tasks_ * i;
    std::swap(variables[0], variables[labels_[i]]);
    std::swap(scores_[0], scores_[labels_[i]]);

    // Update dual variables
    objective_.update_variables(T, K_i[i], variables, &scores_[0]);

    // Put back the ground truth variable
    std::swap(variables[0], variables[labels_[i]]);
  }

  void compute_objectives() override {
    // Compute the three sums independently over all training examples
    result_type regul_sum = 0;
    result_type p_loss_sum = 0;
    result_type d_loss_sum = 0;

    // Compensation variables for the Kahan summation
    result_type regul_comp = 0;
    result_type p_loss_comp = 0;
    result_type d_loss_comp = 0;

    std::vector<data_type> accuracy(num_tasks_);

    for (size_type i = 0; i < num_examples_; ++i) {
      // Let K_i = i'th column of the Gram matrix
      const data_type* K_i = gram_matrix_ + num_examples_ * i;

      if (K_i[i] <= 0) return;

      // Let scores = A * K_i = W' * x_i
      sdca_blas_gemv(T, N, dual_variables_, K_i, &scores_[0]);

      // Put ground truth at 0
      data_type* variables = dual_variables_ + num_tasks_ * i;
      std::swap(variables[0], variables[labels_[i]]);
      std::swap(scores_[0], scores_[labels_[i]]);

      // Count correct predictions
      auto it = std::partition(scores_.begin() + 1, scores_.end(),
        [=](const data_type x){ return x > scores_[0]; });
      accuracy[std::distance(scores_.begin() + 1, it)] += 1;

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

    // Top-k accuracies for all k
    std::partial_sum(accuracy.begin(), accuracy.end(), accuracy.begin());
    std::for_each(accuracy.begin(), accuracy.end(),
      [=](data_type &x){ x /= accuracy.back(); });
    std::copy(accuracy.begin(), accuracy.begin() + 10,
      std::ostream_iterator<data_type>(std::cout, " "));
    std::cout << std::endl;

    // Compute the overall primal/dual objectives and the duality gap
    objective_.primal_dual_gap(regul_sum, p_loss_sum, d_loss_sum,
      primal_, dual_, gap_);
  }

};

}

#endif
