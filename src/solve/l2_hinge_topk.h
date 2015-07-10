#ifndef SDCA_SOLVE_L2_TOPK_LOSS_H
#define SDCA_SOLVE_L2_TOPK_LOSS_H

#include "linalg/linalg.h"
#include "prox/topk_simplex_biased.h"
#include "solvedef.h"

namespace sdca {

template <typename data_type, typename result_type = long double>
struct l2_hinge_topk {
  const difference_type k;
  const data_type rhs;
  const result_type c_div_k;

  l2_hinge_topk(
      const size_type top_k,
      const data_type svm_c
    ) :
      k(static_cast<difference_type>(top_k)),
      rhs(svm_c),
      c_div_k(static_cast<result_type>(svm_c) / static_cast<result_type>(top_k))
  {}

  void update(
      const blas_int num_tasks,
      const size_type label,
      const data_type norm_squared,
      data_type* variables,
      data_type* scores
      ) const {

    data_type a = static_cast<data_type>(1) / norm_squared;
    sdca_blas_axpby(num_tasks, a, scores, -1, variables);

    // Place ground truth at the back
    data_type *scores_back = scores + num_tasks - 1;
    data_type *variables_back = variables + num_tasks - 1;
    std::swap(*scores_back, scores[label]);
    std::swap(*variables_back, variables[label]);

    // Update variables
    a -= *variables_back;
    std::for_each(variables, variables_back, [&](data_type &x){ x += a; });

    // Project onto the topk simplex
    project_topk_simplex_biased(variables, variables_back,
      scores, scores_back, k, rhs);

    // The last one is the sum
    *variables_back = std::accumulate(
      variables, variables_back, static_cast<data_type>(0));

    // Change the sign of all but last one
    std::for_each(variables, variables_back, [](data_type &x){ x = -x; });

    // Put back the ground truth variable
    std::swap(*variables_back, variables[label]);
  }

  void loss(
      const blas_int num_tasks,
      const size_type label,
      const data_type* variables,
      data_type* scores,
      data_type &regularizer,
      data_type &primal_loss,
      data_type &dual_loss
    ) const {

    regularizer = sdca_blas_dot(num_tasks, scores, variables);
    dual_loss = variables[label];

    data_type a = static_cast<data_type>(1) - scores[label];
    std::for_each(scores, scores + num_tasks, [&](data_type &x){ x += a; });
    scores[label] = static_cast<data_type>(0);

    // Sum k largest elements
    std::nth_element(scores, scores + k - 1, scores + num_tasks,
      std::greater<data_type>());
    a = std::accumulate(scores, scores + k, static_cast<data_type>(0));

    // max{0, sum_k_largest} (division by k happens later)
    primal_loss = std::max(static_cast<data_type>(0), a);
  }

  void objective(
      const result_type regularizer,
      const result_type primal_loss,
      const result_type dual_loss,
      result_type &primal_objective,
      result_type &dual_objective,
      result_type &duality_gap
    ) const {
    primal_objective = c_div_k * primal_loss;
    dual_objective = dual_loss;
    duality_gap = primal_objective - dual_objective + regularizer;
    primal_objective += static_cast<result_type>(0.5) * regularizer;
    dual_objective -= static_cast<result_type>(0.5) * regularizer;
  }
};

template <typename data_type, typename result_type = long double>
inline
l2_hinge_topk<data_type, result_type>
make_l2_hinge_topk(const size_type top_k, const data_type svm_c) {
  return l2_hinge_topk<data_type, result_type>(top_k, svm_c);
}

}

#endif
