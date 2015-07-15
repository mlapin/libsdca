#ifndef SDCA_SOLVE_L2_TOPK_LOSS_H
#define SDCA_SOLVE_L2_TOPK_LOSS_H

#include "linalg/linalg.h"
#include "logging/logging.h"
#include "prox/topk_simplex_biased.h"
#include "solvedef.h"

namespace sdca {

template <typename Data, typename Result = long double>
struct l2_hinge_topk {
  const difference_type k;
  const Data rhs;
  const Result c_div_k;

  l2_hinge_topk(
      const size_type top_k,
      const Data svm_c
    ) :
      k(static_cast<difference_type>(top_k)),
      rhs(svm_c),
      c_div_k(static_cast<Result>(svm_c) / static_cast<Result>(top_k))
  {
    LOG_INFO << "objective: l2_hinge_topk ("
      "k = " << top_k << ", "
      "C = " << svm_c << ")" << std::endl;
  }

  void update_variables(
      const blas_int num_tasks,
      const size_type label,
      const Data norm2_inv,
      Data* variables,
      Data* scores
      ) const {

    Data a = norm2_inv;
    sdca_blas_axpby(num_tasks, a, scores, -1, variables);

    // Place ground truth at the back
    Data *scores_back = scores + num_tasks - 1;
    Data *variables_back = variables + num_tasks - 1;
    std::swap(*scores_back, scores[label]);
    std::swap(*variables_back, variables[label]);

    // Update variables
    a -= *variables_back;
    std::for_each(variables, variables_back, [&](Data &x){ x += a; });

    // Project onto the topk simplex
    project_topk_simplex_biased(variables, variables_back,
      scores, scores_back, k, rhs);

    // The last one is the sum
    *variables_back = std::accumulate(
      variables, variables_back, static_cast<Data>(0));

    // Change the sign of all but last one
    std::for_each(variables, variables_back, [](Data &x){ x = -x; });

    // Put back the ground truth variable
    std::swap(*variables_back, variables[label]);
  }

  void regularized_loss(
      const blas_int num_tasks,
      const size_type label,
      const Data* variables,
      Data* scores,
      Data &regularizer,
      Data &primal_loss,
      Data &dual_loss
    ) const {

    regularizer = sdca_blas_dot(num_tasks, scores, variables);
    dual_loss = variables[label];

    Data a = static_cast<Data>(1) - scores[label];
    std::for_each(scores, scores + num_tasks, [&](Data &x){ x += a; });
    scores[label] = static_cast<Data>(0);

    // Sum k largest elements
    std::nth_element(scores, scores + k - 1, scores + num_tasks,
      std::greater<Data>());
    a = std::accumulate(scores, scores + k, static_cast<Data>(0));

    // max{0, sum_k_largest} (division by k happens later)
    primal_loss = std::max(static_cast<Data>(0), a);
  }

  void primal_dual_gap(
      const Result regularizer,
      const Result primal_loss,
      const Result dual_loss,
      Result &primal_objective,
      Result &dual_objective,
      Result &duality_gap
    ) const {
    primal_objective = c_div_k * primal_loss;
    dual_objective = dual_loss;
    duality_gap = primal_objective - dual_objective + regularizer;
    primal_objective += static_cast<Result>(0.5) * regularizer;
    dual_objective -= static_cast<Result>(0.5) * regularizer;
  }
};

template <typename Data, typename Result = long double>
inline
l2_hinge_topk<Data, Result>
make_l2_hinge_topk(
    const size_type top_k,
    const Data svm_c
  ) {
  return l2_hinge_topk<Data, Result>(top_k, svm_c);
}

}

#endif
