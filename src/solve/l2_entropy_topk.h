#ifndef SDCA_SOLVE_L2_ENTROPY_TOPK_LOSS_H
#define SDCA_SOLVE_L2_ENTROPY_TOPK_LOSS_H

#include "prox/topk_entropy.h"
#include "prox/topk_entropy_biased.h"
#include "solvedef.h"
#include "util/util.h"

namespace sdca {

template <typename Data,
          typename Result,
          typename Summation>
struct l2_entropy_topk {
  const difference_type k;
  const Result c;
  const Data coeff;
  const Result log_c;
  const Summation sum;

  l2_entropy_topk(
      const size_type __k,
      const Result __c,
      const Summation __sum
    ) :
      k(static_cast<difference_type>(__k)),
      c(__c),
      coeff(static_cast<Data>(-__c)),
      log_c(std::log(__c)),
      sum(__sum)
  {}

  inline std::string to_string() const {
    std::ostringstream str;
    str << "l2_entropy_topk (k = " << k << ", C = " << c << ")";
    return str.str();
  }

  inline std::string precision_string() const {
    std::ostringstream str;
    str << "summation = " << sum.name() << ", "
      "precision = " << type_traits<Result>::name() << ", "
      "data = " << type_traits<Data>::name();
    return str.str();
  }

  void update_variables(
      const blas_int num_tasks,
      const size_type label,
      const Data norm2_inv,
      Data* variables,
      Data* scores
      ) const {
    // Variables update proceeds in 3 steps:
    // 1. Prepare a vector to project in 'variables'.
    // 2. Perform the proximal step (projection onto the feasible set).
    // 3. Recover the updated dual variables.
    Result norm2 = 1 / static_cast<Result>(norm2_inv);
    Result alpha = c / static_cast<Result>(norm2_inv);

    // 1. Prepare a vector to project in 'variables'.
    sdca_blas_axpby(num_tasks, 1, scores, -static_cast<Data>(norm2), variables);

    // Place ground truth at the back
    Data *scores_back = scores + num_tasks - 1;
    Data *variables_back = variables + num_tasks - 1;
    std::swap(*scores_back, scores[label]);
    std::swap(*variables_back, variables[label]);

    // Complete step 1.
    Data a = -*variables_back;
    std::for_each(variables, variables_back, [&](Data &x){ x += a; });

    // 2. Proximal step (project 'variables', use 'scores' as scratch space)
    prox_topk_entropy_biased(variables, variables_back,
      scores, scores_back, k, alpha, sum);

    // 3. Recover the updated variables
    *variables_back = static_cast<Data>(c * std::min(static_cast<Result>(1),
      sum(variables, variables_back, static_cast<Result>(0)) ));
    std::for_each(variables, variables_back, [=](Data &x){ x *= coeff; });

    // Put back the ground truth variable
    std::swap(*variables_back, variables[label]);
  }

  void regularized_loss(
      const blas_int num_tasks,
      const size_type label,
      const Data* variables,
      Data* scores,
      Result &regularizer,
      Result &primal_loss,
      Result &dual_loss
    ) const {
    regularizer = static_cast<Result>(
      sdca_blas_dot(num_tasks, scores, variables));

    Result comp(0), zero(0), aj(static_cast<Result>(variables[label]));
    dual_loss = (aj < c) ? - (c - aj) * std::log(c - aj) : zero;
    std::for_each(variables, variables + num_tasks, [&](const Result a){
      sum.add((a < zero) ? a * std::log(-a) : zero, dual_loss, comp); });
    sum.add(aj * log_c, dual_loss, comp);

    Data a = scores[label];
    std::for_each(scores, scores + num_tasks, [=](Data &x){ x -= a; });
    scores[label] = scores[num_tasks - 1]; // only need num_tasks - 1 scores

    auto t = thresholds_topk_entropy<Data*, Result, Summation>(
      scores, scores + num_tasks - 1, k, sum);
    if (t.first == scores) {
      primal_loss = t.t; // equals to log(1 + \sum exp scores)
    } else if (t.hi < std::numeric_limits<Result>::epsilon()) {
      primal_loss = 0;
    } else {
      Result num_hi = static_cast<Result>(std::distance(scores, t.first));
      Result sum_hi = sum(scores, t.first, static_cast<Result>(0));
      Result s = t.hi * static_cast<Result>(k);
      primal_loss = - (1 - s) * std::log(1 - s)
        + t.hi * (sum_hi - num_hi * std::log(t.hi)
          + t.t * (static_cast<Result>(k) - num_hi));
    }
  }

  inline void primal_dual_gap(
      const Result regularizer,
      const Result primal_loss,
      const Result dual_loss,
      Result &primal_objective,
      Result &dual_objective,
      Result &duality_gap
    ) const {
    primal_objective = c * primal_loss;
    dual_objective = dual_loss;
    duality_gap = primal_objective - dual_objective + regularizer;
    primal_objective += static_cast<Result>(0.5) * regularizer;
    dual_objective -= static_cast<Result>(0.5) * regularizer;
  }
};

}

#endif
