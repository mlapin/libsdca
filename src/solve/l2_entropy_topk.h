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
  typedef Data data_type;
  typedef Result result_type;
  typedef Summation summation_type;
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
      "data_precision = " << type_traits<Data>::name();
    return str.str();
  }

  void update_variables(
      const blas_int num_tasks,
      const Data norm2,
      Data* variables,
      Data* scores
      ) const {
    Data *first(variables + 1), *last(variables + num_tasks);
    Result alpha(c * static_cast<Result>(norm2));

    // 1. Prepare a vector to project in 'variables'.
    sdca_blas_axpby(num_tasks, 1, scores, -norm2, variables);
    Data a(-variables[0]);
    std::for_each(first, last, [=](Data &x){ x += a; });

    // 2. Proximal step (project 'variables', use 'scores' as scratch space)
    prox_topk_entropy_biased(first, last,
      scores + 1, scores + num_tasks, k, alpha, sum);

    // 3. Recover the updated variables
    *variables = static_cast<Data>(c * std::min(static_cast<Result>(1),
      sum(first, last, static_cast<Result>(0)) ));
    std::for_each(first, last, [=](Data &x){ x *= coeff; });
  }

  void regularized_loss(
      const blas_int num_tasks,
      const Data* variables,
      Data* scores,
      Result &regularizer,
      Result &primal_loss,
      Result &dual_loss
    ) const {
    regularizer = static_cast<Result>(
      sdca_blas_dot(num_tasks, scores, variables));

    Result comp(0), zero(0), aj(static_cast<Result>(variables[0]));
    dual_loss = (aj < c) ? - (c - aj) * std::log(1 - aj / c) : zero;
    std::for_each(variables + 1, variables + num_tasks, [&](const Result a){
      sum.add((a < zero) ? a * std::log(-a) : zero, dual_loss, comp); });
    sum.add(aj * log_c, dual_loss, comp);

    Data *first(scores + 1), *last(scores + num_tasks);
    Data a(-scores[0]);
    std::for_each(first, last, [=](Data &x){ x += a; });

    auto t = thresholds_topk_entropy<Data*, Result>(first, last, k, sum);
    if (t.first == first) {
      primal_loss = t.t; // equals to log(1 + \sum exp scores)
    } else {
      Result num_hi = static_cast<Result>(std::distance(first, t.first));
      Result sum_hi = sum(first, t.first, static_cast<Result>(0));
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
