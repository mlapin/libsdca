#ifndef SDCA_SOLVE_L2_ENTROPY_TOPK_LOSS_H
#define SDCA_SOLVE_L2_ENTROPY_TOPK_LOSS_H

#include "objective_base.h"
#include "prox/topk_entropy.h"
#include "prox/topk_entropy_biased.h"
#include "solvedef.h"

namespace sdca {

template <typename Data,
          typename Result,
          typename Summation>
struct l2_entropy_topk : public objective_base<Data, Result, Summation> {
  typedef objective_base<Data, Result, Summation> base;
  const difference_type k;
  const Result c;
  const Data coeff;
  const Result log_c;

  l2_entropy_topk(
      const size_type __k,
      const Result __c,
      const Summation __sum
    ) :
      base::objective_base(__c, __sum),
      k(static_cast<difference_type>(__k)),
      c(__c),
      coeff(static_cast<Data>(-__c)),
      log_c(std::log(__c))
  {}

  inline std::string to_string() const {
    std::ostringstream str;
    str << "l2_entropy_topk (k = " << k << ", C = " << c << ")";
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
      scores + 1, scores + num_tasks, k, alpha, this->sum);

    // 3. Recover the updated variables
    *variables = static_cast<Data>(c * std::min(static_cast<Result>(1),
      this->sum(first, last, static_cast<Result>(0)) ));
    std::for_each(first, last, [=](Data &x){ x *= coeff; });
  }

  inline Result
  dual_loss(
      const blas_int num_tasks,
      const Data* variables
    ) const {
    Result comp(0), zero(0), aj(static_cast<Result>(variables[0]));
    Result d_loss = (aj < c) ? - (c - aj) * std::log(1 - aj / c) : zero;
    std::for_each(variables + 1, variables + num_tasks, [&](const Result a){
      this->sum.add((a < zero) ? a * std::log(-a) : zero, d_loss, comp); });
    this->sum.add(aj * log_c, d_loss, comp);
    return d_loss;
  }

  inline Result
  primal_loss(
      const blas_int num_tasks,
      Data* scores
    ) const {
    Data *first(scores + 1), *last(scores + num_tasks), a(-scores[0]);
    std::for_each(first, last, [=](Data &x){ x += a; });

    auto t = thresholds_topk_entropy<Data*, Result>(first, last, k, this->sum);
    if (t.first == first) {
      return t.t; // equals to log(1 + \sum exp scores)
    } else {
      Result num_hi = static_cast<Result>(std::distance(first, t.first));
      Result sum_hi = this->sum(first, t.first, static_cast<Result>(0));
      Result s = t.hi * static_cast<Result>(k);
      return (s - 1) * std::log(1 - s) + t.hi * (sum_hi
        - num_hi * std::log(t.hi) + t.t * (static_cast<Result>(k) - num_hi));
    }
  }
};

}

#endif
