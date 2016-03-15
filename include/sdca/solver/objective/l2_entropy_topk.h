#ifndef SDCA_SOLVER_OBJECTIVE_L2_ENTROPY_TOPK_H
#define SDCA_SOLVER_OBJECTIVE_L2_ENTROPY_TOPK_H

#include "sdca/types.h"
#include "sdca/prox/topk_entropy.h"
#include "sdca/prox/topk_entropy_biased.h"
#include "sdca/solver/objective/objective_base.h"

namespace sdca {

template <typename Data,
          typename Result>
struct l2_entropy_topk
    : public objective_base<Data, Result> {

  typedef Data data_type;
  typedef Result result_type;

  typedef objective_base<Data, Result> base;

  const Result c;
  const size_type k;

  const Result c_log_c;


  l2_entropy_topk(
      const Result __c,
      const size_type __k
    ) :
      base::objective_base(__c),
      c(__c),
      k(__k),
      c_log_c(x_log_x(__c))
  {}


  inline std::string to_string() const {
    std::ostringstream str;
    str << "l2_entropy_topk (c: " << c << ", k: " << k <<
           ", precision: " << type_name<Result>() << ")";
    return str.str();
  }


  template <typename Int>
  void update_dual_variables(
      const Int num_classes,
      const Data norm2,
      Data* variables,
      Data* scores
      ) const {
    Data *first(variables + 1), *last(variables + num_classes);
    Result alpha(c * static_cast<Result>(norm2)), zero(0), one(1);

    // 1. Prepare a vector to project in 'variables'.
    sdca_blas_axpby(
      static_cast<blas_int>(num_classes), 1, scores, -norm2, variables);
    Data a(-variables[0]);
    std::for_each(first, last, [=](Data &x){ x += a; });

    // 2. Proximal step (project 'variables', use 'scores' as scratch space)
    prox_topk_entropy_biased(
      first, last, scores + 1, static_cast<diff_type>(k), alpha);

    // 3. Recover the updated variables
    *variables = static_cast<Data>(
      c * std::min(one, std::accumulate(first, last, zero)));
    sdca_blas_scal(
      static_cast<blas_int>(num_classes - 1), static_cast<Data>(-c), first);
  }


  template <typename Int>
  inline Result
  primal_loss(
      const Int num_tasks,
      Data* scores
    ) const {
    Data *first(scores + 1), *last(scores + num_tasks), a(-scores[0]);
    std::for_each(first, last, [=](Data &x){ x += a; });

    auto t = thresholds_topk_entropy<Result>(first, last,
                                             static_cast<diff_type>(k));
    if (t.first == first) {
      return t.t; // equals to log(1 + \sum exp scores)
    } else {
      Result num_hi = static_cast<Result>(std::distance(first, t.first));
      Result sum_hi = std::accumulate(first, t.first, static_cast<Result>(0));
      Result s = t.hi * static_cast<Result>(k);
      return t.hi * (sum_hi + t.t * (static_cast<Result>(k) - num_hi))
        - x_log_x(1 - s) - num_hi * x_log_x(t.hi);
    }
  }


  template <typename Int>
  inline Result
  dual_loss(
      const Int num_classes,
      const Data* variables
    ) const {
    Result d_loss = c_log_c - x_log_x(c - static_cast<Result>(variables[0]));
    std::for_each(variables + 1, variables + num_classes,
      [&](const Result a){ d_loss -= x_log_x(-a); });
    return d_loss;
  }

};

}

#endif
