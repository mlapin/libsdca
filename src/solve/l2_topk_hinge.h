#ifndef SDCA_SOLVE_L2_TOPK_HINGE_LOSS_H
#define SDCA_SOLVE_L2_TOPK_HINGE_LOSS_H

#include "objective_base.h"
#include "prox/knapsack_le_biased.h"
#include "solvedef.h"

namespace sdca {

template <typename Data,
          typename Result,
          typename Summation>
struct l2_topk_hinge : public objective_base<Data, Result, Summation> {
  typedef objective_base<Data, Result, Summation> base;
  const difference_type k;
  const Result c;
  const Result c_div_k;

  l2_topk_hinge(
      const size_type __k,
      const Result __c,
      const Summation __sum
    ) :
      base::objective_base(__c / static_cast<Result>(__k), __sum),
      k(static_cast<difference_type>(__k)),
      c(__c),
      c_div_k(__c / static_cast<Result>(__k))
  {}

  inline std::string to_string() const {
    std::ostringstream str;
    str << "l2_topk_hinge (k = " << k << ", C = " << c << ")";
    return str.str();
  }

  void update_variables(
      const blas_int num_classes,
      const Data norm2,
      Data* variables,
      Data* scores
      ) const {
    Data *first(variables + 1), *last(variables + num_classes), a(1 / norm2);
    Result lo(0), hi(c_div_k), rhs(c), rho(1);

    // 1. Prepare a vector to project in 'variables'.
    sdca_blas_axpby(num_classes, a, scores, -1, variables);
    a -= variables[0];
    std::for_each(first, last, [=](Data &x){ x += a; });

    // 2. Proximal step (project 'variables', use 'scores' as scratch space)
    prox_knapsack_le_biased(first, last,
      scores + 1, scores + num_classes, lo, hi, rhs, rho, this->sum);

    // 3. Recover the updated variables
    *variables = static_cast<Data>(std::min(rhs,
      this->sum(first, last, static_cast<Result>(0)) ));
    std::for_each(first, last, [](Data &x){ x = -x; });
  }

  inline Result
  primal_loss(
      const blas_int num_classes,
      Data* scores
    ) const {
    Data *first(scores + 1), *last(scores + num_classes), a(1 - scores[0]);
    std::for_each(first, last, [=](Data &x){ x += a; });

    // Find k largest elements
    std::nth_element(first, first + k - 1, last, std::greater<Data>());

    // sum_k_largest max{0, score_i} (division by k happens later)
    auto it = std::partition(first, first + k, [](Data x){ return x > 0; });
    return this->sum(first, it, static_cast<Result>(0));
  }
};

}

#endif
