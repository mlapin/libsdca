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
    str << "l2_topk_hinge (k = " << k << ", C = " << c << ", gamma = 0)";
    return str.str();
  }

  void update_variables(
      const blas_int num_tasks,
      const Data norm2,
      Data* variables,
      Data* scores
      ) const {
    Data *first(variables + 1), *last(variables + num_tasks), a(1 / norm2);
    Result lo(0), hi(c_div_k), rhs(c), rho(1);

    // 1. Prepare a vector to project in 'variables'.
    sdca_blas_axpby(num_tasks, a, scores, -1, variables);
    a -= variables[0];
    std::for_each(first, last, [=](Data &x){ x += a; });

    // 2. Proximal step (project 'variables', use 'scores' as scratch space)
    prox_knapsack_le_biased(first, last,
      scores + 1, scores + num_tasks, lo, hi, rhs, rho, this->sum);

    // 3. Recover the updated variables
    *variables = static_cast<Data>(std::min(rhs,
      this->sum(first, last, static_cast<Result>(0)) ));
    std::for_each(first, last, [](Data &x){ x = -x; });
  }

  inline Result
  primal_loss(
      const blas_int num_tasks,
      Data* scores
    ) const {
    Data *first(scores + 1), *last(scores + num_tasks), a(1 - scores[0]);
    std::for_each(first, last, [=](Data &x){ x += a; });

    // Find k largest elements
    std::nth_element(first, first + k - 1, last, std::greater<Data>());

    // sum_k_largest max{0, score_i} (division by k happens later)
    auto it = std::partition(first, first + k, [](Data x){ return x > 0; });
    return this->sum(first, it, static_cast<Result>(0));
  }
};

template <typename Data,
          typename Result,
          typename Summation>
struct l2_topk_hinge_smooth : public objective_base<Data, Result, Summation> {
  typedef objective_base<Data, Result, Summation> base;
  const difference_type k;
  const Result c;
  const Result gamma;
  const Result c_div_k;
  const Result gamma_div_k;
  const Result gamma_div_c;

  l2_topk_hinge_smooth(
      const size_type __k,
      const Result __c,
      const Result __gamma,
      const Summation __sum
    ) :
      base::objective_base(__c / __gamma, __sum),
      k(static_cast<difference_type>(__k)),
      c(__c),
      gamma(__gamma),
      c_div_k(__c / static_cast<Result>(__k)),
      gamma_div_k(__gamma / static_cast<Result>(__k)),
      gamma_div_c(__gamma / __c)
  {}

  inline std::string to_string() const {
    std::ostringstream str;
    str << "l2_topk_hinge (k = " << k << ", C = " << c << ", "
      "gamma = " << gamma << ")";
    return str.str();
  }

  void update_variables(
      const blas_int num_tasks,
      const Data norm2,
      Data* variables,
      Data* scores
      ) const {
    Data *first(variables + 1), *last(variables + num_tasks);
    Result lo(0), hi(c_div_k), rhs(c), rho(static_cast<Result>(norm2) /
      (static_cast<Result>(norm2) + gamma_div_c));

    // 1. Prepare a vector to project in 'variables'.
    Data a(static_cast<Data>(rho) / norm2);
    sdca_blas_axpby(num_tasks, a, scores, -static_cast<Data>(rho), variables);
    a -= variables[0];
    std::for_each(first, last, [=](Data &x){ x += a; });

    // 2. Proximal step (project 'variables', use 'scores' as scratch space)
    prox_knapsack_le_biased(first, last,
      scores + 1, scores + num_tasks, lo, hi, rhs, rho, this->sum);

    // 3. Recover the updated variables
    *variables = static_cast<Data>(std::min(rhs,
      this->sum(first, last, static_cast<Result>(0)) ));
    std::for_each(first, last, [](Data &x){ x = -x; });
  }

  inline Result
  dual_loss(
      const blas_int num_tasks,
      const Data* variables
    ) const {
    Result d_loss(static_cast<Result>(variables[0]));
    d_loss += static_cast<Result>(0.5) * gamma_div_c * (
      d_loss * d_loss - static_cast<Result>(
      sdca_blas_dot(num_tasks, variables, variables)));
    return d_loss;
  }

  inline Result
  primal_loss(
      const blas_int num_tasks,
      Data* scores
    ) const {
    Data *first(scores + 1), *last(scores + num_tasks), a(1 - scores[0]);
    std::for_each(first, last, [=](Data &x){ x += a; });

    // loss = 1/gamma (<p,h> - 1/2 <p,p>), p =prox_{k,gamma}(h), h = c + a
    Result lo(0), hi(gamma_div_k), rhs(gamma);
    auto t = thresholds_knapsack_le(first, last, lo, hi, rhs, this->sum);
    Result ph = dot_prox(t, first, last, this->sum);
    Result pp = dot_prox_prox(t, first, last, this->sum);

    // (division by gamma happens later)
    return ph - static_cast<Result>(0.5) * pp;
  }
};

}

#endif
