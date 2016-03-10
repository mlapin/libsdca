#ifndef SDCA_SOLVER_OBJECTIVE_L2_TOPK_HINGE_H
#define SDCA_SOLVER_OBJECTIVE_L2_TOPK_HINGE_H

#include "sdca/types.h"
#include "sdca/prox/knapsack_le_biased.h"
#include "sdca/solver/objective/objective_base.h"

namespace sdca {

template <typename Data,
          typename Result>
struct l2_topk_hinge
    : public objective_base<Data, Result> {

  typedef Data data_type;
  typedef Result result_type;

  typedef objective_base<Data, Result> base;

  const Result c;
  const size_type k;

  const Result c_div_k;


  l2_topk_hinge(
      const Result __c,
      const size_type __k
    ) :
      base::objective_base(__c / static_cast<Result>(__k)),
      c(__c),
      k(__k),
      c_div_k(__c / static_cast<Result>(__k))
  {}


  inline std::string to_string() const {
    std::ostringstream str;
    str << "l2_topk_hinge (c = " << c << ", gamma = 0, k = " << k << ")";
    return str.str();
  }


  template <typename Int>
  void update_dual_variables(
      const Int num_classes,
      const Data norm2,
      Data* variables,
      Data* scores
      ) const {
    Data *first(variables + 1), *last(variables + num_classes), a(1 / norm2);
    Result lo(0), hi(c_div_k), rhs(c), rho(1), zero(0);

    // 1. Prepare a vector to project in 'variables'.
    sdca_blas_axpby(
      static_cast<blas_int>(num_classes), a, scores, -1, variables);
    a -= variables[0];
    std::for_each(first, last, [=](Data &x){ x += a; });

    // 2. Proximal step (project 'variables', use 'scores' as scratch space)
    prox_knapsack_le_biased(first, last,
      scores + 1, scores + num_classes, lo, hi, rhs, rho);

    // 3. Recover the updated variables
    *variables = static_cast<Data>(
      std::min(rhs, std::accumulate(first, last, zero)));
    sdca_blas_scal(
      static_cast<blas_int>(num_classes - 1), static_cast<Data>(-1), first);
  }


  template <typename Int>
  inline Result
  primal_loss(
      const Int num_classes,
      Data* scores
    ) const {
    Data *first(scores + 1), *last(scores + num_classes), a(1 - scores[0]);
    std::for_each(first, last, [=](Data &x){ x += a; });

    // Find k largest elements
    std::nth_element(first, first + k - 1, last, std::greater<Data>());

    // sum_k_largest max{0, score_i} (division by k happens later)
    auto it = std::partition(first, first + k, [](Data x){ return x > 0; });
    return std::accumulate(first, it, static_cast<Result>(0));
  }
};


template <typename Data,
          typename Result>
struct l2_topk_hinge_smooth
    : public objective_base<Data, Result> {

  typedef Data data_type;
  typedef Result result_type;

  typedef objective_base<Data, Result> base;

  const Result c;
  const Result gamma;
  const size_type k;

  const Result c_div_k;
  const Result gamma_div_k;
  const Result gamma_div_c;
  const Result gamma_div_2c;


  l2_topk_hinge_smooth(
      const Result __c,
      const Result __gamma,
      const size_type __k
    ) :
      base::objective_base(__c / __gamma),
      c(__c),
      gamma(__gamma),
      k(__k),
      c_div_k(__c / static_cast<Result>(__k)),
      gamma_div_k(__gamma / static_cast<Result>(__k)),
      gamma_div_c(__gamma / __c),
      gamma_div_2c(__gamma / (2 * __c))
  {}


  inline std::string to_string() const {
    std::ostringstream str;
    str << "l2_topk_hinge (c = " << c << ", gamma = " << gamma << ", "
      "k = " << k << ")";
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
    Result zero(0), hi(c_div_k), rhs(c), rho(static_cast<Result>(norm2) /
      (static_cast<Result>(norm2) + gamma_div_c));

    // 1. Prepare a vector to project in 'variables'.
    Data a(static_cast<Data>(rho) / norm2);
    sdca_blas_axpby(
      static_cast<blas_int>(num_classes), a, scores,
      -static_cast<Data>(rho), variables);
    a -= variables[0];
    std::for_each(first, last, [=](Data &x){ x += a; });

    // 2. Proximal step (project 'variables', use 'scores' as scratch space)
    prox_knapsack_le_biased(first, last,
      scores + 1, scores + num_classes, zero, hi, rhs, rho);

    // 3. Recover the updated variables
    *variables = static_cast<Data>(
      std::min(rhs, std::accumulate(first, last, zero) ));
    sdca_blas_scal(
      static_cast<blas_int>(num_classes - 1), static_cast<Data>(-1), first);
  }


  template <typename Int>
  inline Result
  dual_loss(
      const Int num_classes,
      const Data* variables
    ) const {
    return static_cast<Result>(variables[0])
      - gamma_div_2c * static_cast<Result>(
      sdca_blas_dot(
        static_cast<blas_int>(num_classes - 1), variables + 1, variables + 1));
  }


  template <typename Int>
  inline Result
  primal_loss(
      const Int num_classes,
      Data* scores
    ) const {
    Data *first(scores + 1), *last(scores + num_classes), a(1 - scores[0]);
    std::for_each(first, last, [=](Data &x){ x += a; });

    // loss = 1/gamma (<h,p> - 1/2 <p,p>), p =prox_{k,gamma}(h), h = c + a
    Result lo(0), hi(gamma_div_k), rhs(gamma);
    auto t = thresholds_knapsack_le(first, last, lo, hi, rhs);
    Result hp = dot_x_prox(t, first, last);
    Result pp = dot_prox_prox(t, first, last);

    // (division by gamma happens later)
    return hp - static_cast<Result>(0.5) * pp;
  }

};

}

#endif
