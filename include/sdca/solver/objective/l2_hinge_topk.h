#ifndef SDCA_SOLVER_OBJECTIVE_L2_HINGE_TOPK_H
#define SDCA_SOLVER_OBJECTIVE_L2_HINGE_TOPK_H

#include "sdca/prox/topk_simplex.h"
#include "sdca/prox/topk_simplex_biased.h"
#include "sdca/solver/solverdef.h"
#include "sdca/solver/objective/base_objective.h"

namespace sdca {

template <typename Data,
          typename Result>
struct l2_hinge_topk
    : public base_objective<Data, Result> {

  typedef base_objective<Data, Result> base;

  const Result c;
  const size_type k;


  l2_hinge_topk(
      const Result __c,
      const size_type __k
    ) :
      base::base_objective(__c / static_cast<Result>(__k)),
      c(__c),
      k(__k)
  {}


  inline std::string to_string() const {
    std::ostringstream str;
    str << "l2_hinge_topk (c = " << c << ", gamma = 0, k = " << k << ")";
    return str.str();
  }


  void update_dual_variables(
      const blas_int num_classes,
      const Data norm2,
      Data* variables,
      Data* scores
      ) const {
    Data *first(variables + 1), *last(variables + num_classes), a(1 / norm2);
    Result rhs(c), rho(1), zero(0);

    // 1. Prepare a vector to project in 'variables'.
    sdca_blas_axpby(num_classes, a, scores, -1, variables);
    a -= variables[0];
    std::for_each(first, last, [=](Data &x){ x += a; });

    // 2. Proximal step (project 'variables', use 'scores' as scratch space)
    prox_topk_simplex_biased(first, last,
      scores + 1, scores + num_classes, k, rhs, rho);

    // 3. Recover the updated variables
    *variables = static_cast<Data>(
      std::min(rhs, std::accumulate(first, last, zero)));
    sdca_blas_scal(num_classes - 1, static_cast<Data>(-1), first);
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

    // max{0, sum_k_largest} (division by k happens later)
    Result zero(0);
    return std::max(zero, std::accumulate(first, first + k, zero));
  }

};


template <typename Data,
          typename Result>
struct l2_hinge_topk_smooth
    : public base_objective<Data, Result> {

  typedef base_objective<Data, Result> base;

  const Result c;
  const Result gamma;
  const size_type k;

  const Result c_div_gamma;
  const Result gamma_div_c;
  const Result gamma_div_2c;


  l2_hinge_topk_smooth(
      const Result __c,
      const Result __gamma,
      const size_type __k
    ) :
      base::base_objective(__c / __gamma),
      c(__c),
      gamma(__gamma),
      k(__k),
      c_div_gamma(__c / __gamma),
      gamma_div_c(__gamma / __c),
      gamma_div_2c(__gamma / (2 * __c))
  {}


  inline std::string to_string() const {
    std::ostringstream str;
    str << "l2_hinge_topk (c = " << c << ", gamma = " << gamma << ", "
      "k = " << k << ")";
    return str.str();
  }


  void update_dual_variables(
      const blas_int num_classes,
      const Data norm2,
      Data* variables,
      Data* scores
      ) const {
    Data *first(variables + 1), *last(variables + num_classes);
    Result rhs(c), zero(0), rho(static_cast<Result>(norm2) /
      (static_cast<Result>(norm2) + gamma_div_c));

    // 1. Prepare a vector to project in 'variables'.
    Data a(static_cast<Data>(rho) / norm2);
    sdca_blas_axpby(num_classes, a, scores, -static_cast<Data>(rho), variables);
    a -= variables[0];
    std::for_each(first, last, [=](Data &x){ x += a; });

    // 2. Proximal step (project 'variables', use 'scores' as scratch space)
    prox_topk_simplex_biased(first, last,
      scores + 1, scores + num_classes, k, rhs, rho);

    // 3. Recover the updated variables
    *variables = static_cast<Data>(
      std::min(rhs, std::accumulate(first, last, zero)));
    sdca_blas_scal(num_classes - 1, static_cast<Data>(-1), first);
  }


  inline Result
  dual_loss(
      const blas_int num_classes,
      const Data* variables
    ) const {
    return static_cast<Result>(variables[0])
      - gamma_div_2c * static_cast<Result>(
      sdca_blas_dot(num_classes - 1, variables + 1, variables + 1));
  }


  inline Result
  primal_loss(
      const blas_int num_classes,
      Data* scores
    ) const {
    Data *first(scores + 1), *last(scores + num_classes), a(1 - scores[0]);
    std::for_each(first, last, [=](Data &x){ x += a; });

    // loss = 1/gamma (<h,p> - 1/2 <p,p>), p =prox_{k,gamma}(h), h = c + a
    auto t = thresholds_topk_simplex(first, last, k, gamma);
    Result hp = dot_x_prox(t, first, last);
    Result pp = dot_prox_prox(t, first, last);

    // (division by gamma happens later)
    return hp - static_cast<Result>(0.5) * pp;
  }

};

}

#endif
