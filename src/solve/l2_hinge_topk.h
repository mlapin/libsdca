#ifndef SDCA_SOLVE_L2_HINGE_TOPK_LOSS_H
#define SDCA_SOLVE_L2_HINGE_TOPK_LOSS_H

#include "prox/topk_simplex.h"
#include "prox/topk_simplex_biased.h"
#include "solvedef.h"
#include "util/util.h"

namespace sdca {

template <typename Data,
          typename Result,
          typename Summation>
struct l2_hinge_topk {
  const difference_type k;
  const Result c;
  const Result c_div_k;
  const Summation sum;

  l2_hinge_topk(
      const size_type __k,
      const Result __c,
      const Summation __sum
    ) :
      k(static_cast<difference_type>(__k)),
      c(__c),
      c_div_k(__c / static_cast<Result>(__k)),
      sum(__sum)
  {}

  inline std::string to_string() const {
    std::ostringstream str;
    str << "l2_hinge_topk (k = " << k << ", C = " << c << ", gamma = 0)";
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
      const Data norm2_inv,
      Data* variables,
      Data* scores
      ) const {
    Data *first(variables + 1), *last(variables + num_tasks);
    Result rhs(c), rho(1);

    // 1. Prepare a vector to project in 'variables'.
    Data a(norm2_inv);
    sdca_blas_axpby(num_tasks, a, scores, -1, variables);
    a -= variables[0];
    std::for_each(first, last, [=](Data &x){ x += a; });

    // 2. Proximal step (project 'variables', use 'scores' as scratch space)
    prox_topk_simplex_biased(first, last,
      scores + 1, scores + num_tasks, k, rhs, rho, sum);

    // 3. Recover the updated variables
    *variables = static_cast<Data>(std::min(rhs,
      sum(first, last, static_cast<Result>(0)) ));
    std::for_each(first, last, [](Data &x){ x = -x; });
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
    dual_loss = static_cast<Result>(variables[0]);

    Data *first(scores + 1), *last(scores + num_tasks);
    Data a(1 - scores[0]);
    std::for_each(scores, scores + num_tasks, [=](Data &x){ x += a; });

    // Find k largest elements
    std::nth_element(first, first + k - 1, last, std::greater<Data>());

    // max{0, sum_k_largest} (division by k happens later)
    primal_loss = std::max(static_cast<Result>(0),
      sum(first, first + k, static_cast<Result>(0)));
  }

  inline void primal_dual_gap(
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

template <typename Data,
          typename Result,
          typename Summation>
struct l2_hinge_topk_smooth {
  const difference_type k;
  const Result c;
  const Result gamma;
  const Result c_div_gamma;
  const Result gamma_div_c;
  const Summation sum;

  l2_hinge_topk_smooth(
      const size_type __k,
      const Result __c,
      const Result __gamma,
      const Summation __sum
    ) :
      k(static_cast<difference_type>(__k)),
      c(__c),
      gamma(__gamma),
      c_div_gamma(__c / __gamma),
      gamma_div_c(__gamma / __c),
      sum(__sum)
  {}

  inline std::string to_string() const {
    std::ostringstream str;
    str << "l2_hinge_topk (k = " << k << ", C = " << c << ", "
      "gamma = " << gamma << ")";
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
      const Data norm2_inv,
      Data* variables,
      Data* scores
      ) const {
    Data *first(variables + 1), *last(variables + num_tasks);
    Result rhs(c), rho(1 / (1 + gamma_div_c * static_cast<Result>(norm2_inv)));

    // 1. Prepare a vector to project in 'variables'.
    Data a(norm2_inv * static_cast<Data>(rho));
    sdca_blas_axpby(num_tasks, a, scores, -static_cast<Data>(rho), variables);
    a -= variables[0];
    std::for_each(first, last, [=](Data &x){ x += a; });

    // 2. Proximal step (project 'variables', use 'scores' as scratch space)
    prox_topk_simplex_biased(first, last,
      scores + 1, scores + num_tasks, k, rhs, rho, sum);

    // 3. Recover the updated variables
    *variables = static_cast<Data>(std::min(rhs,
      sum(first, last, static_cast<Result>(0)) ));
    std::for_each(first, last, [](Data &x){ x = -x; });
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

    dual_loss = static_cast<Result>(variables[0]);
    dual_loss += static_cast<Result>(0.5) * gamma_div_c * (
      dual_loss * dual_loss - static_cast<Result>(
      sdca_blas_dot(num_tasks, variables, variables)));

    Data *first(scores + 1), *last(scores + num_tasks);
    Data a(1 - scores[0]);
    std::for_each(first, last, [=](Data &x){ x += a; });

    // loss = 1/gamma (<p,h> - 1/2 <p,p>), p =prox_{k,gamma}(h), h = c + a
    auto t = thresholds_topk_simplex(first, last, k, gamma, sum);
    Result ph = dot_prox(t, first, last, sum);
    Result pp = dot_prox_prox(t, first, last, sum);

    // (division by gamma happens later)
    primal_loss = ph - static_cast<Result>(0.5) * pp;
  }

  inline void primal_dual_gap(
      const Result regularizer,
      const Result primal_loss,
      const Result dual_loss,
      Result &primal_objective,
      Result &dual_objective,
      Result &duality_gap
    ) const {
    primal_objective = c_div_gamma * primal_loss;
    dual_objective = dual_loss;
    duality_gap = primal_objective - dual_objective + regularizer;
    primal_objective += static_cast<Result>(0.5) * regularizer;
    dual_objective -= static_cast<Result>(0.5) * regularizer;
  }
};

}

#endif
