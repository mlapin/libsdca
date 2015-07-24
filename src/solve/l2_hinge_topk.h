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
  Summation sum;

  l2_hinge_topk(
      const size_type __k,
      const Result __c,
      Summation __sum
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
    Result rhs = c, rho = 1;
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

    // Project onto the feasible set
    project_topk_simplex_biased(variables, variables_back,
      scores, scores_back, k, rhs, rho, sum);

    // The last one is the sum
    *variables_back = static_cast<Data>(std::min(rhs,
      sum(variables, variables_back, static_cast<Result>(0)) ));

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
      Result &regularizer,
      Result &primal_loss,
      Result &dual_loss
    ) const {
    regularizer = static_cast<Result>(
      sdca_blas_dot(num_tasks, scores, variables));
    dual_loss = static_cast<Result>(variables[label]);

    Data a = static_cast<Data>(1) - scores[label];
    std::for_each(scores, scores + num_tasks, [&](Data &x){ x += a; });
    scores[label] = static_cast<Data>(0);

    // Find k largest elements
    std::nth_element(scores, scores + k - 1, scores + num_tasks,
      std::greater<Data>());

    // max{0, sum_k_largest} (division by k happens later)
    primal_loss = std::max(static_cast<Result>(0),
      sum(scores, scores + k, static_cast<Result>(0)));
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
  Summation sum;

  l2_hinge_topk_smooth(
      const size_type __k,
      const Result __c,
      const Result __gamma,
      Summation __sum
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
    Result rhs = c, rho = static_cast<Result>(1) /
      (static_cast<Result>(1) + gamma_div_c * static_cast<Result>(norm2_inv));
    Data a = norm2_inv * static_cast<Data>(rho);
    sdca_blas_axpby(num_tasks, a, scores, -static_cast<Data>(rho), variables);

    // Place ground truth at the back
    Data *scores_back = scores + num_tasks - 1;
    Data *variables_back = variables + num_tasks - 1;
    std::swap(*scores_back, scores[label]);
    std::swap(*variables_back, variables[label]);

    // Update variables
    a -= *variables_back;
    std::for_each(variables, variables_back, [&](Data &x){ x += a; });

    // Project onto the feasible set
    project_topk_simplex_biased(variables, variables_back,
      scores, scores_back, k, rhs, rho, sum);

    // The last one is the sum
    *variables_back = static_cast<Data>(std::min(rhs,
      sum(variables, variables_back, static_cast<Result>(0)) ));

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
      Result &regularizer,
      Result &primal_loss,
      Result &dual_loss
    ) const {
    regularizer = static_cast<Result>(
      sdca_blas_dot(num_tasks, scores, variables));

    dual_loss = static_cast<Result>(variables[label]);
    dual_loss += static_cast<Result>(0.5) * gamma_div_c * (
      dual_loss * dual_loss - static_cast<Result>(
      sdca_blas_dot(num_tasks, variables, variables)));

    Data a = static_cast<Data>(1) - scores[label];
    std::for_each(scores, scores + num_tasks, [&](Data &x){ x += a; });
    scores[label] = static_cast<Data>(0);

    // loss = 1/gamma (<p,h> - 1/2 <p,p>), p =proj_{k,gamma}(h), h = c + a
    auto t = thresholds_topk_simplex(scores, scores + num_tasks, k, gamma, sum);
    Result ph = correlation(t, scores, scores + num_tasks, sum);
    Result pp = norm2(t, scores, scores + num_tasks, sum);

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
