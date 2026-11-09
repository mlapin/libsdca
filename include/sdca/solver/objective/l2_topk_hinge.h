#ifndef SDCA_SOLVER_OBJECTIVE_L2_TOPK_HINGE_H
#define SDCA_SOLVER_OBJECTIVE_L2_TOPK_HINGE_H

#include "sdca/prox/knapsack_le_biased.h"
#include "sdca/solver/objective/objective_base.h"
#include "sdca/utility/types.h"

namespace sdca {

template <typename Data, typename Result>
struct l2_topk_hinge : public objective_base<Data, Result> {
  typedef Data data_type;
  typedef Result result_type;

  typedef objective_base<Data, Result> base;

  const Result c;
  const size_type k;

  const Result c_div_k;

  l2_topk_hinge(const Result c_, const size_type k_)
      : base::objective_base(c_ / static_cast<Result>(k_)),
        c(c_),
        k(k_),
        c_div_k(c_ / static_cast<Result>(k_)) {
    if (!(c > 0) || !std::isfinite(c)) throw std::invalid_argument("C must be finite and positive");
    if (k < 1) throw std::invalid_argument("k must be positive");
  }

  // The same objective in another precision, for the evaluation.
  template <typename D, typename R>
  explicit l2_topk_hinge(const l2_topk_hinge<D, R>& other)
      : l2_topk_hinge(static_cast<Result>(other.c), other.k) {}

  inline std::string to_string() const {
    std::ostringstream str;
    str << "l2_topk_hinge (c: " << c << ", k: " << k << ", precision: " << type_name<Result>()
        << ")";
    return str.str();
  }

  // The projection_precision is accepted for uniformity; top-k projections
  // always use the hardened solver.
  template <typename Int>
  void update_dual_variables(const Int num_classes, const Data norm2, Data* variables, Data* scores,
                             const projection_precision = projection_precision::float64) const {
    detail::check_topk(k, num_classes);
    Data *first(variables + 1), *last(variables + num_classes);
    Result lo(0), hi(c_div_k), rhs(c), rho(1), zero(0);

    // 1. Prepare a vector to project in 'variables', relative to the label's
    // score: a common offset of the scores would erase the margins.
    using Work = detail::hinge_work<Data, Result>;
    detail::topk_hinge_point(num_classes, 1 / static_cast<Work>(norm2), Work(-1), scores,
                             variables);

    // 2. Proximal step (project 'variables', use 'scores' as scratch space)
    prox_knapsack_le_biased(first, last, scores + 1, lo, hi, rhs, rho);

    // 3. Recover the updated variables
    *variables = static_cast<Data>(std::min(rhs, detail::negate_and_sum(first, last, zero)));
  }

  template <typename Int>
  void zero_scores_dual_variables(const Int num_classes, Data* variables) const {
    // Every margin is one: the full mass c, spread evenly.
    detail::zero_scores_dual(num_classes, Int(1), c, variables);
  }

  template <typename Int, typename T>
  inline void feasible_dual(const Int num_classes, T* variables) const {
    feasible::capped(num_classes, variables, T(c) / T(k), T(c));
  }

  template <typename Int>
  inline Result primal_loss(const Int num_classes, Data* scores) const {
    detail::check_topk(k, num_classes);
    using Work = detail::hinge_work<Data, Result>;
    Data *first(scores + 1), *last(scores + num_classes), a(scores[0]);
    // Compensated margins 1 + x - a: a tiny one survives a large x - a.
    std::for_each(first, last,
                  [=](Data& x) { x = static_cast<Data>(detail::margin(Work(x), Work(a))); });

    // Find k largest elements
    std::nth_element(first, first + k - 1, last, std::greater<Data>());

    // sum_k_largest max{0, score_i} (division by k happens later)
    auto it = std::partition(first, first + k, [](Data x) { return x > 0; });
    return std::accumulate(first, it, static_cast<Result>(0));
  }
};

template <typename Data, typename Result>
struct l2_topk_hinge_smooth : public objective_base<Data, Result> {
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

  l2_topk_hinge_smooth(const Result c_, const size_type k_, const Result gamma_)
      : base::objective_base(c_),
        c(c_),
        gamma(gamma_),
        k(k_),
        c_div_k(c_ / static_cast<Result>(k_)),
        gamma_div_k(gamma_ / static_cast<Result>(k_)),
        gamma_div_c(gamma_ / c_),
        gamma_div_2c(gamma_ / (2 * c_)) {
    if (!(c > 0) || !std::isfinite(c)) throw std::invalid_argument("C must be finite and positive");
    if (!(gamma > 0) || !std::isfinite(gamma))
      throw std::invalid_argument("gamma must be finite and positive");
    if (k < 1) throw std::invalid_argument("k must be positive");
  }

  // The same objective in another precision, for the evaluation.
  template <typename D, typename R>
  explicit l2_topk_hinge_smooth(const l2_topk_hinge_smooth<D, R>& other)
      : l2_topk_hinge_smooth(static_cast<Result>(other.c), other.k,
                             static_cast<Result>(other.gamma)) {}

  inline std::string to_string() const {
    std::ostringstream str;
    str << "l2_topk_hinge (c: " << c << ", k: " << k << ", gamma: " << gamma
        << ", precision: " << type_name<Result>() << ")";
    return str.str();
  }

  // The projection_precision is accepted for uniformity; top-k projections
  // always use the hardened solver.
  template <typename Int>
  void update_dual_variables(const Int num_classes, const Data norm2, Data* variables, Data* scores,
                             const projection_precision = projection_precision::float64) const {
    detail::check_topk(k, num_classes);
    Data *first(variables + 1), *last(variables + num_classes);
    Result zero(0), hi(c_div_k), rhs(c),
        rho(static_cast<Result>(norm2) / (static_cast<Result>(norm2) + gamma_div_c));

    // 1. Prepare a vector to project in 'variables', relative to the label's
    // score: a common offset of the scores would erase the margins.
    using Work = detail::hinge_work<Data, Result>;
    detail::topk_hinge_point(num_classes, static_cast<Work>(rho) / static_cast<Work>(norm2),
                             -static_cast<Work>(rho), scores, variables);

    // 2. Proximal step (project 'variables', use 'scores' as scratch space)
    prox_knapsack_le_biased(first, last, scores + 1, zero, hi, rhs, rho);

    // 3. Recover the updated variables
    *variables = static_cast<Data>(std::min(rhs, detail::negate_and_sum(first, last, zero)));
  }

  template <typename Int>
  void zero_scores_dual_variables(const Int num_classes, Data* variables) const {
    // Even margins of one: mass s maximizes s - gamma s^2 / (2 c (m - 1)).
    const Result s = c * std::min(Result(1), Result(num_classes - 1) / gamma);
    detail::zero_scores_dual(num_classes, Int(1), s, variables);
  }

  template <typename Int, typename T>
  inline void feasible_dual(const Int num_classes, T* variables) const {
    feasible::capped(num_classes, variables, T(c) / T(k), T(c));
  }

  template <typename Int>
  inline Result primal_loss(const Int num_classes, Data* scores) const {
    detail::check_topk(k, num_classes);
    using Work = detail::hinge_work<Data, Result>;
    Data *first(scores + 1), *last(scores + num_classes), a(scores[0]);
    // Compensated margins 1 + x - a: a tiny one survives a large x - a.
    std::for_each(first, last,
                  [=](Data& x) { x = static_cast<Data>(detail::margin(Work(x), Work(a))); });

    // loss = 1/gamma (<h,p> - 1/2 <p,p>), p =prox_{k,gamma}(h), h = c + a
    Result lo(0), hi(gamma_div_k), rhs(gamma);
    auto t = thresholds_knapsack_le(first, last, lo, hi, rhs);
    Result hp = dot_x_prox(t, first, last);
    Result pp = dot_prox_prox(t, first, last);

    // Divided by gamma here, scaled term by term if the products overflow or underflow.
    return detail::smooth_hinge_normal(hp, pp / 2) ? (hp - pp / 2) / gamma
                                                   : dot_smooth_hinge(t, first, last, gamma);
  }

  template <typename Int>
  inline Result dual_loss(const Int num_classes, const Data* variables) const {
    return static_cast<Result>(variables[0]) -
           gamma_div_2c * static_cast<Result>(sdca_blas_dot(static_cast<blas_int>(num_classes - 1),
                                                            variables + 1, variables + 1));
  }
};

}  // namespace sdca

#endif
