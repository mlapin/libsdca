#ifndef SDCA_SOLVER_OBJECTIVE_HINGE_H
#define SDCA_SOLVER_OBJECTIVE_HINGE_H

#include <numeric>
#include <string>

#include "sdca/solver/objective/objective_base.h"
#include "sdca/core/lanes.h"

SDCA_ISA_BEGIN

namespace solver_detail {

// The top-k hinge objectives (alpha: the top-k simplex, beta: the capped simplex), smoothed
// for gamma > 0. With gamma = 0, rho = 1 and every step is that of the nonsmooth hinge.
template <typename Data, typename Result, bool Alpha>
struct topk_hinge_objective : objective_base<Data, Result> {
  typedef objective_base<Data, Result> base;

  static const bool shrinks = true;
  static const bool part_steps = true;

  const Result c;
  const Result gamma;
  const size_type k;

  topk_hinge_objective(const Result c_, const size_type k_, const Result gamma_)
      : base(spec_of(c_, k_, gamma_)), c(c_), gamma(gamma_), k(k_) {}
  static objective_spec spec_of(const Result c_, const size_type k_, const Result gamma_) {
    const objective_spec s = {Alpha ? kernel::topk_hinge_alpha : kernel::topk_hinge_beta,
                              double(c_), k_, double(gamma_)};
    return s;
  }

  std::string describe(const char* name) const {
    std::ostringstream str;
    str << name << " (c: " << c << ", k: " << k;
    if (gamma > 0) str << ", gamma: " << gamma;
    str << ", precision: " << type_name<Result>() << ")";
    return str.str();
  }

  // Examples whose margins are all negative have zero blocks; with k = 1 (or beta), so do
  // classes of negative margin; without smoothing at k = 1, classes below another margin.
  screening_rules screening() const {
    screening_rules r;
    r.examples = true;
    r.classes = !Alpha || k == 1;
    r.pairs = k == 1 && gamma == 0;
    return r;
  }

  template <typename Int>
  void update_dual_variables(const Int num_classes, const Data norm2, Data* variables,
                             Data* scores) const {
    this->check_classes(num_classes);
    typedef hinge_work<Data, Result> Work;
    Data *first = variables + 1, *last = variables + num_classes;
    const Result rhs(c), zero(0),
        rho = gamma == 0 ? Result(1) : Result(norm2) / (Result(norm2) + gamma / c);
    // The point to project, relative to the label's score: a common offset of the scores
    // would erase the margins.
    topk_hinge_point(num_classes, Work(rho) / Work(norm2), -Work(rho), scores, variables);
    // The proximal step on the competitors, with the scores as scratch.
    Data *aux = scores + 1, *aux_last = std::copy(first, last, aux);
    const typename std::iterator_traits<Data*>::difference_type kk(k);
    if (Alpha)
      core::apply(core::thresholds_topk_simplex_biased<Result>(aux, aux_last, kk, rhs, rho), first,
                  last);
    else
      core::apply(
          core::thresholds_knapsack_le_biased<Result>(aux, aux_last, zero, c / Result(k), rhs, rho),
          first, last);
    *variables = static_cast<Data>(std::min(rhs, negate_and_sum(first, last, zero)));
  }

  template <typename Int>
  void zero_scores_dual_variables(const Int num_classes, Data* variables) const {
    // Even margins of one: the mass s maximizes s - gamma s^2 / (2 c (m - 1)).
    const Result s = gamma == 0 ? c : c * std::min(Result(1), Result(num_classes - 1) / gamma);
    zero_scores_dual(num_classes, Int(1), s, variables);
  }

  template <typename Int, typename T>
  inline void feasible_dual(const Int num_classes, T* variables) const {
    if (Alpha)
      feasible::topk_simplex(num_classes, variables, k, T(c));
    else
      feasible::capped(num_classes, variables, T(c) / T(k), T(c));
  }

  template <typename Int>
  inline Result dual_loss(const Int num_classes, const Data* variables) const {
    const Result value = static_cast<Result>(variables[0]);
    if (gamma == 0) return value;
    return value - gamma / (2 * c) *
                       static_cast<Result>(sdca_blas_dot(static_cast<blas_int>(num_classes - 1),
                                                         variables + 1, variables + 1));
  }
};

template <typename Data, typename Result, bool Alpha>
const bool topk_hinge_objective<Data, Result, Alpha>::shrinks;
template <typename Data, typename Result, bool Alpha>
const bool topk_hinge_objective<Data, Result, Alpha>::part_steps;

// The multilabel hinge objectives (the two-simplex), smoothed for gamma > 0.
template <typename Data, typename Result>
struct multilabel_hinge_objective : objective_base<Data, Result> {
  typedef objective_base<Data, Result> base;

  static const bool shrinks = true;

  const Result c;
  const Result gamma;

  multilabel_hinge_objective(const Result c_, const Result gamma_)
      : base(spec_of(c_, gamma_)), c(c_), gamma(gamma_) {}
  static objective_spec spec_of(const Result c_, const Result gamma_) {
    const objective_spec s = {kernel::multilabel_hinge, double(c_), 0, double(gamma_)};
    return s;
  }

  std::string describe(const char* name) const {
    std::ostringstream str;
    str << name << " (c: " << c;
    if (gamma > 0) str << ", gamma: " << gamma;
    str << ", precision: " << type_name<Result>() << ")";
    return str.str();
  }

  template <typename Int>
  void update_dual_variables(const Int num_classes, const Int num_labels, const Data norm2,
                             Data* variables, Data* scores) const {
    typedef hinge_work<Data, Result> Work;
    const Work a = gamma == 0 ? 1 / Work(norm2) : Work(1 / (Result(norm2) + gamma / c)),
               b = gamma == 0 ? Work(1) : Work(Result(norm2) / (Result(norm2) + gamma / c));
    // The point to project, relative to the label's score.
    multilabel_hinge_point(num_classes, num_labels, a, -b, scores, variables);
    // The proximal step, with the scores as scratch.
    Data *middle = variables + num_labels, *last = variables + num_classes;
    std::copy(variables, last, scores);
    const auto t = core::thresholds_two_simplex<Result>(
        scores, scores + num_labels, scores + num_labels, scores + num_classes, c);
    core::apply(t.first, variables, middle);
    core::apply(t.second, middle, last);
    for (Data* it = middle; it != last; ++it) *it = -*it;
  }

  template <typename Int>
  void zero_scores_dual_variables(const Int num_classes, const Int num_labels,
                                  Data* variables) const {
    // The mass s maximizes s - gamma s^2 m / (2 c p (m - p)) at zero scores.
    const Result p = Result(num_labels), m = Result(num_classes);
    const Result s = gamma == 0 ? c : c * std::min(Result(1), p * (m - p) / (gamma * m));
    zero_scores_dual(num_classes, num_labels, s, variables);
  }

  template <typename Int, typename T>
  inline void feasible_dual(const Int num_classes, const Int num_labels, T* variables) const {
    feasible::two_simplex(num_classes, num_labels, variables, T(c));
  }

  template <typename Int>
  inline Result dual_loss(const Int num_classes, const Int num_labels,
                          const Data* variables) const {
    const Result labels =
        std::accumulate(variables, variables + num_labels, static_cast<Result>(0));
    if (gamma == 0) return labels;
    const Result others =
        std::accumulate(variables + num_labels, variables + num_classes, static_cast<Result>(0));
    const Result smoothing = gamma / c *
                             static_cast<Result>(sdca_blas_dot(static_cast<blas_int>(num_classes),
                                                               variables, variables));
    return Result(0.5) * (labels - others - smoothing);
  }
};

template <typename Data, typename Result>
const bool multilabel_hinge_objective<Data, Result>::shrinks;

}  // namespace solver_detail

template <typename Data, typename Result>
struct l2_topk_hinge_alpha : solver_detail::topk_hinge_objective<Data, Result, true> {
  typedef Data data_type;
  typedef Result result_type;
  typedef solver_detail::topk_hinge_objective<Data, Result, true> base;
  l2_topk_hinge_alpha(const Result c_, const size_type k_) : base(c_, k_, Result(0)) {}
  template <typename D, typename R>
  explicit l2_topk_hinge_alpha(const l2_topk_hinge_alpha<D, R>& other)
      : base(Result(other.c), other.k, Result(0)) {}
  std::string to_string() const { return this->describe("l2_topk_hinge_alpha"); }
};

template <typename Data, typename Result>
struct l2_topk_hinge_alpha_smooth : solver_detail::topk_hinge_objective<Data, Result, true> {
  typedef Data data_type;
  typedef Result result_type;
  typedef solver_detail::topk_hinge_objective<Data, Result, true> base;
  l2_topk_hinge_alpha_smooth(const Result c_, const size_type k_, const Result gamma_)
      : base(c_, k_, gamma_) {}
  template <typename D, typename R>
  explicit l2_topk_hinge_alpha_smooth(const l2_topk_hinge_alpha_smooth<D, R>& other)
      : base(Result(other.c), other.k, Result(other.gamma)) {}
  std::string to_string() const { return this->describe("l2_topk_hinge_alpha"); }
};

template <typename Data, typename Result>
struct l2_topk_hinge_beta : solver_detail::topk_hinge_objective<Data, Result, false> {
  typedef Data data_type;
  typedef Result result_type;
  typedef solver_detail::topk_hinge_objective<Data, Result, false> base;
  l2_topk_hinge_beta(const Result c_, const size_type k_) : base(c_, k_, Result(0)) {}
  template <typename D, typename R>
  explicit l2_topk_hinge_beta(const l2_topk_hinge_beta<D, R>& other)
      : base(Result(other.c), other.k, Result(0)) {}
  std::string to_string() const { return this->describe("l2_topk_hinge_beta"); }
};

template <typename Data, typename Result>
struct l2_topk_hinge_beta_smooth : solver_detail::topk_hinge_objective<Data, Result, false> {
  typedef Data data_type;
  typedef Result result_type;
  typedef solver_detail::topk_hinge_objective<Data, Result, false> base;
  l2_topk_hinge_beta_smooth(const Result c_, const size_type k_, const Result gamma_)
      : base(c_, k_, gamma_) {}
  template <typename D, typename R>
  explicit l2_topk_hinge_beta_smooth(const l2_topk_hinge_beta_smooth<D, R>& other)
      : base(Result(other.c), other.k, Result(other.gamma)) {}
  std::string to_string() const { return this->describe("l2_topk_hinge_beta"); }
};

template <typename Data, typename Result>
struct l2_multilabel_hinge : solver_detail::multilabel_hinge_objective<Data, Result> {
  typedef Data data_type;
  typedef Result result_type;
  typedef solver_detail::multilabel_hinge_objective<Data, Result> base;
  explicit l2_multilabel_hinge(const Result c_) : base(c_, Result(0)) {}
  template <typename D, typename R>
  explicit l2_multilabel_hinge(const l2_multilabel_hinge<D, R>& other)
      : base(Result(other.c), Result(0)) {}
  std::string to_string() const { return this->describe("l2_multilabel_hinge"); }
};

template <typename Data, typename Result>
struct l2_multilabel_hinge_smooth : solver_detail::multilabel_hinge_objective<Data, Result> {
  typedef Data data_type;
  typedef Result result_type;
  typedef solver_detail::multilabel_hinge_objective<Data, Result> base;
  l2_multilabel_hinge_smooth(const Result c_, const Result gamma_) : base(c_, gamma_) {}
  template <typename D, typename R>
  explicit l2_multilabel_hinge_smooth(const l2_multilabel_hinge_smooth<D, R>& other)
      : base(Result(other.c), Result(other.gamma)) {}
  std::string to_string() const { return this->describe("l2_multilabel_hinge"); }
};

SDCA_ISA_END

#endif
