#ifndef SDCA_SOLVER_OBJECTIVE_L2_MULTILABEL_HINGE_H
#define SDCA_SOLVER_OBJECTIVE_L2_MULTILABEL_HINGE_H

#include "sdca/prox/two_simplex.h"
#include "sdca/solver/objective/objective_base.h"
#include "sdca/utility/types.h"

namespace sdca {

template <typename Data, typename Result>
struct l2_multilabel_hinge : public objective_base<Data, Result> {
  typedef Data data_type;
  typedef Result result_type;

  typedef objective_base<Data, Result> base;

  const Result c;

  l2_multilabel_hinge(const Result c_) : base::objective_base(c_), c(c_) {
    if (!(c > 0) || !std::isfinite(c)) throw std::invalid_argument("C must be finite and positive");
  }

  // The same objective in another precision, for the evaluation.
  template <typename D, typename R>
  explicit l2_multilabel_hinge(const l2_multilabel_hinge<D, R>& other)
      : l2_multilabel_hinge(static_cast<Result>(other.c)) {}

  inline std::string to_string() const {
    std::ostringstream str;
    str << "l2_multilabel_hinge (c: " << c << ", precision: " << type_name<Result>() << ")";
    return str.str();
  }

  // mode: how the two-simplex projection solves (projection_precision).
  template <typename Int>
  void update_dual_variables(
      const Int num_classes, const Int num_labels, const Data norm2, Data* variables, Data* scores,
      const projection_precision mode = projection_precision::float64) const {
    Data *pos_first(variables), *pos_last(variables + num_labels);
    Data *neg_first(variables + num_labels), *neg_last(variables + num_classes);
    Data *pos_scores(scores), *neg_scores(scores + num_labels);

    // 1. Prepare a vector to project in 'variables', relative to the label's
    // score: a common offset of the scores would erase the margins.
    using Work = detail::hinge_work<Data, Result>;
    detail::multilabel_hinge_point(num_classes, num_labels, 1 / static_cast<Work>(norm2), Work(-1),
                                   scores, variables);

    // 2. Proximal step (project 'variables', use 'scores' as scratch space)
    prox_two_simplex(pos_first, pos_last, neg_first, neg_last, pos_scores, neg_scores, c, mode);

    // 3. Recover the updated variables
    for (Data* it = neg_first; it != neg_last; ++it) *it = -*it;
  }

  template <typename Int>
  void zero_scores_dual_variables(const Int num_classes, const Int num_labels,
                                  Data* variables) const {
    detail::zero_scores_dual(num_classes, num_labels, c, variables);
  }

  template <typename Int, typename T>
  inline void feasible_dual(const Int num_classes, const Int num_labels, T* variables) const {
    feasible::two_simplex(num_classes, num_labels, variables, T(c));
  }

  template <typename Int>
  inline Result primal_loss(const Int num_classes, const Int num_labels, Data* scores,
                            const projection_precision = projection_precision::float64) const {
    Data *pos_first(scores), *pos_last(scores + num_labels);
    Data *neg_first(scores + num_labels), *neg_last(scores + num_classes);

    Data min_pos = *std::min_element(pos_first, pos_last);
    Data max_neg = *std::max_element(neg_first, neg_last);

    // The compensated margin keeps a tiny 1 + max_neg - min_pos.
    using Work = detail::hinge_work<Data, Result>;
    Result zero(0), loss = static_cast<Result>(detail::margin<Work>(max_neg, min_pos));
    return std::max(zero, loss);
  }

  template <typename Int>
  inline Result dual_loss(const Int, const Int num_labels, const Data* variables) const {
    return std::accumulate(variables, variables + num_labels, static_cast<Result>(0));
  }
};

template <typename Data, typename Result>
struct l2_multilabel_hinge_smooth : public objective_base<Data, Result> {
  typedef Data data_type;
  typedef Result result_type;

  typedef objective_base<Data, Result> base;

  const Result c;
  const Result gamma;

  const Result gamma_div_c;

  l2_multilabel_hinge_smooth(const Result c_, const Result gamma_)
      : base::objective_base(c_), c(c_), gamma(gamma_), gamma_div_c(gamma_ / c_) {
    if (!(c > 0) || !std::isfinite(c)) throw std::invalid_argument("C must be finite and positive");
    if (!(gamma > 0) || !std::isfinite(gamma))
      throw std::invalid_argument("gamma must be finite and positive");
  }

  // The same objective in another precision, for the evaluation.
  template <typename D, typename R>
  explicit l2_multilabel_hinge_smooth(const l2_multilabel_hinge_smooth<D, R>& other)
      : l2_multilabel_hinge_smooth(static_cast<Result>(other.c), static_cast<Result>(other.gamma)) {
  }

  inline std::string to_string() const {
    std::ostringstream str;
    str << "l2_multilabel_hinge (c: " << c << ", gamma: " << gamma
        << ", precision: " << type_name<Result>() << ")";
    return str.str();
  }

  // mode: how the two-simplex projection solves (projection_precision).
  template <typename Int>
  void update_dual_variables(
      const Int num_classes, const Int num_labels, const Data norm2, Data* variables, Data* scores,
      const projection_precision mode = projection_precision::float64) const {
    Data *pos_first(variables), *pos_last(variables + num_labels);
    Data *neg_first(variables + num_labels), *neg_last(variables + num_classes);
    Data *pos_scores(scores), *neg_scores(scores + num_labels);
    using Work = detail::hinge_work<Data, Result>;
    const Work a = static_cast<Work>(1 / (static_cast<Result>(norm2) + gamma_div_c));
    const Work b =
        static_cast<Work>(static_cast<Result>(norm2) / (static_cast<Result>(norm2) + gamma_div_c));

    // 1. Prepare a vector to project in 'variables', relative to the label's
    // score: a common offset of the scores would erase the margins.
    detail::multilabel_hinge_point(num_classes, num_labels, a, -b, scores, variables);

    // 2. Proximal step (project 'variables', use 'scores' as scratch space)
    prox_two_simplex(pos_first, pos_last, neg_first, neg_last, pos_scores, neg_scores, c, mode);

    // 3. Recover the updated variables
    for (Data* it = neg_first; it != neg_last; ++it) *it = -*it;
  }

  template <typename Int>
  void zero_scores_dual_variables(const Int num_classes, const Int num_labels,
                                  Data* variables) const {
    // Mass s maximizes s - gamma s^2 m / (2 c p (m - p)) at zero scores.
    const Result p = Result(num_labels), m = Result(num_classes);
    const Result s = c * std::min(Result(1), p * (m - p) / (gamma * m));
    detail::zero_scores_dual(num_classes, num_labels, s, variables);
  }

  template <typename Int, typename T>
  inline void feasible_dual(const Int num_classes, const Int num_labels, T* variables) const {
    feasible::two_simplex(num_classes, num_labels, variables, T(c));
  }

  // mode: how the two-simplex projection of the smoothed loss solves.
  template <typename Int>
  inline Result primal_loss(const Int num_classes, const Int num_labels, Data* scores,
                            const projection_precision mode = projection_precision::float64) const {
    Data *pos_first(scores), *pos_last(scores + num_labels);
    Data *neg_first(scores + num_labels), *neg_last(scores + num_classes);

    // Margins 1/2 - (x - z_0) and (x - z_0) + 1/2, or, as in the dual step, the
    // largest pairwise margin in one entry when it nearly cancels.
    using Work = detail::hinge_work<Data, Result>;
    const Work a(0.5), origin = Work(scores[0]);
    const Work low = static_cast<Work>(*std::min_element(pos_first, pos_last));
    const Work high = static_cast<Work>(*std::max_element(neg_first, neg_last));
    if (std::abs((high - low) + 1) < detail::margin_cancellation<Work>() *
                                         (1 + std::abs(low - origin) + std::abs(high - origin))) {
      std::for_each(pos_first, pos_last,
                    [=](Data& x) { x = static_cast<Data>(low - static_cast<Work>(x)); });
      std::for_each(neg_first, neg_last,
                    [=](Data& x) { x = static_cast<Data>(detail::margin(Work(x), low)); });
    } else {
      std::for_each(pos_first, pos_last,
                    [=](Data& x) { x = static_cast<Data>(a - (static_cast<Work>(x) - origin)); });
      std::for_each(neg_first, neg_last,
                    [=](Data& x) { x = static_cast<Data>((static_cast<Work>(x) - origin) + a); });
    }

    auto t = thresholds_two_simplex(pos_first, pos_last, neg_first, neg_last, gamma, mode);

    Result pos_hp = dot_x_prox(t.first, pos_first, pos_last);
    Result pos_pp = dot_prox_prox(t.first, pos_first, pos_last);
    Result neg_hp = dot_x_prox(t.second, neg_first, neg_last);
    Result neg_pp = dot_prox_prox(t.second, neg_first, neg_last);

    // Divided by gamma here, scaled term by term if the products overflow or underflow.
    const Result hp = pos_hp + neg_hp, pp = (pos_pp + neg_pp) / 2;
    return detail::smooth_hinge_normal(hp, pp)
               ? (hp - pp) / gamma
               : dot_smooth_hinge(t.first, pos_first, pos_last, gamma) +
                     dot_smooth_hinge(t.second, neg_first, neg_last, gamma);
  }

  template <typename Int>
  inline Result dual_loss(const Int num_classes, const Int num_labels,
                          const Data* variables) const {
    Result loss1 = std::accumulate(variables, variables + num_labels, static_cast<Result>(0));
    Result loss2 =
        std::accumulate(variables + num_labels, variables + num_classes, static_cast<Result>(0));
    Result smoothing = gamma_div_c * static_cast<Result>(sdca_blas_dot(
                                         static_cast<blas_int>(num_classes), variables, variables));
    return static_cast<Result>(0.5) * (loss1 - loss2 - smoothing);
  }
};

}  // namespace sdca

#endif
