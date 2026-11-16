#ifndef SDCA_SOLVER_OBJECTIVE_H
#define SDCA_SOLVER_OBJECTIVE_H

#include <cmath>
#include <stdexcept>
#include <string>

#include "sdca/loss_kind.h"
#include "sdca/solver/objective/l2_distillation.h"
#include "sdca/solver/objective/l2_cross_entropy.h"
#include "sdca/solver/objective/l2_topk_cross_entropy.h"
#include "sdca/solver/objective/l2_topk_hinge_alpha.h"
#include "sdca/solver/objective/l2_multilabel_cross_entropy.h"
#include "sdca/solver/objective/l2_multilabel_hinge.h"
#include "sdca/solver/objective/l2_topk_hinge_beta.h"

namespace sdca {

template <typename Objective>
struct has_param_k : std::false_type {};

template <typename Data, typename Result>
struct has_param_k<l2_topk_cross_entropy<Data, Result>> : std::true_type {};

template <typename Data, typename Result>
struct has_param_k<l2_topk_hinge_alpha<Data, Result>> : std::true_type {};

template <typename Data, typename Result>
struct has_param_k<l2_topk_hinge_alpha_smooth<Data, Result>> : std::true_type {};

template <typename Data, typename Result>
struct has_param_k<l2_topk_hinge_beta<Data, Result>> : std::true_type {};

template <typename Data, typename Result>
struct has_param_k<l2_topk_hinge_beta_smooth<Data, Result>> : std::true_type {};

template <typename Objective>
struct has_param_gamma : std::false_type {};

template <typename Data, typename Result>
struct has_param_gamma<l2_topk_hinge_alpha_smooth<Data, Result>> : std::true_type {};

template <typename Data, typename Result>
struct has_param_gamma<l2_topk_hinge_beta_smooth<Data, Result>> : std::true_type {};

template <typename Data, typename Result>
struct has_param_gamma<l2_multilabel_hinge_smooth<Data, Result>> : std::true_type {};

template <typename Data, typename Result = Data>
inline l2_cross_entropy<Data, Result> make_objective_l2_cross_entropy(const Result c = 1) {
  return l2_cross_entropy<Data, Result>(c);
}

template <typename Data, typename Result = Data>
inline l2_topk_cross_entropy<Data, Result> make_objective_l2_topk_cross_entropy(const Result c,
                                                                                const size_type k) {
  return l2_topk_cross_entropy<Data, Result>(c, k);
}

template <typename Data, typename Result = Data>
inline l2_topk_hinge_alpha<Data, Result> make_objective_l2_topk_hinge_alpha(const Result c,
                                                                            const size_type k) {
  return l2_topk_hinge_alpha<Data, Result>(c, k);
}

template <typename Data, typename Result = Data>
inline l2_topk_hinge_alpha_smooth<Data, Result> make_objective_l2_topk_hinge_alpha_smooth(
    const Result c, const size_type k, const Result gamma) {
  return l2_topk_hinge_alpha_smooth<Data, Result>(c, k, gamma);
}

template <typename Data, typename Result = Data>
inline l2_topk_hinge_beta<Data, Result> make_objective_l2_topk_hinge_beta(const Result c,
                                                                          const size_type k) {
  return l2_topk_hinge_beta<Data, Result>(c, k);
}

template <typename Data, typename Result = Data>
inline l2_topk_hinge_beta_smooth<Data, Result> make_objective_l2_topk_hinge_beta_smooth(
    const Result c, const size_type k, const Result gamma) {
  return l2_topk_hinge_beta_smooth<Data, Result>(c, k, gamma);
}

template <typename Data, typename Result = Data>
inline l2_multilabel_cross_entropy<Data, Result> make_objective_l2_multilabel_cross_entropy(
    const Result c = 1) {
  return l2_multilabel_cross_entropy<Data, Result>(c);
}

template <typename Data, typename Result = Data>
inline l2_multilabel_hinge<Data, Result> make_objective_l2_multilabel_hinge(const Result c = 1) {
  return l2_multilabel_hinge<Data, Result>(c);
}

template <typename Data, typename Result = Data>
inline l2_multilabel_hinge_smooth<Data, Result> make_objective_l2_multilabel_hinge_smooth(
    const Result c, const Result gamma) {
  return l2_multilabel_hinge_smooth<Data, Result>(c, gamma);
}

template <typename Data, typename Result = Data>
inline l2_distillation<Data, Result> make_objective_l2_distillation(const Result c = 1) {
  return l2_distillation<Data, Result>(c);
}

// The training objectives 0.5 ||W - W0||^2 + c sum_i loss_i by the shared loss
// names (docs/API.md): every loss but the proximal and distillation ones. k is
// the top-k parameter of the top-k losses (1 otherwise) and gamma > 0 smooths
// the hinges (0: nonsmooth, the other losses). The factories below validate
// their arguments (std::invalid_argument); visit_multiclass_objective and
// visit_multilabel_objective call a visitor with the objective type of a spec.
struct objective_spec {
  loss_kind kind;
  double c;
  size_type k;
  double gamma;
};

namespace detail {
inline objective_spec checked_objective(const loss_kind kind, const double c, const size_type k,
                                        const double gamma) {
  if (!(c > 0) || !std::isfinite(c)) throw std::invalid_argument("C must be finite and positive");
  if (k < 1) throw std::invalid_argument("k must be positive");
  if (!(gamma >= 0) || !std::isfinite(gamma))
    throw std::invalid_argument("gamma must be finite and nonnegative");
  objective_spec spec = {kind, c, k, gamma};
  return spec;
}
}  // namespace detail

inline objective_spec cross_entropy_objective(const double c = 1) {
  return detail::checked_objective(loss_kind::cross_entropy, c, 1, 0);
}

inline objective_spec topk_cross_entropy_objective(const double c, const size_type k) {
  return detail::checked_objective(loss_kind::topk_cross_entropy, c, k, 0);
}

inline objective_spec multiclass_hinge_objective(const double c = 1, const double gamma = 0) {
  return detail::checked_objective(loss_kind::multiclass_hinge, c, 1, gamma);
}

inline objective_spec topk_hinge_alpha_objective(const double c, const size_type k,
                                                 const double gamma = 0) {
  return detail::checked_objective(loss_kind::topk_hinge_alpha, c, k, gamma);
}

inline objective_spec topk_hinge_beta_objective(const double c, const size_type k,
                                                const double gamma = 0) {
  return detail::checked_objective(loss_kind::topk_hinge_beta, c, k, gamma);
}

inline objective_spec multilabel_hinge_objective(const double c = 1, const double gamma = 0) {
  return detail::checked_objective(loss_kind::multilabel_hinge, c, 1, gamma);
}

inline objective_spec multilabel_cross_entropy_objective(const double c = 1) {
  return detail::checked_objective(loss_kind::multilabel_cross_entropy, c, 1, 0);
}

// visitor(objective) with the objective of spec on one label per example
// (multiclass_output): l2_cross_entropy, l2_topk_cross_entropy,
// l2_topk_hinge_alpha (multiclass_hinge at k = 1, topk_hinge_alpha) and
// l2_topk_hinge_beta, the _smooth types for gamma > 0, all <Data, Result>.
// Throws std::invalid_argument for the other kinds. The visitor is called once,
// with a prvalue objective, and is typically a functor whose templated
// operator() builds a context and solves.
template <typename Data, typename Result, typename Visitor>
inline void visit_multiclass_objective(const objective_spec& spec, Visitor&& visitor) {
  const Result c = static_cast<Result>(spec.c), gamma = static_cast<Result>(spec.gamma);
  const bool smooth = spec.gamma > 0;
  switch (spec.kind) {
    case loss_kind::cross_entropy:
      visitor(l2_cross_entropy<Data, Result>(c));
      return;
    case loss_kind::topk_cross_entropy:
      visitor(l2_topk_cross_entropy<Data, Result>(c, spec.k));
      return;
    case loss_kind::multiclass_hinge:
    case loss_kind::topk_hinge_alpha: {
      const size_type k = spec.kind == loss_kind::multiclass_hinge ? 1 : spec.k;
      if (smooth)
        visitor(l2_topk_hinge_alpha_smooth<Data, Result>(c, k, gamma));
      else
        visitor(l2_topk_hinge_alpha<Data, Result>(c, k));
      return;
    }
    case loss_kind::topk_hinge_beta:
      if (smooth)
        visitor(l2_topk_hinge_beta_smooth<Data, Result>(c, spec.k, gamma));
      else
        visitor(l2_topk_hinge_beta<Data, Result>(c, spec.k));
      return;
    case loss_kind::multilabel_hinge:
    case loss_kind::multilabel_cross_entropy:
    case loss_kind::proximal_cross_entropy:
    case loss_kind::distillation:
    case loss_kind::proximal_distillation:
      break;
  }
  throw std::invalid_argument(std::string(loss_name(spec.kind)) +
                              " is not an objective of one label per example");
}

// visitor(objective) with the objective of spec on label sets
// (multilabel_output): l2_multilabel_hinge (l2_multilabel_hinge_smooth for
// gamma > 0) or l2_multilabel_cross_entropy, <Data, Result>. Throws
// std::invalid_argument for the other kinds.
template <typename Data, typename Result, typename Visitor>
inline void visit_multilabel_objective(const objective_spec& spec, Visitor&& visitor) {
  const Result c = static_cast<Result>(spec.c);
  switch (spec.kind) {
    case loss_kind::multilabel_hinge:
      if (spec.gamma > 0)
        visitor(l2_multilabel_hinge_smooth<Data, Result>(c, static_cast<Result>(spec.gamma)));
      else
        visitor(l2_multilabel_hinge<Data, Result>(c));
      return;
    case loss_kind::multilabel_cross_entropy:
      visitor(l2_multilabel_cross_entropy<Data, Result>(c));
      return;
    case loss_kind::cross_entropy:
    case loss_kind::topk_cross_entropy:
    case loss_kind::multiclass_hinge:
    case loss_kind::topk_hinge_alpha:
    case loss_kind::topk_hinge_beta:
    case loss_kind::proximal_cross_entropy:
    case loss_kind::distillation:
    case loss_kind::proximal_distillation:
      break;
  }
  throw std::invalid_argument(std::string(loss_name(spec.kind)) +
                              " is not an objective of label sets");
}

}  // namespace sdca

#endif
