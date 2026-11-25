#ifndef SDCA_LOSS_H
#define SDCA_LOSS_H

// The losses of one row of scores (docs/CPP.md): loss_and_gradient of the label
// losses and the distillation losses, on the batch driver of sdca/kernel.h.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "sdca/kernel.h"

namespace sdca {

// The parameters of the losses of loss_and_gradient. A loss takes only its
// own: the fields of the other losses must keep their defaults.
struct loss_parameters {
  // The top-k losses (topk_cross_entropy, topk_hinge_alpha, topk_hinge_beta):
  // 1 <= k < num_classes, required (0: unset).
  std::size_t k = 0;
  // The hinges: smoothing, finite and >= 0 (0: nonsmooth). proximal_cross_entropy:
  // the Moreau step, finite and > 0, required (0: unset).
  double gamma = 0;
};

namespace loss_detail {

// The checks of loss_and_gradient: the kind is a label loss of this target, and takes
// these parameters.
inline void check_row(kernel::Kind kind, std::size_t num_classes, bool sets,
                      const loss_parameters& p) {
  if (num_classes < 2) throw std::invalid_argument("At least two classes are required");
  const kernel::Traits& t = kernel::traits(kind);
  if (t.distillation)
    throw std::invalid_argument(
        "loss_and_gradient takes the label losses; see distillation_loss and "
        "proximal_distillation_loss");
  if (t.multilabel != sets)
    throw std::invalid_argument(std::string(t.name) +
                                (sets ? " takes one label" : " takes a label set"));
  kernel::check_parameters(kind, num_classes, p.k, p.gamma, 1.0);
}

inline void check_distillation(std::size_t n, const void* student, const void* teacher,
                               kernel::Kind kind, double temperature, double gamma) {
  if (n < 2) throw std::invalid_argument("At least two classes are required");
  if (!student || !teacher)
    throw std::invalid_argument("The student and teacher logits must not be null");
  kernel::check_parameters(kind, n, 0, gamma, temperature);
}

// One row of the driver in the arithmetic common_type<Data, Result>, serial; an
// invalid row throws std::invalid_argument, a loss without a kernel std::runtime_error.
template <typename Data, typename Result, typename Targets>
double evaluate_row(kernel::Kind kind, std::size_t k, double gamma, double temperature,
                    std::size_t n, const Data* z, const Targets& t, Data* gradient) {
  const bool float_work = std::is_same<typename std::common_type<Data, Result>::type, float>::value;
  const kernel::Loss f = {kind, k, gamma, temperature, 0.0};
  const kernel::Execution e = {float_work ? kernel::precision_float32 : kernel::precision_float64,
                               1, false};
  double value = 0;
  const kernel::Result r = kernel::evaluate(f, e, 1, n, z, t, &value, gradient);
  if (r.status == kernel::status_unsupported)
    throw std::runtime_error(std::string(kernel::traits(kind).name) +
                             " has no kernel in this build");
  if (r.invalid_rows)
    throw std::invalid_argument(
        "Invalid row: scores must not be NaN or +inf, and labels must be in range with a "
        "positive class");
  return value;
}

}  // namespace loss_detail

// The loss of the scores z (num_classes, in class order) at the label of a
// single-label loss (cross_entropy, topk_cross_entropy, multiclass_hinge,
// topk_hinge_alpha, topk_hinge_beta, proximal_cross_entropy) and, unless gradient
// is null, its gradient (num_classes; a subgradient at the kinks of the nonsmooth
// hinges), in the arithmetic common_type<Data, Result>. A score of -inf masks its
// class. Throws std::invalid_argument for invalid sizes, labels or parameters and
// for an invalid row (NaN, +inf, an invalid row of masked classes).
template <typename Data, typename Result = Data>
double loss_and_gradient(const kernel::Kind kind, const std::size_t num_classes, const Data* z,
                         const std::size_t label, Data* gradient,
                         const loss_parameters& params = loss_parameters()) {
  loss_detail::check_row(kind, num_classes, false, params);
  if (!z) throw std::invalid_argument("The scores must not be null");
  if (label >= num_classes)
    throw std::invalid_argument(kernel::invalid_label_text(static_cast<int64_t>(label)));
  const int64_t y = static_cast<int64_t>(label);
  const kernel::Labels t = {&y, nullptr, 0, false};
  return loss_detail::evaluate_row<Data, Result>(kind, params.k, params.gamma, 1.0, num_classes, z,
                                                 t, gradient);
}

// The loss of a multilabel loss (multilabel_hinge, multilabel_cross_entropy) at
// the label set labels[0, num_labels), sorted without duplicates, as
// loss_and_gradient of one label. multilabel_cross_entropy takes up to every
// class; multilabel_hinge needs a class outside the set.
template <typename Data, typename Result = Data>
double loss_and_gradient(const kernel::Kind kind, const std::size_t num_classes, const Data* z,
                         const std::size_t* labels, const std::size_t num_labels, Data* gradient,
                         const loss_parameters& params = loss_parameters()) {
  loss_detail::check_row(kind, num_classes, true, params);
  if (!z || !labels) throw std::invalid_argument("The scores and labels must not be null");
  if (num_labels < 1 || num_labels > num_classes)
    throw std::invalid_argument("A label set needs between 1 and num_classes labels");
  if (kind == kernel::multilabel_hinge && num_labels == num_classes)
    throw std::invalid_argument("multilabel_hinge needs a class outside the label set");
  std::vector<int64_t> set(num_labels);
  for (std::size_t i = 0; i < num_labels; ++i) {
    if (labels[i] >= num_classes)
      throw std::invalid_argument(kernel::invalid_label_text(static_cast<int64_t>(labels[i])));
    if (i > 0 && labels[i] <= labels[i - 1])
      throw std::invalid_argument("Label sets must be sorted without duplicates");
    set[i] = static_cast<int64_t>(labels[i]);
  }
  const int64_t offsets[2] = {0, static_cast<int64_t>(num_labels)};
  const kernel::LabelSets t = {set.data(), offsets};
  return loss_detail::evaluate_row<Data, Result>(kind, params.k, params.gamma, 1.0, num_classes, z,
                                                 t, gradient);
}

// distillation_loss H(softmax(teacher / temperature), softmax(student / temperature))
// of the n student and teacher logits, and its gradient (q - p) / temperature with
// respect to student if gradient is non-null.
template <typename Data, typename Result = Data>
double distillation_loss(std::size_t n, const Data* student, const Data* teacher,
                         double temperature = 1.0, Data* gradient = nullptr) {
  loss_detail::check_distillation(n, student, teacher, kernel::distillation, temperature, 0.0);
  const kernel::Teacher<Data> t = {teacher};
  return loss_detail::evaluate_row<Data, Result>(kernel::distillation, 0, 0.0, temperature, n,
                                                 student, t, gradient);
}

// proximal_distillation_loss min_u H(softmax(teacher / temperature), softmax(u))
// + |u - student / temperature|^2 / (2 gamma), and its gradient (pi - p) / temperature
// with respect to student if gradient is non-null.
template <typename Data, typename Result = Data>
double proximal_distillation_loss(std::size_t n, const Data* student, const Data* teacher,
                                  double temperature = 1.0, double gamma = 1.0,
                                  Data* gradient = nullptr) {
  loss_detail::check_distillation(n, student, teacher, kernel::proximal_distillation, temperature,
                                  gamma);
  const kernel::Teacher<Data> t = {teacher};
  return loss_detail::evaluate_row<Data, Result>(kernel::proximal_distillation, 0, gamma,
                                                 temperature, n, student, t, gradient);
}

}  // namespace sdca

#endif
