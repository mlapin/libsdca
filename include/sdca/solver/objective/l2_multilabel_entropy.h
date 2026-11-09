#ifndef SDCA_SOLVER_OBJECTIVE_L2_MULTILABEL_ENTROPY_H
#define SDCA_SOLVER_OBJECTIVE_L2_MULTILABEL_ENTROPY_H

#include "sdca/math/entropy.h"
#include "sdca/prox/two_entropy.h"
#include "sdca/solver/objective/objective_base.h"

namespace sdca {

template <typename Data, typename Result>
struct l2_multilabel_entropy : public objective_base<Data, Result> {
  typedef Data data_type;
  typedef Result result_type;

  typedef objective_base<Data, Result> base;

  const Result c;

  l2_multilabel_entropy(const Result c_) : base::objective_base(c_), c(c_) {
    if (!(c > 0) || !std::isfinite(c)) throw std::invalid_argument("C must be finite and positive");
  }

  // The same objective in another precision, for the evaluation.
  template <typename D, typename R>
  explicit l2_multilabel_entropy(const l2_multilabel_entropy<D, R>& other)
      : l2_multilabel_entropy(static_cast<Result>(other.c)) {}

  inline std::string to_string() const {
    std::ostringstream str;
    str << "l2_multilabel_entropy (c: " << c << ", precision: " << type_name<Result>() << ")";
    return str.str();
  }

  template <typename Int>
  void update_dual_variables(const Int num_classes, const Int num_labels, const Data norm2,
                             Data* variables, Data* scores,
                             const projection_precision = projection_precision::float64) const {
    entropy_update(num_classes, num_labels, c, norm2, variables, scores);
  }

  template <typename Int>
  inline Result primal_loss(const Int num_classes, const Int num_labels, Data* scores,
                            const projection_precision = projection_precision::float64) const {
    return entropy_loss<Result>(num_classes, num_labels, scores);
  }

  template <typename Int>
  void zero_scores_dual_variables(const Int num_classes, const Int num_labels,
                                  Data* variables) const {
    // c (q - 1/m), the gradient at zero scores.
    detail::zero_scores_dual(num_classes, num_labels,
                             c * Result(num_classes - num_labels) / Result(num_classes), variables);
  }

  template <typename Int, typename T>
  inline void feasible_dual(const Int num_classes, const Int num_labels, T* variables) const {
    feasible::label_distribution(num_classes, num_labels, variables, T(c));
  }

  template <typename Int>
  inline Result dual_loss(const Int num_classes, const Int num_labels,
                          const Data* variables) const {
    return entropy_dual_loss(num_classes, num_labels, c, variables);
  }
};

}  // namespace sdca

#endif
