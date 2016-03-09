#ifndef SDCA_SOLVER_EVAL_END_H
#define SDCA_SOLVER_EVAL_END_H

#include <numeric>

#include "sdca/math/blas.h"
#include "sdca/solver/eval/types.h"

namespace sdca {

template <typename Int,
          typename Data,
          typename Result,
          template <typename, typename> class Objective>
inline void
eval_end(
    const Int num_classes,
    const Int num_examples,
    const Objective<Data, Result>& obj,
    eval_train<Result, multiclass_output>& e
  ) {
  // Compute the overall primal/dual objectives and their individual terms
  obj.update_all(e.primal, e.dual, e.primal_loss, e.dual_loss, e.regularizer);

  // Top-k accuracies for all k
  std::partial_sum(e.accuracy.begin(), e.accuracy.end(), e.accuracy.begin());
  Result coeff(1 / static_cast<Result>(num_examples));
  sdca_blas_scal(static_cast<Int>(num_classes), coeff, e.accuracy[0]);
}

template <typename Int,
          typename Data,
          typename Result,
          template <typename, typename> class Objective>
inline void
eval_end(
    const Int num_classes,
    const Int num_examples,
    const Objective<Data, Result>& obj,
    eval_test<Result, multiclass_output>& e
  ) {
  // Compute the overall primal/dual objectives and their individual terms
  obj.update_primal_loss(e.primal_loss);

  // Top-k accuracies for all k
  std::partial_sum(e.accuracy.begin(), e.accuracy.end(), e.accuracy.begin());
  Result coeff(1 / static_cast<Result>(num_examples));
  sdca_blas_scal(static_cast<Int>(num_classes), coeff, e.accuracy[0]);
}

}

#endif
