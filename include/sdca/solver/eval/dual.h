#ifndef SDCA_SOLVER_EVAL_DUAL_H
#define SDCA_SOLVER_EVAL_DUAL_H

#include "sdca/solver/eval/types.h"

namespace sdca {

template <typename Int,
          typename Data,
          typename Output,
          typename Objective,
          typename Evaluation>
inline void
eval_dual_loss(
    const Output&,
    const Objective&,
    const Int,
    Data*,
    Evaluation&
  ) {}


template <typename Int,
          typename Data,
          typename Result,
          template <typename, typename> class Objective>
inline void
eval_dual_loss(
    const multiclass_output& out,
    const Objective<Data, Result>& obj,
    const Int i,
    Data* dual_variables,
    eval_train<Result, multiclass_output>& eval
  ) {
  // Swap the ground truth label and a label at 0
  size_type label = out.labels[i];
  std::swap(dual_variables[0], dual_variables[label]);

  // The dual loss computation must not modify the variables
  eval.dual_loss += obj.dual_loss(
    out.num_classes, const_cast<const Data*>(dual_variables));

  // Put back the ground truth
  std::swap(dual_variables[0], dual_variables[label]);
}

}

#endif
