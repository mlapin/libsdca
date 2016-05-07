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
    const Int,
    const Output&,
    const Objective&,
    Data*,
    Evaluation&
  ) {}


template <typename Int,
          typename Data,
          typename Result,
          template <typename, typename> class Objective>
inline void
eval_dual_loss(
    const Int i,
    const multiclass_output& out,
    const Objective<Data, Result>& obj,
    Data* dual_variables,
    eval_train<Result, multiclass_output>& eval
  ) {
  // Swap the ground truth label and a label at 0
  out.move_front(i, dual_variables);

  // The dual loss computation must not modify the variables
  eval.dual_loss += obj.dual_loss(out.num_classes,
                                  const_cast<const Data*>(dual_variables));

  // Put back the ground truth
  out.move_back(i, dual_variables);
}


template <typename Int,
          typename Data,
          typename Result,
          template <typename, typename> class Objective>
inline void
eval_dual_loss(
    const Int i,
    const multilabel_output& out,
    const Objective<Data, Result>& obj,
    Data* dual_variables,
    eval_train<Result, multilabel_output>& eval
  ) {
  out.move_front(i, dual_variables);
  eval.dual_loss += obj.dual_loss(out.num_classes, out.num_labels(i),
                                  const_cast<const Data*>(dual_variables));
  out.move_back(i, dual_variables);
}

}

#endif
