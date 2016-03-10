#ifndef SDCA_SOLVER_EVAL_REGULARIZER_H
#define SDCA_SOLVER_EVAL_REGULARIZER_H

#include "sdca/solver/input.h"
#include "sdca/solver/eval/types.h"

namespace sdca {

template <typename Int,
          typename Data,
          typename Input,
          typename Objective,
          typename Evaluation>
inline void
eval_regularizer_primal(
    const Input&,
    const Objective&,
    const Int,
    const Data*,
    Evaluation&
  ) {}


template <typename Int,
          typename Data,
          typename Result,
          typename Output,
          template <typename, typename> class Objective>
inline void
eval_regularizer_primal(
    const feature_input<Data>& in,
    const Objective<Data, Result>& obj,
    const Int num_classes,
    const Data* primal_variables,
    eval_train<Result, Output>& eval
  ) {
  eval.regularizer += obj.regularizer_primal(
    in.num_dimensions * num_classes, primal_variables);
}


template <typename Int,
          typename Data,
          typename Input,
          typename Objective,
          typename Evaluation>
inline void
eval_regularizer_dual(
    const Input&,
    const Objective&,
    const Int,
    const Data*,
    const Data*,
    Evaluation&
  ) {}


template <typename Int,
          typename Data,
          typename Result,
          typename Output,
          template <typename, typename> class Objective>
inline void
eval_regularizer_dual(
    const kernel_input<Data>&,
    const Objective<Data, Result>& obj,
    const Int num_classes,
    const Data* dual_variables,
    const Data* scores,
    eval_train<Result, Output>& eval
  ) {
  eval.regularizer += obj.regularizer_dual(
    num_classes, dual_variables, scores);
}

}

#endif
