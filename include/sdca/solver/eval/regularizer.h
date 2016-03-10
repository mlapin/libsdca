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
    const Int,
    const Input&,
    const Objective&,
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
    const Int num_classes,
    const feature_input<Data>& in,
    const Objective<Data, Result>& obj,
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
    const Int,
    const Input&,
    const Objective&,
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
    const Int num_classes,
    const kernel_input<Data>&,
    const Objective<Data, Result>& obj,
    const Data* dual_variables,
    const Data* scores,
    eval_train<Result, Output>& eval
  ) {
  eval.regularizer += obj.regularizer_dual(
    num_classes, dual_variables, scores);
}

}

#endif
