#ifndef SDCA_SOLVER_EVAL_REGULARIZER_H
#define SDCA_SOLVER_EVAL_REGULARIZER_H

#include "sdca/solver/data/input.h"
#include "sdca/solver/eval/types.h"

namespace sdca {

template <typename Int,
          typename Input,
          typename Context,
          typename Evaluation>
inline void
eval_regularizer_primal(
    const Int,
    const Input&,
    const Context&,
    Evaluation&
  ) {}


template <typename Int,
          typename Data,
          typename Result,
          typename Output,
          typename Context>
inline void
eval_regularizer_primal(
    const Int num_classes,
    const feature_input<Data>& in,
    const Context& ctx,
    eval_train<Result, Output>& eval
  ) {
  if (ctx.is_prox()) {
    ctx.objective.regularizers_primal(
      in.num_dimensions * num_classes,
      ctx.primal_variables, ctx.primal_initial,
      eval.primal_regularizer, eval.dual_regularizer);
  } else {
    ctx.objective.regularizers_primal(
      in.num_dimensions * num_classes, ctx.primal_variables,
      eval.primal_regularizer, eval.dual_regularizer);
  }
}


template <typename Int,
          typename Data,
          typename Result,
          typename Output,
          typename Context>
inline void
eval_regularizer_primal(
    const Int,
    const model_input<Data>& in,
    const Context& ctx,
    eval_train<Result, Output>& eval
  ) {
  ctx.objective.regularizers_primal(
    in.num_dimensions * in.num_examples,
    ctx.primal_variables, ctx.primal_initial,
    eval.primal_regularizer, eval.dual_regularizer);
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
  obj.regularizers_dual(num_classes, dual_variables, scores,
                        eval.primal_regularizer, eval.dual_regularizer);
}

}

#endif
