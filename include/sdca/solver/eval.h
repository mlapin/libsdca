#ifndef SDCA_SOLVER_EVAL_H
#define SDCA_SOLVER_EVAL_H

#include "sdca/solver/eval/begin.h"
#include "sdca/solver/eval/dual.h"
#include "sdca/solver/eval/end.h"
#include "sdca/solver/eval/primal.h"
#include "sdca/solver/eval/regularizer.h"
#include "sdca/solver/eval/scores.h"
#include "sdca/solver/eval/types.h"

namespace sdca {

template <typename Result,
          typename Data,
          typename Context,
          typename Dataset>
inline void
evaluate_dataset(
    Context& ctx,
    Dataset& d,
    Data* scores
  ) {
  const size_type m = d.num_classes();
  const size_type n = d.num_examples();

  auto& eval = eval_begin(d);

  eval_regularizer_primal(d.in, ctx.objective, m, ctx.primal_variables, eval);

  for (size_type i = 0; i < n; ++i) {
    Data* variables = ctx.dual_variables + m * i;

    eval_scores(i, m, d.in, ctx, scores);

    eval_regularizer_dual(d.in, ctx.objective, m, variables, scores, eval);

    eval_dual_loss(d.out, ctx.objective, i, variables, eval);

    eval_primal_loss(d.out, ctx.objective, i, scores, eval);

  }

  eval_end(m, n, ctx.objective, eval);
}

}

#endif
