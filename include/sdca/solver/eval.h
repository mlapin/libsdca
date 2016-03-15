#ifndef SDCA_SOLVER_EVAL_H
#define SDCA_SOLVER_EVAL_H

#include "sdca/solver/context.h"
#include "sdca/solver/eval/core.h"
#include "sdca/solver/eval/dual.h"
#include "sdca/solver/eval/primal.h"
#include "sdca/solver/eval/regularizer.h"
#include "sdca/solver/eval/scores.h"
#include "sdca/solver/eval/types.h"
#include "sdca/solver/reporting.h"
#include "sdca/solver/scratch.h"

namespace sdca {

template <typename Data,
          template <typename> class Input,
          typename Dataset,
          typename Context>
inline void
evaluate_dataset(
    const Context& ctx,
    Dataset& d,
    solver_scratch<Data, Input>& scratch
  ) {
  const size_type m = d.num_classes();
  const size_type n = d.num_examples();

  auto& eval = eval_begin(d);

  eval_recompute_primal(m, n, d, ctx.dual_variables, ctx.primal_variables);
  eval_regularizer_primal(m, d.in, ctx.objective, ctx.primal_variables, eval);

  assert(m == scratch.scores.size());
  Data* scores = &scratch.scores[0];
  for (size_type i = 0; i < n; ++i) {
    Data* variables = ctx.dual_variables + m * i;

    eval_scores(i, m, d.in, ctx, scores);

    eval_regularizer_dual(m, d.in, ctx.objective, variables, scores, eval);

    eval_dual_loss(i, d.out, ctx.objective, variables, eval);

    eval_primal_loss(i, d.out, ctx.objective, scores, eval);

  }

  eval_end(m, n, ctx.objective, eval);
  reporting::eval_created(ctx, eval);
}


template <typename Data,
          typename Result,
          typename Context>
inline void
check_stopping_criteria(
    Context& ctx
  ) {
  // Nothing to check if the solver is not running
  if (ctx.status != solver_status::solving) return;

  // First, check the relative duality gap
  const auto& criteria = ctx.criteria;
  const auto& evals = ctx.train.evals;
  if (evals.size() > 0) {
    const auto& eval = evals.back();

    Result gap = eval.primal - eval.dual;
    Result max = std::max(std::abs(eval.primal), std::abs(eval.dual));
    Result eps_stop = max * static_cast<Result>(criteria.epsilon);
    Result eps = 64 * std::max(static_cast<Result>(1), max)
      * std::max(std::numeric_limits<Result>::epsilon(),
                 static_cast<Result>(std::numeric_limits<Data>::epsilon()));

    // Check if the relative duality gap is below epsilon
    if (gap < eps_stop) {
      // A large negative duality gap likely indicates an issue in the code
      if (gap < - eps || gap <= - eps_stop) {
        ctx.status = solver_status::failed;
        reporting::solver_stop_failed(gap, eps, eps_stop);
      } else {
        ctx.status = solver_status::solved;
      }
    } else if (evals.size() > 1) {
      // Check if the solver is making progress
      const auto& before = evals.rbegin()[1];
      if (eval.dual + eps * before.dual < before.dual) {
        ctx.status = solver_status::no_progress;
        reporting::solver_stop_no_progress(eval, before);
      }
    }
  }

  // Second, check the runtime limits
  if (ctx.status == solver_status::solving) {
    if (ctx.epoch >= criteria.max_epoch) {
      ctx.status = solver_status::max_epoch;
    } else if (criteria.max_cpu_time > 0 &&
               ctx.cpu_time_now() >= criteria.max_cpu_time) {
      ctx.status = solver_status::max_cpu_time;
    } else if (criteria.max_wall_time > 0 &&
               ctx.wall_time_now() >= criteria.max_wall_time) {
      ctx.status = solver_status::max_wall_time;
    }
  }
}

}

#endif
