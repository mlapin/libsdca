#ifndef SDCA_SOLVER_EVAL_H
#define SDCA_SOLVER_EVAL_H

#include <type_traits>

#include "sdca/solver/data/scratch.h"
#include "sdca/solver/eval/core.h"
#include "sdca/solver/eval/evaluate.h"
#include "sdca/solver/eval/scores.h"
#include "sdca/solver/eval/types.h"
#include "sdca/solver/reporting.h"

namespace sdca {

template <typename Data, template <typename> class Input, typename Dataset, typename Context>
inline void evaluate_dataset(const Context& ctx, Dataset& d, solver_scratch<Data, Input>&,
                             size_type id = 0) {
  auto& eval = eval_begin(d);
  typedef typename std::decay<decltype(eval.primal_loss)>::type Work;
  eval_dataset(ctx, d, eval_objective<Work>(ctx.objective), eval);
  reporting::eval_created(eval, id);
}

template <typename Data, typename Result, typename Context>
inline void check_stopping_criteria(Context& ctx) {
  // Nothing to check if the solver is not running
  if (ctx.status != solver_status::solving) return;

  // First, check the relative duality gap
  const auto& criteria = ctx.criteria;
  const auto& evals = ctx.train.evals;
  if (evals.size() > 0) {
    const auto& eval = evals.back();
    typedef typename std::decay<decltype(eval.primal)>::type Work;

    if (!std::isfinite(eval.primal) || !std::isfinite(eval.dual) ||
        !std::isfinite(eval.gap_error)) {
      ctx.status = solver_status::failed;
      return;
    }
    const Work gap = eval.primal - eval.dual;
    const Work max = std::max(std::abs(eval.primal), std::abs(eval.dual));
    const Work eps_stop = max * static_cast<Work>(criteria.epsilon);

    // Solved only if the gap plus its rounding bound meets the tolerance; a
    // tolerance below that bound is never certified.
    if (gap < -eval.gap_error) {
      ctx.status = solver_status::failed;
      reporting::solver_stop_failed(gap, eval.gap_error, eps_stop);
    } else if (criteria.epsilon > 0 && gap + eval.gap_error <= eps_stop) {
      ctx.status = solver_status::solved;
    } else if (evals.size() > 1) {
      // Check if the solver is making progress (the iterate's storage limits it)
      const auto previous = evals.rbegin();
      const auto& before = previous[1];
      const Work eps = 64 * max *
                           std::max(std::numeric_limits<Work>::epsilon(),
                                    static_cast<Work>(std::numeric_limits<Data>::epsilon())) +
                       eval.gap_error;
      if (eval.dual + eps < before.dual) {
        ctx.status = solver_status::no_progress;
        reporting::solver_stop_no_progress(eval, before);
      }
    }
  }

  // Second, check the runtime limits
  if (ctx.status == solver_status::solving) {
    if (ctx.epoch >= criteria.max_epoch) {
      ctx.status = solver_status::max_epoch;
    } else if (criteria.max_cpu_time > 0 && ctx.cpu_time_now() >= criteria.max_cpu_time) {
      ctx.status = solver_status::max_cpu_time;
    } else if (criteria.max_wall_time > 0 && ctx.wall_time_now() >= criteria.max_wall_time) {
      ctx.status = solver_status::max_wall_time;
    }
  }
}

}  // namespace sdca

#endif
