#ifndef SDCA_SOLVER_REPORTING_H
#define SDCA_SOLVER_REPORTING_H

#include "sdca/solver/solverdef.h"
#include "sdca/solver/eval/types.h"
#include "sdca/utility/logging.h"

namespace sdca {
namespace reporting {

template <typename Context>
inline void
begin_solve(
    const Context& ctx
  ) {
  LOG_INFO << "Solve: " << ctx.to_string() << std::endl;
}


template <typename Context>
inline void
end_solve(
    const Context& ctx
  ) {
  if (ctx.status == solver_status::solved) {
    LOG_INFO << "Solution: " << ctx.status_string() << std::endl;
  } else {
    if (logging::get_level() < logging::level::info) {
      LOG_WARNING << "Solve: " << ctx.to_string() << std::endl;
    }
    LOG_WARNING << "Solution: " << ctx.status_string() << std::endl;
  }
}


template <typename Context>
inline void
end_epoch(
    const Context& ctx,
    const bool is_evaluated
  ) {
  if (is_evaluated) return;
  LOG_DEBUG <<
    "  "
    "epoch: " << ctx.epoch << ", "
    "cpu_time: " << ctx.cpu_time() <<
    " (" << ctx.solve_time.cpu.elapsed <<
    " + " << ctx.eval_time.cpu.elapsed << "), "
    "wall_time: " << ctx.wall_time() <<
    " (" << ctx.solve_time.wall.elapsed <<
    " + " << ctx.eval_time.wall.elapsed << ")" <<
    std::endl;
}


template <typename Result,
          typename Output>
inline void
eval_created(
    const eval_train<Result, Output>& eval,
    const size_type
  ) {
  LOG_VERBOSE <<
    "  " << eval.to_string() << std::endl;
}


template <typename Result,
          typename Output>
inline void
eval_created(
    const eval_test<Result, Output>& eval,
    const size_type id
  ) {
  LOG_VERBOSE <<
    "  eval on set #" << id << ": " << eval.to_string() << std::endl;
}


template <typename Result>
inline void
solver_stop_failed(
    const Result absolute_gap,
    const Result eps_machine,
    const Result eps_user
  ) {
  LOG_WARNING <<
    "Warning: negative duality gap; "
    "absolute_gap: " << absolute_gap << ", "
    "eps_machine: " << eps_machine << ", "
    "eps_user: " << eps_user <<
    std::endl;
}


template <typename Result,
          typename Output>
inline void
solver_stop_no_progress(
    const eval_train<Result, Output>& eval,
    const eval_train<Result, Output>& before
  ) {
  LOG_WARNING <<
    "Warning: dual objective decreased; "
    "dual: " << eval.dual << ", "
    "dual_before: " << before.dual << ", "
    "difference: " << eval.dual - before.dual <<
    std::endl;
}

}
}

#endif
