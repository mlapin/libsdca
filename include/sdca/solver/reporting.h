#ifndef SDCA_SOLVER_REPORTING_H
#define SDCA_SOLVER_REPORTING_H

#include "sdca/solver/solverdef.h"
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
    const Context& ctx
  ) {
  LOG_DEBUG <<
    "  "
    "epoch: " << std::setw(3) << ctx.epoch << std::setw(0) << ", "
    "wall_time: " << ctx.wall_time() <<
    " (" << ctx.solve_time.wall.elapsed <<
    " + " << ctx.eval_time.wall.elapsed << "), "
    "cpu_time: " << ctx.cpu_time() <<
    " (" << ctx.solve_time.cpu.elapsed <<
    " + " << ctx.eval_time.cpu.elapsed << ")" <<
    std::endl;
}


template <typename Context,
          typename Evaluation>
inline void
eval_created(
    const Context& ctx,
    const Evaluation& eval
  ) {
  LOG_VERBOSE <<
    "  "
    "epoch: " << std::setw(3) << ctx.epoch << std::setw(0) << ", " <<
    eval.to_string() << ", "
    "wall_time: " << ctx.wall_time_now() << ", "
    "cpu_time: " << ctx.cpu_time_now() <<
    std::endl;
}


template <typename Result>
inline void
solver_stop_failed(
    const Result absolute_gap,
    const Result eps_machine,
    const Result eps_user
  ) {
  LOG_WARNING <<
    "Warning: negative duality gap. "
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
    "Warning: dual objective decreased. "
    "dual: " << eval.dual << ", "
    "dual_before: " << before.dual << ", "
    "difference: " << eval.dual - before.dual <<
    std::endl;
}

}
}

#endif
