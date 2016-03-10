#ifndef SDCA_SOLVER_REPORTING_H
#define SDCA_SOLVER_REPORTING_H

#include "sdca/solver/context.h"
#include "sdca/utility/logging.h"

namespace sdca {

inline std::string
to_string(
    solver_status __status
  ) {
  switch (__status) {
    case solver_status::none:
      return "none";
    case solver_status::solving:
      return "solving";
    case solver_status::solved:
      return "solved";
    case solver_status::no_progress:
      return "no_progress";
    case solver_status::max_epoch:
      return "max_epoch";
    case solver_status::max_cpu_time:
      return "max_cpu_time";
    case solver_status::max_wall_time:
      return "max_wall_time";
    case solver_status::failed:
      return "failed";
  }
}

namespace reporting {

template <typename Context>
inline void
end_solve(
    const Context& ctx
  ) {
  LOG_INFO <<
    "status: " << to_string(ctx.status) << ", "
    "epoch: " << ctx.epoch << ", "
    "wall_time: " << ctx.wall_time() <<
    " (" << ctx.solve_time.wall.elapsed <<
    " + " << ctx.eval_time.wall.elapsed << "), "
    "cpu_time: " << ctx.cpu_time() <<
    " (" << ctx.solve_time.cpu.elapsed <<
    " + " << ctx.eval_time.cpu.elapsed << ")" <<
    std::endl;
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


template <typename Result,
          typename Context>
inline void
eval_created(
    const Context& ctx,
    const eval_train<Result, multiclass_output>& eval
  ) {
  std::ostringstream str;
  auto offset = std::min(static_cast<std::size_t>(5), eval.accuracy.size());
  std::copy(eval.accuracy.begin(), eval.accuracy.begin() + offset,
    std::ostream_iterator<Result>(str, ", "));
  LOG_VERBOSE <<
    "  "
    "epoch: " << std::setw(3) << ctx.epoch << std::setw(0) << ", "
    "accuracy: " << str.str() <<
    "relative_gap: " << relative_gap(eval.primal, eval.dual) << ", "
    "absolute_gap: " << eval.primal - eval.dual << ", "
    "primal: " << eval.primal << ", "
    "dual: " << eval.dual << ", "
    "primal_loss: " << eval.primal_loss << ", "
    "dual_loss: " << eval.dual_loss << ", "
    "regularizer: " << eval.regularizer << ", "
    "wall_time: " << ctx.wall_time_now() << ", "
    "cpu_time: " << ctx.cpu_time_now() <<
    std::endl;
}


template <typename Result,
          typename Context>
inline void
eval_created(
    const Context& ctx,
    const eval_test<Result, multiclass_output>& eval
  ) {
  std::ostringstream str;
  auto offset = std::min(static_cast<std::size_t>(5), eval.accuracy.size());
  std::copy(eval.accuracy.begin(), eval.accuracy.begin() + offset,
    std::ostream_iterator<Result>(str, ", "));
  LOG_VERBOSE <<
    "  "
    "epoch: " << std::setw(3) << ctx.epoch << std::setw(0) << ", "
    "accuracy: " << str.str() <<
    "primal_loss: " << eval.primal_loss << ", "
    "wall_time: " << ctx.wall_time_now() << ", "
    "cpu_time: " << ctx.cpu_time_now() <<
    std::endl;
}


template <typename Result,
          typename Output,
          typename Context>
inline void
stop_solved(
    const Context& ctx,
    const eval_train<Result, Output>& eval
  ) {
  assert(ctx.status == solver_status::solved);
  LOG_DEBUG <<
    "  "
    "stop: " << to_string(ctx.status) << ", "
    "relative_gap: " << relative_gap(eval.primal, eval.dual) << ", "
    "epsilon: " << ctx.criteria.epsilon <<
    std::endl;
}


template <typename Result,
          typename Output,
          typename Context>
inline void
stop_failed(
    const Context& ctx,
    const eval_train<Result, Output>& eval
  ) {
  assert(ctx.status == solver_status::failed);
  LOG_DEBUG <<
    "  "
    "stop: " << to_string(ctx.status) << ", "
    "primal: " << eval.primal << ", "
    "dual: " << eval.dual << ", "
    "absolute_gap: " << eval.primal - eval.dual <<
    std::endl;
}


template <typename Result,
          typename Output,
          typename Context>
inline void
stop_no_progress(
    const Context& ctx,
    const eval_train<Result, Output>& eval,
    const eval_train<Result, Output>& before
  ) {
  assert(ctx.status == solver_status::no_progress);
  LOG_DEBUG <<
    "  "
    "stop: " << to_string(ctx.status) << ", "
    "dual: " << eval.dual << ", "
    "dual_before: " << before.dual << ", "
    "difference: " << eval.dual - before.dual <<
    std::endl;
}


template <typename Context>
inline void
stop_max_epoch(
    const Context& ctx
  ) {
  assert(ctx.status == solver_status::max_epoch);
  LOG_DEBUG <<
    "  "
    "stop: " << to_string(ctx.status) << ", "
    "epoch: " << ctx.epoch << ", "
    "max_epoch: " << ctx.criteria.max_epoch <<
    std::endl;
}


template <typename Context>
inline void
stop_max_cpu_time(
    const Context& ctx
  ) {
  assert(ctx.status == solver_status::max_cpu_time);
  LOG_DEBUG <<
    "  "
    "stop: " << to_string(ctx.status) << ", "
    "cpu_time: " << ctx.cpu_time() << ", "
    "max_cpu_time: " << ctx.criteria.max_cpu_time <<
    std::endl;
}


template <typename Context>
inline void
stop_max_wall_time(
    const Context& ctx
  ) {
  assert(ctx.status == solver_status::max_wall_time);
  LOG_DEBUG <<
    "  "
    "stop: " << to_string(ctx.status) << ", "
    "wall_time: " << ctx.wall_time() << ", "
    "max_wall_time: " << ctx.criteria.wall_cpu_time <<
    std::endl;
}

}

}

#endif
