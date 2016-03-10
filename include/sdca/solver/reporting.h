#ifndef SDCA_SOLVER_REPORTING_H
#define SDCA_SOLVER_REPORTING_H

#include "sdca/solver/context.h"
#include "sdca/utility/logging.h"

namespace sdca {

inline std::string
to_string(solver_status __status) {
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
end_solve(const Context& ctx) {
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
end_epoch(const Context& ctx) {
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


}

}

#endif
