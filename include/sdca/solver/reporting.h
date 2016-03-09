#ifndef SDCA_SOLVER_REPORTING_H
#define SDCA_SOLVER_REPORTING_H

#include "sdca/utility/logging.h"

namespace sdca {

static const char* solver_status_name[] = {
  "none",
  "solving",
  "solved",
  "no_progress",
  "max_epoch",
  "max_cpu_time",
  "max_wall_time",
  "failed"
};


namespace reporting {

template <typename Context>
inline void
end_solve(const Context& ctx) {
  LOG_INFO <<
    "status: " << solver_status_name[ctx.status] << ", "
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
