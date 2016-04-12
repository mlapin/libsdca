#ifndef SDCA_SOLVER_SOLVERDEF_H
#define SDCA_SOLVER_SOLVERDEF_H

#include <cassert>
#include <sstream>

#include "sdca/utility/types.h"

namespace sdca {

enum class solver_status {
  none = 0,
  solving,
  solved,
  no_progress,
  max_epoch,
  max_cpu_time,
  max_wall_time,
  failed
};


inline std::string
solver_status_name(
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
  assert(false);
  return "unknown";
}


struct stopping_criteria {
  size_type eval_epoch = 10;
  size_type max_epoch = 1000;
  double epsilon = 1e-3;
  double max_cpu_time = 0;
  double max_wall_time = 0;
  bool eval_on_start = false;


  inline std::string
  to_string() const {
    std::ostringstream str;
    str << "epsilon: " << epsilon << ", "
           "eval_epoch: " << eval_epoch << ", "
           "max_epoch: " << max_epoch << ", "
           "max_cpu_time: " << max_cpu_time << ", "
           "max_wall_time: " << max_wall_time << ", "
           "eval_on_start: " << eval_on_start;
    return str.str();
  }
};

}

#endif
