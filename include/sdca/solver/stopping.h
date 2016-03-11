#ifndef SDCA_SOLVER_STOPPING_CRITERIA_H
#define SDCA_SOLVER_STOPPING_CRITERIA_H

#include <sstream>

#include "sdca/types.h"

namespace sdca {

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
    str << "epsilon = " << epsilon << ", "
           "eval_epoch = " << eval_epoch << ", "
           "max_epoch = " << max_epoch << ", "
           "max_cpu_time = " << max_cpu_time << ", "
           "max_wall_time = " << max_wall_time << ", "
           "eval_on_start = " << eval_on_start;
    return str.str();
  }
};

}

#endif
