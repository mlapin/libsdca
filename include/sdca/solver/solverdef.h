#ifndef SDCA_SOLVER_SOLVERDEF_H
#define SDCA_SOLVER_SOLVERDEF_H

#include <iterator>
#include <iomanip>
#include <sstream>
#include <vector>

namespace sdca {

typedef typename std::size_t size_type;
typedef typename std::ptrdiff_t difference_type;


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

typedef typename std::underlying_type<solver_status>::type solver_status_type;


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




template <typename Field>
struct model_info {
  typedef Field field_type;

  std::vector<std::pair<const char*, field_type>> fields;


  inline void add(
      const char* name,
      const field_type value) {
    fields.emplace_back(std::make_pair(name, value));
  }

};

}

#endif
