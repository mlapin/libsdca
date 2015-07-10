#ifndef SDCA_SOLVE_SOLVEDEF_H
#define SDCA_SOLVE_SOLVEDEF_H

#include <chrono>
#include <limits>

namespace sdca {

enum class status {
  none = 0,
  solving,
  solved,
  dual_decreased,
  max_num_epoch,
  max_cpu_time,
  max_wall_time
};

static const char* status_name[] = {
  "none",
  "solving",
  "solved",
  "dual_decreased",
  "max_num_epoch",
  "max_cpu_time",
  "max_wall_time"
};

using wall_clock = std::chrono::high_resolution_clock;
typedef typename std::chrono::time_point<wall_clock> wall_time_point;
typedef typename std::clock_t cpu_time_point;
typedef typename std::underlying_type<status>::type status_type;
typedef typename std::size_t size_type;
typedef typename std::ptrdiff_t difference_type;

struct stopping_criteria {
  size_type check_epoch = 10;
  size_type max_num_epoch = 100;
  double max_cpu_time = 0;
  double max_wall_time = 0;
  double epsilon = 1e-3;
};

template <typename Result>
struct state {
  typedef Result result_type;
  size_type epoch;
  double cpu_time;
  double wall_time;
  result_type primal;
  result_type dual;
  result_type gap;

  state() :
      epoch(0),
      cpu_time(0),
      wall_time(0),
      primal(std::numeric_limits<result_type>::infinity()),
      dual(-std::numeric_limits<result_type>::infinity()),
      gap(std::numeric_limits<result_type>::infinity())
    {}

  state(
      const size_type __epoch,
      const double __cpu_time,
      const double __wall_time,
      const result_type __primal,
      const result_type __dual,
      const result_type __gap
    ) :
      epoch(__epoch),
      cpu_time(__cpu_time),
      wall_time(__wall_time),
      primal(__primal),
      dual(__dual),
      gap(__gap)
    {}
};

}

#endif
