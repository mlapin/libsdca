#ifndef SDCA_SOLVE_SOLVEDEF_H
#define SDCA_SOLVE_SOLVEDEF_H

#include <chrono>
#include <limits>
#include <sstream>
#include <vector>

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
typedef typename std::underlying_type<solver_status>::type solver_status_type;
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

using wall_clock = std::chrono::high_resolution_clock;
typedef typename std::chrono::time_point<wall_clock> wall_time_point;
typedef typename std::clock_t cpu_time_point;
typedef typename std::size_t size_type;
typedef typename std::ptrdiff_t difference_type;

struct stopping_criteria {
  bool check_on_start = false;
  size_type check_epoch = 1;
  size_type max_epoch = 1000;
  double max_cpu_time = 0;
  double max_wall_time = 0;
  double epsilon = 1e-3;

  inline std::string
  to_string() const {
    std::ostringstream str;
    str << "epsilon = " << epsilon << ", "
           "check_on_start = " << check_on_start << ", "
           "check_epoch = " << check_epoch << ", "
           "max_epoch = " << max_epoch << ", "
           "max_cpu_time = " << max_cpu_time << ", "
           "max_wall_time = " << max_wall_time;
    return str.str();
  }
};

template <typename Data>
struct dataset {
  typedef Data data_type;
  size_type num_dimensions = 0;
  size_type num_examples = 0;
  size_type num_tasks = 0;
  std::vector<size_type> labels;
  data_type* data = nullptr;

  inline std::string
  to_string() const {
    std::ostringstream str;
    str << "num_dimensions = " << num_dimensions << ", "
           "num_examples = " << num_examples << ", "
           "num_tasks = " << num_tasks;
    return str.str();
  }
};

template <typename Data,
          typename Result>
struct solver_context {
  typedef Data data_type;
  typedef Result result_type;
  bool is_dual = false;
  stopping_criteria criteria;
  std::vector<dataset<Data>> datasets;
  data_type* primal_variables = nullptr;
  data_type* dual_variables = nullptr;
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

template <typename Result>
struct statistic {
  typedef Result result_type;
  result_type primal;
  result_type dual;
  result_type gap;
  std::vector<result_type> performance;
};

}

#endif
