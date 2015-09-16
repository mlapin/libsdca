#ifndef SDCA_SOLVE_SOLVEDEF_H
#define SDCA_SOLVE_SOLVEDEF_H

#include <iterator>
#include <iomanip>
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
           "num_tasks = " << num_tasks << ", "
           "num_examples = " << num_examples;
    return str.str();
  }
};

template <typename Data>
struct solver_context {
  typedef Data data_type;
  bool is_dual = false;
  stopping_criteria criteria;
  std::vector<dataset<data_type>> datasets;
  data_type* primal_variables = nullptr;
  data_type* dual_variables = nullptr;
};

template <typename Result>
struct train_point {
  typedef Result result_type;
  result_type primal;
  result_type dual;
  result_type gap;
  result_type primal_loss;
  result_type dual_loss;
  result_type regularizer;
  size_type epoch;
  double cpu_time;
  double wall_time;
  double solve_cpu_time;
  double solve_wall_time;
  double eval_cpu_time;
  double eval_wall_time;

  train_point(
      const result_type __primal,
      const result_type __dual,
      const result_type __gap,
      const result_type __p_loss,
      const result_type __d_loss,
      const result_type __regul,
      const size_type __epoch,
      const double __cpu_time,
      const double __wall_time,
      const double __solve_cpu,
      const double __solve_wall,
      const double __eval_cpu,
      const double __eval_wall
  ) :
    primal(__primal), dual(__dual), gap(__gap),
    primal_loss(__p_loss), dual_loss(__d_loss), regularizer(__regul),
    epoch(__epoch), cpu_time(__cpu_time), wall_time(__wall_time),
    solve_cpu_time(__solve_cpu), solve_wall_time(__solve_wall),
    eval_cpu_time(__eval_cpu), eval_wall_time(__eval_wall)
  {}
};

template <typename Result>
struct test_point {
  typedef Result result_type;
  result_type loss;
  std::vector<result_type> accuracy;

  inline std::string
  to_string() const {
    std::ostringstream str;
    str << "loss = " << std::setprecision(4) << loss << ", accuracy = ";
    long offset = std::min(5L, static_cast<long>(accuracy.size()));
    std::copy(accuracy.begin(), accuracy.begin() + offset,
      std::ostream_iterator<result_type>(str, ", "));
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
