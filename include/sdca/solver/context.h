#ifndef SDCA_SOLVER_CONTEXT_H
#define SDCA_SOLVER_CONTEXT_H

#include "sdca/solver/dataset.h"
#include "sdca/solver/objective.h"
#include "sdca/solver/stopping.h"
#include "sdca/utility/stopwatch.h"

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
status_name(
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


template <typename Data,
          typename Result,
          template <typename> class Input,
          typename Output,
          template <typename, typename> class Objective>
struct solver_context {
  typedef Data data_type;
  typedef Result result_type;
  typedef Input<Data> input_type;
  typedef Output output_type;
  typedef Objective<Data, Result> objective_type;

  typedef eval_train<Result, Output> train_eval_type;
  typedef eval_test<Result, Output> test_eval_type;

  typedef dataset<input_type, output_type, train_eval_type> train_set_type;
  typedef dataset<input_type, output_type, test_eval_type> test_set_type;

  train_set_type train;
  std::vector<test_set_type> test;

  objective_type objective;
  stopping_criteria criteria;

  data_type* primal_variables = nullptr;
  data_type* dual_variables = nullptr;

  solver_status status = solver_status::none;
  size_type epoch = 0;
  stopwatch solve_time;
  stopwatch eval_time;


  solver_context(
      train_set_type&& __train_set,
      objective_type&& __objective,
      data_type* __dual_variables,
      data_type* __primal_variables = nullptr
    ) :
      train(std::move(__train_set)),
      objective(std::move(__objective)),
      primal_variables(__primal_variables),
      dual_variables(__dual_variables)
  {}


  inline std::string to_string() const {
    std::ostringstream str;
    str.copyfmt(std::cout);
    str << objective.to_string() << ", " <<
           "stopping_criteria (" << criteria.to_string() << "), " <<
           train.to_string();
    return str.str();
  }


  inline std::string status_string() const {
    std::ostringstream str;
    str.copyfmt(std::cout);
    str << "status: " << status_name(status) << ", "
           "epoch: " << epoch << ", ";
    if (train.evals.size() > 0) {
      str << "eval: " << train.evals.back().to_string() << ", ";
    }
    str << "wall_time: " << wall_time() <<
           " (solve: " << solve_time.wall.elapsed <<
           ", eval: " << eval_time.wall.elapsed << "), "
           "cpu_time: " << cpu_time() <<
           " (solve: " << solve_time.cpu.elapsed <<
           ", eval: " << eval_time.cpu.elapsed << ")";
    return str.str();
  }


  void add_test(input_type&& in, output_type&& out) {
    test.emplace_back(std::move(in), std::move(out));
  }


  bool is_dual() const { return primal_variables == nullptr; }


  double cpu_time() const {
    return solve_time.cpu.elapsed + eval_time.cpu.elapsed;
  }


  double wall_time() const {
    return solve_time.wall.elapsed + eval_time.wall.elapsed;
  }


  double cpu_time_now() const {
    return solve_time.cpu.elapsed_now() + eval_time.cpu.elapsed_now();
  }


  double wall_time_now() const {
    return solve_time.wall.elapsed_now() + eval_time.wall.elapsed_now();
  }

};


template <typename Data,
          typename Result,
          template <typename> class Input,
          typename Output,
          template <typename, typename> class Objective>
inline solver_context<Data, Result, Input, Output, Objective>
make_context(
    Input<Data>&& in,
    Output&& out,
    Objective<Data, Result>&& objective,
    Data* dual_variables,
    Data* primal_variables
  ) {
  return solver_context<Data, Result, Input, Output, Objective>(
    make_dataset_train<Result>(std::move(in), std::move(out)),
    std::move(objective), dual_variables, primal_variables);
}


template <typename Data,
          typename Result,
          typename Output,
          template <typename, typename> class Objective>
inline solver_context<Data, Result, kernel_input, Output, Objective>
make_context(
    kernel_input<Data>&& in,
    Output&& out,
    Objective<Data, Result>&& objective,
    Data* dual_variables
  ) {
  return solver_context<Data, Result, kernel_input, Output, Objective>(
    make_dataset_train<Result>(std::move(in), std::move(out)),
    std::move(objective), dual_variables);
}

}

#endif
