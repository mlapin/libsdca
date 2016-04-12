#ifndef SDCA_SOLVER_CONTEXT_H
#define SDCA_SOLVER_CONTEXT_H

#include "sdca/solver/data.h"
#include "sdca/solver/objective.h"
#include "sdca/solver/solverdef.h"
#include "sdca/utility/stopwatch.h"

namespace sdca {

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
    str << "status: " << solver_status_name(status);
    if (train.evals.size() > 0) {
      str << ", " << train.evals.back().to_string(true);
    }
    str << ", cpu_time: " << cpu_time() <<
             " (solve: " << solve_time.cpu.elapsed <<
             ", eval: " << eval_time.cpu.elapsed << ")"
           ", wall_time: " << wall_time() <<
             " (solve: " << solve_time.wall.elapsed <<
             ", eval: " << eval_time.wall.elapsed << ")";
    return str.str();
  }


  void add_test(input_type&& in, output_type&& out) {
    // TODO: verify dimensions
    test.emplace_back(std::move(in), std::move(out));
  }


  std::string status_name() const { return solver_status_name(status); }


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
