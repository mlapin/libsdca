#ifndef SDCA_SOLVER_CONTEXT_H
#define SDCA_SOLVER_CONTEXT_H

#include "sdca/solver/dataset.h"
#include "sdca/solver/objective.h"
#include "sdca/solver/stopping_criteria.h"
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

  typedef eval_dataset<input_type, output_type, train_eval_type> train_set_type;
  typedef eval_dataset<input_type, output_type, test_eval_type> test_set_type;

  train_set_type train;
  std::vector<test_set_type> test;

  objective_type objective;
  stopping_criteria criteria;

  data_type* primal_variables;
  data_type* dual_variables;

  size_type epoch;
  solver_status status;
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


  void add_test(test_set_type&& d) { test.push_back(std::move(d)); }


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
          typename Iterator,
          template <typename, typename> class Objective>
inline solver_context<Data, Result,
                      feature_input, multiclass_output, Objective>
make_context_multiclass(
    Objective<Data, Result>&& objective,
    const size_type num_dimensions,
    const size_type num_examples,
    const Data* features,
    Iterator labels,
    Data* dual_variables,
    Data* primal_variables
  ) {
  return solver_context<Data, Result,
                        feature_input, multiclass_output, Objective>(
    make_dataset_train_feature_in_multiclass_out<Result>(
      num_dimensions, num_examples, features, labels),
    std::move(objective),
    dual_variables,
    primal_variables);
}


template <typename Data,
          typename Result,
          typename Iterator,
          template <typename, typename> class Objective>
inline solver_context<Data, Result,
                      kernel_input, multiclass_output, Objective>
make_context_multiclass(
    Objective<Data, Result>&& objective,
    const size_type num_examples,
    const Data* kernel,
    Iterator labels,
    Data* dual_variables
  ) {
  return solver_context<Data, Result,
                        kernel_input, multiclass_output, Objective>(
    make_dataset_train_kernel_in_multiclass_out<Result>(
      num_examples, kernel, labels),
    std::move(objective),
    dual_variables);
}

}

#endif
