#ifndef SDCA_SOLVER_CONTEXT_H
#define SDCA_SOLVER_CONTEXT_H

#include "sdca/solver/dataset.h"
#include "sdca/utility/stopwatch.h"

namespace sdca {

template <typename Data,
          typename Result,
          template <typename> class Input,
          typename Output>
struct solver_context {
  typedef Data data_type;
  typedef Result result_type;
  typedef Input<Data> input_type;
  typedef Output output_type;

  typedef eval_dataset<Input<Data>, Output, train_point<Result>> train_set_type;
  typedef eval_dataset<Input<Data>, Output, test_point<Result>> test_set_type;

  train_set_type train;
  std::vector<test_set_type> test;
  stopping_criteria criteria;

  data_type* primal_variables;
  data_type* dual_variables;

  stopwatch solve_time;
  stopwatch eval_time;


  solver_context(
      train_set_type&& __train_set,
      data_type* __dual_variables,
      data_type* __primal_variables = nullptr
    ) :
      train(std::move(__train_set)),
      primal_variables(__primal_variables),
      dual_variables(__dual_variables)
  {}


  void add_test(test_set_type&& d) { test.push_back(std::move(d)); }

  bool is_dual() const { return primal_variables == nullptr; }

};


template <typename Result = double,
          typename Data,
          typename Iterator>
inline solver_context<Data, Result, feature_input, multiclass_output>
make_context_multiclass(
    const size_type num_dimensions,
    const size_type num_examples,
    const Data* features,
    Iterator labels,
    Data* dual_variables,
    Data* primal_variables
  ) {
  return solver_context<Data, Result, feature_input, multiclass_output>(
    make_dataset_train_feature_in_multiclass_out<Result>(
      num_dimensions, num_examples, features, labels),
    dual_variables,
    primal_variables);
}


template <typename Result = double,
          typename Data,
          typename Iterator>
inline solver_context<Data, Result, kernel_input, multiclass_output>
make_context_multiclass(
    const size_type num_examples,
    const Data* kernel,
    Iterator labels,
    Data* dual_variables
  ) {
  return solver_context<Data, Result, kernel_input, multiclass_output>(
    make_dataset_train_kernel_in_multiclass_out<Result>(
      num_examples, kernel, labels),
    dual_variables);
}

}

#endif
