#ifndef SDCA_SOLVER_DATASET_H
#define SDCA_SOLVER_DATASET_H

#include "sdca/solver/input.h"
#include "sdca/solver/output.h"
#include "sdca/solver/eval_type.h"

namespace sdca {

template <typename Input,
          typename Output>
struct dataset {
  typedef Input input_type;
  typedef Output output_type;

  input_type in;
  output_type out;


  dataset(
      Input&& __in,
      Output&& __out
    ) :
      in(std::move(__in)),
      out(std::move(__out))
  {}


  inline size_type num_dimensions() const { return in.num_dimensions; }

  inline size_type num_examples() const { return in.num_examples; }

  inline size_type num_classes() const { return out.num_classes; }


  inline std::string
  to_string() const {
    std::ostringstream str;
    str << in.to_string() << ", " << out.to_string();
    return str.str();
  }

};


template <typename Input,
          typename Output,
          typename Evaluation>
struct eval_dataset
    : public dataset<Input, Output> {
  typedef dataset<Input, Output> base;

  typedef Input input_type;
  typedef Output output_type;
  typedef Evaluation eval_type;

  std::vector<eval_type> evals;


  eval_dataset(
      Input&& __in,
      Output&& __out
    ) :
      base::dataset(std::move(__in), std::move(__out))
  {}

};


template <typename Data,
          typename Iterator>
inline dataset<feature_input<Data>,
               multiclass_output>
make_dataset_feature_in_multiclass_out(
    const size_type num_dimensions,
    const size_type num_examples,
    const Data* features,
    Iterator labels
  ) {
  return dataset<feature_input<Data>,
                 multiclass_output>(
    make_input_feature(num_dimensions, num_examples, features),
    make_output_multiclass(labels, labels + num_examples));
}


template <typename Data,
          typename Iterator>
inline dataset<kernel_input<Data>,
               multiclass_output>
make_dataset_kernel_in_multiclass_out(
    const size_type num_train_examples,
    const size_type num_examples,
    const Data* kernel,
    Iterator labels
  ) {
  return dataset<kernel_input<Data>,
                 multiclass_output>(
    make_input_kernel(num_train_examples, num_examples, kernel),
    make_output_multiclass(labels, labels + num_examples));
}


template <typename Result = double,
          typename Data,
          typename Iterator>
inline eval_dataset<feature_input<Data>,
                    multiclass_output,
                    eval_train<Result, multiclass_output>>
make_dataset_train_feature_in_multiclass_out(
    const size_type num_dimensions,
    const size_type num_examples,
    const Data* features,
    Iterator labels
  ) {
  return eval_dataset<feature_input<Data>,
                      multiclass_output,
                      eval_train<Result, multiclass_output>>(
    make_input_feature(num_dimensions, num_examples, features),
    make_output_multiclass(labels, labels + num_examples));
}


template <typename Result = double,
          typename Data,
          typename Iterator>
inline eval_dataset<kernel_input<Data>,
                    multiclass_output,
                    eval_train<Result, multiclass_output>>
make_dataset_train_kernel_in_multiclass_out(
    const size_type num_examples,
    const Data* kernel,
    Iterator labels
  ) {
  return eval_dataset<kernel_input<Data>,
                      multiclass_output,
                      eval_train<Result, multiclass_output>>(
    make_input_kernel(num_examples, num_examples, kernel),
    make_output_multiclass(labels, labels + num_examples));
}


template <typename Result = double,
          typename Data,
          typename Iterator>
inline eval_dataset<feature_input<Data>,
                    multiclass_output,
                    eval_test<Result, multiclass_output>>
make_dataset_test_feature_in_multiclass_out(
    const size_type num_dimensions,
    const size_type num_examples,
    const Data* features,
    Iterator labels
  ) {
  return eval_dataset<feature_input<Data>,
                      multiclass_output,
                      eval_test<Result, multiclass_output>>(
    make_input_feature(num_dimensions, num_examples, features),
    make_output_multiclass(labels, labels + num_examples));
}


template <typename Result = double,
          typename Data,
          typename Iterator>
inline eval_dataset<kernel_input<Data>,
                    multiclass_output,
                    eval_test<Result, multiclass_output>>
make_dataset_test_kernel_in_multiclass_out(
    const size_type num_train_examples,
    const size_type num_examples,
    const Data* kernel,
    Iterator labels
  ) {
  return eval_dataset<kernel_input<Data>,
                      multiclass_output,
                      eval_test<Result, multiclass_output>>(
    make_input_kernel(num_train_examples, num_examples, kernel),
    make_output_multiclass(labels, labels + num_examples));
}

}

#endif
