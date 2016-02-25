#ifndef SDCA_SOLVER_DATASET_H
#define SDCA_SOLVER_DATASET_H

#include <algorithm>
#include <stdexcept>

#include "sdca/solver/solverdef.h"

namespace sdca {

template <typename Data>
struct feature_input {
  typedef Data data_type;

  // The feature matrix must be num_dimensions-by-num_examples
  // in column-major order (i.e. num_dimensions changes fastest)
  size_type num_dimensions = 0;
  size_type num_examples = 0;
  const data_type* features = nullptr;


  feature_input(
      const size_type __num_dimensions,
      const size_type __num_examples,
      const data_type* __features
    ) :
      num_dimensions(__num_dimensions),
      num_examples(__num_examples),
      features(__features)
  {}


  inline std::string
  to_string() const {
    std::ostringstream str;
    str << "features (" << num_dimensions << "-by-" << num_examples << ")";
    return str.str();
  }

};


template <typename Data>
inline feature_input<Data>
make_input_feature(
    const size_type num_dimensions,
    const size_type num_examples,
    const Data* features
  ) {
  return feature_input<Data>(num_dimensions, num_examples, features);
}


template <typename Data>
struct kernel_input {
  typedef Data data_type;

  // The kernel matrix must be num_train_examples-by-num_examples
  // in column-major order (i.e. num_train_examples changes fastest)
  size_type num_train_examples = 0;
  size_type num_examples = 0;
  const data_type* kernel = nullptr;


  kernel_input(
      const size_type __num_train_examples,
      const size_type __num_examples,
      const data_type* __kernel
    ) :
      num_train_examples(__num_train_examples),
      num_examples(__num_examples),
      kernel(__kernel)
  {}


  inline std::string
  to_string() const {
    std::ostringstream str;
    str << "kernel (" << num_train_examples << "-by-" << num_examples << ")";
    return str.str();
  }

};


template <typename Data>
inline kernel_input<Data>
make_input_kernel(
    const size_type num_train_examples,
    const size_type num_examples,
    const Data* kernel
  ) {
  return kernel_input<Data>(num_train_examples, num_examples, kernel);
}


struct multiclass_output {
  size_type num_classes = 0;
  std::vector<size_type> labels;


  multiclass_output(
      const size_type __num_classes,
      std::vector<size_type>& __labels
    ) :
      num_classes(__num_classes),
      labels(std::move(__labels))
  {}


  multiclass_output(multiclass_output&&) = default;


  inline std::string
  to_string() const {
    std::ostringstream str;
    str << "labels (" << num_classes << "-by-" << labels.size() << ")";
    return str.str();
  }

};


template <typename Iterator>
inline multiclass_output
make_output_multiclass(
    Iterator first,
    Iterator last
  ) {
  std::vector<size_type> v(first, last);
  auto minmax = std::minmax_element(v.begin(), v.end());
  if (*minmax.first == 1) {
    std::for_each(v.begin(), v.end(), [](size_type &x){ x -= 1; });
  } else if (*minmax.first != 0) {
    throw std::invalid_argument("Invalid class labels range.");
  }

  return multiclass_output(*minmax.second + 1, v);
}


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

  std::vector<eval_type> eval;


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
                    train_point<Result>>
make_dataset_train_feature_in_multiclass_out(
    const size_type num_dimensions,
    const size_type num_examples,
    const Data* features,
    Iterator labels
  ) {
  return eval_dataset<feature_input<Data>,
                      multiclass_output,
                      train_point<Result>>(
    make_input_feature(num_dimensions, num_examples, features),
    make_output_multiclass(labels, labels + num_examples));
}


template <typename Result = double,
          typename Data,
          typename Iterator>
inline eval_dataset<kernel_input<Data>,
                    multiclass_output,
                    train_point<Result>>
make_dataset_train_kernel_in_multiclass_out(
    const size_type num_examples,
    const Data* kernel,
    Iterator labels
  ) {
  return eval_dataset<kernel_input<Data>,
                      multiclass_output,
                      train_point<Result>>(
    make_input_kernel(num_examples, num_examples, kernel),
    make_output_multiclass(labels, labels + num_examples));
}


template <typename Result = double,
          typename Data,
          typename Iterator>
inline eval_dataset<feature_input<Data>,
                    multiclass_output,
                    test_point<Result>>
make_dataset_test_feature_in_multiclass_out(
    const size_type num_dimensions,
    const size_type num_examples,
    const Data* features,
    Iterator labels
  ) {
  return eval_dataset<feature_input<Data>,
                      multiclass_output,
                      test_point<Result>>(
    make_input_feature(num_dimensions, num_examples, features),
    make_output_multiclass(labels, labels + num_examples));
}


template <typename Result = double,
          typename Data,
          typename Iterator>
inline eval_dataset<kernel_input<Data>,
                    multiclass_output,
                    test_point<Result>>
make_dataset_test_kernel_in_multiclass_out(
    const size_type num_train_examples,
    const size_type num_examples,
    const Data* kernel,
    Iterator labels
  ) {
  return eval_dataset<kernel_input<Data>,
                      multiclass_output,
                      test_point<Result>>(
    make_input_kernel(num_train_examples, num_examples, kernel),
    make_output_multiclass(labels, labels + num_examples));
}


}

#endif
