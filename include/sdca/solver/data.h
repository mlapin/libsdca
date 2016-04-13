#ifndef SDCA_SOLVER_DATA_H
#define SDCA_SOLVER_DATA_H

#include <numeric>

#include "sdca/solver/data/dataset.h"
#include "sdca/solver/data/input.h"
#include "sdca/solver/data/output.h"
#include "sdca/solver/data/scratch.h"
#include "sdca/solver/eval/types.h"

namespace sdca {

template <typename Result = double,
          typename Data,
          template <typename> class Input,
          typename Output>
inline dataset<Input<Data>, Output, eval_train<Result, Output>>
make_dataset_train(
    Input<Data>&& in,
    Output&& out
  ) {
  return dataset<Input<Data>, Output, eval_train<Result, Output>>(
    std::move(in), std::move(out));
}


template <typename Result = double,
          typename Data,
          template <typename> class Input,
          typename Output>
inline dataset<Input<Data>, Output, eval_test<Result, Output>>
make_dataset_test(
    Input<Data>&& in,
    Output&& out
  ) {
  return dataset<Input<Data>, Output, eval_test<Result, Output>>(
    std::move(in), std::move(out));
}


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
inline kernel_input<Data>
make_input_kernel(
    const size_type num_train_examples,
    const size_type num_examples,
    const Data* kernel
  ) {
  return kernel_input<Data>(num_train_examples, num_examples, kernel);
}


template <typename Data>
inline kernel_input<Data>
make_input_kernel(
    const size_type num_examples,
    const Data* kernel
  ) {
  return kernel_input<Data>(num_examples, num_examples, kernel);
}


template <typename Iterator>
inline multiclass_output
make_output_multiclass(
    Iterator first,
    Iterator last
  ) {
  std::vector<size_type> v(first, last);
  auto minmax = validate_labels(v.begin(), v.end());
  return multiclass_output(*minmax.second + 1, v);
}


// Most efficient: A vector of labels and a vector of offsets per example
// Directly corresponds to Matlab's sparse matrix format as follows:
//    labels:  ir array (row index = class label);
//    offsets: jc array (num_labels per example = jc[j+1] - jc[j]).
template <typename Iterator>
inline multilabel_output
make_output_multilabel(
    Iterator label_first,
    Iterator label_last,
    Iterator offset_first,
    Iterator offset_last
  ) {
  std::vector<size_type> v(label_first, label_last);
  std::vector<size_type> u(offset_first, offset_last);
  auto minmax = validate_labels(v.begin(), v.end());
  const size_type num_classes = *minmax.second + 1;
  validate_labels_and_offsets(num_classes, v, u);
  return multilabel_output(num_classes, v, u);
}


// A vector of labels per example (i.e., a vector of vectors)
template <typename Type>
inline multilabel_output
make_output_multilabel(
    std::vector<std::vector<Type>>& labels
  ) {
  std::vector<size_type> v;
  std::vector<size_type> u;
  u.push_back(0); // zero offset
  std::for_each(labels.begin(), labels.end(), [&](std::vector<Type>& yi){
    v.insert(v.end(), yi.begin(), yi.end());
    u.push_back(u.back() + yi.size());
  });
  auto minmax = validate_labels(v.begin(), v.end());
  const size_type num_classes = *minmax.second + 1;
  validate_labels_and_offsets(num_classes, v, u);
  return multilabel_output(num_classes, v, u);
}


// Special case: multiclass setting (a vector of one label per example)
template <typename Iterator>
inline multilabel_output
make_output_multilabel(
    Iterator first,
    Iterator last
  ) {
  std::vector<size_type> v(first, last);
  std::vector<sdca::size_type> u(v.size() + 1);
  std::iota(u.begin(), u.end(), 0);
  auto minmax = validate_labels(v.begin(), v.end());
  const size_type num_classes = *minmax.second + 1;
  validate_labels_and_offsets(num_classes, v, u);
  return multilabel_output(num_classes, v, u);
}

}

#endif
