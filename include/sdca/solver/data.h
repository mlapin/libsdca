#ifndef SDCA_SOLVER_DATA_H
#define SDCA_SOLVER_DATA_H

#include <numeric>
#include <type_traits>

#include "sdca/solver/data/dataset.h"
#include "sdca/solver/data/input.h"
#include "sdca/solver/data/output.h"
#include "sdca/solver/data/scratch.h"
#include "sdca/solver/eval/types.h"

namespace sdca {

template <typename Result = double, typename Data, template <typename> class Input, typename Output>
inline dataset<Input<Data>, Output, eval_train<Result, Output>> make_dataset_train(Input<Data>&& in,
                                                                                   Output&& out) {
  return dataset<Input<Data>, Output, eval_train<Result, Output>>(std::move(in), std::move(out));
}

template <typename Result = double, typename Data, template <typename> class Input, typename Output>
inline dataset<Input<Data>, Output, eval_test<Result, Output>> make_dataset_test(Input<Data>&& in,
                                                                                 Output&& out) {
  return dataset<Input<Data>, Output, eval_test<Result, Output>>(std::move(in), std::move(out));
}

template <typename Data>
inline feature_input<Data> make_input_feature(const size_type num_dimensions,
                                              const size_type num_examples, const Data* features) {
  return feature_input<Data>(num_dimensions, num_examples, features);
}

template <typename Data>
inline kernel_input<Data> make_input_kernel(const size_type num_train_examples,
                                            const size_type num_examples, const Data* kernel) {
  return kernel_input<Data>(num_train_examples, num_examples, kernel);
}

template <typename Data>
inline kernel_input<Data> make_input_kernel(const size_type num_examples, const Data* kernel) {
  return kernel_input<Data>(num_examples, num_examples, kernel);
}

// One label in [0, num_classes) per example.
template <typename Iterator>
inline multiclass_output make_output_multiclass(Iterator first, Iterator last,
                                                const size_type num_classes) {
  return multiclass_output(num_classes, copy_class_labels(first, last));
}

// Label sets as a label array and offsets (num_examples + 1, from 0): the
// sorted labels of example i are labels[offsets[i]] .. labels[offsets[i + 1] - 1],
// as the row indices (ir) and column offsets (jc) of a sparse matrix.
template <typename Iterator, typename OffsetIterator>
inline multilabel_output make_output_multilabel(Iterator label_first, Iterator label_last,
                                                OffsetIterator offset_first,
                                                OffsetIterator offset_last,
                                                const size_type num_classes) {
  return multilabel_output(num_classes, copy_class_labels(label_first, label_last),
                           copy_class_labels(offset_first, offset_last));
}

// Label sets as one sorted label vector per example.
template <typename Type>
inline multilabel_output make_output_multilabel(const std::vector<std::vector<Type>>& labels,
                                                const size_type num_classes) {
  std::vector<size_type> v;
  std::vector<size_type> u(1, 0);
  for (const auto& yi : labels) {
    const auto checked = copy_class_labels(yi.begin(), yi.end());
    v.insert(v.end(), checked.begin(), checked.end());
    u.push_back(v.size());
  }
  return multilabel_output(num_classes, std::move(v), std::move(u));
}

template <typename Data>
inline distribution_output make_output_distribution(const size_type num_classes,
                                                    const size_type num_examples,
                                                    const Data* targets) {
  if (!targets) throw std::invalid_argument("Required buffers must not be NULL");
  std::vector<double> q(targets, targets + num_examples * num_classes);
  return distribution_output(num_classes, num_examples, std::move(q));
}

}  // namespace sdca

#endif
