#ifndef SDCA_SOLVER_DATA_INPUT_H
#define SDCA_SOLVER_DATA_INPUT_H

#include <sstream>
#include <stdexcept>

#include "sdca/utility/types.h"

namespace sdca {

template <typename Data>
struct feature_input {
  typedef Data data_type;

  // The feature matrix must be num_dimensions-by-num_examples
  // in column-major order (i.e. num_dimensions changes fastest)
  size_type num_dimensions = 0;
  size_type num_examples = 0;
  const data_type* features = nullptr;

  feature_input(const size_type num_dimensions_, const size_type num_examples_,
                const data_type* features_)
      : num_dimensions(num_dimensions_), num_examples(num_examples_), features(features_) {}

  inline std::string to_string() const {
    std::ostringstream str;
    str << "features (num_dimensions: " << num_dimensions << ", num_examples: " << num_examples
        << ", precision: " << type_name<Data>() << ")";
    return str.str();
  }
};

template <typename Data>
struct kernel_input {
  typedef Data data_type;

  // The kernel matrix must be num_train_examples-by-num_examples
  // in column-major order (i.e. num_train_examples changes fastest)
  size_type num_train_examples = 0;
  size_type num_examples = 0;
  const data_type* kernel = nullptr;

  kernel_input(const size_type num_train_examples_, const size_type num_examples_,
               const data_type* kernel_)
      : num_train_examples(num_train_examples_), num_examples(num_examples_), kernel(kernel_) {}

  inline std::string to_string() const {
    std::ostringstream str;
    str << "kernel (num_train_examples: " << num_train_examples
        << ", num_examples: " << num_examples << ", precision: " << type_name<Data>() << ")";
    return str.str();
  }
};

template <typename Data>
inline void validate_test_input(const feature_input<Data>& train, const feature_input<Data>& test) {
  if (train.num_dimensions != test.num_dimensions)
    throw std::invalid_argument("Training and test feature dimensions differ");
}

template <typename Data>
inline void validate_test_input(const kernel_input<Data>& train, const kernel_input<Data>& test) {
  if (train.num_train_examples != test.num_train_examples)
    throw std::invalid_argument("Test kernel must have one row per training example");
}

}  // namespace sdca

#endif
