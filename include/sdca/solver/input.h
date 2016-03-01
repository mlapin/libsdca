#ifndef SDCA_SOLVER_INPUT_H
#define SDCA_SOLVER_INPUT_H

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

}

#endif
