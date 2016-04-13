#ifndef SDCA_SOLVER_DATA_INPUT_H
#define SDCA_SOLVER_DATA_INPUT_H

#include <sstream>

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
    str << "features (num_dimensions: " << num_dimensions <<
           ", num_examples: " << num_examples <<
           ", precision: " << type_name<Data>() << ")";
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
    str << "kernel (num_train_examples: " << num_train_examples <<
           ", num_examples: " << num_examples <<
           ", precision: " << type_name<Data>() << ")";
    return str.str();
  }

};

}

#endif
