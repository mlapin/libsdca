#ifndef SDCA_SOLVER_SCRATCH_H
#define SDCA_SOLVER_SCRATCH_H

#include <vector>

#include "sdca/solver/dataset.h"

namespace sdca {

template <typename Data,
          template <typename> class Input>
struct solver_scratch {};


template <typename Data>
struct solver_scratch<Data, feature_input> {
  typedef Data data_type;
  typedef feature_input<Data> input_type;

  std::vector<data_type> scores;
  std::vector<data_type> variables;
  std::vector<data_type> norms;


  template <typename Dataset>
  void init(const Dataset& d) {
    scores.resize(d.num_classes());
    variables.resize(d.num_classes());
    norms.resize(d.num_examples());
  }
};


template <typename Data>
struct solver_scratch<Data, kernel_input> {
  typedef Data data_type;
  typedef kernel_input<Data> input_type;

  std::vector<data_type> scores;


  template <typename Dataset>
  void init(const Dataset& d) {
    scores.resize(d.num_classes());
  }
};

}

#endif
