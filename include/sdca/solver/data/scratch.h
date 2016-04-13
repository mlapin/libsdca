#ifndef SDCA_SOLVER_SCRATCH_H
#define SDCA_SOLVER_SCRATCH_H

#include <vector>

#include "sdca/math/blas.h"
#include "sdca/solver/data/input.h"

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

    const auto n = d.num_examples();
    norms.resize(n);

    const auto dim = d.num_dimensions();
    const blas_int D = static_cast<blas_int>(dim);
    const Data* features = d.in.features;
    for (size_type i = 0; i < n; ++i) {
      const Data* x_i = features + dim * i;
      norms[i] = sdca_blas_dot(D, x_i, x_i);
    }
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
