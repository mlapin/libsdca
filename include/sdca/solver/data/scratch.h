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

  std::vector<data_type> norms;
  std::vector<data_type> scores;
  std::vector<data_type> variables;


  template <typename Dataset>
  void init(const Dataset& d) {
    auto n = d.num_examples();
    norms.resize(n);

    scores.resize(d.num_classes());
    variables.resize(d.num_classes());

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


template <typename Data>
struct solver_scratch<Data, model_input> {
  typedef Data data_type;
  typedef model_input<Data> input_type;

  data_type lipschitz = 0;
  std::vector<data_type> scores;
  std::vector<data_type> a;
  std::vector<data_type> x;


  template <typename Dataset>
  void init(const Dataset& d) {
    auto dim = d.num_dimensions();
    auto m = d.num_classes();
    scores.resize(m);
    a.resize(m);
    x.resize(dim);

    lipschitz = sdca_blas_nrm2(static_cast<blas_int>(dim * m), d.in.model);
    lipschitz *= lipschitz / 2;
  }
};

}

#endif
