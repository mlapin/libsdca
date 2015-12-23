#ifndef SDCA_PYTHON_PY_PROX_H
#define SDCA_PYTHON_PY_PROX_H

#include "../prox/prox.h"
#include "py_util.h"

#include <string>
#include <iostream>

using namespace sdca;

namespace py_prox {

template <typename Data,
          typename Result,
          typename Summation>
void
py_main(
    Data* A,
    Result* X,
    proxOpts<Result>* opts,
    const Summation sum
    ) {

  Result lo = opts->lo;
  Result hi = opts->hi;
  Result rhs = opts->rhs;
  Result rho = opts->rho;

  std::ptrdiff_t m = static_cast<std::ptrdiff_t>(opts->m);
  std::ptrdiff_t n = static_cast<std::ptrdiff_t>(opts->n);
  std::ptrdiff_t k = static_cast<std::ptrdiff_t>(opts->k);

  std::vector<Data> aux(static_cast<std::size_t>(m));
  Data* first = X;
  Data* last = first + m*n;
  Data* aux_first = &aux[0];
  Data* aux_last = aux_first + m;
  std::string prox = opts->prox;
  if (prox == "knapsack" || prox == "knapsack_eq") {
    prox_knapsack_eq<Data*, Result, Summation>(
      m, first, last, aux_first, aux_last, lo, hi, rhs, sum);
  } else if (prox == "knapsack_le") {
    prox_knapsack_le<Data*, Result, Summation>(
      m, first, last, aux_first, aux_last, lo, hi, rhs, sum);
  } else if (prox == "knapsack_le_biased") {
    prox_knapsack_le_biased<Data*, Result, Summation>(
      m, first, last, aux_first, aux_last, lo, hi, rhs, rho, sum);
  } else if (prox == "topk_simplex") {
    prox_topk_simplex<Data*, Result, Summation>(
      m, first, last, aux_first, aux_last, k, rhs, sum);
  } else if (prox == "topk_simplex_biased") {
    prox_topk_simplex_biased<Data*, Result, Summation>(
      m, first, last, aux_first, aux_last, k, rhs, rho, sum);
  } else if (prox == "topk_cone") {
    prox_topk_cone<Data*, Result, Summation>(
      m, first, last, aux_first, aux_last, k, sum);
  } else if (prox == "topk_cone_biased") {
    prox_topk_cone_biased<Data*, Result, Summation>(
      m, first, last, aux_first, aux_last, k, rho, sum);
  } else {
    std::cout << "You passed: " << opts->prox << std::endl;
    throw std::runtime_error(
        "invalid value passed to py_prox in opts.prox"
        "struct.  See code for list of valid options."
      );
  }
}

// py_prox(A, X, opts)
// result is stored in X
template <typename Data,
          typename Result>
void
py_prox(
    Data* A,
    Result* X,
    proxOpts<Result>* opts
    ) {
            //  copy A into X
  std::ptrdiff_t m = static_cast<std::ptrdiff_t>(opts->m);
  std::ptrdiff_t n = static_cast<std::ptrdiff_t>(opts->n);
  memcpy(X, A, sizeof(Data) * m * n);
  if (opts->summation == "standard" || opts->summation == "default") {
    std_sum<Data*, Result> sum;
    py_main<Data, Result, std_sum<Data*, Result>>(
      A, X, opts, sum);
  } else if (opts->summation == "kahan") {
    kahan_sum<Data*, Result> sum;
    py_main<Data, Result, kahan_sum<Data*, Result>>(
      A, X, opts, sum);
  } else {
    std::cout << "You passed: " << opts->summation << std::endl;
    throw std::runtime_error(
        "invalid value passed to py_prox in opts.summation"
        "Valid options are standard or kahan."
      );
  }
}

// py_prox(A, opts)
// result overwrites A
template <typename Data,
          typename Result>
void
py_prox_inplace(
    Data* A,
    proxOpts<Result>* opts
  ) {
            //  let X point to A
  Result* X = const_cast<Data*>(A);
  if (opts->summation == "standard" || opts->summation == "default") {
    std_sum<Data*, Data> sum;
    py_main<Data, Result, std_sum<Data*, Result>>(
        A, X, opts, sum);
  } else if (opts->summation == "kahan") {
    kahan_sum<Data*, Data> sum;
    py_main<Data, Result, kahan_sum<Data*, Result>>(
        A, X, opts, sum);
  } else {
    std::cout << "You passed: " << opts->summation << std::endl;
    throw std::runtime_error(
        "invalid value passed to py_prox in opts.summation"
        "Valid options are standard or kahan."
      );
  }
}

// cython annoyance, needed if we want to release the gil
void py_prox_(
    double* A,
    double* X,
    proxOpts<double>* opts
  ) {
  py_prox<double, double>(A, X, opts);
}

void py_prox_inplace_(
    double* A,
    proxOpts<double>* opts
  ) {
  py_prox_inplace<double, double>(A, opts);
}

}

#endif
