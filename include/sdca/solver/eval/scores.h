#ifndef SDCA_SOLVER_EVAL_SCORES_H
#define SDCA_SOLVER_EVAL_SCORES_H

#include "sdca/math/blas.h"
#include "sdca/solver/data/input.h"

namespace sdca {

template <typename Int,
          typename Data,
          typename Context>
inline void
eval_scores(
    const Int i,
    const Int num_classes,
    const feature_input<Data>& in,
    const Context& ctx,
    Data* scores
  ) {
  // Let scores = W' * x_i
  sdca_blas_gemv(static_cast<blas_int>(in.num_dimensions),
                 static_cast<blas_int>(num_classes),
                 ctx.primal_variables,
                 in.features + in.num_dimensions * i,
                 scores,
                 CblasTrans);
}


template <typename Int,
          typename Data,
          typename Context>
inline void
eval_scores(
    const Int i,
    const Int num_classes,
    const kernel_input<Data>& in,
    const Context& ctx,
    Data* scores
  ) {
  // Let scores = A * K_i = W' * x_i
  sdca_blas_gemv(static_cast<blas_int>(num_classes),
                 static_cast<blas_int>(in.num_train_examples),
                 ctx.dual_variables,
                 in.kernel + in.num_train_examples * i,
                 scores);
}


template <typename Int,
          typename Data,
          typename Context>
inline void
eval_scores(
    const Int i,
    const Int num_classes,
    const model_input<Data>& in,
    const Context& ctx,
    Data* scores
  ) {
  // Let scores = W' * x_i
  sdca_blas_gemv(static_cast<blas_int>(in.num_dimensions),
                 static_cast<blas_int>(num_classes),
                 in.model,
                 ctx.primal_variables + in.num_dimensions * i,
                 scores,
                 CblasTrans);
}

}

#endif
