#ifndef SDCA_SOLVER_UPDATE_H
#define SDCA_SOLVER_UPDATE_H

#include "sdca/solver/context.h"
#include "sdca/solver/eval/scores.h"
#include "sdca/solver/scratch.h"

namespace sdca {

template <typename Data,
          typename Result,
          template <typename, typename> class Objective>
inline void
update_dual_variables(
    const size_type i,
    const size_type m,
    const Data norm2,
    const multiclass_output& out,
    const Objective<Data, Result>& obj,
    Data* variables,
    Data* scores
  ) {
  size_type label = out.labels[i];
  std::swap(variables[0], variables[label]);
  std::swap(scores[0], scores[label]);
  obj.update_dual_variables(m, norm2, variables, scores);
  std::swap(variables[0], variables[label]);
}


template <typename Data,
          typename Result,
          template <typename, typename> class Objective>
inline void
update_dual_variables(
    const size_type i,
    const size_type m,
    const Data norm2,
    const multilabel_output& out,
    const Objective<Data, Result>& obj,
    Data* variables,
    Data* scores
  ) {
  out.move_front(i, variables, scores);
  obj.update_dual_variables(m, out.num_labels(i), norm2, variables, scores);
  out.move_back(i, variables);
}


template <typename Data,
          typename Context>
inline void
update_variables(
    const size_type i,
    Context& ctx,
    solver_scratch<Data, feature_input>& scratch
  ) {
  const auto& d = ctx.train;
  const Data norm2 = scratch.norms[i];

  if (norm2 <= 0) return;

  const size_type m = d.num_classes();
  Data* scores = &scratch.scores[0];
  eval_scores(i, m, d.in, ctx, scores);

  // Update dual variables
  const blas_int M = static_cast<blas_int>(m);
  Data* var_copy = &scratch.variables[0];
  Data* variables = ctx.dual_variables + m * i;
  sdca_blas_copy(M, variables, var_copy);
  update_dual_variables(i, m, norm2, d.out, ctx.objective, variables, scores);

  // Update primal variables
  sdca_blas_axpy(M, -1, variables, var_copy);
  Data diff = sdca_blas_asum(M, var_copy);
  if (diff > std::numeric_limits<Data>::epsilon()) {
    const size_type dim = d.in.num_dimensions;
    const blas_int D = static_cast<blas_int>(dim);
    const Data* x_i = d.in.features + dim * i;
    sdca_blas_ger(D, M, -1, x_i, var_copy, ctx.primal_variables);
  }
}


template <typename Data,
          typename Context>
inline void
update_variables(
    const size_type i,
    Context& ctx,
    solver_scratch<Data, kernel_input>& scratch
  ) {
  const auto& d = ctx.train;
  const Data norm2 = d.in.kernel[d.in.num_train_examples * i + i];

  if (norm2 <= 0) return;

  const size_type m = d.num_classes();
  Data* scores = &scratch.scores[0];
  eval_scores(i, m, d.in, ctx, scores);

  Data* variables = ctx.dual_variables + m * i;
  update_dual_variables(i, m, norm2, d.out, ctx.objective, variables, scores);
}

}

#endif
