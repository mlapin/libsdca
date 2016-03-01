#ifndef SDCA_SOLVER_EVALUATION_H
#define SDCA_SOLVER_EVALUATION_H

#include "sdca/math/blas.h"
#include "sdca/solver/dataset.h"

namespace sdca {

template <typename Data,
          typename Context>
inline void
compute_scores(
    const size_type i,
    const size_type num_classes,
    const Context& ctx,
    const feature_input<Data>& in,
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


template <typename Data,
          typename Context>
inline void
compute_scores(
    const size_type i,
    const size_type num_classes,
    const Context& ctx,
    const kernel_input<Data>& in,
    Data* scores
  ) {
  // Let scores = A * K_i = W' * x_i
  sdca_blas_gemv(static_cast<blas_int>(num_classes),
                 static_cast<blas_int>(in.num_train_examples),
                 ctx.dual_variables,
                 in.kernel + in.num_train_examples * i,
                 scores);
}


inline void
eval_begin(

  ) {
  stats.accuracy.resize(num_classes_);
  auto acc_first = stats.accuracy.begin();
  auto acc_last = stats.accuracy.end();

  // Compute the three terms independently
  regularizer_ = objective_.regularizer_primal(D * T, primal_variables_);
  primal_loss_ = 0;
  dual_loss_ = 0;
}


template <typename Data,
          typename Objective,
          typename Output,
          typename Evaluation>
inline void
eval_dual(
    const size_type,
    const size_type,
    Objective&,
    Output&,
    Evaluation&,
    Data*,
    Data*
  ) {}


template <typename Data,
          typename Objective,
          typename Output,
          typename Evaluation>
inline void
eval_primal(
    const size_type,
    const size_type,
    Objective&,
    Output&,
    Evaluation&,
    Data*
  ) {}


template <typename Data,
          typename Result,
          template <typename, typename> class Objective>
inline void
eval_dual(
    const size_type i,
    const size_type num_classes,
    Objective<Data, Result>& obj,
    multiclass_output& out,
    eval_train<Result, multiclass_output>& eval,
    Data* variables,
    Data* scores
  ) {
  // The regularizer does not depend on the ground truth label
  eval.regularizer += obj.regularizer_dual(num_classes, variables, scores);

  // Swap the ground truth label and a label at 0
  size_type label = out.labels[i];
  std::swap(variables[0], variables[label]);

  // The dual loss computation must not modify the variables
  eval.dual_loss += obj.dual_loss(num_classes,
                                  const_cast<const Data*>(variables));

  // Put back the ground truth
  std::swap(variables[0], variables[label]);
}


template <typename Data,
          typename Result,
          template <typename, typename> class Objective,
          template <typename, typename> class Evaluation>
inline void
eval_primal(
    const size_type i,
    const size_type num_classes,
    Objective<Data, Result>& obj,
    multiclass_output& out,
    Evaluation<Result, multiclass_output>& eval,
    Data* scores
  ) {
  // Put the ground truth score at 0
  size_type label = out.labels[i];
  std::swap(scores[0], scores[label]);

  // Count correct predictions - re-orders the scores!
  auto it = std::partition(scores + 1, scores + num_classes,
    [=](const Data& x){ return x >= scores[0]; });
  eval.accuracy[std::distance(scores + 1, it)] += 1;

  // Increment the primal loss (may re-order the scores)
  eval.primal_loss += obj.primal_loss(num_classes, scores);
}


template <typename Data,
          typename Result,
          typename Context,
          typename Dataset>
inline void
evaluate_dataset(
    Context& ctx,
    Dataset& d,
    Data* scores
  ) {
  const size_type m = d.num_classes();
  const size_type n = d.num_examples();

  eval_begin();

  for (size_type i = 0; i < n; ++i) {
    Data* variables = ctx.dual_variables + m * i;

    compute_scores(i, m, ctx, d.in, scores);

    eval_dual(i, m, ctx.objective, d.out, d.evals.back(), variables, scores);

    eval_primal(i, m, ctx.objective, d.out, d.evals.back(), scores);

  }

  eval_end();
}

}

#endif
