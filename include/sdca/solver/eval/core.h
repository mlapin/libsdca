#ifndef SDCA_SOLVER_EVAL_CORE_H
#define SDCA_SOLVER_EVAL_CORE_H

#include <cmath>
#include <limits>
#include <numeric>

#include "sdca/math/blas.h"
#include "sdca/solver/data.h"

namespace sdca {

template <typename Result, typename Input, typename Output,
          template <typename, typename> class Evaluation>
inline Evaluation<Result, Output>& eval_begin(
    dataset<Input, Output, Evaluation<Result, Output>>& d) {
  // Create a new eval using the default c'tor
  d.evals.resize(d.evals.size() + 1);
  return d.evals.back();
}

template <typename Result, typename Input, template <typename, typename> class Evaluation>
inline Evaluation<Result, multiclass_output>& eval_begin(
    dataset<Input, multiclass_output, Evaluation<Result, multiclass_output>>& d) {
  // Create a new eval using the default c'tor
  d.evals.resize(d.evals.size() + 1);
  Evaluation<Result, multiclass_output>& eval = d.evals.back();

  // Allocate storage for the multiclass performance metric
  eval.accuracy.resize(d.out.num_classes);
  return eval;
}

template <typename Result, typename Input, template <typename, typename> class Evaluation>
inline Evaluation<Result, distribution_output>& eval_begin(
    dataset<Input, distribution_output, Evaluation<Result, distribution_output>>& d) {
  d.evals.resize(d.evals.size() + 1);
  Evaluation<Result, distribution_output>& eval = d.evals.back();
  eval.accuracy.resize(d.out.num_classes);
  return eval;
}

// Combines the terms of a training evaluation (with the objective in the
// evaluation's arithmetic); bound is their rounding in units of epsilon.
template <typename Result, typename Output, typename Context, typename Objective>
inline void eval_finish(const Context& ctx, const Objective& obj, const Result bound,
                        eval_train<Result, Output>& e) {
  // Set epoch and timings
  e.epoch = ctx.epoch;
  e.solve_time_cpu = ctx.solve_time.cpu.elapsed_now();
  e.solve_time_wall = ctx.solve_time.wall.elapsed_now();
  e.eval_time_cpu = ctx.eval_time.cpu.elapsed_now();
  e.eval_time_wall = ctx.eval_time.wall.elapsed_now();

  // Compute the overall primal/dual objectives and their individual terms
  obj.update_all(e.primal, e.dual, e.primal_loss, e.dual_loss, e.primal_regularizer,
                 e.dual_regularizer);

  // Epsilon is twice the unit roundoff: a factor of two over the first-order
  // bound, which gets the final sums here.
  e.gap_error = std::numeric_limits<Result>::epsilon() *
                (bound + 2 * (std::abs(e.primal) + std::abs(e.dual)));
}

template <typename Int, typename Result, typename Context, typename Objective>
inline void eval_end(const Int num_classes, const Int num_examples, const Context& ctx,
                     const Objective& obj, const Result bound,
                     eval_train<Result, multiclass_output>& e) {
  eval_finish(ctx, obj, bound, e);

  // Top-k accuracies for all k
  std::partial_sum(e.accuracy.begin(), e.accuracy.end(), e.accuracy.begin());
  Result coeff(1 / static_cast<Result>(num_examples));
  sdca_blas_scal(static_cast<blas_int>(num_classes), coeff, &e.accuracy[0]);
}

template <typename Int, typename Result, typename Context, typename Objective>
inline void eval_end(const Int, const Int num_examples, const Context& ctx, const Objective& obj,
                     const Result bound, eval_train<Result, multilabel_output>& e) {
  eval_finish(ctx, obj, bound, e);
  e.rank_loss /= static_cast<Result>(num_examples);
}

template <typename Int, typename Result, typename Objective>
inline void eval_end(const Int num_classes, const Int num_examples, const Objective& obj,
                     eval_test<Result, multiclass_output>& e) {
  obj.update_primal_loss(e.primal_loss);

  // Top-k accuracies for all k
  std::partial_sum(e.accuracy.begin(), e.accuracy.end(), e.accuracy.begin());
  Result coeff(1 / static_cast<Result>(num_examples));
  sdca_blas_scal(static_cast<blas_int>(num_classes), coeff, &e.accuracy[0]);
}

template <typename Int, typename Result, typename Objective>
inline void eval_end(const Int, const Int num_examples, const Objective& obj,
                     eval_test<Result, multilabel_output>& e) {
  obj.update_primal_loss(e.primal_loss);
  e.rank_loss /= static_cast<Result>(num_examples);
}

template <typename Int, typename Result, typename Context, typename Objective>
inline void eval_end(const Int num_classes, const Int num_examples, const Context& ctx,
                     const Objective& obj, const Result bound,
                     eval_train<Result, distribution_output>& e) {
  eval_finish(ctx, obj, bound, e);
  std::partial_sum(e.accuracy.begin(), e.accuracy.end(), e.accuracy.begin());
  Result coeff(1 / static_cast<Result>(num_examples));
  sdca_blas_scal(static_cast<blas_int>(num_classes), coeff, &e.accuracy[0]);
}

template <typename Int, typename Result, typename Objective>
inline void eval_end(const Int num_classes, const Int num_examples, const Objective& obj,
                     eval_test<Result, distribution_output>& e) {
  obj.update_primal_loss(e.primal_loss);
  std::partial_sum(e.accuracy.begin(), e.accuracy.end(), e.accuracy.begin());
  Result coeff(1 / static_cast<Result>(num_examples));
  sdca_blas_scal(static_cast<blas_int>(num_classes), coeff, &e.accuracy[0]);
}

}  // namespace sdca

#endif
