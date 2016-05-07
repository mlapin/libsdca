#ifndef SDCA_SOLVER_UPDATE_H
#define SDCA_SOLVER_UPDATE_H

#include "sdca/prox/proxdef.h"
#include "sdca/solver/data/input.h"
#include "sdca/solver/data/output.h"
#include "sdca/solver/data/scratch.h"
#include "sdca/solver/eval/scores.h"

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
  // TODO: introduce num_labels to obj.update_dual_variables
  // and merge this and the function below
  out.move_front(i, variables, scores);
  obj.update_dual_variables(m, norm2, variables, scores);
  out.move_back(i, variables);
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


template <typename Data>
inline Data
guess_lipschitz_constant(
  const size_type i,
  const multiclass_output& out,
  const solver_scratch<Data, model_input>& scratch
  ) {
  return scratch.norms[out.labels[i]];
}


template <typename Data>
inline Data
guess_lipschitz_constant(
  const size_type i,
  const multilabel_output& out,
  const solver_scratch<Data, model_input>& scratch
  ) {
  Data max(std::numeric_limits<Data>::lowest());
  auto label = out.labels_cbegin(i);
  auto label_end = out.labels_cend(i);
  for (; label != label_end; ++label) {
    max = (scratch.norms[*label] > max) ? scratch.norms[*label] : max;
  }
  return max;
}


/*
 * This update step is based on the ADMM splitting scheme.
 *
 * [1] Parikh N, Boyd S.
 *     Proximal Algorithms.
 *     Foundations and Trends in Optimization. 2014 Jan 13;1(3):127-239.
 *
 * [2] Boyd S, Parikh N, Chu E, Peleato B, Eckstein J.
 *     Distributed optimization and statistical learning via
 *     the alternating direction method of multipliers.
 *     Foundations and Trends in Machine Learning. 2011 Jan 1;3(1):1-22.
 */
template <typename Data,
          typename Context>
inline void
update_variables(
    const size_type i,
    Context& ctx,
    solver_scratch<Data, model_input>& scratch
  ) {
  const auto& obj = ctx.objective;
  const auto& dataset = ctx.train;
  const auto& out = dataset.out;

  const Data lambda = guess_lipschitz_constant(i, out, scratch);
  if (lambda <= 0) return;

  const size_type d = dataset.num_dimensions();
  const size_type m = dataset.num_classes();
  const blas_int M = static_cast<blas_int>(m);

  const Data* W = dataset.in.model; // d-by-m
  const Data* x_i_0 = ctx.primal_initial + d * i;

  // Primal and dual variables to update
  Data* x_i = ctx.primal_variables + d * i;
  Data* z = ctx.dual_variables + m * i;

  // x, r, u - m-dimensional arrays (scratch space)
  Data* x = &scratch.scores[0];
  Data* r = &scratch.r[0];
  Data* u = &scratch.u[0];

  // Initialize u = 0; x and r are set below
  Data norm_u(0), r_pri(0), r_duo(0);
  std::fill(scratch.u.begin(), scratch.u.end(), 0);

  // The linearized ADMM method, see Section 4.4.2 in [1].
  Data eps = numeric_defaults<Data>::epsilon_relative();
  std::size_t max_num_iter = numeric_defaults<Data>::max_num_iter();
  for (std::size_t iter = 0; iter < max_num_iter; ++iter) {
    // p =
    // x = prox_f(x - p)
    sdca_blas_copy(M, z, x);
    sdca_blas_axpy(M, -1, u, x); // x = z - u
    out.move_front(i, x);
    obj.prox_f(m, out.num_labels(i), lambda, x, r); // r is scratch space
    out.move_back(i, x);

    Data norm_x = sdca_blas_nrm2(M, x);
    sdca_blas_copy(M, x, r); // r = x

    // z = prox_g(x + u)
    sdca_blas_swap(M, x, z); // x = z_old
    sdca_blas_axpy(M, 1, u, z); // z = x + u
    obj.prox_g(d, m, lambda, W, x_i_0, z, x_i);

    Data norm_z = sdca_blas_nrm2(M, z);

    // r = r - z = x - z        -- primal residual
    // x = x - z = z_old - z    -- dual residual
    sdca_blas_axpy(M, -1, z, r);
    sdca_blas_axpy(M, -1, z, x);

    // Stopping criterion (relative suboptimality)
    r_pri = sdca_blas_nrm2(M, r);
    r_duo = sdca_blas_nrm2(M, x);

    Data one(1);
    bool pri_optimal = r_pri <= eps * std::max(one, std::max(norm_x, norm_z));
    bool duo_optimal = r_duo <= eps * std::max(one, norm_u);
    if (pri_optimal && duo_optimal) break;

    // u = u + x - z = u + r
    sdca_blas_axpy(M, 1, r, u);
    norm_u = sdca_blas_nrm2(M, u);
  }

  // Recompute x_i if the last change in z was significant
//  if (r_duo > std::numeric_limits<Data>::epsilon()) {
//    obj.compute_features(d, m, W, x_i_0, z, x_i);
//  }
}

}

#endif
