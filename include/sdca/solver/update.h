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
 * Warning: slow convergence; the implementation is not well tested
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

  const Data lip = scratch.lipschitz;
  if (lip <= 0) return;

  const size_type d = dataset.num_dimensions();
  const size_type m = dataset.num_classes();
  const blas_int D = static_cast<blas_int>(d);
  const blas_int M = static_cast<blas_int>(m);

  const Data* W = dataset.in.model; // d-by-m
  const Data* x_i_0 = ctx.primal_initial + d * i;

  // Primal and dual variables to update
  Data* x = ctx.dual_variables + m * i;
  Data* z = ctx.primal_variables + d * i;

  // Initialization
  assert(scratch.scores.size() == m);
  assert(scratch.a.size() == m);
  assert(scratch.x.size() == d);
  Data mu(1); // note: obj.prox methods expect the inverted parameter (1/rho)
  Data* r = &scratch.scores[0]; // m-dim
  Data* a = &scratch.a[0]; // m-dim
  Data* u = &scratch.x[0]; // d-dim
  sdca_blas_copy(D, x_i_0, u);

  // The linearized ADMM method, see Section 4.4.2 in [1].
  Data eps = static_cast<Data>(ctx.criteria.epsilon);
  std::size_t max_num_iter = numeric_defaults<Data>::max_num_iter();
  for (std::size_t iter = 0; iter < max_num_iter; ++iter) {
    // p = 1/L W' (Wx - z + u)
    sdca_blas_axpy(D, -1, u, z); // z = z - u
    sdca_blas_gemv(D, M, W, x, z, CblasNoTrans, 1, -1); // z = Wx - z

    // x = prox_f(x - p)
    sdca_blas_copy(M, x, a); // a = x_old
    sdca_blas_gemv(D, M, W, z, x, CblasTrans, -1/lip, 1); // x = x - 1/L W' z
    out.move_front(i, x);
    obj.prox_f(m, out.num_labels(i), mu, x, r); // r is scratch space
    out.move_back(i, x);

    // z = prox_g(Wx + u)
    sdca_blas_gemv(D, M, W, x, u, CblasNoTrans, 1, 1); // u = Wx + u
    obj.prox_g(d, lip, x_i_0, const_cast<const Data*>(u), z);

    // Stopping criterion (relative suboptimality)
    sdca_blas_axpy(M, -1, x, a); // a = a - x = x_old - x_new
    Data norm_x = sdca_blas_nrm2(M, x);
    Data norm_a = sdca_blas_nrm2(M, a); // residual
    if (norm_a <= eps * norm_x) break;

    // u = u + Wx - z = u - z
    sdca_blas_axpy(D, -1, z, u);
  }
}

}

#endif
