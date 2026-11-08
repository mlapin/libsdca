#ifndef SDCA_SOLVER_UPDATE_H
#define SDCA_SOLVER_UPDATE_H

#include <algorithm>
#include <type_traits>
#include <vector>

#include "sdca/prox/proxdef.h"
#include "sdca/solver/data/input.h"
#include "sdca/solver/data/output.h"
#include "sdca/solver/data/scratch.h"
#include "sdca/solver/eval/scores.h"
#include "sdca/solver/objective.h"
#include "sdca/solver/screening.h"

namespace sdca {

namespace detail {

// obj.update_dual_variables(args..., mode), or without mode for an objective
// that does not take one (a user-defined objective).
template <typename Objective, typename... Args>
inline auto update_block(const Objective& obj, const projection_precision mode, int,
                         Args... args) -> decltype(obj.update_dual_variables(args..., mode)) {
  return obj.update_dual_variables(args..., mode);
}
template <typename Objective, typename... Args>
inline void update_block(const Objective& obj, const projection_precision, long, Args... args) {
  obj.update_dual_variables(args...);
}

template <typename Data, typename Context, typename Output, typename Objective, typename Tag>
inline int part_step(const size_type, Context&, solver_scratch<Data, feature_input>&, const Output&,
                     const Objective&, const bool, const char*, Tag) {
  return -1;
}

// The step of example i on part of its block, the other classes held at
// zero: an exact step on that part, whose scores and update of W cost the
// part alone. The part is the label and the classes of nonzero dual, with,
// unless support_only, every class not marked in screened (if given), and
// zero classes added up to k. Returns whether the block is unchanged, or -1
// if the part is every class.
template <typename Var, typename Context, typename Objective>
inline int part_step(const size_type i, Context& ctx, solver_scratch<Var, feature_input>& scratch,
                     const multiclass_output& out, const Objective& obj, const bool support_only,
                     const char* screened, std::true_type) {
  typedef typename std::common_type<Var, double>::type Work;
  const size_type m = out.num_classes, dim = ctx.train.in.num_dimensions, y = out.labels[i];
  Var* variables = ctx.dual_variables + m * i;
  size_type* classes = &scratch.classes[0];
  size_type n = 0;
  classes[n++] = y;
  for (size_type c = 0; c < m; ++c)
    if (c != y && (variables[c] != 0 || (!support_only && !(screened && screened[c]))))
      classes[n++] = c;
  if (n == 1) return 1;  // A zero block stays zero.
  for (size_type c = 0; n <= obj.k && c < m; ++c)
    if (c != y && variables[c] == 0 && std::find(classes, classes + n, c) == classes + n)
      classes[n++] = c;
  if (n == m) return -1;

  const auto* x_i = ctx.train.in.features + dim * i;
  Var *scores = &scratch.scores[0], *old = &scratch.variables[0], *part = &scratch.part[0];
  for (size_type q = 0; q < n; ++q) {
    Work sum;
    blas_dots<1>(static_cast<blas_int>(dim), ctx.primal_variables + dim * classes[q], 0, x_i, &sum);
    scores[q] = static_cast<Var>(sum);
    old[q] = part[q] = variables[classes[q]];
  }
  // A failed step throws before the block changes.
  update_block(obj, ctx.criteria.projection, 0, n, scratch.norms[i], part, scores);

  bool same = true;
  for (size_type q = 0; q < n; ++q) {
    variables[classes[q]] = part[q];
    const Var delta = old[q] - part[q];
    if (delta == 0) continue;
    same = false;
    const Var scale = -delta;
    Var* w = ctx.primal_variables + dim * classes[q];
    for (size_type j = 0; j < dim; ++j) w[j] += static_cast<Var>(x_i[j]) * scale;
  }
  return same ? 1 : 0;
}

}  // namespace detail

template <typename Data, typename Result, template <typename, typename> class Objective>
inline void update_dual_variables(const size_type i, const size_type m, const Data norm2,
                                  const multiclass_output& out, const Objective<Data, Result>& obj,
                                  Data* variables, Data* scores,
                                  const projection_precision mode = projection_precision::float64) {
  out.move_front(i, variables, scores);
  try {
    detail::update_block(obj, mode, 0, m, norm2, variables, scores);
  } catch (...) {
    out.move_back(i, variables);
    throw;
  }
  out.move_back(i, variables);
}

template <typename Data, typename Result, template <typename, typename> class Objective>
inline void update_dual_variables(const size_type i, const size_type m, const Data norm2,
                                  const multilabel_output& out, const Objective<Data, Result>& obj,
                                  Data* variables, Data* scores,
                                  const projection_precision mode = projection_precision::float64) {
  out.move_front(i, variables, scores);
  try {
    detail::update_block(obj, mode, 0, m, out.num_labels(i), norm2, variables, scores);
  } catch (...) {
    out.move_back(i, variables);
    throw;
  }
  out.move_back(i, variables);
}

template <typename Data, typename Result, template <typename, typename> class Objective>
inline void update_dual_variables(const size_type i, const size_type m, const Data norm2,
                                  const distribution_output& out,
                                  const Objective<Data, Result>& obj, Data* variables, Data* scores,
                                  const projection_precision mode = projection_precision::float64) {
  detail::update_block(obj, mode, 0, m, out.row(i), norm2, variables, scores);
}

// A row of zero norm has zero scores: its block maximizes the dual alone.
template <typename Data, typename Result, template <typename, typename> class Objective>
inline void zero_scores_dual_variables(const size_type i, const size_type m,
                                       const multiclass_output& out,
                                       const Objective<Data, Result>& obj, Data* variables) {
  out.move_front(i, variables);
  try {
    obj.zero_scores_dual_variables(m, variables);
  } catch (...) {
    out.move_back(i, variables);  // The objective throws before writing the block.
    throw;
  }
  out.move_back(i, variables);
}

template <typename Data, typename Result, template <typename, typename> class Objective>
inline void zero_scores_dual_variables(const size_type i, const size_type m,
                                       const multilabel_output& out,
                                       const Objective<Data, Result>& obj, Data* variables) {
  out.move_front(i, variables);
  try {
    obj.zero_scores_dual_variables(m, out.num_labels(i), variables);
  } catch (...) {
    out.move_back(i, variables);  // The objective throws before writing the block.
    throw;
  }
  out.move_back(i, variables);
}

template <typename Data, typename Result, template <typename, typename> class Objective>
inline void zero_scores_dual_variables(const size_type i, const size_type m,
                                       const distribution_output& out,
                                       const Objective<Data, Result>& obj, Data* variables) {
  obj.zero_scores_dual_variables(m, out.row(i), variables);
}

// The coordinate step of example i: its dual block and W. Returns whether the
// step left the block, and so W, unchanged. Unless whole, a single-label
// hinge block steps on its label and its classes of nonzero dual alone; with
// screened (m marks), on the classes of nonzero dual or not marked. The
// scratch, W and the duals hold Var (Context::variable_type).
template <typename Var, typename Context>
inline bool update_variables(const size_type i, Context& ctx,
                             solver_scratch<Var, feature_input>& scratch, const bool whole = true,
                             const char* screened = nullptr) {
  const auto& d = ctx.train;
  const Var norm2 = scratch.norms[i];
  const size_type m = d.num_classes(), dim = d.in.num_dimensions;
  const auto* x_i = d.in.features + dim * i;
  if ((!whole || screened) && norm2 > 0) {
    const int same = detail::part_step(
        i, ctx, scratch, d.out, ctx.step_objective, !whole, screened,
        typename detail::has_part_steps<typename Context::objective_type>::type());
    if (same >= 0) return same == 1;
  }

  // Update dual variables
  Var* old = &scratch.variables[0];
  Var* variables = ctx.dual_variables + m * i;
  std::copy(variables, variables + m, old);
  if (norm2 <= 0) {
    zero_scores_dual_variables(i, m, d.out, ctx.step_objective, variables);
  } else {
    Var* scores = &scratch.scores[0];
    column_dots(dim, m, ctx.primal_variables, x_i, scores);
    try {
      update_dual_variables(i, m, norm2, d.out, ctx.step_objective, variables, scores,
                            ctx.criteria.projection);
    } catch (...) {
      // A failed step leaves the block as it was, so that W = X'A still holds.
      std::copy(old, old + m, variables);
      throw;
    }
  }

  // Update primal variables: W -= x_i (old - new)'
  bool same = true;
  for (size_type c = 0; c < m; ++c) {
    old[c] -= variables[c];
    same = same && old[c] == 0;
  }
  if (same) return true;
  rank_one_update(dim, m, x_i, old, ctx.primal_variables);
  return false;
}

template <typename Var, typename Context>
inline bool update_variables(const size_type i, Context& ctx,
                             solver_scratch<Var, kernel_input>& scratch, const bool = true,
                             const char* = nullptr) {
  const auto& d = ctx.train;
  const Var norm2 = static_cast<Var>(d.in.kernel[d.in.num_train_examples * i + i]);
  const size_type m = d.num_classes();
  Var* variables = ctx.dual_variables + m * i;

  // A zero diagonal entry of a positive semidefinite kernel zeroes its row.
  if (norm2 <= 0) {
    zero_scores_dual_variables(i, m, d.out, ctx.step_objective, variables);
    return false;
  }

  Var* scores = &scratch.scores[0];
  eval_scores(i, m, d.in, ctx, scores);

  std::vector<Var>& old = scratch.variables;
  std::copy(variables, variables + m, old.begin());
  try {
    update_dual_variables(i, m, norm2, d.out, ctx.step_objective, variables, scores,
                          ctx.criteria.projection);
  } catch (...) {
    std::copy(old.begin(), old.end(), variables);
    throw;
  }
  return std::equal(old.begin(), old.end(), variables);
}

}  // namespace sdca

#endif
