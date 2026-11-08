#ifndef SDCA_SOLVER_EVAL_DUAL_H
#define SDCA_SOLVER_EVAL_DUAL_H

#include "sdca/solver/data/output.h"

namespace sdca {

// The feasible block (class order) of stored duals in the evaluation's
// arithmetic, and its conjugate loss; the stored duals are not modified.
template <typename Work, typename Data, typename Objective>
inline Work eval_dual_block(const size_type i, const multiclass_output& out, const Objective& obj,
                            const Data* variables, Work* block) {
  const size_type m = out.num_classes;
  for (size_type j = 0; j < m; ++j) block[j] = static_cast<Work>(variables[j]);
  out.move_front(i, block);
  obj.feasible_dual(m, block);
  const Work loss = obj.dual_loss(m, const_cast<const Work*>(block));
  out.move_back(i, block);
  return loss;
}

template <typename Work, typename Data, typename Objective>
inline Work eval_dual_block(const size_type i, const multilabel_output& out, const Objective& obj,
                            const Data* variables, Work* block) {
  const size_type m = out.num_classes, p = out.num_labels(i);
  for (size_type j = 0; j < m; ++j) block[j] = static_cast<Work>(variables[j]);
  out.move_front(i, block);
  obj.feasible_dual(m, p, block);
  const Work loss = obj.dual_loss(m, p, const_cast<const Work*>(block));
  out.move_back(i, block);
  return loss;
}

template <typename Work, typename Data, typename Objective>
inline Work eval_dual_block(const size_type i, const distribution_output& out, const Objective& obj,
                            const Data* variables, Work* block) {
  const size_type m = out.num_classes;
  const double* q = out.row(i);
  for (size_type j = 0; j < m; ++j) block[j] = static_cast<Work>(variables[j]);
  obj.feasible_dual(m, q, block);
  return obj.dual_loss(m, q, const_cast<const Work*>(block));
}

}  // namespace sdca

#endif
