#ifndef SDCA_SOLVER_EVAL_PRIMAL_H
#define SDCA_SOLVER_EVAL_PRIMAL_H

#include <algorithm>
#include <cassert>
#include <iterator>

#include "sdca/solver/output.h"

namespace sdca {

template <typename Int,
          typename Data,
          typename Output,
          typename Objective,
          typename Evaluation>
inline void
eval_primal_loss(
    const Output&,
    const Objective&,
    const Int,
    Data*,
    Evaluation&
  ) {}


template <typename Int,
          typename Data,
          typename Result,
          template <typename, typename> class Objective,
          template <typename, typename> class Evaluation>
inline void
eval_primal_loss(
    const multiclass_output& out,
    const Objective<Data, Result>& obj,
    const Int i,
    Data* scores,
    Evaluation<Result, multiclass_output>& eval
  ) {
  assert(eval.accuracy.size() == out.num_classes);

  // Put the ground truth score at 0
  size_type label = out.labels[i];
  std::swap(scores[0], scores[label]);

  // Count correct predictions - re-orders the scores!
  auto it = std::partition(scores + 1, scores + out.num_classes,
    [=](const Data& x){ return x >= scores[0]; });
  eval.accuracy[std::distance(scores + 1, it)] += 1;

  // Increment the primal loss (may re-order the scores)
  eval.primal_loss += obj.primal_loss(out.num_classes, scores);
}

}

#endif
