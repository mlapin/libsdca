#ifndef SDCA_SOLVER_EVAL_PRIMAL_H
#define SDCA_SOLVER_EVAL_PRIMAL_H

#include <algorithm>
#include <cassert>
#include <iterator>

#include "sdca/solver/data/output.h"

namespace sdca {

template <typename Int,
          typename Data,
          typename Output,
          typename Objective,
          typename Evaluation>
inline void
eval_primal_loss(
    const Int,
    const Output&,
    const Objective&,
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
    const Int i,
    const multiclass_output& out,
    const Objective<Data, Result>& obj,
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
  eval.accuracy[static_cast<std::size_t>(std::distance(scores + 1, it))] += 1;

  // Increment the primal loss (may re-order the scores)
  eval.primal_loss += obj.primal_loss(out.num_classes, scores);
}


template <typename Int,
          typename Data,
          typename Result,
          template <typename, typename> class Objective,
          template <typename, typename> class Evaluation>
inline void
eval_primal_loss(
    const Int i,
    const multilabel_output& out,
    const Objective<Data, Result>& obj,
    Data* scores,
    Evaluation<Result, multilabel_output>& eval
  ) {
  out.move_front(i, scores);

  size_type num_classes = out.num_classes, num_labels = out.num_labels(i);
  Data *pos_first(scores), *pos_last(scores + num_labels);
  Data *neg_first(scores + num_labels), *neg_last(scores + num_classes);

  // Compute the ranking loss
  for (Data* gt = pos_first; gt != pos_last; ++gt) {
    auto it = std::partition(neg_first, neg_last,
      [=](const Data& x){ return x >= *gt; });
    eval.rank_loss += static_cast<Result>(std::distance(neg_first, it));
  }
  eval.rank_loss /=
    static_cast<Result>(num_labels * (num_classes - num_labels));

  // Increment the primal loss (may re-order the scores)
  eval.primal_loss += obj.primal_loss(num_classes, num_labels, scores);
}

}

#endif
