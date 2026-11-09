#ifndef SDCA_SOLVER_EVAL_PRIMAL_H
#define SDCA_SOLVER_EVAL_PRIMAL_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>

#include "sdca/solver/data/output.h"
#include "sdca/solver/objective/l2_entropy.h"
#include "sdca/solver/objective/l2_multilabel_entropy.h"

namespace sdca {

// A bound on the l1 norm of the scaled loss's gradient from its unscaled
// value: 2c in general (the norm of a dual block).
template <typename Work, typename Objective>
inline Work loss_slope(const Objective& obj, const Work, const size_type) {
  return 2 * static_cast<Work>(obj.c);
}

// Softmax: 2 (1 - p_y) <= 2 min(1, loss).
template <typename Work>
inline Work loss_slope(const l2_entropy<Work, Work>& obj, const Work loss, const size_type) {
  return 2 * obj.c * std::min(Work(1), loss);
}

// Pinsker: |softmax - q|_1 <= sqrt(2 KL(q, softmax)), and KL = loss - log p.
template <typename Work>
inline Work loss_slope(const l2_multilabel_entropy<Work, Work>& obj, const Work loss,
                       const size_type p) {
  const Work kl = std::max(Work(0), loss - std::log(static_cast<Work>(p)));
  return obj.c * std::min(Work(2), std::sqrt(2 * kl));
}

// The example's (unscaled) primal loss, its loss_slope and its metric (see
// add_metric); the scores are reordered.
template <typename Work, typename Objective>
inline Work eval_primal_loss(const size_type i, const multiclass_output& out, const Objective& obj,
                             const projection_precision, Work* scores, Work& slope, Work& metric) {
  // Put the ground truth score at 0
  size_type label = out.labels[i];
  std::swap(scores[0], scores[label]);

  // The competitors scoring at least the label: the top-k accuracies count the
  // example for every k above (re-orders the scores).
  auto it = std::partition(scores + 1, scores + out.num_classes,
                           [=](const Work& x) { return x >= scores[0]; });
  metric = static_cast<Work>(std::distance(scores + 1, it));

  const Work loss = obj.primal_loss(out.num_classes, scores);
  slope = loss_slope(obj, loss, 1);
  return loss;
}

template <typename Work, typename Objective>
inline Work eval_primal_loss(const size_type i, const multilabel_output& out, const Objective& obj,
                             const projection_precision mode, Work* scores, Work& slope,
                             Work& metric) {
  out.move_front(i, scores);

  size_type num_classes = out.num_classes, num_labels = out.num_labels(i);
  Work *pos_first(scores), *pos_last(scores + num_labels);
  Work *neg_first(scores + num_labels), *neg_last(scores + num_classes);

  // Compute the ranking loss
  Work rank_loss(0);
  for (Work* gt = pos_first; gt != pos_last; ++gt) {
    auto it = std::partition(neg_first, neg_last, [=](const Work& x) { return x >= *gt; });
    rank_loss += static_cast<Work>(std::distance(neg_first, it));
  }
  metric = rank_loss / static_cast<Work>(num_labels * (num_classes - num_labels));

  const Work loss = obj.primal_loss(num_classes, num_labels, scores, mode);
  slope = loss_slope(obj, loss, num_labels);
  return loss;
}

// Adds an example's metric of eval_primal_loss to the evaluation.
template <typename Work, typename Evaluation>
inline void add_metric(const multiclass_output&, const Work metric, Evaluation& eval) {
  eval.accuracy[static_cast<std::size_t>(metric)] += 1;
}

template <typename Work, typename Evaluation>
inline void add_metric(const multilabel_output&, const Work metric, Evaluation& eval) {
  eval.rank_loss += metric;
}

template <typename Work, typename Objective>
inline Work eval_primal_loss(const size_type i, const distribution_output& out,
                             const Objective& obj, const projection_precision, Work* scores,
                             Work& slope, Work& metric) {
  const double* q = out.row(i);
  const size_type m = out.num_classes;
  size_type label = static_cast<size_type>(std::max_element(q, q + m) - q);
  const Work target_score = scores[label];
  size_type count = 0;
  for (size_type j = 0; j < m; ++j)
    if (j != label && scores[j] >= target_score) ++count;
  metric = static_cast<Work>(count);
  const Work loss = obj.primal_loss(m, q, scores);
  slope = loss_slope(obj, loss, 1);
  return loss;
}

template <typename Work, typename Evaluation>
inline void add_metric(const distribution_output&, const Work metric, Evaluation& eval) {
  eval.accuracy[static_cast<std::size_t>(metric)] += 1;
}

}  // namespace sdca

#endif
