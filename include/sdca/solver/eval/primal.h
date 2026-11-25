#ifndef SDCA_SOLVER_EVAL_PRIMAL_H
#define SDCA_SOLVER_EVAL_PRIMAL_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "sdca/kernel.h"
#include "sdca/solver/data/output.h"
#include "sdca/solver/eval/linalg.h"
#include "sdca/solver/objective/entropy.h"
#include "sdca/solver/objective/hinge.h"
#include "sdca/core/lanes.h"

SDCA_ISA_BEGIN

// A bound on the l1 norm of the scaled loss's gradient from its unscaled
// value: 2c in general (the norm of a dual block).
template <typename Work, typename Objective>
inline Work loss_slope(const Objective& obj, const Work, const size_type) {
  return 2 * static_cast<Work>(obj.c);
}

// Softmax: 2 (1 - p_y) <= 2 min(1, loss).
template <typename Work>
inline Work loss_slope(const l2_cross_entropy<Work, Work>& obj, const Work loss, const size_type) {
  return 2 * obj.c * std::min(Work(1), loss);
}

// Pinsker: |softmax - q|_1 <= sqrt(2 KL(q, softmax)), and KL = loss - log p.
template <typename Work>
inline Work loss_slope(const l2_multilabel_cross_entropy<Work, Work>& obj, const Work loss,
                       const size_type p) {
  const Work kl = std::max(Work(0), loss - std::log(static_cast<Work>(p)));
  return obj.c * std::min(Work(2), std::sqrt(2 * kl));
}

namespace solver_detail {

// Values of count rows of m scores by the loss kernel, serially (the evaluation's threads
// run blocks of rows).
template <typename Targets>
inline void kernel_values(const kernel::Loss& loss, const size_type count, const size_type m,
                          const double* z, const Targets& t, double* values) {
  const kernel::Execution e = {kernel::precision_float64, 1, false};
  if (kernel::evaluate(loss, e, count, m, z, t, values, static_cast<double*>(nullptr)).status !=
      kernel::status_ok)
    throw std::runtime_error("The loss kernel does not take the objective's targets");
}

inline std::vector<int64_t> signed_indices(const std::vector<size_type>& x) {
  return std::vector<int64_t>(x.begin(), x.end());
}

// The scores of example i not below that of class y: its competitors at least as high.
inline double competitors(const size_type m, const double* z, const size_type y) {
  const double zy = z[y];
  return static_cast<double>(linalg::count_at_least(m, z, zy) - (zy >= zy ? 1 : 0));
}

}  // namespace solver_detail

// The primal losses and metrics of blocks of rows of m scores in class order: values(first,
// count, z, loss) of the examples [first, first + count) by the objective's loss kernel,
// metric(i, z) (see add_metric) and slope(i, loss) (loss_slope).
template <typename Output, typename Objective>
struct primal_rows;

template <typename Objective>
struct primal_rows<multiclass_output, Objective> {
  const multiclass_output& out;
  const Objective& obj;
  const std::vector<int64_t> labels;

  primal_rows(const multiclass_output& out_, const Objective& obj_)
      : out(out_), obj(obj_), labels(solver_detail::signed_indices(out_.labels)) {}

  double metric(const size_type i, const double* z) const {
    return solver_detail::competitors(out.num_classes, z, out.labels[i]);
  }
  double slope(const size_type, const double value) const { return loss_slope(obj, value, 1); }

  void values(const size_type first, const size_type count, const double* z, double* value) const {
    const size_type m = out.num_classes;
    const kernel::Labels t = {labels.data() + first, nullptr, 0, false};
    solver_detail::kernel_values(obj.loss(), count, m, z, t, value);
  }
};

template <typename Objective>
struct primal_rows<multilabel_output, Objective> {
  const multilabel_output& out;
  const Objective& obj;
  const std::vector<int64_t> labels, offsets;

  primal_rows(const multilabel_output& out_, const Objective& obj_)
      : out(out_),
        obj(obj_),
        labels(solver_detail::signed_indices(out_.labels)),
        offsets(solver_detail::signed_indices(out_.offsets)) {}

  // The rank loss: pairs of a label and a non-label scoring at least as high, over all pairs.
  double metric(const size_type i, const double* z) const {
    const size_type m = out.num_classes, p = out.num_labels(i);
    const size_type* first = &out.labels[out.offsets[i]];
    size_type pairs = 0;
    for (size_type l = 0; l < p; ++l) {
      const double zl = z[first[l]];
      pairs += linalg::count_at_least(m, z, zl);
      for (size_type q = 0; q < p; ++q) pairs -= z[first[q]] >= zl;
    }
    return static_cast<double>(pairs) / static_cast<double>(p * (m - p));
  }
  double slope(const size_type i, const double value) const {
    return loss_slope(obj, value, out.num_labels(i));
  }

  void values(const size_type first, const size_type count, const double* z, double* value) const {
    const size_type m = out.num_classes;
    const kernel::LabelSets t = {labels.data(), offsets.data() + first};
    solver_detail::kernel_values(obj.loss(), count, m, z, t, value);
  }
};

template <typename Objective>
struct primal_rows<distribution_output, Objective> {
  const distribution_output& out;
  const Objective& obj;

  primal_rows(const distribution_output& out_, const Objective& obj_) : out(out_), obj(obj_) {}

  // The competitors of the most probable class (the first of them).
  double metric(const size_type i, const double* z) const {
    const double* q = out.row(i);
    const size_type m = out.num_classes;
    return solver_detail::competitors(m, z, size_type(std::max_element(q, q + m) - q));
  }
  double slope(const size_type, const double value) const { return loss_slope(obj, value, 1); }

  void values(const size_type first, const size_type count, const double* z, double* value) const {
    const size_type m = out.num_classes;
    const kernel::Probabilities<double> t = {out.row(first), nullptr};
    solver_detail::kernel_values(obj.loss(), count, m, z, t, value);
  }
};

// Adds an example's metric to the evaluation: the top-k accuracies count an example for every
// k above its competitors; the rank losses add.
template <typename Work, typename Evaluation>
inline void add_metric(const multiclass_output&, const Work metric, Evaluation& eval) {
  eval.accuracy[static_cast<std::size_t>(metric)] += 1;
}

template <typename Work, typename Evaluation>
inline void add_metric(const multilabel_output&, const Work metric, Evaluation& eval) {
  eval.rank_loss += metric;
}

template <typename Work, typename Evaluation>
inline void add_metric(const distribution_output&, const Work metric, Evaluation& eval) {
  eval.accuracy[static_cast<std::size_t>(metric)] += 1;
}

SDCA_ISA_END

#endif
