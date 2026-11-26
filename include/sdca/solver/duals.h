#ifndef SDCA_SOLVER_DUALS_H
#define SDCA_SOLVER_DUALS_H

// A caller's dual variables (n, m), row-major, for a warm start: feasibility for a training
// objective to the tolerances of docs/API.md, the label entry that makes a single-label row
// sum to zero as stored, and the weights W = center + A' X of the evaluation.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "sdca/solver/eval/linalg.h"
#include "sdca/solver/eval/scores.h"
#include "sdca/solver/objective.h"
#include "sdca/utility/parallel.h"
#include "sdca/core/lanes.h"

SDCA_ISA_BEGIN
namespace duals {

inline void infeasible(const std::string& message) { throw std::invalid_argument(message); }

// The hinge constraints on a row v with the positive classes mark of mass (their sum):
// the positive entries are nonnegative, the mass at most c, and the top-k cap or cone holds.
template <typename D>
void check_hinge_row(const std::vector<double>& v, const char* mark, double mass,
                     const objective_spec& o, double slack, const std::string& what) {
  const std::size_t m = v.size(), k = std::max<std::size_t>(1, o.k);
  for (std::size_t j = 0; j < m; ++j)
    if (mark[j] && v[j] < -slack) infeasible(what + " is outside the feasible set");
  if (mass > o.c + slack) infeasible(what + " is outside the feasible set");
  if (o.kind == kernel::topk_hinge_beta) {
    for (std::size_t j = 0; j < m; ++j)
      if (-v[j] > o.c / double(k) + slack) infeasible(what + " violates the top-k cap");
  } else if (o.kind != kernel::multilabel_hinge) {
    // The top-k projections clip at the cap s / k exactly, their mass is within gamma_{m+1}
    // of s, so the cap of the block's mass errs by that over k.
    const double cone = slack + rounding_bound<D>(m + 1) * o.c / double(k);
    for (std::size_t j = 0; j < m; ++j)
      if (-v[j] > mass / double(k) + cone) infeasible(what + " violates the top-k cone");
  }
}

// Throws std::invalid_argument unless the duals a (n, m) are finite and feasible for the
// objective: each row sums to zero relative to its largest entry, to the rounding of a
// block whose pivot is the sum of the others in D and is stored in D (gamma_m(D)), plus that
// of this sum in double (gamma_{m-1}), and the sign and box
// constraints hold to the rounding of m entries in D against a mass of at most c (slack =
// gamma_m(D) c), the top-k cone also to the projections' mass error over k. probabilities
// (n, m), if not null, are the targets of the distillation objective; else positives(i, mark) marks the positive classes of row i
// in mark (m entries, zeroed) and returns their count.
template <typename D, typename Positives>
void check_feasible(const D* a, std::size_t n, std::size_t m, const objective_spec& o,
                    const double* probabilities, Positives positives, const char* name) {
  const double eps = rounding_bound<D>(m) + rounding_bound<double>(m - 1),
               slack = rounding_bound<D>(m) * o.c;
  const std::string what(name);
  std::vector<double> v(m);
  std::vector<char> mark(m);
  for (std::size_t i = 0; i < n; ++i) {
    double largest = 0;
    for (std::size_t j = 0; j < m; ++j) {
      v[j] = double(a[i * m + j]);
      if (!std::isfinite(v[j])) infeasible(what + " must be finite");
      largest = std::max(largest, std::abs(v[j]));
    }
    const double scale = largest > 0 ? largest : 1;
    double sum = 0, total = 0;
    for (double x : v) {
      sum += x / scale;
      total += std::abs(x / scale);
    }
    if (std::abs(sum) > eps * total)
      infeasible(what + " rows must sum to zero to storage precision");
    if (probabilities) {
      const double* q = probabilities + i * m;
      for (std::size_t j = 0; j < m; ++j)
        if (v[j] > o.c * q[j] + slack || v[j] < o.c * (q[j] - 1) - slack)
          infeasible(what + " is outside the feasible set of the distillation objective");
      continue;
    }
    std::fill(mark.begin(), mark.end(), char(0));
    const std::size_t count = positives(i, mark.data());
    double mass = 0;
    for (std::size_t j = 0; j < m; ++j) {
      if (!mark[j] && v[j] > 0)
        infeasible(what + ": the entries of the other classes must be nonpositive");
      if (mark[j]) mass += v[j];
    }
    if (o.kind == kernel::cross_entropy || o.kind == kernel::multilabel_cross_entropy) {
      for (std::size_t j = 0; j < m; ++j)
        if (mark[j] && v[j] > o.c / double(count) + slack)
          infeasible(what + " is outside the feasible set of the cross-entropy objective");
      continue;
    }
    check_hinge_row<D>(v, mark.data(), mass, o, slack, what);
  }
}

// The label entry of each single-label row: minus the sum of the others in double, so
// that the row sums to zero as stored. label(i) is the label of row i.
template <typename D, typename Label>
void rebuild_label_entries(D* a, std::size_t n, std::size_t m, Label label) {
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t y = label(i);
    double others = 0;
    for (std::size_t j = 0; j < m; ++j)
      if (j != y) others += double(a[i * m + j]);
    a[i * m + y] = static_cast<D>(-others);
  }
}

// The weights w (m, d), row c = center_c + sum_i a_ic x_i (center may be null), rounded
// once: the sums of the evaluation's recomputed W, in its chunks of examples, on
// num_threads threads (stopping_criteria::num_threads).
template <typename D>
void primal_of_dual(const D* x, const D* a, const D* center, size_type n, size_type d, size_type m,
                    std::size_t num_threads, D* w) {
  const size_type chunk = eval_chunk(n);
  const std::vector<double> block(a, a + n * m);
  std::vector<double> H(d * m, 0.0);
  ::sdca::detail::parallel_rows(m, n * d, num_threads, [&](size_type first, size_type last) {
    std::vector<size_type> rows(chunk);
    for (size_type r0 = 0; r0 < n; r0 += chunk)
      linalg::row_sums(d, m, std::min(chunk, n - r0), x + d * r0, d, &block[m * r0], first, last,
                       rows.data(), H.data());
  });
  for (size_type j = 0; j < d * m; ++j)
    w[j] = static_cast<D>(center ? static_cast<double>(center[j]) + H[j] : H[j]);
}

}  // namespace duals
SDCA_ISA_END

#endif
