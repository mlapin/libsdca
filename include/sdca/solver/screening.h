#ifndef SDCA_SOLVER_SCREENING_H
#define SDCA_SOLVER_SCREENING_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "sdca/math/blas.h"
#include "sdca/solver/data/input.h"
#include "sdca/solver/data/output.h"
#include "sdca/solver/objective.h"
#include "sdca/solver/solverdef.h"
#include "sdca/utility/parallel.h"

namespace sdca {

namespace detail {

// The hinge objectives, the ones that shrink: their dual blocks sit at the
// bounds of their sets (zero, or the full mass on few classes), which the
// entropies' blocks never reach.
template <typename Objective>
struct shrinks : std::false_type {};
template <typename Data, typename Result>
struct shrinks<l2_hinge_topk<Data, Result>> : std::true_type {};
template <typename Data, typename Result>
struct shrinks<l2_hinge_topk_smooth<Data, Result>> : std::true_type {};
template <typename Data, typename Result>
struct shrinks<l2_topk_hinge<Data, Result>> : std::true_type {};
template <typename Data, typename Result>
struct shrinks<l2_topk_hinge_smooth<Data, Result>> : std::true_type {};
template <typename Data, typename Result>
struct shrinks<l2_multilabel_hinge<Data, Result>> : std::true_type {};
template <typename Data, typename Result>
struct shrinks<l2_multilabel_hinge_smooth<Data, Result>> : std::true_type {};

// The single-label hinges: their steps treat the classes other than the label
// alike, so a step on some of them, the others held at zero, is a step of the
// objective on fewer classes.
template <typename Objective>
struct has_part_steps : std::false_type {};
template <typename Data, typename Result>
struct has_part_steps<l2_hinge_topk<Data, Result>> : std::true_type {};
template <typename Data, typename Result>
struct has_part_steps<l2_hinge_topk_smooth<Data, Result>> : std::true_type {};
template <typename Data, typename Result>
struct has_part_steps<l2_topk_hinge<Data, Result>> : std::true_type {};
template <typename Data, typename Result>
struct has_part_steps<l2_topk_hinge_smooth<Data, Result>> : std::true_type {};

}  // namespace detail

// Which duals of a single-label hinge the margins m_j = 1 + z_j - z_y of the
// optimum fix at zero. The loss of every one is zero, with zero gradient,
// where every m_j < 0: then the block is zero (examples). With m_j < 0, class
// j's dual is zero (classes) where the loss sums its margins' positive parts
// (topk_hinge_beta) or k = 1 (the alpha loss of k > 1 averages its k largest
// margins, negative ones too). Without smoothing and with k = 1, the dual of
// j is also zero where another margin exceeds m_j (pairs): the loss is the
// largest of 0 and the m_j.
struct screening_rules {
  bool examples = false, classes = false, pairs = false;
};

template <typename Objective>
inline screening_rules screening_rules_of(const Objective&) {
  return screening_rules();
}
template <typename Data, typename Result>
inline screening_rules screening_rules_of(const l2_hinge_topk<Data, Result>& obj) {
  screening_rules r;
  r.examples = true;
  r.classes = r.pairs = obj.k == 1;
  return r;
}
template <typename Data, typename Result>
inline screening_rules screening_rules_of(const l2_hinge_topk_smooth<Data, Result>& obj) {
  screening_rules r;
  r.examples = true;
  r.classes = obj.k == 1;
  return r;
}
template <typename Data, typename Result>
inline screening_rules screening_rules_of(const l2_topk_hinge<Data, Result>& obj) {
  screening_rules r;
  r.examples = r.classes = true;
  r.pairs = obj.k == 1;
  return r;
}
template <typename Data, typename Result>
inline screening_rules screening_rules_of(const l2_topk_hinge_smooth<Data, Result>&) {
  screening_rules r;
  r.examples = r.classes = true;
  return r;
}

// Gap-safe screening (Ndiaye et al., JMLR 2017): the primal is 1-strongly
// convex in W, so |W - W*| <= r = sqrt(2 gap) for the certified gap of any
// W and feasible A, and every margin difference of example i at W* lies
// within b_i = sqrt(2) |x_i| r of its value at W (Cauchy-Schwarz). Marks
// (with 1) the classes of examples whose dual at the optimum the rules fix at
// zero, and an example whose whole block is; marks only grow. W is
// dim-by-m; norms are the |x_i|. Returns the examples marked whole.
template <typename Data, typename Weight>
inline size_type screen_examples(const screening_rules& rules, const size_type n,
                                 const size_type dim, const multiclass_output& out, const Data* X,
                                 const Weight* W, const std::vector<double>& norms,
                                 const double gap, const std::size_t num_threads,
                                 std::vector<char>& example, std::vector<char>& klass) {
  const size_type m = out.num_classes;
  const double eps = std::numeric_limits<double>::epsilon();
  double max_w = 0;
  for (size_type c = 0; c < m; ++c) {
    double sum = 0;
    for (size_type j = 0; j < dim; ++j) sum += double(W[dim * c + j]) * double(W[dim * c + j]);
    max_w = std::max(max_w, std::sqrt(sum));
  }
  // The radius and the rounding of the scores (a dot product of dim terms)
  // and margins, with room for the norms' own rounding.
  const double r = std::sqrt(2 * gap) * (1 + 16 * eps),
               slack = 4 * double(dim + 4) * eps + 4 * double(std::numeric_limits<Data>::epsilon());
  detail::parallel_rows(n, dim * m, num_threads, [&](size_type first, size_type last) {
    std::vector<double> z(m);
    for (size_type i = first; i < last; ++i) {
      if (example[i]) continue;
      const Data* x = X + dim * i;
      for (size_type c = 0; c < m; ++c) {
        double sum = 0;
        for (size_type j = 0; j < dim; ++j) sum += double(W[dim * c + j]) * double(x[j]);
        z[c] = sum;
      }
      const size_type y = out.labels[i];
      const double zy = z[y];
      double top = -std::numeric_limits<double>::infinity(), scale = 1 + std::abs(zy);
      for (size_type c = 0; c < m; ++c) {
        if (c == y) continue;
        z[c] = 1 + z[c] - zy;
        top = std::max(top, z[c]);
        scale = std::max(scale, 1 + std::abs(z[c]) + std::abs(zy));
      }
      const double b =
          std::sqrt(2.0) * norms[i] * r * (1 + 8 * eps) + slack * (norms[i] * max_w + scale);
      char* marks = &klass[m * i];
      if (rules.examples && top + b < 0) {
        example[i] = 1;
        std::fill(marks, marks + m, char(1));
        marks[y] = 0;
        continue;
      }
      if (!rules.classes) continue;
      const double bar = rules.pairs ? std::max(0.0, top) : 0.0;
      for (size_type c = 0; c < m; ++c)
        if (c != y && bar - z[c] > b) marks[c] = 1;
    }
  });
  return static_cast<size_type>(std::count(example.begin(), example.end(), char(1)));
}

}  // namespace sdca

#endif
