#ifndef SDCA_SOLVER_SCREENING_H
#define SDCA_SOLVER_SCREENING_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "sdca/solver/data/input.h"
#include "sdca/solver/data/output.h"
#include "sdca/solver/eval/linalg.h"
#include "sdca/solver/objective.h"
#include "sdca/solver/solverdef.h"
#include "sdca/utility/parallel.h"
#include "sdca/core/lanes.h"

SDCA_ISA_BEGIN

namespace solver_detail {

// Objective::shrinks: the hinge objectives, whose dual blocks sit at the bounds of their sets
// (zero, or the full mass on few classes), which the entropies' blocks never reach.
// Objective::part_steps: the single-label hinges, whose steps treat the classes other than
// the label alike, so that a step on some of them, the others held at zero, is a step of the
// objective on fewer classes. Both false for an objective without them.
template <typename Objective, typename = void>
struct shrinks : std::false_type {};
template <typename Objective>
struct shrinks<Objective, decltype(void(Objective::shrinks))>
    : std::integral_constant<bool, Objective::shrinks> {};

template <typename Objective, typename = void>
struct has_part_steps : std::false_type {};
template <typename Objective>
struct has_part_steps<Objective, decltype(void(Objective::part_steps))>
    : std::integral_constant<bool, Objective::part_steps> {};

template <typename Objective>
inline auto screening_rules_of(const Objective& obj, int) -> decltype(obj.screening()) {
  return obj.screening();
}
template <typename Objective>
inline screening_rules screening_rules_of(const Objective&, long) {
  return screening_rules();
}

}  // namespace solver_detail

// Which duals of a single-label hinge the margins m_j = 1 + z_j - z_y of the optimum fix at
// zero (Objective::screening). The loss of every one is zero, with zero gradient, where every
// m_j < 0: then the block is zero (examples). With m_j < 0, class j's dual is zero (classes)
// where the loss sums its margins' positive parts (topk_hinge_beta) or k = 1 (the alpha loss
// of k > 1 averages its k largest margins, negative ones too). Without smoothing and with
// k = 1, the dual of j is also zero where another margin exceeds m_j (pairs): the loss is the
// largest of 0 and the m_j.
template <typename Objective>
inline screening_rules screening_rules_of(const Objective& obj) {
  return solver_detail::screening_rules_of(obj, 0);
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
  std::vector<double> squares(m);
  linalg::pair_dots(dim, m, W, dim, W, dim, squares.data());
  double max_w = 0;
  for (size_type c = 0; c < m; ++c) max_w = std::max(max_w, std::sqrt(squares[c]));
  // The radius and the rounding of the scores (a dot product of dim terms)
  // and margins, with room for the norms' own rounding.
  const double r = std::sqrt(2 * gap) * (1 + 16 * eps),
               slack = 4 * (linalg::dot_steps(dim) + 4) * eps +
                       4 * double(std::numeric_limits<Data>::epsilon());
  ::sdca::detail::parallel_rows(n, dim * m, num_threads, [&](size_type first, size_type last) {
    std::vector<double> z(m);
    for (size_type i = first; i < last; ++i) {
      if (example[i]) continue;
      linalg::dots(dim, m, W, dim, nullptr, 1, X + dim * i, 0, z.data(), 0);
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

SDCA_ISA_END

#endif
