#ifndef SDCA_SOLVER_SCRATCH_H
#define SDCA_SOLVER_SCRATCH_H

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "sdca/solver/eval/linalg.h"
#include "sdca/solver/data/input.h"
#include "sdca/solver/solverdef.h"
#include "sdca/utility/parallel.h"
#include "sdca/core/lanes.h"

SDCA_ISA_BEGIN

// The coordinate steps divide by c * norm2: c, and the squared norm and c
// times it of every nonzero row, must be normal in the data type. A zero row
// takes a block of size c.
template <typename Data, typename C>
inline void check_step_scale(const Data norm2, const C c) {
  typedef typename std::common_type<Data, C, double>::type Wide;
  const Wide scaled = static_cast<Wide>(c) * static_cast<Wide>(norm2);
  if (!std::isnormal(static_cast<Data>(c)) ||
      (norm2 != 0 && (!std::isnormal(norm2) || !std::isnormal(static_cast<Data>(scaled)))))
    throw std::invalid_argument("C, norm2, and C*norm2 must be normal in the data type");
}

// The scratch of the coordinate steps, in the type of the iterate
// (solver_context::variable_type), which may be wider than the data.
template <typename Var, template <typename> class Input>
struct solver_scratch {};

template <typename Var>
struct solver_scratch<Var, feature_input> {
  typedef Var data_type;

  std::vector<data_type> norms;
  std::vector<data_type> scores;
  std::vector<data_type> variables;
  // The classes of a step on part of a block, and its block.
  std::vector<size_type> classes;
  std::vector<data_type> part;

  // The squared row norms of ctx.train, checked in the data type against
  // ctx.objective.c.
  template <typename Context>
  void init(const Context& ctx) {
    typedef typename Context::data_type Data;
    const auto& d = ctx.train;
    const size_type n = d.num_examples(), dim = d.num_dimensions();
    std::vector<double> squares(n);
    linalg::pair_dots(dim, n, d.in.features, dim, d.in.features, dim, squares.data());
    norms.resize(n);
    scores.resize(d.num_classes());
    variables.resize(d.num_classes());
    classes.resize(d.num_classes());
    part.resize(d.num_classes());
    for (size_type i = 0; i < n; ++i) {
      const double norm2 = squares[i];
      const Data stored = static_cast<Data>(norm2);
      if (!std::isfinite(stored))
        throw std::invalid_argument("Feature norms must be finite in the data type");
      check_step_scale(stored, ctx.objective.c);
      norms[i] = static_cast<Var>(norm2);
    }
  }
};

// Throws unless the n-by-n kernel has a nonnegative diagonal, is symmetric and satisfies
// |K_ij| <= sqrt(K_ii K_jj), to the tolerances of the criteria (kernel_symmetry,
// kernel_cauchy_schwarz). This rejects non-finite entries and grossly indefinite kernels; it
// does not prove positive semidefiniteness. Blocks of rows run on criteria.num_threads.
template <typename Data>
inline void check_training_kernel(const size_type n, const Data* K,
                                  const stopping_criteria& criteria) {
  const double symmetry = criteria.kernel_symmetry, cauchy_schwarz = criteria.kernel_cauchy_schwarz;
  std::vector<double> root(n);
  for (size_type i = 0; i < n; ++i) {
    const double diagonal = static_cast<double>(K[n * i + i]);
    if (!(diagonal >= 0)) throw std::invalid_argument("A kernel diagonal must be nonnegative");
    root[i] = std::sqrt(diagonal);
  }
  // Block row b checks the pairs j < i of its rows i, a tile at a time.
  const size_type block = 64, blocks = (n + block - 1) / block;
  auto rows = [=, &root](size_type first, size_type last) {
    for (size_type b = first; b < last; ++b) {
      const size_type i0 = b * block, i1 = std::min(n, i0 + block);
      for (size_type j0 = 0; j0 < i1; j0 += block)
        for (size_type i = i0; i < i1; ++i)
          for (size_type j = j0; j < std::min(i, j0 + block); ++j) {
            const double kij = static_cast<double>(K[n * i + j]);
            const double kji = static_cast<double>(K[n * j + i]);
            const double scale = root[i] * root[j];
            if (!(std::abs(kij - kji) <= symmetry * scale)) {
              std::ostringstream text;
              text << "Training kernel must be symmetric: |K_ij - K_ji| exceeds " << symmetry
                   << " sqrt(K_ii K_jj)";
              throw std::invalid_argument(text.str());
            }
            if (!(std::abs(kij) <= (1 + cauchy_schwarz) * scale))
              throw std::invalid_argument(
                  "Training kernel must be positive semidefinite: |K_ij| exceeds "
                  "sqrt(K_ii K_jj)");
          }
    }
  };
  ::sdca::detail::parallel_rows(blocks, block * n, criteria.num_threads, rows);
}

template <typename Var>
struct solver_scratch<Var, kernel_input> {
  typedef Var data_type;

  std::vector<data_type> scores;
  std::vector<data_type> variables;

  // Checks the training kernel of ctx (check_training_kernel) and its
  // diagonal against ctx.objective.c.
  template <typename Context>
  void init(const Context& ctx) {
    const auto& d = ctx.train;
    const size_type n = d.in.num_train_examples;
    scores.resize(d.num_classes());
    variables.resize(d.num_classes());
    check_training_kernel(n, d.in.kernel, ctx.criteria);
    for (size_type i = 0; i < n; ++i) check_step_scale(d.in.kernel[n * i + i], ctx.objective.c);
  }
};

SDCA_ISA_END

#endif
