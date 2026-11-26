#ifndef SDCA_SOLVER_EVAL_SCORES_H
#define SDCA_SOLVER_EVAL_SCORES_H

#include <algorithm>
#include <type_traits>

#include "sdca/math/blas.h"
#include "sdca/solver/data/input.h"
#include "sdca/solver/eval/linalg.h"
#include "sdca/core/lanes.h"

SDCA_ISA_BEGIN

// Long sums over examples go in chunks of about sqrt(n) terms: their rounding then grows
// with chunk_steps(n), not with n.
inline size_type eval_chunk(const size_type n) {
  size_type chunk = 8;
  while (chunk * chunk < n) chunk *= 2;
  return chunk;
}

// Roundings per term of a chunked sum of n terms.
inline double chunk_steps(const size_type n) {
  const size_type chunk = eval_chunk(n);
  return static_cast<double>(std::min(n, chunk) + n / chunk + 2);
}

// The per-example steps of a feature model take the products of sdca/solver/eval/linalg.h
// whatever the BLAS backend: at the sizes of one example an external call costs more than it
// saves, and they accumulate in double. W and the scores may be wider than the data (double
// weights on float features).

// scores = W x of the weights W (m, dim), one row per class.
template <typename Var, typename Data>
inline void column_dots(const size_type dim, const size_type m, const Var* W, const Data* x,
                        Var* scores) {
  linalg::dots(dim, m, W, dim, nullptr, 1, x, 0, scores, 0);
}

// W -= delta x' on the classes whose delta is nonzero: a coordinate step changes few classes
// of a hinge block.
template <typename Var, typename Data>
inline void rank_one_update(const size_type dim, const size_type m, const Data* x, const Var* delta,
                            Var* W) {
  linalg::rank_one_update(dim, m, W, dim, nullptr, x, delta);
}

// scores = A' k of the duals A (n, m) at a kernel row k, in chunks of eval_chunk(n) rows.
template <typename Var, typename Data>
inline void kernel_dots(const size_type m, const size_type n, const Var* A, const Data* k,
                        Var* scores) {
  linalg::column_sums(n, m, A, k, eval_chunk(n), scores, nullptr);
}

// The scores (n, m) of n examples at once: features x (n, dim) times the
// transposed weights (m, dim), or a cross-kernel x (n, dim) to dim training
// examples times their duals (dim, m). One gemm of an external BLAS backend in the
// storage type, else the products of linalg.h; every array is row-major and used in place.
template <typename T>
inline void batch_scores(const bool kernel, const size_type n, const size_type dim,
                         const size_type m, const T* x, const T* coefficients, T* scores) {
#if defined(BLAS_BUILTIN)
  if (!kernel) {
    linalg::dots(dim, m, coefficients, dim, nullptr, n, x, dim, scores, m);
    return;
  }
  for (size_type i = 0; i < n; ++i)
    linalg::column_sums(dim, m, coefficients, x + dim * i, eval_chunk(dim), scores + m * i,
                        nullptr);
#else
  // In BLAS terms the scores are an m-by-n column-major product with the
  // dim-by-n examples: of the transposed dim-by-m weights, or of the m-by-dim duals.
  const blas_int M = static_cast<blas_int>(m), D = static_cast<blas_int>(dim);
  sdca_blas_gemm(M, static_cast<blas_int>(n), D, coefficients, kernel ? M : D, x, D, scores,
                 kernel ? CblasNoTrans : CblasTrans);
#endif
}

template <typename Int, typename Data, typename Context, typename Var>
inline void eval_scores(const Int i, const Int num_classes, const feature_input<Data>& in,
                        const Context& ctx, Var* scores) {
  column_dots(in.num_dimensions, num_classes, ctx.primal_variables,
              in.features + in.num_dimensions * i, scores);
}

template <typename Int, typename Data, typename Context, typename Var>
inline void eval_scores(const Int i, const Int num_classes, const kernel_input<Data>& in,
                        const Context& ctx, Var* scores) {
  kernel_dots(num_classes, in.num_train_examples, ctx.dual_variables,
              in.kernel + in.num_train_examples * i, scores);
}

SDCA_ISA_END

#endif
