#ifndef SDCA_SOLVER_EVAL_SCORES_H
#define SDCA_SOLVER_EVAL_SCORES_H

#include <algorithm>
#include <type_traits>

#include "sdca/math/blas.h"
#include "sdca/solver/data/input.h"

namespace sdca {

// The per-example steps of a feature model use the built-in loops whatever
// the BLAS backend: at the sizes of one example an external call costs more
// than it saves (2-4x slower), and the built-in dot products accumulate
// float storage in double. W and the scores may be wider than the data
// (double weights on float features).

// scores = W' x of a dim-by-m column-major W, four columns at a time.
template <typename Var, typename Data>
inline void column_dots(const size_type dim, const size_type m, const Var* W, const Data* x,
                        Var* scores) {
  typedef typename std::common_type<Var, double>::type Work;
  const blas_int D = static_cast<blas_int>(dim);
  size_type c = 0;
  for (; c + 4 <= m; c += 4) {
    Work sums[4];
    detail::blas_dots<4>(D, W + dim * c, dim, x, sums);
    for (size_type q = 0; q < 4; ++q) scores[c + q] = static_cast<Var>(sums[q]);
  }
  for (; c < m; ++c) {
    Work sum;
    detail::blas_dots<1>(D, W + dim * c, 0, x, &sum);
    scores[c] = static_cast<Var>(sum);
  }
}

// W -= x delta' on the classes whose delta is nonzero: a coordinate step
// changes few classes of a hinge block.
template <typename Var, typename Data>
inline void rank_one_update(const size_type dim, const size_type m, const Data* x, const Var* delta,
                            Var* W) {
  for (size_type c = 0; c < m; ++c) {
    if (delta[c] == 0) continue;
    const Var scale = -delta[c];
    Var* w = W + dim * c;
    for (size_type j = 0; j < dim; ++j) w[j] += static_cast<Var>(x[j]) * scale;
  }
}

// scores = A k of the m-by-n duals A at a kernel column k: one long gemv, for
// the BLAS backend; wider duals than the kernel take the built-in loop.
template <typename T>
inline void kernel_dots(const size_type m, const size_type n, const T* A, const T* k, T* scores) {
  sdca_blas_gemv(static_cast<blas_int>(m), static_cast<blas_int>(n), A, k, scores);
}

template <typename Var, typename Data>
inline void kernel_dots(const size_type m, const size_type n, const Var* A, const Data* k,
                        Var* scores) {
  std::fill(scores, scores + m, Var(0));
  for (size_type l = 0; l < n; ++l) {
    const Var kl = static_cast<Var>(k[l]);
    const Var* a = A + m * l;
    for (size_type c = 0; c < m; ++c) scores[c] += a[c] * kl;
  }
}

template <typename Int, typename Data, typename Context, typename Var>
inline void eval_scores(const Int i, const Int num_classes, const feature_input<Data>& in,
                        const Context& ctx, Var* scores) {
  // Let scores = W' * x_i
  column_dots(in.num_dimensions, num_classes, ctx.primal_variables,
              in.features + in.num_dimensions * i, scores);
}

template <typename Int, typename Data, typename Context, typename Var>
inline void eval_scores(const Int i, const Int num_classes, const kernel_input<Data>& in,
                        const Context& ctx, Var* scores) {
  // Let scores = A * K_i = W' * x_i
  kernel_dots(num_classes, in.num_train_examples, ctx.dual_variables,
              in.kernel + in.num_train_examples * i, scores);
}

}  // namespace sdca

#endif
