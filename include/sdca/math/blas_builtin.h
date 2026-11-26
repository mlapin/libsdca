#ifndef SDCA_MATH_BLAS_BUILTIN_H
#define SDCA_MATH_BLAS_BUILTIN_H

// The built-in scal and dot (batch_scores takes the products of linalg.h in place of gemm):
// the same bits on every instruction set and thread count.
#ifdef BLAS_BUILTIN
#include <cstddef>
#include "sdca/math/blas.h"
#include "sdca/solver/eval/linalg.h"

namespace sdca {
namespace detail {
// Scalars are non-deduced so alpha=1 works with float arrays, as in CBLAS.
template <typename T>
struct blas_scalar {
  using type = T;
};
}  // namespace detail

template <typename T>
inline void sdca_blas_scal(blas_int n, typename detail::blas_scalar<T>::type alpha, T* x) {
  for (blas_int i = 0; i < n; ++i) x[i] *= alpha;
}
template <typename T>
inline T sdca_blas_dot(blas_int n, const T* x, const T* y) {
  return n > 0 ? static_cast<T>(linalg::dot(std::size_t(n), x, y)) : T(0);
}

}  // namespace sdca
#endif  // BLAS_BUILTIN
#endif
