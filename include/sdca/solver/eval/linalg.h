#ifndef SDCA_SOLVER_EVAL_LINALG_H
#define SDCA_SOLVER_EVAL_LINALG_H

// The products of sdca/solver/eval/products.h on the instruction set that sdca::kernel::isa()
// picks: the same bits on every one. Must not be first included inside an instruction-set
// pass of sdca/kernel.h.
#include <cstddef>

#include "sdca/kernel.h"
#define SDCA_EACH_ISA "sdca/solver/eval/products.h"
#include "sdca/core/each_isa.h"

namespace sdca {
namespace linalg {

#if defined(__x86_64__)
#define SDCA_LINALG_CALL(call)           \
  switch (::sdca::kernel::isa()) {       \
    case 4:                              \
      return isa_avx512::products::call; \
    case 3:                              \
      return isa_avx2::products::call;   \
    default:                             \
      return isa_scalar::products::call; \
  }
#else
#define SDCA_LINALG_CALL(call) return native::products::call;
#endif

using native::products::dot_steps;

template <typename Wt, typename Xt, typename O>
inline void dots(const std::size_t n, const std::size_t rows, const Wt* W, const std::size_t ldw,
                 const std::size_t* index, const std::size_t count, const Xt* X,
                 const std::size_t ldx, O* out, const std::size_t ldo) {
  SDCA_LINALG_CALL(dots(n, rows, W, ldw, index, count, X, ldx, out, ldo))
}

template <typename T, typename U, typename O>
inline void pair_dots(const std::size_t n, const std::size_t count, const T* A,
                      const std::size_t lda, const U* B, const std::size_t ldb, O* out) {
  SDCA_LINALG_CALL(pair_dots(n, count, A, lda, B, ldb, out))
}

// <a, b> of n entries, in the order of dots.
template <typename T, typename U>
inline double dot(const std::size_t n, const T* a, const U* b) {
  double out;
  pair_dots(n, 1, a, 0, b, 0, &out);
  return out;
}

template <typename T, typename Xt>
inline void rank_one_update(const std::size_t n, const std::size_t rows, T* W,
                            const std::size_t ldw, const std::size_t* index, const Xt* x,
                            const T* delta) {
  SDCA_LINALG_CALL(rank_one_update(n, rows, W, ldw, index, x, delta))
}

template <typename T, typename K, typename O>
inline void column_sums(const std::size_t n, const std::size_t m, const T* A, const K* k,
                        const std::size_t chunk, O* out, double* magnitude) {
  SDCA_LINALG_CALL(column_sums(n, m, A, k, chunk, out, magnitude))
}

template <typename Xt>
inline void row_sums(const std::size_t n, const std::size_t m, const std::size_t count, const Xt* X,
                     const std::size_t ldx, const double* a, const std::size_t first,
                     const std::size_t last, std::size_t* rows, double* H) {
  SDCA_LINALG_CALL(row_sums(n, m, count, X, ldx, a, first, last, rows, H))
}

inline std::size_t count_at_least(const std::size_t n, const double* x, const double value) {
  SDCA_LINALG_CALL(count_at_least(n, x, value))
}

inline double max_abs(const std::size_t n, const double* x) { SDCA_LINALG_CALL(max_abs(n, x)) }

#undef SDCA_LINALG_CALL

}  // namespace linalg
}  // namespace sdca

#endif
