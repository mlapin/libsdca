#ifndef SDCA_MATH_BLAS_BUILTIN_H
#define SDCA_MATH_BLAS_BUILTIN_H

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>
#include "sdca/math/blas.h"
#if defined(__GNUC__) && defined(__SSE2__)
#include <emmintrin.h>
#endif

// The dot product kernels of detail serve every backend (the solver's
// per-example steps); the sdca_blas_* functions only the built-in one.
namespace sdca {
namespace detail {
// Scalars are non-deduced so alpha=1 works with float arrays, as in CBLAS.
template <typename T>
struct blas_scalar {
  using type = T;
};

// Dot products out[c] of x with the columns a + c * lda, c < C. Each block of
// four products adds as (p0 + p2, p1 + p3) to two partial sums per column,
// which are added at the end, followed by the tail in order. The order
// depends on n alone; the lanes vectorize under strict IEEE flags. a and x
// may differ in type (double weights, float features).
template <int C, typename T, typename X, typename Work>
inline void blas_dots(blas_int n, const T* a, std::size_t lda, const X* x, Work* out) {
  Work s[C][2] = {};
  blas_int i = 0;
  for (; i + 4 <= n; i += 4)
    for (int c = 0; c < C; ++c)
      for (int l = 0; l < 2; ++l)
        s[c][l] += Work(a[c * lda + i + l]) * Work(x[i + l]) +
                   Work(a[c * lda + i + l + 2]) * Work(x[i + l + 2]);
  for (int c = 0; c < C; ++c) {
    out[c] = s[c][0] + s[c][1];
    for (blas_int j = i; j < n; ++j) out[c] += Work(a[c * lda + j]) * Work(x[j]);
  }
}
#if defined(__GNUC__)
// The same sums on explicit 2-lane vectors, which compilers do not derive
// reliably from the scalar lanes above. Inlined and unrolled by hand, so
// that the sums stay in registers at any optimization level.
typedef double blas_v2 __attribute__((vector_size(16)));
inline __attribute__((always_inline)) void blas_load2(const double* p, blas_v2& v) {
  std::memcpy(&v, p, sizeof v);
}
inline __attribute__((always_inline)) void blas_load2(const float* p, blas_v2& v) {
#if defined(__SSE2__)
  v = _mm_cvtps_pd(_mm_castsi128_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p))));
#else
  v = blas_v2{double(p[0]), double(p[1])};
#endif
}
// s += (p0 + p2, p1 + p3) for the products p of a[0..3] and (u0, u1).
template <typename T>
inline __attribute__((always_inline)) void blas_step(const T* a, const blas_v2& u0,
                                                     const blas_v2& u1, blas_v2& s) {
  blas_v2 v0, v1;
  blas_load2(a, v0);
  blas_load2(a + 2, v1);
  s += v0 * u0 + v1 * u1;
}
// The two partial sums added, then the products a[j] x[j], i <= j < n.
template <typename T, typename X>
inline __attribute__((always_inline)) double blas_tail(const blas_v2& s, blas_int i, blas_int n,
                                                       const T* a, const X* x) {
  double sum = s[0] + s[1];
  for (; i < n; ++i) sum += double(a[i]) * double(x[i]);
  return sum;
}
template <int C, typename T, typename X>
inline __attribute__((always_inline)) void blas_dots(blas_int n, const T* a, std::size_t lda,
                                                     const X* x, double* out) {
  static_assert(C == 1 || C == 4, "one or four columns");
  blas_v2 s0 = {}, s1 = {}, s2 = {}, s3 = {}, u0, u1;
  blas_int i = 0;
  for (; i + 4 <= n; i += 4) {
    blas_load2(x + i, u0);
    blas_load2(x + i + 2, u1);
    blas_step(a + i, u0, u1, s0);
    if (C == 4) {
      blas_step(a + lda + i, u0, u1, s1);
      blas_step(a + 2 * lda + i, u0, u1, s2);
      blas_step(a + 3 * lda + i, u0, u1, s3);
    }
  }
  out[0] = blas_tail(s0, i, n, a, x);
  if (C == 4) {
    out[1] = blas_tail(s1, i, n, a + lda, x);
    out[2] = blas_tail(s2, i, n, a + 2 * lda, x);
    out[3] = blas_tail(s3, i, n, a + 3 * lda, x);
  }
}
#endif
}  // namespace detail
}  // namespace sdca

#ifdef BLAS_BUILTIN
namespace sdca {
template <typename T>
inline void sdca_blas_scal(blas_int n, typename detail::blas_scalar<T>::type alpha, T* x) {
  for (blas_int i = 0; i < n; ++i) x[i] *= alpha;
}
template <typename T>
inline void sdca_blas_copy(blas_int n, const T* x, T* y) {
  for (blas_int i = 0; i < n; ++i) y[i] = x[i];
}
template <typename T>
inline void sdca_blas_swap(blas_int n, T* x, T* y) {
  for (blas_int i = 0; i < n; ++i) std::swap(x[i], y[i]);
}
template <typename T>
inline void sdca_blas_axpy(blas_int n, typename detail::blas_scalar<T>::type alpha, const T* x,
                           T* y) {
  if (alpha == T(0)) return;
  for (blas_int i = 0; i < n; ++i) y[i] += alpha * x[i];
}
template <typename T>
inline void sdca_blas_axpby(blas_int n, typename detail::blas_scalar<T>::type alpha, const T* x,
                            typename detail::blas_scalar<T>::type beta, T* y) {
  // Match the generic external wrapper's scal followed by axpy.
  sdca_blas_scal(n, beta, y);
  sdca_blas_axpy(n, alpha, x, y);
}
template <typename T>
inline T sdca_blas_dot(blas_int n, const T* x, const T* y) {
  typename std::common_type<T, double>::type sum;
  detail::blas_dots<1>(n, x, 0, y, &sum);
  return static_cast<T>(sum);
}
template <typename T>
inline T sdca_blas_asum(blas_int n, const T* x) {
  using Work = typename std::common_type<T, double>::type;
  Work sum(0);
  for (blas_int i = 0; i < n; ++i) sum += std::abs(Work(x[i]));
  return static_cast<T>(sum);
}
template <typename T>
inline T sdca_blas_nrm2(blas_int n, const T* x) {
  using Work = typename std::common_type<T, double>::type;
  Work scale(0), sum(1);
  bool infinite = false;
  for (blas_int i = 0; i < n; ++i) {
    const Work value = std::abs(Work(x[i]));
    if (std::isnan(value)) return std::numeric_limits<T>::quiet_NaN();
    if (std::isinf(value)) {
      infinite = true;
      continue;
    }
    if (value == 0) continue;
    if (scale < value) {
      const Work r = scale / value;
      sum = 1 + sum * r * r;
      scale = value;
    } else {
      const Work r = value / scale;
      sum += r * r;
    }
  }
  return infinite ? std::numeric_limits<T>::infinity() : static_cast<T>(scale * std::sqrt(sum));
}
template <typename T>
inline void sdca_blas_gemv(blas_int m, blas_int n, const T* a, const T* x, T* y,
                           CBLAS_TRANSPOSE trans = CblasNoTrans,
                           typename detail::blas_scalar<T>::type alpha = 1,
                           typename detail::blas_scalar<T>::type beta = 0) {
  if (m <= 0 || n <= 0) return;
  using Work = typename std::common_type<T, double>::type;
  if (trans == CblasNoTrans) {
    // Traverse column-major A contiguously, retaining each output's original
    // summation order and double accumulation for float storage.
    Work local[256];
    std::vector<Work> heap(m > 256 ? std::size_t(m) : 0);
    Work* sums = heap.empty() ? local : heap.data();
    std::fill(sums, sums + m, Work(0));
    if (alpha != T(0))
      for (blas_int j = 0; j < n; ++j) {
        const Work xj = Work(x[j]);
        const T* col = a + std::size_t(j) * std::size_t(m);
        for (blas_int i = 0; i < m; ++i) sums[i] += Work(col[i]) * xj;
      }
    for (blas_int i = 0; i < m; ++i)
      y[i] = (beta == T(0) ? T(0) : beta * y[i]) + alpha * static_cast<T>(sums[i]);
    return;
  }
  // Four outputs at a time share the loads of x.
  for (blas_int i = 0; i < n; i += 4) {
    Work sums[4] = {};
    const T* col = a + std::size_t(i) * std::size_t(m);
    if (alpha != T(0) && i + 4 <= n)
      detail::blas_dots<4>(m, col, std::size_t(m), x, sums);
    else if (alpha != T(0))
      for (blas_int q = 0; i + q < n; ++q)
        detail::blas_dots<1>(m, col + std::size_t(q) * std::size_t(m), 0, x, sums + q);
    for (blas_int q = 0; q < 4 && i + q < n; ++q)
      y[i + q] = (beta == T(0) ? T(0) : beta * y[i + q]) + alpha * static_cast<T>(sums[q]);
  }
}
template <typename T>
inline void sdca_blas_ger(blas_int m, blas_int n, typename detail::blas_scalar<T>::type alpha,
                          const T* x, const T* y, T* a) {
  if (alpha == T(0)) return;
  // The reference order: alpha y_j once per column, skipping zero multipliers.
  for (blas_int j = 0; j < n; ++j) {
    if (y[j] == T(0)) continue;
    const T scaled = alpha * y[j];
    T* column = a + std::size_t(j) * std::size_t(m);
    for (blas_int i = 0; i < m; ++i) column[i] += x[i] * scaled;
  }
}
template <typename T>
inline void sdca_blas_gemm(blas_int m, blas_int n, blas_int k, const T* a, blas_int lda, const T* b,
                           blas_int ldb, T* c, CBLAS_TRANSPOSE trans_a = CblasNoTrans,
                           CBLAS_TRANSPOSE trans_b = CblasNoTrans,
                           typename detail::blas_scalar<T>::type alpha = 1,
                           typename detail::blas_scalar<T>::type beta = 0) {
  using Work = typename std::common_type<T, double>::type;
  if (m <= 0 || n <= 0) return;
  // Contiguous column sums keep the ordinary path cache-friendly. This is
  // a small reference backend, not a replacement for a tuned GEMM kernel.
  Work local[256];
  std::vector<Work> heap(m > 256 ? std::size_t(m) : 0);
  Work* sums = heap.empty() ? local : heap.data();
  for (blas_int j = 0; j < n; ++j) {
    std::fill(sums, sums + m, Work(0));
    if (alpha != T(0)) {
      for (blas_int p = 0; p < k; ++p) {
        const std::size_t bi = trans_b == CblasNoTrans
                                   ? std::size_t(p) + std::size_t(j) * std::size_t(ldb)
                                   : std::size_t(j) + std::size_t(p) * std::size_t(ldb);
        const Work value = Work(b[bi]);
        if (trans_a == CblasNoTrans) {
          const T* col = a + std::size_t(p) * std::size_t(lda);
          for (blas_int i = 0; i < m; ++i) sums[i] += Work(col[i]) * value;
        } else {
          for (blas_int i = 0; i < m; ++i)
            sums[i] += Work(a[std::size_t(p) + std::size_t(i) * std::size_t(lda)]) * value;
        }
      }
    }
    for (blas_int i = 0; i < m; ++i) {
      const std::size_t ci = std::size_t(i) + std::size_t(j) * std::size_t(m);
      c[ci] = (beta == T(0) ? T(0) : beta * c[ci]) + alpha * static_cast<T>(sums[i]);
    }
  }
}

}  // namespace sdca
#endif  // BLAS_BUILTIN
#endif
