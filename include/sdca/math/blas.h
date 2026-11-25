#ifndef SDCA_MATH_BLAS_H
#define SDCA_MATH_BLAS_H

// The BLAS of the solver: an external CBLAS (BLAS_DEFAULT_LOCAL_HEADER, linked by the builds of
// src/native_build.txt), else the built-in products (BLAS_BUILTIN). Without either the
// built-in products stand in for a BLAS nobody chose (SDCA_BLAS_FALLBACK), and the solver logs
// a warning once.
#if !defined(BLAS_BUILTIN) && !defined(BLAS_DEFAULT_LOCAL_HEADER)
#define BLAS_BUILTIN
#ifndef SDCA_BLAS_FALLBACK
#define SDCA_BLAS_FALLBACK
#endif
#endif
#if defined(BLAS_BUILTIN) && defined(BLAS_DEFAULT_LOCAL_HEADER)
#error Select only one BLAS backend
#endif

#if defined(BLAS_DEFAULT_LOCAL_HEADER)
#include "sdca/math/cblas.h"
#endif

namespace sdca {

typedef int blas_int;

// The BLAS of this build, as sdca_blas_backend() of the C ABI reports it: "external" (with
// the name of its library, "external: <SDCA_BLAS_NAME>", when the build names it), "builtin",
// or "fallback" (built-in because no external BLAS was found).
inline const char* blas_backend() {
#if defined(BLAS_DEFAULT_LOCAL_HEADER) && defined(SDCA_BLAS_NAME)
  return "external: " SDCA_BLAS_NAME;
#elif defined(BLAS_DEFAULT_LOCAL_HEADER)
  return "external";
#elif defined(SDCA_BLAS_FALLBACK)
  return "fallback";
#else
  return "builtin";
#endif
}

#ifndef BLAS_BUILTIN
using ::CBLAS_TRANSPOSE;
using ::CblasNoTrans;
using ::CblasTrans;

inline void sdca_blas_scal(const blas_int n, const float alpha, float* X) {
  cblas_sscal(n, alpha, X, 1);
}

inline void sdca_blas_scal(const blas_int n, const double alpha, double* X) {
  cblas_dscal(n, alpha, X, 1);
}

inline float sdca_blas_dot(const blas_int n, const float* X, const float* Y) {
  return cblas_sdot(n, X, 1, Y, 1);
}

inline double sdca_blas_dot(const blas_int n, const double* X, const double* Y) {
  return cblas_ddot(n, X, 1, Y, 1);
}

inline void sdca_blas_gemm(const blas_int m, const blas_int n, const blas_int k, const float* A,
                           const blas_int lda, const float* B, const blas_int ldb, float* C,
                           const CBLAS_TRANSPOSE transA = CblasNoTrans,
                           const CBLAS_TRANSPOSE transB = CblasNoTrans, const float alpha = 1,
                           const float beta = 0) {
  if (m == 0 || n == 0) return;
  cblas_sgemm(CblasColMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, m);
}

inline void sdca_blas_gemm(const blas_int m, const blas_int n, const blas_int k, const double* A,
                           const blas_int lda, const double* B, const blas_int ldb, double* C,
                           const CBLAS_TRANSPOSE transA = CblasNoTrans,
                           const CBLAS_TRANSPOSE transB = CblasNoTrans, const double alpha = 1,
                           const double beta = 0) {
  if (m == 0 || n == 0) return;
  cblas_dgemm(CblasColMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, m);
}

#endif  // !BLAS_BUILTIN
}  // namespace sdca

#include "sdca/math/blas_builtin.h"

#endif
