#ifndef SDCA_UTIL_BLAS_H
#define SDCA_UTIL_BLAS_H

#if defined(BLAS_MKL)
  //#include <mkl.h> // produces warnings with -pedantic
  #include <mkl_cblas.h>
#elif defined(BLAS_ACCELERATE)
  #include <Accelerate/Accelerate.h>
#elif defined(BLAS_DEFAULT)
  #include <cblas.h>
#elif defined(BLAS_DEFAULT_LOCAL_HEADER)
  #include "cblas.h"
#endif

namespace sdca {

#if defined(MKL_INT)
  typedef MKL_INT blas_int;
#else
  typedef int blas_int;
#endif

inline void
sdca_blas_scal(
    const blas_int n,
    const float alpha,
    float* X
  ) {
  cblas_sscal(n, alpha, X, 1);
}

inline void
sdca_blas_scal(
    const blas_int n,
    const double alpha,
    double* X
  ) {
  cblas_dscal(n, alpha, X, 1);
}

inline void
sdca_blas_copy(
    const blas_int n,
    const float* X,
    float* Y
  ) {
  cblas_scopy(n, X, 1, Y, 1);
}

inline void
sdca_blas_copy(
    const blas_int n,
    const double* X,
    double* Y
  ) {
  cblas_dcopy(n, X, 1, Y, 1);
}

inline void
sdca_blas_axpy(
    const blas_int n,
    const float alpha,
    const float* X,
    float* Y
  ) {
  cblas_saxpy(n, alpha, X, 1, Y, 1);
}

inline void
sdca_blas_axpy(
    const blas_int n,
    const double alpha,
    const double* X,
    double* Y
  ) {
  cblas_daxpy(n, alpha, X, 1, Y, 1);
}

inline void
sdca_blas_axpby(
    const blas_int n,
    const float alpha,
    const float* X,
    const float beta,
    float* Y
  ) {
#if defined(cblas_saxpby)
  cblas_saxpby(n, alpha, X, 1, beta, Y, 1);
#else
  cblas_sscal(n, beta, Y, 1);
  cblas_saxpy(n, alpha, X, 1, Y, 1);
#endif
}

inline void
sdca_blas_axpby(
    const blas_int n,
    const double alpha,
    const double* X,
    const double beta,
    double* Y
  ) {
#if defined(cblas_daxpby)
  cblas_daxpby(n, alpha, X, 1, beta, Y, 1);
#else
  cblas_dscal(n, beta, Y, 1);
  cblas_daxpy(n, alpha, X, 1, Y, 1);
#endif
}

inline float
sdca_blas_dot(
    const blas_int n,
    const float* X,
    const float* Y
  ) {
  return cblas_sdot(n, X, 1, Y, 1);
}

inline double
sdca_blas_dot(
    const blas_int n,
    const double* X,
    const double* Y
  ) {
  return cblas_ddot(n, X, 1, Y, 1);
}

inline float
sdca_blas_asum(
    const blas_int n,
    const float* X
  ) {
  return cblas_sasum(n, X, 1);
}

inline double
sdca_blas_asum(
    const blas_int n,
    const double* X
  ) {
  return cblas_dasum(n, X, 1);
}

inline float
sdca_blas_nrm2(
    const blas_int n,
    const float* X
  ) {
  return cblas_snrm2(n, X, 1);
}

inline double
sdca_blas_nrm2(
    const blas_int n,
    const double* X
  ) {
  return cblas_dnrm2(n, X, 1);
}

inline void
sdca_blas_gemv(
    const blas_int m,
    const blas_int n,
    const float* A,
    const float* X,
    float* Y,
    const CBLAS_TRANSPOSE transA = CblasNoTrans,
    const float alpha = 1,
    const float beta = 0
    ) {
  cblas_sgemv(CblasColMajor, transA, m, n, alpha, A, m,
              X, 1, beta, Y, 1);
}

inline void
sdca_blas_gemv(
    const blas_int m,
    const blas_int n,
    const double* A,
    const double* X,
    double* Y,
    const CBLAS_TRANSPOSE transA = CblasNoTrans,
    const double alpha = 1,
    const double beta = 0
    ) {
  cblas_dgemv(CblasColMajor, transA, m, n, alpha, A, m,
              X, 1, beta, Y, 1);
}

inline void
sdca_blas_ger(
    const blas_int m,
    const blas_int n,
    const float alpha,
    const float* X,
    const float* Y,
    float* A
    ) {
  cblas_sger(CblasColMajor, m, n, alpha, X, 1, Y, 1, A, m);
}

inline void
sdca_blas_ger(
    const blas_int m,
    const blas_int n,
    const double alpha,
    const double* X,
    const double* Y,
    double* A
    ) {
  cblas_dger(CblasColMajor, m, n, alpha, X, 1, Y, 1, A, m);
}

inline void
sdca_blas_gemm(
    const blas_int m,
    const blas_int n,
    const blas_int k,
    const float* A,
    const blas_int lda,
    const float* B,
    const blas_int ldb,
    float* C,
    const CBLAS_TRANSPOSE transA = CblasNoTrans,
    const CBLAS_TRANSPOSE transB = CblasNoTrans,
    const float alpha = 1,
    const float beta = 0
    ) {
  cblas_sgemm(CblasColMajor, transA, transB, m, n, k,
              alpha, A, lda, B, ldb, beta, C, m);
}

inline void
sdca_blas_gemm(
    const blas_int m,
    const blas_int n,
    const blas_int k,
    const double* A,
    const blas_int lda,
    const double* B,
    const blas_int ldb,
    double* C,
    const CBLAS_TRANSPOSE transA = CblasNoTrans,
    const CBLAS_TRANSPOSE transB = CblasNoTrans,
    const double alpha = 1,
    const double beta = 0
    ) {
  cblas_dgemm(CblasColMajor, transA, transB, m, n, k,
              alpha, A, lda, B, ldb, beta, C, m);
}

}

#endif
