#ifndef SDCA_LINALG_LINALG_H
#define SDCA_LINALG_LINALG_H

#include <algorithm>

#if defined(BLAS_MKL)
  //#include <mkl.h> // produces warnings with -pedantic
  #include <mkl_cblas.h>
#elif defined(BLAS_ACCELERATE)
  #include <Accelerate/Accelerate.h>
#elif defined(BLAS_DEFAULT)
  #include <cblas.h>
#endif

namespace sdca {

#if defined(BLAS_MKL)
  typedef MKL_INT index_type;
#else
  typedef int index_type;
#endif

inline
void sdca_blas_scal(
    const index_type n,
    const float alpha,
    float* X
  ) {
  cblas_sscal(n, alpha, X, 1);
}

inline
void sdca_blas_scal(
    const index_type n,
    const double alpha,
    double* X
  ) {
  cblas_dscal(n, alpha, X, 1);
}

inline
void sdca_blas_copy(
    const index_type n,
    const float* X,
    float* Y
  ) {
  cblas_scopy(n, X, 1, Y, 1);
}

inline
void sdca_blas_copy(
    const index_type n,
    const double* X,
    double* Y
  ) {
  cblas_dcopy(n, X, 1, Y, 1);
}

inline
void sdca_blas_axpy(
    const index_type n,
    const float alpha,
    const float* X,
    float* Y
  ) {
  cblas_saxpy(n, alpha, X, 1, Y, 1);
}

inline
void sdca_blas_axpy(
    const index_type n,
    const double alpha,
    const double* X,
    double* Y
  ) {
  cblas_daxpy(n, alpha, X, 1, Y, 1);
}

inline
void sdca_blas_axpby(
    const index_type n,
    const float alpha,
    const float* X,
    const float beta,
    float* Y
  ) {
#if defined(__MKL_CBLAS_H__)
  cblas_saxpby(n, alpha, X, 1, beta, Y, 1);
#else
  cblas_sscal(n, beta, Y, 1);
  cblas_saxpy(n, alpha, X, 1, Y, 1);
#endif
}

inline
void sdca_blas_axpby(
    const index_type n,
    const double alpha,
    const double* X,
    const double beta,
    double* Y
  ) {
#if defined(__MKL_CBLAS_H__)
  cblas_daxpby(n, alpha, X, 1, beta, Y, 1);
#else
  cblas_dscal(n, beta, Y, 1);
  cblas_daxpy(n, alpha, X, 1, Y, 1);
#endif
}

inline
float sdca_blas_dot(
    const index_type n,
    const float* X,
    const float* Y
  ) {
  return cblas_sdot(n, X, 1, Y, 1);
}

inline
double sdca_blas_dot(
    const index_type n,
    const double* X,
    const double* Y
  ) {
  return cblas_ddot(n, X, 1, Y, 1);
}

inline
float sdca_blas_asum(
    const index_type n,
    const float* X
  ) {
  return cblas_sasum(n, X, 1);
}

inline
double sdca_blas_asum(
    const index_type n,
    const double* X
  ) {
  return cblas_dasum(n, X, 1);
}

inline
void sdca_blas_gemv(
    const index_type m,
    const index_type n,
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

inline
void sdca_blas_gemv(
    const index_type m,
    const index_type n,
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

inline
void sdca_blas_ger(
    const index_type m,
    const index_type n,
    const float alpha,
    const float* X,
    const float* Y,
    float* A
    ) {
  cblas_sger(CblasColMajor, m, n, alpha, X, 1, Y, 1, A, m);
}

inline
void sdca_blas_ger(
    const index_type m,
    const index_type n,
    const double alpha,
    const double* X,
    const double* Y,
    double* A
    ) {
  cblas_dger(CblasColMajor, m, n, alpha, X, 1, Y, 1, A, m);
}

}
