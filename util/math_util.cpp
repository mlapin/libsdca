#include <algorithm>

#include "math_util.hpp"

namespace sdca {

const IndexType kIndexOne = static_cast<IndexType>(1);


template <>
void sdca_blas_scal<float>(
    const IndexType n,
    const float alpha,
    float *X
  ) {
  cblas_sscal(n, alpha, X, kIndexOne);
}

template <>
void sdca_blas_scal<double>(
    const IndexType n,
    const double alpha,
    double *X
  ) {
  cblas_dscal(n, alpha, X, kIndexOne);
}



template <>
void sdca_blas_copy<float>(
    const IndexType n,
    const float *__restrict__ X,
    float *__restrict__ Y
  ) {
  cblas_scopy(n, X, kIndexOne, Y, kIndexOne);
}

template <>
void sdca_blas_copy<double>(
    const IndexType n,
    const double *__restrict__ X,
    double *__restrict__ Y
  ) {
  cblas_dcopy(n, X, kIndexOne, Y, kIndexOne);
}



template <>
void sdca_blas_axpy<float>(
    const IndexType n,
    const float alpha,
    const float *__restrict__ X,
    float *__restrict__ Y
  ) {
  cblas_saxpy(n, alpha, X, kIndexOne, Y, kIndexOne);
}

template <>
void sdca_blas_axpy<double>(
    const IndexType n,
    const double alpha,
    const double *__restrict__ X,
    double *__restrict__ Y
  ) {
  cblas_daxpy(n, alpha, X, kIndexOne, Y, kIndexOne);
}



template <>
void sdca_blas_axpby<float>(
    const IndexType n,
    const float alpha,
    const float *__restrict__ X,
    const float beta,
    float *__restrict__ Y
  ) {
#if defined(__MKL_CBLAS_H__)
  cblas_saxpby(n, alpha, X, kIndexOne, beta, Y, kIndexOne);
#else
  cblas_sscal(n, beta, Y, kIndexOne);
  cblas_saxpy(n, alpha, X, kIndexOne, Y, kIndexOne);
#endif
}

template <>
void sdca_blas_axpby<double>(
    const IndexType n,
    const double alpha,
    const double *__restrict__ X,
    const double beta,
    double *__restrict__ Y
  ) {
#if defined(__MKL_CBLAS_H__)
  cblas_daxpby(n, alpha, X, kIndexOne, beta, Y, kIndexOne);
#else
  cblas_dscal(n, beta, Y, kIndexOne);
  cblas_daxpy(n, alpha, X, kIndexOne, Y, kIndexOne);
#endif
}



template <>
float sdca_blas_dot(
    const IndexType n,
    const float *X,
    const float *Y
  ) {
  return cblas_sdot(n, X, kIndexOne, Y, kIndexOne);
}

template <>
double sdca_blas_dot(
    const IndexType n,
    const double *X,
    const double *Y
  ) {
  return cblas_ddot(n, X, kIndexOne, Y, kIndexOne);
}



template <>
float sdca_blas_asum(
    const IndexType n,
    const float *X
  ) {
  return cblas_sasum(n, X, kIndexOne);
}

template <>
double sdca_blas_asum(
    const IndexType n,
    const double *X
  ) {
  return cblas_dasum(n, X, kIndexOne);
}



template <>
void sdca_blas_gemv<float>(
    const IndexType m,
    const IndexType n,
    const float *__restrict__ A,
    const float *__restrict__ X,
    float *__restrict__ Y,
    const CBLAS_TRANSPOSE transA,
    const float alpha,
    const float beta
    ) {
  cblas_sgemv(CblasColMajor, transA, m, n, alpha, A, m,
              X, kIndexOne, beta, Y, kIndexOne);
}

template <>
void sdca_blas_gemv<double>(
    const IndexType m,
    const IndexType n,
    const double *__restrict__ A,
    const double *__restrict__ X,
    double *__restrict__ Y,
    const CBLAS_TRANSPOSE transA,
    const double alpha,
    const double beta
    ) {
  cblas_dgemv(CblasColMajor, transA, m, n, alpha, A, m,
              X, kIndexOne, beta, Y, kIndexOne);
}



template <>
void sdca_blas_ger<float>(
    const IndexType m,
    const IndexType n,
    const float alpha,
    const float *X,
    const float *Y,
    float *A
    ) {
  cblas_sger(CblasColMajor, m, n, alpha, X, kIndexOne, Y, kIndexOne, A, m);
}

template <>
void sdca_blas_ger<double>(
    const IndexType m,
    const IndexType n,
    const double alpha,
    const double *X,
    const double *Y,
    double *A
    ) {
  cblas_dger(CblasColMajor, m, n, alpha, X, kIndexOne, Y, kIndexOne, A, m);
}

}
