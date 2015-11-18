#ifndef SDCA_UTIL_BLAS_H
#define SDCA_UTIL_BLAS_H

#if defined(BLAS_MKL)
  //#include <mkl.h> // produces warnings with -pedantic
  #include <mkl_cblas.h>
#elif defined(BLAS_ACCELERATE)
  #include <Accelerate/Accelerate.h>
#elif defined(BLAS_MATLAB)
  #include <blas.h>
  #include <cstddef>
  typedef std::ptrdiff_t ptrdiff_t; // Matlab does not define ptrdiff_t
#elif defined(BLAS_DEFAULT)
  #include <cblas.h>
#elif defined(BLAS_DEFAULT_LOCAL_HEADER)
  #include "cblas.h"
#endif

#if !defined(CBLAS_ORDER)
enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
#endif

#if !defined(CBLAS_TRANSPOSE)
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
#endif

namespace sdca {

#if defined(MKL_INT)
  typedef MKL_INT blas_int;
#elif defined(BLAS_MATLAB)
  typedef ptrdiff_t blas_int;
#else
  typedef int blas_int;
#endif

inline void
sdca_blas_scal(
    const blas_int n,
    const float alpha,
    float* X
  ) {
#if defined(cblas_sscal)
  cblas_sscal(n, alpha, X, 1);
#else
  blas_int inc(1);
  sscal(const_cast<blas_int*>(&n), const_cast<float*>(&alpha), X, &inc);
#endif
}

inline void
sdca_blas_scal(
    const blas_int n,
    const double alpha,
    double* X
  ) {
#if defined(cblas_dscal)
  cblas_dscal(n, alpha, X, 1);
#else
  blas_int inc(1);
  dscal(const_cast<blas_int*>(&n), const_cast<double*>(&alpha), X, &inc);
#endif
}

inline void
sdca_blas_copy(
    const blas_int n,
    const float* X,
    float* Y
  ) {
#if defined(cblas_scopy)
  cblas_scopy(n, X, 1, Y, 1);
#else
  blas_int inc(1);
  scopy(const_cast<blas_int*>(&n), const_cast<float*>(X), &inc, Y, &inc);
#endif
}

inline void
sdca_blas_copy(
    const blas_int n,
    const double* X,
    double* Y
  ) {
#if defined(cblas_dcopy)
  cblas_dcopy(n, X, 1, Y, 1);
#else
  blas_int inc(1);
  dcopy(const_cast<blas_int*>(&n), const_cast<double*>(X), &inc, Y, &inc);
#endif
}

inline void
sdca_blas_axpy(
    const blas_int n,
    const float alpha,
    const float* X,
    float* Y
  ) {
#if defined(cblas_saxpy)
  cblas_saxpy(n, alpha, X, 1, Y, 1);
#else
  blas_int inc(1);
  saxpy(const_cast<blas_int*>(&n), const_cast<float*>(&alpha),
        const_cast<float*>(X), &inc, Y, &inc);
#endif
}

inline void
sdca_blas_axpy(
    const blas_int n,
    const double alpha,
    const double* X,
    double* Y
  ) {
#if defined(cblas_daxpy)
  cblas_daxpy(n, alpha, X, 1, Y, 1);
#else
  blas_int inc(1);
  daxpy(const_cast<blas_int*>(&n), const_cast<double*>(&alpha),
        const_cast<double*>(X), &inc, Y, &inc);
#endif
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
#elif defined(cblas_sscal)
  cblas_sscal(n, beta, Y, 1);
  cblas_saxpy(n, alpha, X, 1, Y, 1);
#else
  blas_int inc(1);
  sscal(const_cast<blas_int*>(&n), const_cast<float*>(&beta), Y, &inc);
  saxpy(const_cast<blas_int*>(&n), const_cast<float*>(&alpha),
        const_cast<float*>(X), &inc, Y, &inc);
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
#elif defined(cblas_dscal)
  cblas_dscal(n, beta, Y, 1);
  cblas_daxpy(n, alpha, X, 1, Y, 1);
#else
  blas_int inc(1);
  dscal(const_cast<blas_int*>(&n), const_cast<double*>(&beta), Y, &inc);
  daxpy(const_cast<blas_int*>(&n), const_cast<double*>(&alpha),
        const_cast<double*>(X), &inc, Y, &inc);
#endif
}

inline float
sdca_blas_dot(
    const blas_int n,
    const float* X,
    const float* Y
  ) {
#if defined(cblas_sdot)
  return cblas_sdot(n, X, 1, Y, 1);
#else
  blas_int inc(1);
  return sdot(const_cast<blas_int*>(&n), const_cast<float*>(X), &inc,
              const_cast<float*>(Y), &inc);
#endif
}

inline double
sdca_blas_dot(
    const blas_int n,
    const double* X,
    const double* Y
  ) {
#if defined(cblas_ddot)
  return cblas_ddot(n, X, 1, Y, 1);
#else
  blas_int inc(1);
  return ddot(const_cast<blas_int*>(&n), const_cast<double*>(X), &inc,
              const_cast<double*>(Y), &inc);
#endif
}

inline float
sdca_blas_asum(
    const blas_int n,
    const float* X
  ) {
#if defined(cblas_sasum)
  return cblas_sasum(n, X, 1);
#else
  blas_int inc(1);
  return sasum(const_cast<blas_int*>(&n), const_cast<float*>(X), &inc);
#endif
}

inline double
sdca_blas_asum(
    const blas_int n,
    const double* X
  ) {
#if defined(cblas_dasum)
  return cblas_dasum(n, X, 1);
#else
  blas_int inc(1);
  return dasum(const_cast<blas_int*>(&n), const_cast<double*>(X), &inc);
#endif
}

inline float
sdca_blas_nrm2(
    const blas_int n,
    const float* X
  ) {
#if defined(cblas_snrm2)
  return cblas_snrm2(n, X, 1);
#else
  blas_int inc(1);
  return snrm2(const_cast<blas_int*>(&n), const_cast<float*>(X), &inc);
#endif
}

inline double
sdca_blas_nrm2(
    const blas_int n,
    const double* X
  ) {
#if defined(cblas_dnrm2)
  return cblas_dnrm2(n, X, 1);
#else
  blas_int inc(1);
  return dnrm2(const_cast<blas_int*>(&n), const_cast<double*>(X), &inc);
#endif
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
#if defined(cblas_sgemv)
  cblas_sgemv(CblasColMajor, transA, m, n, alpha, A, m,
              X, 1, beta, Y, 1);
#else
  blas_int inc(1);
  char trans((transA == CblasNoTrans) ? 'N' : 'T');
  sgemv(&trans, const_cast<blas_int*>(&m), const_cast<blas_int*>(&n),
        const_cast<float*>(&alpha),
        const_cast<float*>(A), const_cast<blas_int*>(&m),
        const_cast<float*>(X), &inc, const_cast<float*>(&beta), Y, &inc);
#endif
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
#if defined(cblas_dgemv)
  cblas_dgemv(CblasColMajor, transA, m, n, alpha, A, m,
              X, 1, beta, Y, 1);
#else
  blas_int inc(1);
  char trans((transA == CblasNoTrans) ? 'N' : 'T');
  dgemv(&trans, const_cast<blas_int*>(&m), const_cast<blas_int*>(&n),
        const_cast<double*>(&alpha),
        const_cast<double*>(A), const_cast<blas_int*>(&m),
        const_cast<double*>(X), &inc, const_cast<double*>(&beta), Y, &inc);
#endif
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
#if defined(cblas_sger)
  cblas_sger(CblasColMajor, m, n, alpha, X, 1, Y, 1, A, m);
#else
  blas_int inc(1);
  sger(const_cast<blas_int*>(&m), const_cast<blas_int*>(&n),
       const_cast<float*>(&alpha), const_cast<float*>(X), &inc,
       const_cast<float*>(Y), &inc, A, const_cast<blas_int*>(&m));
#endif
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
#if defined(cblas_dger)
  cblas_dger(CblasColMajor, m, n, alpha, X, 1, Y, 1, A, m);
#else
  blas_int inc(1);
  dger(const_cast<blas_int*>(&m), const_cast<blas_int*>(&n),
       const_cast<double*>(&alpha), const_cast<double*>(X), &inc,
       const_cast<double*>(Y), &inc, A, const_cast<blas_int*>(&m));
#endif
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
#if defined(cblas_sgemm)
  cblas_sgemm(CblasColMajor, transA, transB, m, n, k,
              alpha, A, lda, B, ldb, beta, C, m);
#else
  char transa((transA == CblasNoTrans) ? 'N' : 'T');
  char transb((transB == CblasNoTrans) ? 'N' : 'T');
  sgemm(&transa, &transb, const_cast<blas_int*>(&m), const_cast<blas_int*>(&n),
        const_cast<blas_int*>(&k), const_cast<float*>(&alpha),
        const_cast<float*>(A), const_cast<blas_int*>(&lda),
        const_cast<float*>(B), const_cast<blas_int*>(&ldb),
        const_cast<float*>(&beta), C, const_cast<blas_int*>(&m));
#endif
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
#if defined(cblas_dgemm)
  cblas_dgemm(CblasColMajor, transA, transB, m, n, k,
              alpha, A, lda, B, ldb, beta, C, m);
#else
  char transa((transA == CblasNoTrans) ? 'N' : 'T');
  char transb((transB == CblasNoTrans) ? 'N' : 'T');
  dgemm(&transa, &transb, const_cast<blas_int*>(&m), const_cast<blas_int*>(&n),
        const_cast<blas_int*>(&k), const_cast<double*>(&alpha),
        const_cast<double*>(A), const_cast<blas_int*>(&lda),
        const_cast<double*>(B), const_cast<blas_int*>(&ldb),
        const_cast<double*>(&beta), C, const_cast<blas_int*>(&m));
#endif
}

}

#endif
