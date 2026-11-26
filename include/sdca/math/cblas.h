// The CBLAS functions libsdca calls, declared for linking an external BLAS without its
// headers (BLAS_DEFAULT_LOCAL_HEADER). The signatures and enum values are those of the
// standard C interface to the BLAS.
#ifndef CBLAS_H
#define CBLAS_H

// SDCA_CBLAS_PREFIX: a prefix of the symbols, such as scipy_ for the OpenBLAS of the package
// scipy-openblas32 (scipy_cblas_sdot), the BLAS of the Linux wheels.
#ifdef SDCA_CBLAS_PREFIX
#define SDCA_CBLAS_JOIN_(prefix, name) prefix##name
#define SDCA_CBLAS_JOIN(prefix, name) SDCA_CBLAS_JOIN_(prefix, name)
#define cblas_sdot SDCA_CBLAS_JOIN(SDCA_CBLAS_PREFIX, cblas_sdot)
#define cblas_ddot SDCA_CBLAS_JOIN(SDCA_CBLAS_PREFIX, cblas_ddot)
#define cblas_sscal SDCA_CBLAS_JOIN(SDCA_CBLAS_PREFIX, cblas_sscal)
#define cblas_dscal SDCA_CBLAS_JOIN(SDCA_CBLAS_PREFIX, cblas_dscal)
#define cblas_sgemm SDCA_CBLAS_JOIN(SDCA_CBLAS_PREFIX, cblas_sgemm)
#define cblas_dgemm SDCA_CBLAS_JOIN(SDCA_CBLAS_PREFIX, cblas_dgemm)
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };

float cblas_sdot(const int N, const float* X, const int incX, const float* Y, const int incY);
double cblas_ddot(const int N, const double* X, const int incX, const double* Y, const int incY);
void cblas_sscal(const int N, const float alpha, float* X, const int incX);
void cblas_dscal(const int N, const double alpha, double* X, const int incX);

void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                 const float alpha, const float* A, const int lda, const float* B, const int ldb,
                 const float beta, float* C, const int ldc);
void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                 const double alpha, const double* A, const int lda, const double* B, const int ldb,
                 const double beta, double* C, const int ldc);

#ifdef __cplusplus
}
#endif

#endif  // CBLAS_H
