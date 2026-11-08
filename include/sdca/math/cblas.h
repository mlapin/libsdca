// The CBLAS functions libsdca calls, declared for linking an external BLAS
// without its headers (BLAS_DEFAULT_LOCAL_HEADER). The signatures and enum
// values are those of the standard C interface to the BLAS.
#ifndef CBLAS_H
#define CBLAS_H

#ifdef __cplusplus
extern "C" {
#endif

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };

float cblas_sdot(const int N, const float* X, const int incX, const float* Y, const int incY);
double cblas_ddot(const int N, const double* X, const int incX, const double* Y, const int incY);
float cblas_snrm2(const int N, const float* X, const int incX);
double cblas_dnrm2(const int N, const double* X, const int incX);
float cblas_sasum(const int N, const float* X, const int incX);
double cblas_dasum(const int N, const double* X, const int incX);

void cblas_sswap(const int N, float* X, const int incX, float* Y, const int incY);
void cblas_dswap(const int N, double* X, const int incX, double* Y, const int incY);
void cblas_scopy(const int N, const float* X, const int incX, float* Y, const int incY);
void cblas_dcopy(const int N, const double* X, const int incX, double* Y, const int incY);
void cblas_saxpy(const int N, const float alpha, const float* X, const int incX, float* Y,
                 const int incY);
void cblas_daxpy(const int N, const double alpha, const double* X, const int incX, double* Y,
                 const int incY);
void cblas_sscal(const int N, const float alpha, float* X, const int incX);
void cblas_dscal(const int N, const double alpha, double* X, const int incX);

void cblas_sgemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA, const int M,
                 const int N, const float alpha, const float* A, const int lda, const float* X,
                 const int incX, const float beta, float* Y, const int incY);
void cblas_dgemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA, const int M,
                 const int N, const double alpha, const double* A, const int lda, const double* X,
                 const int incX, const double beta, double* Y, const int incY);
void cblas_sger(const enum CBLAS_ORDER order, const int M, const int N, const float alpha,
                const float* X, const int incX, const float* Y, const int incY, float* A,
                const int lda);
void cblas_dger(const enum CBLAS_ORDER order, const int M, const int N, const double alpha,
                const double* X, const int incX, const double* Y, const int incY, double* A,
                const int lda);

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
