#ifndef SDCA_MATH_UTIL_HPP
#define SDCA_MATH_UTIL_HPP

#if defined(BLAS_MKL)
  //#include <mkl.h> // produces warnings with -pedantic
  #include <mkl_cblas.h>
#elif defined(BLAS_DEFAULT)
  #include <cblas.h>
#endif

namespace sdca {

#if defined(BLAS_MKL)
  typedef MKL_INT IndexType;
#elif defined(BLAS_DEFAULT)
  typedef int IndexType;
#else
  typedef std::ptrdiff_t IndexType;
#endif

template <typename RealType>
void sdca_blas_scal(
    const IndexType n,
    const RealType alpha,
    RealType *X
  );

template <typename RealType>
void sdca_blas_copy(
    const IndexType n,
    const RealType *__restrict__ X,
    RealType *__restrict__ Y
  );

template <typename RealType>
void sdca_blas_axpy(
    const IndexType n,
    const RealType alpha,
    const RealType *__restrict__ X,
    RealType *__restrict__ Y
  );

template <typename RealType>
void sdca_blas_axpby(
    const IndexType n,
    const RealType alpha,
    const RealType *__restrict__ X,
    const RealType beta,
    RealType *__restrict__ Y
  );

template <typename RealType>
RealType sdca_blas_dot(
    const IndexType n,
    const RealType *X,
    const RealType *Y
  );

template <typename RealType>
RealType sdca_blas_asum(
    const IndexType n,
    const RealType *X
  );

template <typename RealType>
void sdca_blas_gemv(
    const IndexType m,
    const IndexType n,
    const RealType *__restrict__ A,
    const RealType *__restrict__ X,
    RealType *__restrict__ Y,
    const CBLAS_TRANSPOSE transA = CblasNoTrans,
    const RealType alpha = static_cast<RealType>(1),
    const RealType beta = static_cast<RealType>(0)
    );

template <typename RealType>
void sdca_blas_ger(
    const IndexType m,
    const IndexType n,
    const RealType alpha,
    const RealType *X,
    const RealType *Y,
    RealType *A
    );

}

#endif // SDCA_MATH_UTIL_HPP
