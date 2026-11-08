#include "sdca/math/blas.h"
#include "test_util.h"

template <typename T>
void check_blas_vectors() {
  T x[] = {T(1), T(-2), T(3)}, y[] = {T(4), T(5), T(6)}, z[3];
  sdca::sdca_blas_copy(3, x, z);
  sdca::sdca_blas_scal(3, 2, z);
  EXPECT_EQ(z[1], T(-4));
  sdca::sdca_blas_swap(3, z, y);
  EXPECT_EQ(y[2], T(6));
  EXPECT_EQ(z[0], T(4));
  sdca::sdca_blas_axpy(3, 2, x, z);
  EXPECT_EQ(z[0], T(6));
  EXPECT_EQ(z[1], T(1));
  EXPECT_EQ(z[2], T(12));
  sdca::sdca_blas_axpby(3, 2, x, T(.5), z);
  EXPECT_EQ(z[0], T(5));
  EXPECT_EQ(z[1], T(-3.5));
  EXPECT_EQ(z[2], T(12));
  EXPECT_EQ(sdca::sdca_blas_dot(3, x, x), T(14));
  EXPECT_EQ(sdca::sdca_blas_asum(3, x), T(6));
  sdca::sdca_blas_axpy(3, 0, x, z);
  EXPECT_EQ(z[0], T(5));
  EXPECT_EQ(sdca::sdca_blas_dot(0, x, x), T(0));
  EXPECT_EQ(sdca::sdca_blas_asum(0, x), T(0));
  EXPECT_EQ(sdca::sdca_blas_nrm2(0, x), T(0));
  const T large = std::sqrt(std::numeric_limits<T>::max()) * T(2);
  const T small = std::numeric_limits<T>::min() * T(8);
  for (T scale : {large, small}) {
    T v[] = {T(3) * scale, T(4) * scale, T(0)};
    EXPECT_NEAR(static_cast<double>(sdca::sdca_blas_nrm2(3, v) / (T(5) * scale)), 1., 1e-6);
  }
}
TEST(BlasTest, vectors_float) { check_blas_vectors<float>(); }
TEST(BlasTest, vectors_double) { check_blas_vectors<double>(); }

template <typename T>
void check_blas_matrices() {
  for (int m : {3, 257})
    for (auto ta : {CblasNoTrans, CblasTrans, CblasConjTrans})
      for (auto tb : {CblasNoTrans, CblasTrans, CblasConjTrans}) {
        const int n = 4, k = 5, lda = (ta == CblasNoTrans ? m : k) + 2,
                  ldb = (tb == CblasNoTrans ? k : n) + 3;
        std::vector<T> a(std::size_t(lda * (ta == CblasNoTrans ? k : m)), T(99));
        std::vector<T> b(std::size_t(ldb * (tb == CblasNoTrans ? n : k)), T(99));
        for (int i = 0; i < m; ++i)
          for (int p = 0; p < k; ++p)
            a[std::size_t(ta == CblasNoTrans ? i + p * lda : p + i * lda)] = T((i + 2 * p) % 7 - 3);
        for (int p = 0; p < k; ++p)
          for (int j = 0; j < n; ++j)
            b[std::size_t(tb == CblasNoTrans ? p + j * ldb : j + p * ldb)] = T((p + 3 * j) % 5 - 2);
        for (T alpha : {T(0), T(1.25)})
          for (T beta : {T(0), T(.25)}) {
            std::vector<T> c(std::size_t(m * n),
                             beta == 0 ? std::numeric_limits<T>::quiet_NaN() : T(2));
            sdca::sdca_blas_gemm(m, n, k, a.data(), lda, b.data(), ldb, c.data(), ta, tb, alpha,
                                 beta);
            for (int j = 0; j < n; ++j)
              for (int i = 0; i < m; ++i) {
                int sum = 0;
                for (int p = 0; p < k; ++p) sum += ((i + 2 * p) % 7 - 3) * ((p + 3 * j) % 5 - 2);
                EXPECT_EQ(c[std::size_t(i + j * m)], alpha * T(sum) + beta * T(2));
              }
          }
      }
  T a[] = {1, 2, 3, 4, 5, 6}, x[] = {2, 3, 4}, y[3];
  for (auto trans : {CblasNoTrans, CblasTrans, CblasConjTrans}) {
    std::fill(y, y + 3, std::numeric_limits<T>::quiet_NaN());
    sdca::sdca_blas_gemv(2, 3, a, x, y, trans, 1, 0);
    if (trans == CblasNoTrans) {
      EXPECT_EQ(y[0], T(31));
      EXPECT_EQ(y[1], T(40));
    } else {
      EXPECT_EQ(y[0], T(8));
      EXPECT_EQ(y[1], T(18));
      EXPECT_EQ(y[2], T(28));
    }
    sdca::sdca_blas_gemv(2, 3, a, x, y, trans, 0, T(.5));
    EXPECT_EQ(y[0], trans == CblasNoTrans ? T(15.5) : T(4));
  }
  T b[6] = {0};
  sdca::sdca_blas_ger(2, 3, 2, x, x, b);
  EXPECT_EQ(b[0], T(8));
  EXPECT_EQ(b[5], T(24));
  sdca::sdca_blas_ger(2, 3, 0, x, x, b);
  EXPECT_EQ(b[0], T(8));
  sdca::sdca_blas_gemm(2, 3, 0, a, 2, a, 3, b, CblasNoTrans, CblasNoTrans, 1, 0);
  for (T value : b) EXPECT_EQ(value, T(0));
  b[0] = T(42);
  sdca::sdca_blas_gemm(0, 3, 2, a, 2, a, 3, b);
  sdca::sdca_blas_gemv(0, 3, a, x, b);
  EXPECT_EQ(b[0], T(42));
}
TEST(BlasTest, matrices_float) { check_blas_matrices<float>(); }
TEST(BlasTest, matrices_double) { check_blas_matrices<double>(); }

#if defined(BLAS_BUILTIN) && !defined(__FAST_MATH__)
TEST(BlasTest, builtin_nonfinite_and_zero_coefficients) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();
  double x[] = {inf, inf}, y[] = {0, 0};
  EXPECT_EQ(sdca::sdca_blas_nrm2(2, x), inf);
  x[1] = nan;
  EXPECT_TRUE(std::isnan(sdca::sdca_blas_nrm2(2, x)));
  sdca::sdca_blas_gemv(1, 2, x, x, y, CblasNoTrans, 0, 0);
  EXPECT_EQ(y[0], 0.);
  sdca::sdca_blas_gemm(1, 2, 2, x, 1, x, 2, y, CblasNoTrans, CblasNoTrans, 0, 0);
  EXPECT_EQ(y[0], 0.);
  EXPECT_EQ(y[1], 0.);
}

TEST(BlasTest, builtin_sum_order_depends_on_length_only) {
  // Two partial sums, each adding pairs of products (i, i + 2) in blocks of
  // four, then the tail in order.
  auto reference = [](int n, const double* x, const double* y) {
    double s[2] = {};
    for (int i = 0; i + 4 <= n; i += 4)
      for (int l = 0; l < 2; ++l) s[l] += x[i + l] * y[i + l] + x[i + l + 2] * y[i + l + 2];
    double sum = s[0] + s[1];
    for (int i = n / 4 * 4; i < n; ++i) sum += x[i] * y[i];
    return sum;
  };
  std::mt19937 gen(5);
  std::normal_distribution<double> normal(0, 1);
  for (int n : {3, 4, 37, 1001}) {
    const int cols = 5;  // a block of four outputs and one more
    const std::size_t size = std::size_t(cols * n);
    std::vector<double> a(size), x(static_cast<std::size_t>(n)), copy(size + 3);
    for (auto& v : a) v = normal(gen) * std::exp(10 * normal(gen));
    for (auto& v : x) v = normal(gen);
    for (int offset = 0; offset < 4; ++offset) {  // alignment
      std::copy(a.begin(), a.end(), copy.begin() + offset);
      const double* b = copy.data() + offset;
      double y[cols];
      sdca::sdca_blas_gemv(n, cols, b, x.data(), y, CblasTrans);
      for (int c = 0; c < cols; ++c) EXPECT_EQ(y[c], reference(n, &a[std::size_t(c * n)], &x[0]));
      EXPECT_EQ(sdca::sdca_blas_dot(n, b, x.data()), reference(n, a.data(), x.data()));
    }
  }
}
#endif

template <typename T>
void check_gemv_scratch_boundaries() {
  for (int m : {1, 255, 256, 257, 1024})
    for (auto trans : {CblasNoTrans, CblasTrans, CblasConjTrans})
      for (T alpha : {T(0), T(1.25)})
        for (T beta : {T(0), T(.25)}) {
          const int n = 7, outputs = trans == CblasNoTrans ? m : n;
          const int inputs = trans == CblasNoTrans ? n : m;
          std::vector<T> a(std::size_t(m * n)), x(std::size_t(inputs), T(0));
          std::vector<T> y(std::size_t(outputs),
                           beta == 0 ? std::numeric_limits<T>::quiet_NaN() : T(2));
          for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
              a[std::size_t(i + j * m)] =
                  alpha == 0 ? std::numeric_limits<T>::quiet_NaN() : T((i + 2 * j) % 7 - 3);
          for (int j = 0; j < inputs; ++j) x[std::size_t(j)] = T(j % 5 - 2);
          sdca::sdca_blas_gemv(m, n, a.data(), x.data(), y.data(), trans, alpha, beta);
          for (int i = 0; i < outputs; ++i) {
            int sum = 0;
            for (int j = 0; j < inputs; ++j)
              sum +=
                  (trans == CblasNoTrans ? (i + 2 * j) % 7 - 3 : (j + 2 * i) % 7 - 3) * (j % 5 - 2);
            EXPECT_EQ(y[std::size_t(i)], alpha * T(sum) + beta * T(2));
          }
        }
}
TEST(BlasTest, gemv_scratch_boundaries_float) { check_gemv_scratch_boundaries<float>(); }
TEST(BlasTest, gemv_scratch_boundaries_double) { check_gemv_scratch_boundaries<double>(); }
