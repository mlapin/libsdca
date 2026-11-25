#include <cstring>

#include "sdca/math/blas.h"
#include "sdca/solver.h"
#include "test_util.h"

template <typename T>
void check_blas_vectors() {
  T x[] = {T(1), T(-2), T(3)}, z[] = {T(1), T(-2), T(3)};
  sdca::sdca_blas_scal(3, 2, z);
  EXPECT_EQ(z[1], T(-4));
  EXPECT_EQ(sdca::sdca_blas_dot(3, x, x), T(14));
  EXPECT_EQ(sdca::sdca_blas_dot(3, x, z), T(28));
  EXPECT_EQ(sdca::sdca_blas_dot(0, x, x), T(0));
}
TEST(BlasTest, vectors_float) { check_blas_vectors<float>(); }
TEST(BlasTest, vectors_double) { check_blas_vectors<double>(); }

#if !defined(BLAS_BUILTIN)
// The gemm of batch_scores, in every transposition, with exact small-integer sums.
template <typename T>
void check_blas_gemm() {
  for (int m : {3, 257})
    for (auto ta : {sdca::CblasNoTrans, sdca::CblasTrans})
      for (auto tb : {sdca::CblasNoTrans, sdca::CblasTrans}) {
        const int n = 4, k = 5, lda = (ta == sdca::CblasNoTrans ? m : k) + 2,
                  ldb = (tb == sdca::CblasNoTrans ? k : n) + 3;
        std::vector<T> a(std::size_t(lda * (ta == sdca::CblasNoTrans ? k : m)), T(99));
        std::vector<T> b(std::size_t(ldb * (tb == sdca::CblasNoTrans ? n : k)), T(99));
        for (int i = 0; i < m; ++i)
          for (int p = 0; p < k; ++p)
            a[std::size_t(ta == sdca::CblasNoTrans ? i + p * lda : p + i * lda)] =
                T((i + 2 * p) % 7 - 3);
        for (int p = 0; p < k; ++p)
          for (int j = 0; j < n; ++j)
            b[std::size_t(tb == sdca::CblasNoTrans ? p + j * ldb : j + p * ldb)] =
                T((p + 3 * j) % 5 - 2);
        std::vector<T> c(std::size_t(m * n), std::numeric_limits<T>::quiet_NaN());
        sdca::sdca_blas_gemm(m, n, k, a.data(), lda, b.data(), ldb, c.data(), ta, tb);
        for (int j = 0; j < n; ++j)
          for (int i = 0; i < m; ++i) {
            int sum = 0;
            for (int p = 0; p < k; ++p) sum += ((i + 2 * p) % 7 - 3) * ((p + 3 * j) % 5 - 2);
            EXPECT_EQ(c[std::size_t(i + j * m)], T(sum));
          }
      }
  T a[] = {1, 2, 3, 4, 5, 6}, b[] = {42};
  sdca::sdca_blas_gemm(0, 3, 2, a, 2, a, 3, b);
  EXPECT_EQ(b[0], T(42));
}
TEST(BlasTest, gemm_float) { check_blas_gemm<float>(); }
TEST(BlasTest, gemm_double) { check_blas_gemm<double>(); }
#endif

// Batch scores of row-major arrays: x (n, d) times the transposed weights (m, d), and a
// cross-kernel (n, d) times the duals (d, m). Small integers make every sum exact, so the
// one gemm agrees with plain loops and with the per-example kernels of a fit.
template <typename T>
void check_batch_scores() {
  const sdca::size_type n = 6;
  // Class counts around the four-column blocks of the built-in backend.
  const sdca::size_type classes[] = {1, 3, 4, 9}, dimensions[] = {1, 5, 8};
  for (sdca::size_type m : classes)
    for (sdca::size_type d : dimensions) {
      std::vector<T> x(n * d), w(m * d);
      std::vector<T> a(d * m);
      auto at = [](sdca::size_type i, sdca::size_type j, sdca::size_type cols) {
        return i * cols + j;
      };
      auto xv = [](sdca::size_type i, sdca::size_type j) { return int((2 * i + 3 * j) % 7) - 3; };
      auto wv = [](sdca::size_type c, sdca::size_type j) { return int((5 * c + j) % 5) - 2; };
      auto av = [](sdca::size_type l, sdca::size_type c) { return int((l + 4 * c) % 9) - 4; };
      for (sdca::size_type i = 0; i < n; ++i)
        for (sdca::size_type j = 0; j < d; ++j) x[at(i, j, d)] = T(xv(i, j));
      for (sdca::size_type c = 0; c < m; ++c)
        for (sdca::size_type j = 0; j < d; ++j) w[at(c, j, d)] = T(wv(c, j));
      for (sdca::size_type l = 0; l < d; ++l)
        for (sdca::size_type c = 0; c < m; ++c) a[at(l, c, m)] = T(av(l, c));
      const T nan = std::numeric_limits<T>::quiet_NaN();
      std::vector<T> z(n * m, nan), k(z), row(m);
      sdca::batch_scores(false, n, d, m, x.data(), w.data(), z.data());
      sdca::batch_scores(true, n, d, m, x.data(), a.data(), k.data());
      for (sdca::size_type i = 0; i < n; ++i) {
        for (sdca::size_type c = 0; c < m; ++c) {
          int features = 0, kernel = 0;
          for (sdca::size_type j = 0; j < d; ++j) {
            features += xv(i, j) * wv(c, j);
            kernel += xv(i, j) * av(j, c);
          }
          EXPECT_EQ(z[at(i, c, m)], T(features));
          EXPECT_EQ(k[at(i, c, m)], T(kernel));
        }
        const T* example = x.data() + d * i;
        sdca::native::column_dots(d, m, w.data(), example, row.data());
        for (sdca::size_type c = 0; c < m; ++c) EXPECT_EQ(row[at(0, c, m)], z[at(i, c, m)]);
        sdca::native::kernel_dots(m, d, a.data(), example, row.data());
        for (sdca::size_type c = 0; c < m; ++c) EXPECT_EQ(row[at(0, c, m)], k[at(i, c, m)]);
      }
      // No examples: nothing is written.
      z.assign(z.size(), T(42));
      sdca::batch_scores(false, 0, d, m, x.data(), w.data(), z.data());
      sdca::batch_scores(true, 0, d, m, x.data(), a.data(), z.data());
      for (T value : z) EXPECT_EQ(value, T(42));
    }
}
TEST(BlasTest, batch_scores_float) { check_batch_scores<float>(); }
TEST(BlasTest, batch_scores_double) { check_batch_scores<double>(); }

// Rows shorter than a block of lanes skip the lanes and keep their bits: products of every
// magnitude, cancelling and signed-zero ones, and float entries widened exactly.
template <typename W, typename X>
void check_short_dots() {
  namespace products = sdca::native::products;
  std::mt19937 gen(7);
  std::normal_distribution<double> normal(0, 1);
  const std::size_t rows = 6, count = 3;
  auto same_bits = [](double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; };
  for (std::size_t n = 1; n < products::kSums; ++n) {
    std::vector<W> w(rows * n);
    std::vector<X> x(count * n);
    for (auto& v : w) v = W(normal(gen) * std::exp(8 * normal(gen)));
    for (auto& v : x) v = X(normal(gen));
    for (std::size_t j = 0; j < n; ++j) {
      w[j] = W(-0.0);                // row 0: products of -0
      w[n + j] = W(j % 2 ? 1 : -1);  // row 1: products that cancel
      x[j] = X(1);
    }
    std::vector<double> got(count * rows), pairs(count);
    products::dots(n, rows, w.data(), n, nullptr, count, x.data(), n, got.data(), rows);
    products::pair_dots(n, count, w.data(), n, x.data(), n, pairs.data());
    for (std::size_t e = 0; e < count; ++e) {
      const X* xe = x.data() + n * e;
      for (std::size_t r = 0; r < rows; ++r) {
        const W* wr = w.data() + n * r;
        double want;
        products::dot_tile<1, 1>(n, &wr, &xe, &want, 1);
        EXPECT_TRUE(same_bits(got[rows * e + r], want)) << n << " " << e << " " << r;
        if (r == e) {
          EXPECT_TRUE(same_bits(pairs[e], want)) << n << " " << e;
        }
      }
    }
  }
}
TEST(BlasTest, short_dots_double) { check_short_dots<double, double>(); }
TEST(BlasTest, short_dots_float) { check_short_dots<float, float>(); }
TEST(BlasTest, short_dots_double_weights_float_features) { check_short_dots<double, float>(); }

// The rank-one update is entry by entry: w + x * (-delta) rounded once per entry in T, on the
// rows of nonzero delta alone.
template <typename T, typename X>
void check_rank_one_update() {
  std::mt19937 gen(11);
  std::normal_distribution<double> normal(0, 1);
  for (std::size_t n : {1, 3, 8, 37}) {
    const std::size_t rows = 3;
    std::vector<T> w(rows * n), delta = {T(0.5), T(0), T(-3)};
    std::vector<X> x(n);
    for (auto& v : w) v = T(normal(gen));
    for (auto& v : x) v = X(normal(gen));
    std::vector<T> want(w);
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t j = 0; j < n; ++j)
        if (delta[r] != 0) want[n * r + j] = w[n * r + j] + T(x[j]) * -delta[r];
    sdca::linalg::rank_one_update(n, rows, w.data(), n, nullptr, x.data(), delta.data());
    EXPECT_EQ(std::memcmp(w.data(), want.data(), w.size() * sizeof(T)), 0) << n;
  }
}
TEST(BlasTest, rank_one_update_double) { check_rank_one_update<double, double>(); }
TEST(BlasTest, rank_one_update_float) { check_rank_one_update<float, float>(); }
TEST(BlasTest, rank_one_update_double_weights_float_features) {
  check_rank_one_update<double, float>();
}

#if defined(BLAS_BUILTIN)
TEST(BlasTest, builtin_dot_is_linalg_dot_at_every_alignment) {
  std::mt19937 gen(5);
  std::normal_distribution<double> normal(0, 1);
  for (int n : {3, 4, 37, 1001}) {
    std::vector<double> a(static_cast<std::size_t>(n)), x(a.size()), copy(a.size() + 3);
    for (auto& v : a) v = normal(gen) * std::exp(10 * normal(gen));
    for (auto& v : x) v = normal(gen);
    const double expected = sdca::linalg::dot(a.size(), a.data(), x.data());
    for (int offset = 0; offset < 4; ++offset) {
      std::copy(a.begin(), a.end(), copy.begin() + offset);
      EXPECT_EQ(sdca::sdca_blas_dot(n, copy.data() + offset, x.data()), expected) << n;
    }
  }
}
#endif
