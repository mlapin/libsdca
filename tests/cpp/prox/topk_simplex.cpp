#include "sdca/prox/topk_simplex.h"
#include "test_util.h"

template <typename Type>
inline void test_prox_topk_simplex_check_feasible(const ptrdiff_t k, const Type rhs, const Type eps,
                                                  std::vector<Type>& v) {
  sdca::prox_topk_simplex(v.begin(), v.end(), k, rhs);

  Type sum = std::accumulate(v.begin(), v.end(), static_cast<Type>(0));

  Type lo(0), hi(sum / static_cast<Type>(k));
  std::for_each(v.begin(), v.end(), [=](const Type x) { ASSERT_GE(x, lo); });
  std::for_each(v.begin(), v.end(), [=](const Type x) { ASSERT_LE(x, hi + eps); });
  ASSERT_LE(sum, rhs + eps);
}

template <typename Type>
inline void test_prox_topk_simplex_set_params(const std::vector<Type>& v, std::mt19937& gen,
                                              std::uniform_int_distribution<ptrdiff_t>& d_k,
                                              std::uniform_real_distribution<Type>& d_rhs,
                                              ptrdiff_t& k, Type& rhs, Type& eps) {
  k = d_k(gen);
  rhs = d_rhs(gen);
  Type max(*std::max_element(v.begin(), v.end()));
  eps = std::numeric_limits<Type>::epsilon() * std::max(static_cast<Type>(1), std::abs(max)) *
        static_cast<Type>(v.size());
}

template <typename Type>
inline void test_prox_topk_simplex_feasible(const int pow_from, const int pow_to, const Type tol) {
  std::mt19937 gen(1);
  std::uniform_int_distribution<ptrdiff_t> d_k(1, 10);
  std::uniform_real_distribution<Type> d_rhs(0, 10);

  Type rhs, eps;
  ptrdiff_t k;
  std::vector<Type> v;

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_prox_topk_simplex_set_params(v, gen, d_k, d_rhs, k, rhs, eps);
      test_prox_topk_simplex_check_feasible(k, rhs, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_simplex_set_params(v, gen, d_k, d_rhs, k, rhs, eps);
      test_prox_topk_simplex_check_feasible(k, rhs, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_simplex_set_params(v, gen, d_k, d_rhs, k, rhs, eps);
      test_prox_topk_simplex_check_feasible(k, rhs, tol * eps, v);
    }
  }

  for (int i = 0; i < 100; ++i) {
    for (int p = pow_from; p < pow_to; ++p) {
      test_populate_real(25, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(25, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_simplex_set_params(v, gen, d_k, d_rhs, k, rhs, eps);
      test_prox_topk_simplex_check_feasible(k, rhs, tol * eps, v);
    }
  }
}

TEST(ProxTopKSimplexTest, test_prox_feasible_float) {
#ifndef __FAST_MATH__
  test_prox_topk_simplex_feasible<float>(-3, 3, 5);
#else
  test_prox_topk_simplex_feasible<float>(-3, 3, 8);
#endif
}

TEST(ProxTopKSimplexTest, test_prox_feasible_double) {
#ifndef __FAST_MATH__
  test_prox_topk_simplex_feasible<double>(-6, 6, 4);
#else
  test_prox_topk_simplex_feasible<double>(-6, 6, 8);
#endif
}

// Tiny scores at k = 1: kept below the cap, cut to a tiny cap above it.
template <typename Type>
inline void test_prox_topk_simplex_tiny(const std::vector<Type>& scales) {
  const Type tol = 8 * std::numeric_limits<Type>::epsilon();
  for (const Type c : scales) {
    std::vector<Type> x{c, 0, -c};
    sdca::prox_topk_simplex(x.begin(), x.end(), 1, static_cast<Type>(1));
    EXPECT_EQ(x[0], c);
    EXPECT_EQ(x[1], 0);
    EXPECT_EQ(x[2], 0);

    for (const Type ratio : {static_cast<Type>(1e-4), static_cast<Type>(1e-1)}) {
      const Type rhs = c * ratio;
      if (rhs < std::numeric_limits<Type>::min()) continue;
      std::vector<Type> y{c, 0, -c};
      sdca::prox_topk_simplex(y.begin(), y.end(), 1, rhs);
      EXPECT_NEAR(y[0], rhs, tol * rhs);
      EXPECT_EQ(y[1], 0);
      EXPECT_EQ(y[2], 0);
    }
  }
}

template <typename Type>
inline void test_prox_topk_simplex_invalid() {
  const Type nan = std::numeric_limits<Type>::quiet_NaN();
  const Type inf = std::numeric_limits<Type>::infinity();
  for (const Type rhs : {static_cast<Type>(-1), -std::numeric_limits<Type>::min(), nan, inf}) {
    std::vector<Type> x{-1, -2};
    EXPECT_THROW(sdca::prox_topk_simplex(x.begin(), x.end(), 1, rhs), std::invalid_argument);
  }
  std::vector<Type> x{1, nan};
  EXPECT_THROW(sdca::prox_topk_simplex(x.begin(), x.end(), 1, static_cast<Type>(1)),
               std::invalid_argument);
}

// proj(c * x, c * rhs) = c * proj(x, rhs) over many decades of c, with KKT verification.
template <typename Type>
inline void test_prox_topk_simplex_homogeneous(const int pow_from, const int pow_to,
                                               const int pow_step, const Type tol) {
  std::mt19937 gen(11);
  std::normal_distribution<Type> d(0, 1);
  for (const std::size_t n : {std::size_t(2), std::size_t(5), std::size_t(20)}) {
    for (int i = 0; i < 30; ++i) {
      std::vector<Type> x(n);
      for (auto& v : x) v = d(gen);
      const auto k = static_cast<std::ptrdiff_t>(1 + static_cast<std::size_t>(i) % n);
      const Type rhs = static_cast<Type>(i % 3 + 1) / 2;
      std::vector<Type> p = x;
      sdca::prox_topk_simplex(p.begin(), p.end(), k, rhs);
      ASSERT_LE(topk_cone_simplex_kkt_residual(x, p, k, rhs, Type(0)), tol);
      Type max = 0;
      for (const Type v : p) max = std::max(max, v);
      for (int e = pow_from; e <= pow_to; e += pow_step) {
        const Type c = static_cast<Type>(std::pow(10., e));
        std::vector<Type> q(n);
        for (std::size_t j = 0; j < n; ++j) q[j] = c * x[j];
        sdca::prox_topk_simplex(q.begin(), q.end(), k, c * rhs);
        for (std::size_t j = 0; j < n; ++j) ASSERT_NEAR(q[j] / c, p[j], tol * (1 + max));
      }
    }
  }
}

TEST(ProxTopKSimplexTest, test_prox_tiny_float) {
  test_prox_topk_simplex_tiny<float>({1e-7f, 1e-30f, 1e-40f, std::numeric_limits<float>::min(),
                                      std::numeric_limits<float>::denorm_min()});
}

TEST(ProxTopKSimplexTest, test_prox_tiny_double) {
  test_prox_topk_simplex_tiny<double>({1e-15, 1e-16, 1e-300, 1e-310,
                                       std::numeric_limits<double>::min(),
                                       std::numeric_limits<double>::denorm_min()});
}

TEST(ProxTopKSimplexTest, test_prox_invalid_float) { test_prox_topk_simplex_invalid<float>(); }

TEST(ProxTopKSimplexTest, test_prox_invalid_double) { test_prox_topk_simplex_invalid<double>(); }

TEST(ProxTopKSimplexTest, test_prox_homogeneous_float) {
  test_prox_topk_simplex_homogeneous<float>(-30, 30, 2, 64 * std::numeric_limits<float>::epsilon());
}

TEST(ProxTopKSimplexTest, test_prox_homogeneous_double) {
  test_prox_topk_simplex_homogeneous<double>(-290, 290, 10,
                                             64 * std::numeric_limits<double>::epsilon());
}

// Analytic projections from 1e-30 to near overflow: s [3, 1, -1] at k = 1,
// rhs = s is s [1, 0, 0]; at k = 2, s [2, 2, 0] (the cone point) for
// rhs = 4s and s [1, 1, 0] for rhs = 2s. [1e308, 1e308] at k = 1 is halved.
TEST(ProxTopKSimplexTest, test_prox_scale_sweep_double) {
  std::vector<double> scales{1e300, 3e307};
  for (int p = -30; p <= 30; ++p) scales.push_back(std::pow(10.0, p));
  const double tol = 4 * std::numeric_limits<double>::epsilon();
  struct {
    std::ptrdiff_t k;
    double rhs;
    std::vector<double> x;
  } cases[] = {{1, 1, {1, 0, 0}}, {2, 4, {2, 2, 0}}, {2, 2, {1, 1, 0}}};
  for (const double s : scales) {
    // Intermediate sums of entries near 1e308 need the exponent of the retry.
    if (s > 1e307 && !wide_long_double()) continue;
    for (const auto& c : cases) {
      std::vector<double> x{3 * s, s, -s};
      sdca::prox_topk_simplex(x.begin(), x.end(), c.k, c.rhs * s);
      for (std::size_t i = 0; i < 3; ++i) EXPECT_NEAR(x[i], c.x[i] * s, tol * s) << "s = " << s;
    }
  }
  std::vector<double> x{1e308, 1e308};
  sdca::prox_topk_simplex(x.begin(), x.end(), 1, 1e308);
  EXPECT_EQ(x, (std::vector<double>{5e307, 5e307}));
}
