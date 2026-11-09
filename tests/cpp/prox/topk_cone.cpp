#include "sdca/prox/topk_cone.h"
#include "test_util.h"

template <typename Type>
inline void test_prox_topk_cone_check_feasible(const ptrdiff_t k, const Type eps,
                                               std::vector<Type>& v) {
  sdca::prox_topk_cone(v.begin(), v.end(), k);

  Type sum = std::accumulate(v.begin(), v.end(), static_cast<Type>(0));

  Type lo(0), hi(sum / static_cast<Type>(k));
  std::for_each(v.begin(), v.end(), [=](const Type x) { ASSERT_GE(x, lo); });
  std::for_each(v.begin(), v.end(), [=](const Type x) { ASSERT_LE(x, hi + eps); });
}

template <typename Type>
inline void test_prox_topk_cone_set_params(const std::vector<Type>& v, std::mt19937& gen,
                                           std::uniform_int_distribution<ptrdiff_t>& d_k,
                                           ptrdiff_t& k, Type& eps) {
  k = d_k(gen);
  Type max(*std::max_element(v.begin(), v.end()));
  eps = std::numeric_limits<Type>::epsilon() * std::max(static_cast<Type>(1), std::abs(max)) *
        static_cast<Type>(v.size());
}

template <typename Type>
inline void test_prox_topk_cone_feasible(const int pow_from, const int pow_to, const Type tol) {
  std::mt19937 gen(1);
  std::uniform_int_distribution<ptrdiff_t> d_k(1, 10);

  Type eps;
  ptrdiff_t k;
  std::vector<Type> v;

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_prox_topk_cone_set_params(v, gen, d_k, k, eps);
      test_prox_topk_cone_check_feasible(k, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_cone_set_params(v, gen, d_k, k, eps);
      test_prox_topk_cone_check_feasible(k, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_cone_set_params(v, gen, d_k, k, eps);
      test_prox_topk_cone_check_feasible(k, tol * eps, v);
    }
  }

  for (int i = 0; i < 100; ++i) {
    for (int p = pow_from; p < pow_to; ++p) {
      test_populate_real(25, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(25, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_cone_set_params(v, gen, d_k, k, eps);
      test_prox_topk_cone_check_feasible(k, tol * eps, v);
    }
  }
}

TEST(ProxTopKConeTest, test_prox_feasible_float) { test_prox_topk_cone_feasible<float>(-3, 3, 1); }

TEST(ProxTopKConeTest, test_prox_feasible_double) {
  test_prox_topk_cone_feasible<double>(-6, 6, 1);
}

// At k = 1 the cone is the nonnegative orthant: tiny scores are kept.
template <typename Type>
inline void test_prox_topk_cone_tiny(const std::vector<Type>& scales) {
  for (const Type c : scales) {
    std::vector<Type> x{c, 0, -c};
    sdca::prox_topk_cone(x.begin(), x.end(), 1);
    EXPECT_EQ(x[0], c);
    EXPECT_EQ(x[1], 0);
    EXPECT_EQ(x[2], 0);

    std::vector<Type> y{c / 2, 0, c};
    sdca::prox_topk_cone(y.begin(), y.end(), 1);
    EXPECT_EQ(y[0], c / 2);
    EXPECT_EQ(y[1], 0);
    EXPECT_EQ(y[2], c);

    // At k = 2, two equal tiny scores are kept.
    std::vector<Type> z{c, c, -c};
    sdca::prox_topk_cone(z.begin(), z.end(), 2);
    EXPECT_EQ(z[0], c);
    EXPECT_EQ(z[1], c);
    EXPECT_EQ(z[2], 0);
  }
  const Type nan = std::numeric_limits<Type>::quiet_NaN();
  const Type inf = std::numeric_limits<Type>::infinity();
  for (const Type bad : {nan, inf, -inf}) {
    std::vector<Type> x{1, bad, 0};
    EXPECT_THROW(sdca::prox_topk_cone(x.begin(), x.end(), 1), std::invalid_argument);
  }
}

// proj(c * x) = c * proj(x) over many decades of c, with KKT verification.
template <typename Type>
inline void test_prox_topk_cone_homogeneous(const int pow_from, const int pow_to,
                                            const int pow_step, const Type tol) {
  std::mt19937 gen(7);
  std::normal_distribution<Type> d(0, 1);
  for (const std::size_t n : {std::size_t(2), std::size_t(5), std::size_t(20)}) {
    for (int i = 0; i < 20; ++i) {
      std::vector<Type> x(n);
      for (auto& v : x) v = d(gen);
      const auto k = static_cast<std::ptrdiff_t>(1 + static_cast<std::size_t>(i) % n);
      std::vector<Type> p = x;
      sdca::prox_topk_cone(p.begin(), p.end(), k);
      ASSERT_LE(
          topk_cone_simplex_kkt_residual(x, p, k, std::numeric_limits<Type>::infinity(), Type(0)),
          tol);
      Type max = 0;
      for (const Type v : p) max = std::max(max, v);
      for (int e = pow_from; e <= pow_to; e += pow_step) {
        const Type c = static_cast<Type>(std::pow(10., e));
        std::vector<Type> q(n);
        for (std::size_t j = 0; j < n; ++j) q[j] = c * x[j];
        sdca::prox_topk_cone(q.begin(), q.end(), k);
        for (std::size_t j = 0; j < n; ++j) ASSERT_NEAR(q[j] / c, p[j], tol * (1 + max));
      }
    }
  }
}

TEST(ProxTopKConeTest, test_prox_tiny_float) {
  test_prox_topk_cone_tiny<float>({1e-8f, 1e-30f, 1e-40f, std::numeric_limits<float>::min(),
                                   std::numeric_limits<float>::denorm_min()});
}

TEST(ProxTopKConeTest, test_prox_tiny_double) {
  test_prox_topk_cone_tiny<double>({1e-16, 1e-300, 1e-310, std::numeric_limits<double>::min(),
                                    std::numeric_limits<double>::denorm_min()});
}

TEST(ProxTopKConeTest, test_prox_homogeneous_float) {
  test_prox_topk_cone_homogeneous<float>(-30, 30, 2, 64 * std::numeric_limits<float>::epsilon());
}

TEST(ProxTopKConeTest, test_prox_homogeneous_double) {
  test_prox_topk_cone_homogeneous<double>(-290, 290, 10,
                                          64 * std::numeric_limits<double>::epsilon());
}

// The scales of the analytic sweeps: 1e-30 to 1e30 and near overflow, where
// sums of the scores overflow and the search runs in power-of-two units.
inline std::vector<double> test_topk_scales() {
  std::vector<double> scales{1e300, 5e307};
  for (int p = -30; p <= 30; ++p) scales.push_back(std::pow(10.0, p));
  return scales;
}

inline void test_expect_block(const std::vector<double>& x, const std::vector<double>& expected,
                              const double s) {
  const double tol = 4 * std::numeric_limits<double>::epsilon() * s;
  ASSERT_EQ(x.size(), expected.size());
  for (std::size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(x[i], expected[i], tol) << "s = " << s;
}

// s [3, 1, -1] projects to max(a, 0) at k = 1 (feasible) and to s [2, 2, 0]
// at k = 2 (the cap equals the free entry: (3 - x) + (1 - x) = 0).
TEST(ProxTopKConeTest, test_prox_scale_sweep_double) {
  for (const double s : test_topk_scales()) {
    std::vector<double> x{3 * s, s, -s};
    sdca::prox_topk_cone(x.begin(), x.end(), 1);
    test_expect_block(x, {3 * s, s, 0}, s);
    x = {3 * s, s, -s};
    sdca::prox_topk_cone(x.begin(), x.end(), 2);
    test_expect_block(x, {2 * s, 2 * s, 0}, s);
  }
  for (const std::ptrdiff_t k : {1, 2}) {
    std::vector<double> x{1e308, 1e308};
    sdca::prox_topk_cone(x.begin(), x.end(), k);
    EXPECT_EQ(x, (std::vector<double>{1e308, 1e308}));
  }
}

// When rounding at a tie rejects every active set, the search takes the
// one that violates its conditions least, never a silent zero.
TEST(ProxTopKConeTest, test_least_violating_candidate) {
  const double inf = std::numeric_limits<double>::infinity();
  sdca::detail::topk_cone_best<double> best;
  EXPECT_EQ(best.u, 0u);
  EXPECT_EQ(best.m, 0u);
  best.update(0, 1, 2, 1, inf, -inf, 0, 3);    // hi + t < max_M by 1
  best.update(0.5, 2, 2, 1, inf, -inf, 1, 1);  // passes by 0.5
  best.update(0, 0, 2, 1, inf, -inf, 0, 2);    // violated by 2
  EXPECT_EQ(best.violation, -0.5);
  EXPECT_EQ(best.u, 1u);
  EXPECT_EQ(best.m, 1u);
}
