#include "sdca/prox/knapsack_le_biased.h"
#include "sdca/prox/topk_cone_biased.h"
#include "test_util.h"

template <typename Type>
inline void test_prox_topk_cone_biased_check_feasible(const ptrdiff_t k, const Type rho,
                                                      const Type eps, std::vector<Type>& v) {
  sdca::prox_topk_cone_biased(v.begin(), v.end(), k, rho);

  Type sum = std::accumulate(v.begin(), v.end(), static_cast<Type>(0));

  Type lo(0), hi(sum / static_cast<Type>(k));
  std::for_each(v.begin(), v.end(), [=](const Type x) { ASSERT_GE(x, lo); });
  std::for_each(v.begin(), v.end(), [=](const Type x) { ASSERT_LE(x, hi + eps); });
}

template <typename Type>
inline void test_prox_topk_cone_biased_set_params(const std::vector<Type>& v, std::mt19937& gen,
                                                  std::uniform_int_distribution<ptrdiff_t>& d_k,
                                                  std::uniform_real_distribution<Type>& d_rho,
                                                  ptrdiff_t& k, Type& rho, Type& eps) {
  k = d_k(gen);
  rho = d_rho(gen);
  Type max(*std::max_element(v.begin(), v.end()));
  eps = std::numeric_limits<Type>::epsilon() * std::max(static_cast<Type>(1), std::abs(max)) *
        static_cast<Type>(v.size());
}

template <typename Type>
inline void test_prox_topk_cone_biased_feasible(const int pow_from, const int pow_to,
                                                const Type tol) {
  std::mt19937 gen(1);
  std::uniform_int_distribution<ptrdiff_t> d_k(1, 10);
  std::uniform_real_distribution<Type> d_rho(0, 2);

  Type rho, eps;
  ptrdiff_t k;
  std::vector<Type> v;

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_prox_topk_cone_biased_set_params(v, gen, d_k, d_rho, k, rho, eps);
      test_prox_topk_cone_biased_check_feasible(k, rho, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_cone_biased_set_params(v, gen, d_k, d_rho, k, rho, eps);
      test_prox_topk_cone_biased_check_feasible(k, rho, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_cone_biased_set_params(v, gen, d_k, d_rho, k, rho, eps);
      test_prox_topk_cone_biased_check_feasible(k, rho, tol * eps, v);
    }
  }

  for (int i = 0; i < 100; ++i) {
    for (int p = pow_from; p < pow_to; ++p) {
      test_populate_real(25, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(25, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_cone_biased_set_params(v, gen, d_k, d_rho, k, rho, eps);
      test_prox_topk_cone_biased_check_feasible(k, rho, tol * eps, v);
    }
  }
}

TEST(ProxTopKConeBiasedTest, test_prox_feasible_float) {
  test_prox_topk_cone_biased_feasible<float>(-3, 3, 1);
}

TEST(ProxTopKConeBiasedTest, test_prox_feasible_double) {
  test_prox_topk_cone_biased_feasible<double>(-6, 6, 1);
}

// At k = 1 the biased cone projection of tiny scores is max(a, 0) / (1 + rho).
template <typename Type>
inline void test_prox_topk_cone_biased_tiny(const std::vector<Type>& scales) {
  const Type tol = 8 * std::numeric_limits<Type>::epsilon();
  const Type tiny = std::numeric_limits<Type>::denorm_min();
  for (const Type c : scales) {
    for (const Type rho : {static_cast<Type>(1), static_cast<Type>(3)}) {
      std::vector<Type> x{c, 0, -c};
      sdca::prox_topk_cone_biased(x.begin(), x.end(), 1, rho);
      EXPECT_NEAR(x[0], c / (1 + rho), tol * c + tiny);
      EXPECT_EQ(x[1], 0);
      EXPECT_EQ(x[2], 0);
    }
  }
}

TEST(ProxTopKConeBiasedTest, test_prox_tiny_float) {
  test_prox_topk_cone_biased_tiny<float>({1e-8f, 1e-30f, 1e-40f, std::numeric_limits<float>::min(),
                                          std::numeric_limits<float>::denorm_min()});
}

TEST(ProxTopKConeBiasedTest, test_prox_tiny_double) {
  test_prox_topk_cone_biased_tiny<double>({1e-16, 1e-300, 1e-310,
                                           std::numeric_limits<double>::min(),
                                           std::numeric_limits<double>::denorm_min()});
}

// Analytic projections from 1e-30 to near overflow: s [1, 1] gives s / 3
// (x + <1, x> = a), s [3, 1, -1] gives s [3/2, 0, 0] at the cap, k = 1.
TEST(ProxTopKConeBiasedTest, test_prox_scale_sweep_double) {
  std::vector<double> scales{1e300, 5e307, 1e308};
  for (int p = -30; p <= 30; ++p) scales.push_back(std::pow(10.0, p));
  const double tol = 4 * std::numeric_limits<double>::epsilon();
  for (const double s : scales) {
    std::vector<double> x{s, s};
    sdca::prox_topk_cone_biased(x.begin(), x.end(), 1, 1.);
    for (const double v : x) EXPECT_NEAR(v, s / 3, tol * s) << "s = " << s;
    if (s > 5e307) continue;
    x = {3 * s, s, -s};
    sdca::prox_topk_cone_biased(x.begin(), x.end(), 1, 1.);
    EXPECT_NEAR(x[0], 1.5 * s, tol * s) << "s = " << s;
    EXPECT_EQ(x[1], 0);
    EXPECT_EQ(x[2], 0);
  }
}

// Large k at large n: the searches bisect each |U| (quadratic before), so
// this finishes in milliseconds; the caps and signs hold.
TEST(ProxTopKConeBiasedTest, test_prox_large_k) {
  std::mt19937 gen(1);
  std::normal_distribution<double> normal(0, 1);
  std::vector<double> a(40000);
  for (double& v : a) v = normal(gen);
  const std::ptrdiff_t k = 20000;
  for (const double rho : {0.0, 1.0}) {
    std::vector<double> x = a;
    sdca::prox_topk_cone_biased(x.begin(), x.end(), k, rho);
    const double sum = std::accumulate(x.begin(), x.end(), 0.0);
    EXPECT_GT(sum, 0);
    for (const double v : x) {
      EXPECT_GE(v, 0);
      EXPECT_LE(v, sum / double(k) * (1 + 1e-12));
    }
  }
  std::vector<double> x = a;
  sdca::prox_knapsack_le_biased(x.begin(), x.end(), 0.0, 1.0 / double(k), 1.0, 1.0);
  EXPECT_LE(std::accumulate(x.begin(), x.end(), 0.0), 1 + 1e-12);
  for (const double v : x) {
    EXPECT_GE(v, 0);
    EXPECT_LE(v, 1.0 / double(k));
  }
}
