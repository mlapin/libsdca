#include "sdca/prox/knapsack_le_biased.h"
#include "test_util.h"

template <typename Type>
inline void test_prox_knapsack_le_biased_check_feasible(const Type lo, const Type hi,
                                                        const Type rhs, const Type rho,
                                                        const Type eps, std::vector<Type>& v) {
  sdca::prox_knapsack_le_biased(v.begin(), v.end(), lo, hi, rhs, rho);

  std::for_each(v.begin(), v.end(), [=](const Type x) { ASSERT_GE(x, lo); });
  std::for_each(v.begin(), v.end(), [=](const Type x) { ASSERT_LE(x, hi); });
  Type sum = std::accumulate(v.begin(), v.end(), static_cast<Type>(0));
  ASSERT_LE(sum, rhs + eps);
}

template <typename Type>
inline void test_prox_knapsack_le_biased_set_params(const std::vector<Type>& v, std::mt19937& gen,
                                                    std::uniform_real_distribution<Type>& d_lo,
                                                    std::uniform_real_distribution<Type>& d_hi,
                                                    std::uniform_real_distribution<Type>& d_rhs,
                                                    std::uniform_real_distribution<Type>& d_rho,
                                                    Type& lo, Type& hi, Type& rhs, Type& rho,
                                                    Type& eps) {
  Type size = static_cast<Type>(v.size());
  for (;;) {
    lo = d_lo(gen);
    hi = d_hi(gen);
    rhs = d_rhs(gen);
    rho = d_rho(gen);
    if (lo <= hi && lo * size <= rhs && hi * size >= rhs) break;
  }
  Type max(*std::max_element(v.begin(), v.end()));
  eps = std::numeric_limits<Type>::epsilon() * std::max(static_cast<Type>(1), std::abs(max)) *
        static_cast<Type>(v.size());
}

template <typename Type>
inline void test_prox_knapsack_le_biased_feasible(const int pow_from, const int pow_to,
                                                  const Type tol) {
  std::mt19937 gen(1);
  std::uniform_real_distribution<Type> d_lo(-2, 0.5);
  std::uniform_real_distribution<Type> d_hi(-0.5, 2);
  std::uniform_real_distribution<Type> d_rhs(-5, 5);
  std::uniform_real_distribution<Type> d_rho(0, 2);

  Type lo, hi, rhs, rho, eps;
  std::vector<Type> v;

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_prox_knapsack_le_biased_set_params(v, gen, d_lo, d_hi, d_rhs, d_rho, lo, hi, rhs, rho,
                                              eps);
      test_prox_knapsack_le_biased_check_feasible(lo, hi, rhs, rho, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_knapsack_le_biased_set_params(v, gen, d_lo, d_hi, d_rhs, d_rho, lo, hi, rhs, rho,
                                              eps);
      test_prox_knapsack_le_biased_check_feasible(lo, hi, rhs, rho, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_knapsack_le_biased_set_params(v, gen, d_lo, d_hi, d_rhs, d_rho, lo, hi, rhs, rho,
                                              eps);
      test_prox_knapsack_le_biased_check_feasible(lo, hi, rhs, rho, tol * eps, v);
    }
  }

  for (int i = 0; i < 100; ++i) {
    for (int p = pow_from; p < pow_to; ++p) {
      test_populate_real(25, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(25, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_knapsack_le_biased_set_params(v, gen, d_lo, d_hi, d_rhs, d_rho, lo, hi, rhs, rho,
                                              eps);
      test_prox_knapsack_le_biased_check_feasible(lo, hi, rhs, rho, tol * eps, v);
    }
  }
}

TEST(ProxKnapsackLEBiasedTest, test_prox_feasible_float) {
#ifndef __FAST_MATH__
  test_prox_knapsack_le_biased_feasible<float>(-3, 3, 256);
#else
  test_prox_knapsack_le_biased_feasible<float>(-3, 3, 256);
#endif
}

TEST(ProxKnapsackLEBiasedTest, test_prox_feasible_double) {
#ifndef __FAST_MATH__
  // Sums accumulate in long double; in double they round up to 32 times more.
  test_prox_knapsack_le_biased_feasible<double>(-6, 6, 256 * (wide_long_double() ? 1 : 32));
#else
  test_prox_knapsack_le_biased_feasible<double>(-3, 3, 256);
#endif
}

// The optimum when the sum bound is inactive: x = clamp(a - rho * s) with
// s = <1, x>, by bisection on s in long double.
template <typename Type>
inline std::vector<long double> test_knapsack_le_biased_unbound(const std::vector<Type>& a,
                                                                const Type lo, const Type hi,
                                                                const Type rho) {
  using L = long double;
  auto clamp = [=](const long double v) { return std::max(L(lo), std::min(v, L(hi))); };
  const long double n = static_cast<long double>(a.size());
  long double s_lo = n * L(lo), s_hi = n * L(hi);
  for (int it = 0; it < 200; ++it) {
    const long double s = (s_lo + s_hi) / 2;
    long double sum = 0;
    for (const Type v : a) sum += clamp(L(v) - L(rho) * s);
    (sum > s ? s_lo : s_hi) = s;
  }
  std::vector<long double> x;
  for (const Type v : a) x.push_back(clamp(L(v) - L(rho) * s_lo));
  return x;
}

// KKT: x = clamp(a - t), t >= rho * <1, x>, and t = rho * <1, x> unless the
// sum bound is active.
template <typename Type>
inline void test_knapsack_le_biased_kkt(const std::vector<Type>& a, const std::vector<Type>& x,
                                        const Type lo, const Type hi, const Type rhs,
                                        const Type rho, const long double tol) {
  using L = long double;
  long double s = 0;
  for (const Type v : x) s += L(v);
  ASSERT_LE(s, L(rhs) + tol);
  long double t_min = L(rho) * s, t_max = std::numeric_limits<long double>::infinity();
  if (s < L(rhs) - tol) t_max = L(rho) * s;
  for (std::size_t i = 0; i < a.size(); ++i) {
    ASSERT_GE(x[i], lo);
    ASSERT_LE(x[i], hi);
    const long double ai = L(a[i]);
    if (x[i] == lo) t_min = std::max(t_min, ai - L(lo));
    if (x[i] == hi) t_max = std::min(t_max, ai - L(hi));
    if (lo < x[i] && x[i] < hi) {
      t_min = std::max(t_min, ai - L(x[i]));
      t_max = std::min(t_max, ai - L(x[i]));
    }
  }
  EXPECT_LE(t_min, t_max + tol);
}

template <typename Type>
inline void test_prox_knapsack_le_biased_redundant() {
  const Type third = static_cast<Type>(1) / 3;
  const Type tol = 8 * std::numeric_limits<Type>::epsilon();
  const struct {
    Type lo, hi, rhs, rho, x;
  } cases[] = {{0, 1, 3, 1, third}, {0, 1, 2, 1, third}, {0, 1, 3, 0, 1},
               {-2, -1, 0, 1, -1},  {-2, -1, -2, 0, -1}, {-2, -1, 0, 0, -1}};
  for (const auto& c : cases) {
    std::vector<Type> x{1, 1};
    sdca::prox_knapsack_le_biased(x.begin(), x.end(), c.lo, c.hi, c.rhs, c.rho);
    EXPECT_NEAR(x[0], c.x, tol);
    EXPECT_NEAR(x[1], c.x, tol);
  }

  std::mt19937 gen(3);
  std::normal_distribution<Type> d_a(0, 2);
  std::uniform_real_distribution<Type> d_u(0, 1);
  for (const std::size_t n : {std::size_t(1), std::size_t(2), std::size_t(5), std::size_t(50)}) {
    for (int i = 0; i < 50; ++i) {
      std::vector<Type> a(n);
      for (auto& v : a) v = d_a(gen);
      const Type hi = (i % 2 ? -1 : 1) * (d_u(gen) + static_cast<Type>(0.1));
      const Type lo = hi - 1 - 2 * d_u(gen);
      const Type count = static_cast<Type>(n);
      const Type rhs = i % 3 ? count * hi + d_u(gen) * count : count * hi;
      for (const Type rho : {Type(0), Type(1) / 2, Type(3)}) {
        std::vector<Type> x = a;
        sdca::prox_knapsack_le_biased(x.begin(), x.end(), lo, hi, rhs, rho);
        const auto ref = test_knapsack_le_biased_unbound(a, lo, hi, rho);
        const long double scale = static_cast<long double>(1 + std::abs(lo) + std::abs(hi));
        for (std::size_t j = 0; j < n; ++j)
          EXPECT_NEAR(x[j], ref[j], static_cast<long double>(16 * tol) * scale);
        test_knapsack_le_biased_kkt(
            a, x, lo, hi, rhs, rho,
            static_cast<long double>(16 * tol) * scale * static_cast<long double>(count));
      }
    }
  }

  const Type nan = std::numeric_limits<Type>::quiet_NaN();
  std::vector<Type> bad{1, nan};
  EXPECT_THROW(
      sdca::prox_knapsack_le_biased(bad.begin(), bad.end(), Type(0), Type(1), Type(3), Type(1)),
      std::invalid_argument);
  std::vector<Type> x{1, 1};
  EXPECT_THROW(
      sdca::prox_knapsack_le_biased(x.begin(), x.end(), Type(1), Type(2), Type(1), Type(1)),
      std::invalid_argument);
  EXPECT_THROW(sdca::prox_knapsack_le_biased(x.begin(), x.end(), Type(0), Type(1), nan, Type(1)),
               std::invalid_argument);
}

TEST(ProxKnapsackLEBiasedTest, test_prox_redundant_bound_float) {
  test_prox_knapsack_le_biased_redundant<float>();
}

TEST(ProxKnapsackLEBiasedTest, test_prox_redundant_bound_double) {
  test_prox_knapsack_le_biased_redundant<double>();
}

// rhs = n lo with lo > 0 (here up to its rounding) leaves one feasible point,
// every entry at lo, however far above it the scores are.
template <typename Type>
inline void test_prox_knapsack_le_biased_tight_lower() {
  const Type lo = Type(0.1), hi = Type(3.1);
  for (const Type rho : {Type(1e-3), Type(1), Type(1e3)}) {
    for (const Type rhs : {Type(1), Type(10) * lo}) {
      std::vector<Type> x{Type(5099.43), Type(-0.01), 0,          Type(2363), Type(4640),
                          Type(33),      0,           Type(-0.5), Type(2392), 0};
      sdca::prox_knapsack_le_biased(x.begin(), x.end(), lo, hi, rhs, rho);
      for (const Type v : x) EXPECT_EQ(v, lo) << "rho = " << rho << ", rhs = " << rhs;
    }
  }
}

TEST(ProxKnapsackLEBiasedTest, test_prox_tight_lower_bound_float) {
  test_prox_knapsack_le_biased_tight_lower<float>();
}

TEST(ProxKnapsackLEBiasedTest, test_prox_tight_lower_bound_double) {
  test_prox_knapsack_le_biased_tight_lower<double>();
}

// Tiny scores are projected like rescaled ones: [s, s] in [0, 1]^2 with
// <1, x> <= 1 and rho = 1 is s / 3 (stationarity x + <1, x> = a) up to
// s = 3/2, then the sum bound gives 1/2.
template <typename Type>
inline void test_prox_knapsack_le_biased_tiny_scores() {
  const Type tol = 4 * std::numeric_limits<Type>::epsilon();
  for (const Type rhs : {Type(1), Type(1e-20)}) {
    std::vector<Type> x{Type(1e-20), Type(1e-20)};
    sdca::prox_knapsack_le_biased(x.begin(), x.end(), Type(0), Type(1), rhs, Type(1));
    for (const Type v : x) EXPECT_NEAR(v, Type(1e-20) / 3, tol * Type(1e-20));
  }
  for (int p = -30; p <= 30; ++p) {
    const Type s = static_cast<Type>(std::pow(10.0L, p));
    std::vector<Type> a{s, s}, x = a;
    sdca::prox_knapsack_le_biased(x.begin(), x.end(), Type(0), Type(1), Type(1), Type(1));
    const Type expected = s <= Type(1.5) ? s / 3 : Type(0.5);
    for (const Type v : x) EXPECT_NEAR(v, expected, tol * expected) << "s = " << s;
    EXPECT_LE(knapsack_kkt_residual(a, x, Type(0), Type(1), Type(1), Type(1), false), 4 * tol);
  }
}

TEST(ProxKnapsackLEBiasedTest, test_prox_tiny_scores_float) {
  test_prox_knapsack_le_biased_tiny_scores<float>();
}

TEST(ProxKnapsackLEBiasedTest, test_prox_tiny_scores_double) {
  test_prox_knapsack_le_biased_tiny_scores<double>();
}

// a = s [3, 1, -1] in [0, 2s]: unbound, x = s [3/2, 0, 0] (x_1 = 3 - x_1);
// with rhs = s the bound binds at t = 2s >= rho rhs. Up to near overflow.
TEST(ProxKnapsackLEBiasedTest, test_prox_scale_sweep_double) {
  const double tol = 4 * std::numeric_limits<double>::epsilon();
  std::vector<double> scales{1e306, 5e307};
  for (int p = -30; p <= 30; ++p) scales.push_back(std::pow(10.0, p));
  for (const double s : scales) {
    // Intermediate sums of entries near 1e308 need the exponent of the retry.
    if (s > 1e307 && !wide_long_double()) continue;
    const std::vector<double> a{3 * s, s, -s};
    for (const double rhs : {3 * s, s}) {
      std::vector<double> x = a;
      sdca::prox_knapsack_le_biased(x.begin(), x.end(), 0., 2 * s, rhs, 1.);
      const double x0 = rhs == s ? s : 1.5 * s;
      EXPECT_NEAR(x[0], x0, tol * x0) << "s = " << s;
      EXPECT_EQ(x[1], 0) << "s = " << s;
      EXPECT_EQ(x[2], 0) << "s = " << s;
      EXPECT_LE(knapsack_kkt_residual(a, x, 0., 2 * s, rhs, 1., false), 4 * tol);
    }
  }
  // Intermediate sums of 1e308 entries overflow: the optimum is 1e308 / 3.
  std::vector<double> x{1e308, 1e308};
  sdca::prox_knapsack_le_biased(x.begin(), x.end(), 0., 1e308, 1e308, 1.);
  for (const double v : x) EXPECT_NEAR(v, 1e308 / 3, tol * 1e308);
}

// Opposite saturated masses cancel exactly: the middle entry carries rhs.
TEST(ProxKnapsackLEBiasedTest, test_prox_signed_saturated_bounds) {
  std::vector<double> x{-1e25, 0, 1e25};
  sdca::prox_knapsack_le_biased(x.begin(), x.end(), -1e20, 1e20, -1., 1.);
  EXPECT_EQ(x, (std::vector<double>{-1e20, -1, 1e20}));
}

// Signed random problems at scales 1e-30 to 1e30 satisfy the KKT conditions
// to the rounding of each condition's own terms.
template <typename Type>
inline void test_prox_knapsack_le_biased_kkt_sweep(const int pow_from, const int pow_to) {
  std::mt19937 gen(7);
  std::normal_distribution<Type> d_a(0, 1);
  std::uniform_real_distribution<Type> d_u(0, 1);
  const Type tol = 64 * std::numeric_limits<Type>::epsilon();
  for (int p = pow_from; p <= pow_to; p += 2) {
    const Type s = static_cast<Type>(std::pow(10.0L, p));
    for (const std::size_t n : {std::size_t(2), std::size_t(5), std::size_t(20)}) {
      for (int i = 0; i < 20; ++i) {
        std::vector<Type> a(n);
        for (auto& v : a) v = s * d_a(gen);
        const Type lo = i % 2 ? -s * d_u(gen) : Type(0), hi = s * (d_u(gen) + Type(0.1));
        const Type count = static_cast<Type>(n);
        const Type rhs = count * (lo + (hi - lo) * d_u(gen));
        const Type rho = static_cast<Type>(std::pow(10.0, i % 5 - 2));
        std::vector<Type> x = a;
        sdca::prox_knapsack_le_biased(x.begin(), x.end(), lo, hi, rhs, rho);
        EXPECT_LE(knapsack_kkt_residual(a, x, lo, hi, rhs, rho, false), tol)
            << "s = " << s << " n = " << n << " i = " << i;
      }
    }
  }
}

TEST(ProxKnapsackLEBiasedTest, test_prox_kkt_sweep_float) {
  test_prox_knapsack_le_biased_kkt_sweep<float>(-30, 30);
}

TEST(ProxKnapsackLEBiasedTest, test_prox_kkt_sweep_double) {
  test_prox_knapsack_le_biased_kkt_sweep<double>(-30, 30);
}

// The scalar root that replaces a failed active-set search (rounding at a
// tie) solves t = rho <1, clamp(a - t)> by bisection, below the cap t_max.
template <typename Type>
inline void test_knapsack_le_biased_root() {
  const Type inf = std::numeric_limits<Type>::infinity();
  const Type eps = std::numeric_limits<Type>::epsilon();
  std::vector<Type> a{Type(1e-20), Type(1e-20)};
  auto t = sdca::detail::knapsack_le_biased_root(a.begin(), a.end(), Type(0), Type(1), Type(1), inf,
                                                 Type(1));
  EXPECT_NEAR(t.t, Type(2e-20) / 3, 4 * eps * Type(1e-20));
  // a = [3, 1, -1] (sorted) in [0, 2]: x = [3/2, 0, 0] at t = 3/2.
  a = {3, 1, -1};
  t = sdca::detail::knapsack_le_biased_root(a.begin(), a.end(), Type(0), Type(2), Type(1), inf,
                                            Type(1));
  EXPECT_NEAR(t.t, Type(1.5), 4 * eps);
  EXPECT_EQ(std::distance(a.begin(), t.first), 0);
  EXPECT_EQ(std::distance(a.begin(), t.last), 1);
  t = sdca::detail::knapsack_le_biased_root(a.begin(), a.end(), Type(0), Type(2), Type(1), Type(1),
                                            Type(1));
  EXPECT_EQ(t.t, Type(1));
}

TEST(ProxKnapsackLEBiasedTest, test_root_fallback) {
  test_knapsack_le_biased_root<float>();
  test_knapsack_le_biased_root<double>();
  test_knapsack_le_biased_root<long double>();
}
