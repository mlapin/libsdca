#include "sdca/prox/two_simplex.h"
#include "test_util.h"

template <typename Type>
inline void test_prox_two_simplex_check_feasible(const ptrdiff_t p, const Type rhs, const Type eps,
                                                 std::vector<Type>& v) {
  ASSERT_TRUE(p > 0);
  ASSERT_TRUE(static_cast<std::size_t>(p) < v.size());
  sdca::prox_two_simplex(v.begin(), v.begin() + p, v.end(), rhs);

  Type lo(0), hi(rhs);
  std::for_each(v.begin(), v.end(), [=](const Type x) { ASSERT_GE(x, lo); });
  std::for_each(v.begin(), v.end(), [=](const Type x) { ASSERT_LE(x, hi); });

  Type sum1 = std::accumulate(v.begin(), v.begin() + p, static_cast<Type>(0));
  Type sum2 = std::accumulate(v.begin() + p, v.end(), static_cast<Type>(0));
  ASSERT_LE(sum1, rhs + eps);
  ASSERT_LE(sum2, rhs + eps);
  ASSERT_NEAR(sum1, sum2, eps);
}

template <typename Type>
inline void test_prox_two_simplex_set_params(const std::vector<Type>& v, std::mt19937& gen,
                                             std::uniform_int_distribution<ptrdiff_t>& d_p,
                                             std::uniform_real_distribution<Type>& d_rhs,
                                             ptrdiff_t& p, Type& rhs, Type& eps) {
  p = d_p(gen);
  rhs = d_rhs(gen);
  Type max(*std::max_element(v.begin(), v.end()));
  eps = std::numeric_limits<Type>::epsilon() * std::max(static_cast<Type>(1), std::abs(max)) *
        static_cast<Type>(v.size());
}

template <typename Type>
inline void test_prox_two_simplex_feasible(const int pow_from, const int pow_to, const Type tol) {
  std::mt19937 gen(1);
  std::uniform_int_distribution<ptrdiff_t> d_p(1, 10);
  std::uniform_real_distribution<Type> d_rhs(0, 5);

  ptrdiff_t p;
  Type rhs, eps;
  std::vector<Type> v;

  // One special case (also test this in debug mode!)
  p = 1;
  rhs = 2;
  v.push_back(static_cast<Type>(-0.49371069182389915));
  for (int i = 0; i < 158; ++i) {
    v.push_back(static_cast<Type>(0.49371069182390021));
  }
  eps = 4 * std::numeric_limits<Type>::epsilon() * static_cast<Type>(v.size());
  test_prox_two_simplex_check_feasible(p, rhs, tol * eps, v);
  v.clear();

  for (int pow = pow_from; pow < pow_to; ++pow) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, pow, pow + 1, static_cast<Type>(1), gen, v);
      test_prox_two_simplex_set_params(v, gen, d_p, d_rhs, p, rhs, eps);
      test_prox_two_simplex_check_feasible(p, rhs, tol * eps, v);
    }
  }

  for (int pow = pow_from; pow < pow_to; ++pow) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, pow, pow + 1, -static_cast<Type>(1), gen, v);
      test_prox_two_simplex_set_params(v, gen, d_p, d_rhs, p, rhs, eps);
      test_prox_two_simplex_check_feasible(p, rhs, tol * eps, v);
    }
  }

  for (int pow = pow_from; pow < pow_to; ++pow) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, pow, pow + 1, static_cast<Type>(1), gen, v);
      test_populate_real(100, pow, pow + 1, -static_cast<Type>(1), gen, v);
      test_prox_two_simplex_set_params(v, gen, d_p, d_rhs, p, rhs, eps);
      test_prox_two_simplex_check_feasible(p, rhs, tol * eps, v);
    }
  }

  for (int i = 0; i < 100; ++i) {
    for (int pow = pow_from; pow < pow_to; ++pow) {
      test_populate_real(25, pow, pow + 1, static_cast<Type>(1), gen, v);
      test_populate_real(25, pow, pow + 1, -static_cast<Type>(1), gen, v);
      test_prox_two_simplex_set_params(v, gen, d_p, d_rhs, p, rhs, eps);
      test_prox_two_simplex_check_feasible(p, rhs, tol * eps, v);
    }
  }
}

TEST(ProxTwoSimplexTest, test_prox_feasible_float) {
  test_prox_two_simplex_feasible<float>(-3, 3, 1);
}

TEST(ProxTwoSimplexTest, test_prox_feasible_double) {
  test_prox_two_simplex_feasible<double>(-6, 6, 1);
}

template <typename Type>
inline void test_prox_two_simplex_kkt_sweep() {
  std::mt19937 gen(17);
  std::normal_distribution<Type> d(0, 1);
  const Type tol = 32 * std::numeric_limits<Type>::epsilon();
  for (int pow = -30; pow <= 30; pow += 4) {
    const Type s = static_cast<Type>(std::pow(10.0L, pow));
    for (const std::size_t n : {std::size_t(3), std::size_t(10)}) {
      for (int i = 0; i < 10; ++i) {
        std::vector<Type> v(n);
        for (auto& x : v) x = s * d(gen);
        const std::ptrdiff_t p = 1 + i % static_cast<int>(n - 1);
        const Type rhs = s * static_cast<Type>(1 + i % 3) / 2;
        const std::vector<Type> a(v.begin(), v.begin() + p), b(v.begin() + p, v.end());
        sdca::prox_two_simplex(v.begin(), v.begin() + p, v.end(), rhs);
        const std::vector<Type> x(v.begin(), v.begin() + p), y(v.begin() + p, v.end());
        EXPECT_LE(two_simplex_kkt_residual(a, b, x, y, rhs), tol) << "s = " << s << " n = " << n;
      }
    }
  }
}

TEST(ProxTwoSimplexTest, test_prox_kkt_sweep) {
  test_prox_two_simplex_kkt_sweep<float>();
  test_prox_two_simplex_kkt_sweep<double>();
}

// Both projection precisions agree to rounding on ordinary blocks; the huge
// pair beside a small mass (extended's case) keeps its small entries in both.
TEST(ProxTwoSimplexTest, test_projection_precisions_agree) {
  std::mt19937 gen(19);
  std::normal_distribution<double> d(0, 1);
  const double eps = std::numeric_limits<double>::epsilon();
  for (int i = 0; i < 2000; ++i) {
    const std::size_t n = 2 + std::size_t(i) % 40;
    const std::ptrdiff_t p = 1 + i % static_cast<int>(n - 1);
    std::vector<double> v(n);
    for (auto& x : v) x = d(gen);
    const double rhs = std::ldexp(1., i % 9 - 4);
    std::vector<double> fast = v, extended = v;
    sdca::prox_two_simplex(fast.begin(), fast.begin() + p, fast.end(), rhs,
                           sdca::projection_precision::float64);
    sdca::prox_two_simplex(extended.begin(), extended.begin() + p, extended.end(), rhs,
                           sdca::projection_precision::extended);
    double scale = rhs;
    for (double x : v) scale = std::max(scale, std::abs(x));
    for (std::size_t j = 0; j < n; ++j) EXPECT_NEAR(fast[j], extended[j], 8 * eps * scale) << i;
  }
  for (const auto mode :
       {sdca::projection_precision::float64, sdca::projection_precision::extended}) {
    for (double gap : {1e13, 1e15, 1e16}) {
      // Beside 1e16, 1 needs more than the 53 bits of double; in double, the
      // extended solver resolves it beside 1e13.
      const bool extended = mode == sdca::projection_precision::extended;
      if (gap > (wide_long_double() ? 1e16 : extended ? 1e13 : 1e15)) continue;
      double a[] = {gap, 1, 0}, b[] = {gap, 0, 0};
      sdca::prox_two_simplex(a, a + 3, b, b + 3, 2 * gap, mode);
      EXPECT_NEAR(a[1], .8, 1e-15);
      EXPECT_NEAR(b[1], .2, 1e-15);
    }
  }
}
