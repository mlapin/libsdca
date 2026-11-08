#include "sdca/prox/knapsack_eq.h"
#include "test_util.h"

template <typename Type>
inline void test_prox_knapsack_eq_check_feasible(const Type lo, const Type hi, const Type rhs,
                                                 const Type eps, std::vector<Type>& v) {
  sdca::prox_knapsack_eq(v.begin(), v.end(), lo, hi, rhs);

  std::for_each(v.begin(), v.end(), [=](const Type x) { ASSERT_GE(x, lo); });
  std::for_each(v.begin(), v.end(), [=](const Type x) { ASSERT_LE(x, hi); });
  Type sum = std::accumulate(v.begin(), v.end(), static_cast<Type>(0));
  ASSERT_NEAR(rhs, sum, eps);
}

template <typename Type>
inline void test_prox_knapsack_eq_set_params(const std::vector<Type>& v, std::mt19937& gen,
                                             std::uniform_real_distribution<Type>& d_lo,
                                             std::uniform_real_distribution<Type>& d_hi,
                                             std::uniform_real_distribution<Type>& d_rhs, Type& lo,
                                             Type& hi, Type& rhs, Type& eps) {
  Type size = static_cast<Type>(v.size());
  for (;;) {
    lo = d_lo(gen);
    hi = d_hi(gen);
    rhs = d_rhs(gen);
    if (lo <= hi && lo * size <= rhs && hi * size >= rhs) break;
  }
  Type max(*std::max_element(v.begin(), v.end()));
  eps = std::numeric_limits<Type>::epsilon() * std::max(static_cast<Type>(1), std::abs(max)) *
        static_cast<Type>(v.size());
}

template <typename Type>
inline void test_prox_knapsack_eq_feasible(const int pow_from, const int pow_to, const Type tol) {
  std::mt19937 gen(1);
  std::uniform_real_distribution<Type> d_lo(-2, 0.5);
  std::uniform_real_distribution<Type> d_hi(-0.5, 2);
  std::uniform_real_distribution<Type> d_rhs(-5, 5);

  Type lo, hi, rhs, eps;
  std::vector<Type> v;

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_prox_knapsack_eq_set_params(v, gen, d_lo, d_hi, d_rhs, lo, hi, rhs, eps);
      test_prox_knapsack_eq_check_feasible(lo, hi, rhs, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_knapsack_eq_set_params(v, gen, d_lo, d_hi, d_rhs, lo, hi, rhs, eps);
      test_prox_knapsack_eq_check_feasible(lo, hi, rhs, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_knapsack_eq_set_params(v, gen, d_lo, d_hi, d_rhs, lo, hi, rhs, eps);
      test_prox_knapsack_eq_check_feasible(lo, hi, rhs, tol * eps, v);
    }
  }

  for (int i = 0; i < 100; ++i) {
    for (int p = pow_from; p < pow_to; ++p) {
      test_populate_real(25, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(25, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_knapsack_eq_set_params(v, gen, d_lo, d_hi, d_rhs, lo, hi, rhs, eps);
      test_prox_knapsack_eq_check_feasible(lo, hi, rhs, tol * eps, v);
    }
  }
}

TEST(ProxKnapsackEQTest, test_prox_feasible_float) {
#ifndef __FAST_MATH__
  test_prox_knapsack_eq_feasible<float>(-3, 3, 256);
#else
  test_prox_knapsack_eq_feasible<float>(-3, 3, 512);
#endif
}

TEST(ProxKnapsackEQTest, test_prox_feasible_double) {
#ifndef __FAST_MATH__
  // Sums accumulate in long double; in double they round up to 64 times more.
  test_prox_knapsack_eq_feasible<double>(-6, 6, 256 * (wide_long_double() ? 1 : 64));
#else
  test_prox_knapsack_eq_feasible<double>(-3, 3, 512);
#endif
}

// Opposite saturated masses are combined before rhs: the free entry carries
// the small rhs, also when the bound products round (checked in long double).
TEST(ProxKnapsackEQTest, test_prox_signed_saturated_bounds) {
  std::vector<double> x{-1e25, 0, 1e25};
  sdca::prox_knapsack_eq(x.begin(), x.end(), -1e20, 1e20, -1.);
  EXPECT_EQ(x, (std::vector<double>{-1e20, -1, 1e20}));

  const double hi = std::ldexp(1.0, 67) - std::ldexp(1.0, 14), lo = -1.5 * hi;
  std::vector<double> y{1e30, 1e30, 1e30, 0, -1e30, -1e30};
  sdca::prox_knapsack_eq(y.begin(), y.end(), lo, hi, -1.);
  if (std::numeric_limits<long double>::digits >= 64) {
    const long double mass = 3.0L * static_cast<long double>(hi) +
                             2.0L * static_cast<long double>(lo);  // exact with 64 digits
    EXPECT_NE(static_cast<long double>(3 * hi + 2 * lo), mass);    // the products round
    EXPECT_EQ(y[3], static_cast<double>(-1 - mass));
  }
  EXPECT_EQ(y[0], hi);
  EXPECT_EQ(y[5], lo);
}

// Signed problems at scales 1e-30 to 1e30 satisfy the KKT conditions to the
// rounding of each condition's own terms.
template <typename Type>
inline void test_prox_knapsack_eq_kkt_sweep() {
  std::mt19937 gen(11);
  std::normal_distribution<Type> d_a(0, 1);
  std::uniform_real_distribution<Type> d_u(0, 1);
  const Type tol = 64 * std::numeric_limits<Type>::epsilon();
  for (int p = -30; p <= 30; p += 2) {
    const Type s = static_cast<Type>(std::pow(10.0L, p));
    for (const std::size_t n : {std::size_t(2), std::size_t(5), std::size_t(20)}) {
      for (int i = 0; i < 20; ++i) {
        std::vector<Type> a(n);
        for (auto& v : a) v = s * d_a(gen);
        const Type lo = -s * d_u(gen), hi = s * (d_u(gen) + Type(0.1));
        const Type rhs = static_cast<Type>(n) * (lo + (hi - lo) * d_u(gen));
        std::vector<Type> x = a;
        sdca::prox_knapsack_eq(x.begin(), x.end(), lo, hi, rhs);
        EXPECT_LE(knapsack_kkt_residual(a, x, lo, hi, rhs, Type(0), true), tol)
            << "s = " << s << " n = " << n << " i = " << i;
      }
    }
  }
}

TEST(ProxKnapsackEQTest, test_prox_kkt_sweep) {
  test_prox_knapsack_eq_kkt_sweep<float>();
  test_prox_knapsack_eq_kkt_sweep<double>();
}
