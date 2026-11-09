#include <algorithm>
#include <vector>
#include "sdca/math/exp_rows.h"
#include "sdca/math/entropy.h"
#include "sdca/math/log_exp.h"
#include "sdca/math/prox_rows.h"
#include "test_util.h"

template <typename Type, typename Function>
inline void test_log_exp_traits_one(Type x, Type y, Function fun) {
  auto is_finite_and_normal = [&](Type z) -> bool {
    return std::isfinite(z) && std::isnormal(z) && std::isfinite(fun(z)) && std::isnormal(fun(z));
  };
  EXPECT_TRUE(is_finite_and_normal(x));
#ifndef __FAST_MATH__
  EXPECT_FALSE(is_finite_and_normal(y));
#else
  // This is just a dummy test to keep the y variable used
  EXPECT_FALSE(std::isnan(y));
#endif
}

template <typename Type>
inline void test_log_exp_traits() {
  Type x;
  x = sdca::log_traits<Type>::min_arg();
  test_log_exp_traits_one(x, prev_float(x), [](Type z) { return std::log(z); });
  x = sdca::log_traits<Type>::max_arg();
  test_log_exp_traits_one(x, next_float(x), [](Type z) { return std::log(z); });

  x = sdca::exp_traits<Type>::min_arg();
  test_log_exp_traits_one(x, prev_float(x), [](Type z) { return std::exp(z); });
  x = sdca::exp_traits<Type>::max_arg();
  test_log_exp_traits_one(x, next_float(x), [](Type z) { return std::exp(z); });
}

template <typename Type, typename Result>
inline void test_log_sum_exp_compare(const Result eps, const std::vector<Type>& v) {
  Result sum(0);
  std::for_each(v.begin(), v.end(), [&](const Type x) { sum += std::exp(static_cast<Result>(x)); });

  Result log_sum = sdca::log_sum_exp<Result>(v.begin(), v.end());
  Result log_1_sum = sdca::log_1_sum_exp<Result>(v.begin(), v.end());

  ASSERT_TRUE(std::abs(std::log(sum) - log_sum) < eps);
  ASSERT_TRUE(std::abs(std::log(1 + sum) - log_1_sum) < eps);

  log_sum = 0;
  log_1_sum = 0;
  sdca::log_sum_exp<Result>(v.begin(), v.end(), log_sum, log_1_sum);

  ASSERT_TRUE(std::abs(std::log(sum) - log_sum) < eps);
  ASSERT_TRUE(std::abs(std::log(1 + sum) - log_1_sum) < eps);
}

template <typename Type>
inline void test_log_sum_exp_finite(const std::vector<Type>& v) {
  ASSERT_TRUE(std::isfinite(sdca::log_sum_exp(v.begin(), v.end())));
  ASSERT_TRUE(std::isfinite(sdca::log_1_sum_exp(v.begin(), v.end())));
  Type lse(-1), lse1(-1);
  sdca::log_sum_exp(v.begin(), v.end(), lse, lse1);
  ASSERT_TRUE(std::isfinite(lse));
  ASSERT_TRUE(std::isfinite(lse1));
}

template <typename Type, typename Result = Type>
inline void test_log_sum_exp(const int pow_from, const int pow_to) {
  std::mt19937 gen(1);
  Result eps = 1024 * std::numeric_limits<Result>::epsilon();

  std::vector<Type> v;
  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    test_populate_real(10000, p, p + 1, static_cast<Type>(1), gen, v);
    test_log_sum_exp_compare(eps, v);
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    test_populate_real(10000, p, p + 1, -static_cast<Type>(1), gen, v);
    test_log_sum_exp_compare(eps, v);
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    test_populate_real(5000, p, p + 1, static_cast<Type>(1), gen, v);
    test_populate_real(5000, p, p + 1, -static_cast<Type>(1), gen, v);
    test_log_sum_exp_compare(eps, v);
  }

  for (int p = pow_from; p < pow_to; ++p) {
    test_populate_real(1000, p, p + 1, static_cast<Type>(1), gen, v);
    test_populate_real(1000, p, p + 1, -static_cast<Type>(1), gen, v);
    test_log_sum_exp_compare<Type, Result>(eps, v);
  }
}

template <typename Type, typename Result = Type>
inline void test_log_sum_exp_special_cases(const int pow_from, const int pow_to,
                                           const short int eps_tol = 1) {
  std::mt19937 gen(1);
  std::vector<Type> v;

  // Empty input
  Type lse(-1), lse1(-1);
  sdca::log_sum_exp(v.begin(), v.begin(), lse, lse1);
  ASSERT_TRUE(lse == 0);
  ASSERT_TRUE(lse1 == 0);
  ASSERT_TRUE(sdca::log_sum_exp(v.begin(), v.begin()) == 0);
  ASSERT_TRUE(sdca::log_1_sum_exp(v.begin(), v.begin()) == 0);

  // Single element
  Type eps = eps_tol * std::numeric_limits<Type>::epsilon();
  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    test_populate_real(1, p, p + 1, static_cast<Type>(1), gen, v);
    sdca::log_sum_exp(v.begin(), v.end(), lse, lse1);
    ASSERT_EQ(v.front(), lse);
    ASSERT_EQ(v.front(), sdca::log_sum_exp<Type>(v.begin(), v.end()));
    if (v.front() <= sdca::exp_traits<Type>::max_arg()) {
      Type lse1_ref = std::log1p(std::exp(v.front()));
      ASSERT_TRUE(std::abs(lse1_ref - lse1) < eps);
      ASSERT_TRUE(std::abs(lse1_ref - sdca::log_1_sum_exp<Type>(v.begin(), v.end())) < eps);
    }
  }

  // Test overflow
  v.clear();
  test_populate_real(10000, pow_from, pow_to, static_cast<Type>(1), gen, v);
  test_log_sum_exp_finite(v);
  v.clear();
  test_populate_real(10000, pow_from, pow_to, -static_cast<Type>(1), gen, v);
  test_log_sum_exp_finite(v);
  v.clear();
  test_populate_real(5000, pow_from, pow_to, static_cast<Type>(1), gen, v);
  test_populate_real(5000, pow_from, pow_to, -static_cast<Type>(1), gen, v);
  test_log_sum_exp_finite(v);
}

TEST(LogExpTest, log_exp_traits) {
  test_log_exp_traits<float>();
  test_log_exp_traits<double>();
  test_log_exp_traits<long double>();
}

TEST(LogExpTest, log_sum_exp_extensive) {
  test_log_sum_exp<float, double>(-16, 2);
  test_log_sum_exp<double, double>(-16, 2);
  // Scores of 10^3 overflow exp in double.
  if (wide_long_double())
    test_log_sum_exp<long double, long double>(-24, 3);
  else
    test_log_sum_exp<long double, long double>(-16, 2);
}

TEST(LogExpTest, log_sum_exp_special_cases) {
  test_log_sum_exp_special_cases<float, double>(-16, 16);
  test_log_sum_exp_special_cases<double, double>(-16, 16);
  test_log_sum_exp_special_cases<long double, long double>(-24, 24, 1024);
}

TEST(ExpRowsTest, exp_normal_within_an_ulp_of_libm) {
  // exp_normal is within an ulp of the true exp, libm within half an ulp.
  double worst = 0;
  for (int i = 0; i <= 200000; ++i) {
    const double x = -708.0 * i / 200000.0, expected = std::exp(x);
    const double ulp = std::nextafter(expected, 1e300) - expected;
    worst = std::max(worst, std::abs(sdca::detail::exp_normal(x) - expected) / ulp);
  }
  EXPECT_LE(worst, 1.5);
}

TEST(ExpRowsTest, sums_and_terms_skip_one_entry_on_any_path) {
  for (std::size_t n : {2, 7, 33, 300}) {
    std::vector<double> z(n), terms(n), reference(n);
    for (std::size_t j = 0; j < n; ++j) z[j] = std::sin(1.0 + j) * 30 - (j % 5 == 0 ? 650 : 0);
    for (std::size_t skip : {std::size_t(0), n / 2, n - 1}) {
      for (bool normal : {true, false}) {
        const double sum = sdca::detail::exp_sum(n, z.data(), skip, 1.0, terms.data(), normal);
        const double alone =
            sdca::detail::exp_sum(n, z.data(), skip, 1.0, static_cast<double*>(nullptr), normal);
        long double expected = 0;
        for (std::size_t j = 0; j < n; ++j)
          if (j != skip) expected += std::exp(static_cast<long double>(z[j]) - 1);
        EXPECT_EQ(sum, alone);
        EXPECT_NEAR(sum, static_cast<double>(expected), 4e-16 * sum);
        EXPECT_EQ(terms[skip], 0);
      }
      // Terms below double's normal range take libm's exp.
      z[skip == 0 ? 1 : 0] = -800;
      EXPECT_EQ(n > 2, sdca::detail::exp_sum(n, z.data(), skip, 1.0, terms.data(), false) > 0);
      EXPECT_EQ(terms[skip == 0 ? 1 : 0], 0);
      z[skip == 0 ? 1 : 0] = 0;
    }
  }
}

TEST(ProxRowsTest, log1p_lane_within_an_ulp) {
  double worst = 0;
  for (int i = -2000; i <= 2000; ++i) {
    const double scaled = std::sinh(i / 100.0);
    for (double x : {scaled * 1e-9, std::tanh(i / 300.0), scaled}) {
      if (!(x > -1)) continue;
      const long double expected = std::log1p(static_cast<long double>(x));
      const double rounded = static_cast<double>(expected);
      const double ulp = std::nextafter(std::fabs(rounded), 1e300) - std::fabs(rounded);
      const long double error =
          std::fabs(static_cast<long double>(sdca::detail::log1p_lane(x)) - expected);
      if (ulp > 0)
        worst = std::max(worst, static_cast<double>(error / static_cast<long double>(ulp)));
    }
  }
  EXPECT_LE(worst, 1.0);
  EXPECT_EQ(sdca::detail::log1p_lane(0.0), 0.0);
  EXPECT_EQ(sdca::detail::log1p_lane(1e-300), 1e-300);
}

TEST(ProxRowsTest, probabilities_match_the_scalar_function) {
  for (double alpha : {1e-3, 1.0, 50.0}) {
    const double log_alpha = std::log(alpha);
    std::vector<double> v(100), prob(100);
    for (std::size_t j = 0; j < v.size(); ++j)
      v[j] = 6 * std::sin(1.0 + 3.0 * j) - (j % 9 ? 0 : 40);
    for (double dt : {-2.0, 0.5, 30.0, 800.0}) {
      sdca::detail::probability_kernel(v.size(), v.data(), 0.0, dt, 0.0, alpha, log_alpha,
                                       prob.data());
      for (std::size_t j = 0; j < v.size(); ++j) {
        const double h = v[j] - dt;
        const double expected = sdca::detail::probability_scalar(h, alpha, log_alpha);
        // Two roundings apart; the scalar function's exp(h - W) (h + log alpha < -18) carries
        // the rounding of h - W, |h| / 2 ulp (the kernel's exp(h) W(y) / y does not).
        const double ulps = 4 + (h + log_alpha < -18 ? std::fabs(h) / 2 : 0.0);
        EXPECT_NEAR(prob[j], expected, ulps * 2.3e-16 * expected) << alpha << " " << dt << " " << j;
      }
    }
  }
}

TEST(ProxRowsTest, residuals_match_entropy_residual) {
  // Whole blocks: the kernel reads row_lanes entries.
  const std::size_t n = 2 * sdca::detail::row_lanes;
  std::vector<double> t(n), q(n), log_q(n), guess(n), r(n);
  for (double alpha : {1e-2, 1.0, 100.0}) {
    for (std::size_t l = 0; l < n; ++l) {
      q[l] = std::exp(-0.3 * static_cast<double>(l));
      log_q[l] = std::log(q[l]);
      const double half = alpha * q[l] / 2;
      // Arguments inside the Halley window, one exactly zero.
      t[l] = l == 3 ? 0.0 : (half + 0.3) * std::sin(2.0 + l);
      guess[l] = l % 2 ? std::numeric_limits<double>::quiet_NaN() : 0.1 * q[l];
    }
    for (std::size_t begin = 0; begin < n; begin += sdca::detail::row_lanes) {
      const std::size_t count = std::min(n - begin, sdca::detail::row_lanes);
      sdca::detail::residual_kernel(count, t.data() + begin, q.data() + begin, log_q.data() + begin,
                                    guess.data() + begin, alpha, r.data() + begin);
    }
    for (std::size_t l = 0; l < n; ++l) {
      const double expected =
          sdca::detail::entropy_residual(t[l], alpha, q[l], std::log(alpha), log_q[l]);
      // Both meet the residual tolerance 8 eps |t| on alpha r - log1p(-r / q).
      const double slope = alpha + 1 / (q[l] - expected);
      EXPECT_NEAR(r[l], expected, 32 * 2.220446049250313e-16 * std::fabs(t[l]) / slope + 1e-300)
          << alpha << " " << l;
    }
    EXPECT_EQ(r[3], 0.0);
  }
}
