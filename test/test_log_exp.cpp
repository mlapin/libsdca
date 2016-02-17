#include <cstdio>

#include "gtest/gtest.h"
#include "sdca/math/log_exp.h"

#include "test_util.h"

template <typename Type, typename Function>
inline void
test_log_exp_traits_one(Type x, Type y, Function fun) {
  auto is_finite_and_normal = [&](Type z) -> bool {
    return std::isfinite(z) && std::isnormal(z)
        && std::isfinite(fun(z)) && std::isnormal(fun(z));
  };
  EXPECT_TRUE(is_finite_and_normal(x));
  EXPECT_FALSE(is_finite_and_normal(y));
}

template <typename Type>
inline void
test_log_exp_traits() {
  Type x;
  x = sdca::log_traits<Type>::min_arg();
  test_log_exp_traits_one(x, prev_float(x), [](Type z){ return std::log(z); });
  x = sdca::log_traits<Type>::max_arg();
  test_log_exp_traits_one(x, next_float(x), [](Type z){ return std::log(z); });

  x = sdca::exp_traits<Type>::min_arg();
  test_log_exp_traits_one(x, prev_float(x), [](Type z){ return std::exp(z); });
  x = sdca::exp_traits<Type>::max_arg();
  test_log_exp_traits_one(x, next_float(x), [](Type z){ return std::exp(z); });
}

template <typename Type,
          typename Result = Type>
inline void
test_log_sum_exp_compare(const Type eps, const std::vector<Type>& v) {
  Type sum(0);
  std::for_each(v.begin(), v.end(), [&](const Type x){ sum += std::exp(x); });
  ASSERT_NEAR(std::log(sum),
              sdca::log_sum_exp<Result>(v.begin(), v.end()),
              eps);
  ASSERT_NEAR(std::log(1 + sum),
              sdca::log_1_sum_exp<Result>(v.begin(), v.end()),
              eps);
  Result lse(-1), lse1(-1);
  sdca::log_sum_exp<Result>(v.begin(), v.end(), lse, lse1);
  ASSERT_NEAR(std::log(sum),
              static_cast<Type>(lse),
              eps);
  ASSERT_NEAR(std::log(1 + sum),
              static_cast<Type>(lse1),
              eps);
}

template <typename Type>
inline void
test_log_sum_exp_finite(const std::vector<Type>& v) {
  ASSERT_TRUE(std::isfinite(sdca::log_sum_exp(v.begin(), v.end())));
  ASSERT_TRUE(std::isfinite(sdca::log_1_sum_exp(v.begin(), v.end())));
  Type lse(-1), lse1(-1);
  sdca::log_sum_exp(v.begin(), v.end(), lse, lse1);
  ASSERT_TRUE(std::isfinite(lse));
  ASSERT_TRUE(std::isfinite(lse1));
}

template <typename Type,
          typename Result = Type>
inline void
test_log_sum_exp(const int pow_from, const int pow_to) {
  std::mt19937 gen(1);
  Type eps = 1024 * std::numeric_limits<Type>::epsilon();

  std::vector<Type> v;
  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    test_populate(10000, p, p + 1, static_cast<Type>(1), gen, v);
    test_log_sum_exp_compare(eps, v);
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    test_populate(10000, p, p + 1, -static_cast<Type>(1), gen, v);
    test_log_sum_exp_compare(eps, v);
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    test_populate(5000, p, p + 1, static_cast<Type>(1), gen, v);
    test_populate(5000, p, p + 1, -static_cast<Type>(1), gen, v);
    test_log_sum_exp_compare(eps, v);
  }

  for (int p = pow_from; p < pow_to; ++p) {
    test_populate(1000, p, p + 1, static_cast<Type>(1), gen, v);
    test_populate(1000, p, p + 1, -static_cast<Type>(1), gen, v);
    test_log_sum_exp_compare<Type, Result>(eps, v);
  }
}

template <typename Type,
          typename Result = Type>
inline void
test_log_sum_exp_special_cases(const int pow_from, const int pow_to) {
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
  Type eps = 4 * std::numeric_limits<Type>::epsilon();
  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    test_populate(1, p, p + 1, static_cast<Type>(1), gen, v);
    sdca::log_sum_exp(v.begin(), v.end(), lse, lse1);
    ASSERT_EQ(v.front(), lse);
    ASSERT_EQ(v.front(), sdca::log_sum_exp<Type>(v.begin(), v.end()));
    Type lse1_ref = std::log1p(std::exp(v.front()));
    if (std::isfinite(lse1_ref)) {
      ASSERT_NEAR(lse1_ref, lse1, eps);
      ASSERT_NEAR(lse1_ref, sdca::log_1_sum_exp<Type>(v.begin(), v.end()), eps);
    }
  }

  // Test overflow
  v.clear();
  test_populate(10000, pow_from, pow_to, static_cast<Type>(1), gen, v);
  test_log_sum_exp_finite(v);
  v.clear();
  test_populate(10000, pow_from, pow_to, -static_cast<Type>(1), gen, v);
  test_log_sum_exp_finite(v);
  v.clear();
  test_populate(5000, pow_from, pow_to, static_cast<Type>(1), gen, v);
  test_populate(5000, pow_from, pow_to, -static_cast<Type>(1), gen, v);
  test_log_sum_exp_finite(v);
}

TEST(LogExpTest, log_exp_traits) {
  test_log_exp_traits<float>();
  test_log_exp_traits<double>();
  test_log_exp_traits<long double>();
}

TEST(LogExpTest, log_sum_exp_extensive) {
  test_log_sum_exp<float, double>(-8, 1);
  test_log_sum_exp<double, double>(-16, 2);
  test_log_sum_exp<long double, long double>(-24, 4);
}

TEST(LogExpTest, log_sum_exp_special_cases) {
  test_log_sum_exp_special_cases<float, double>(-8, 8);
  test_log_sum_exp_special_cases<double, double>(-16, 16);
  test_log_sum_exp_special_cases<long double, long double>(-24, 24);
}
