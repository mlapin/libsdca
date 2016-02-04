#include <cstdio>

#include "gtest/gtest.h"
#include "sdca/math/lambert.h"

#include "test_util.h"

template <typename Type>
inline void
test_lambert_w_exp_pos(const Type tol, const std::vector<Type>& v) {
  std::for_each(v.begin(), v.end(), [=](const Type &x){ ASSERT_NEAR(
    x, sdca::lambert_w_exp_inverse(sdca::lambert_w_exp(x)),
    tol * std::max(static_cast<Type>(1), x)); });
}

template <typename Type>
inline void
test_lambert_w_exp_neg(const Type tol, const std::vector<Type>& v) {
  std::for_each(v.begin(), v.end(), [=](const Type &x){ ASSERT_NEAR(
    std::exp(x), sdca::x_exp_x(sdca::lambert_w_exp(x)),
    tol); });
}

TEST(LambertTest, lambert_w_exp_float) {
  std::mt19937 gen(1);
  float tol = 4 * std::numeric_limits<float>::epsilon();
  std::printf("lambert_w_exp_float: tol = %e\n", tol);

  std::vector<float> v;
  test_populate(10000, -8, 8, 1.0f, gen, v);
  test_add_0_1_eps_min_max(1.0f, v);
  test_lambert_w_exp_pos(tol, v);

  v.clear();
  test_populate(10000, -8, 8, -1.0f, gen, v);
  test_add_0_1_eps_min_max(-1.0f, v);
  test_lambert_w_exp_neg(tol, v);
}

TEST(LambertTest, lambert_w_exp_double) {
  std::mt19937 gen(1);
  auto tol = 4 * std::numeric_limits<double>::epsilon();
  std::printf("lambert_w_exp_double: tol = %e\n", tol);

  std::vector<double> v;
  test_populate(10000, -16, 16, 1.0, gen, v);
  test_add_0_1_eps_min_max(1.0, v);
  test_lambert_w_exp_pos(tol, v);

  v.clear();
  test_populate(10000, -16, 16, -1.0, gen, v);
  test_add_0_1_eps_min_max(-1.0, v);
  test_lambert_w_exp_neg(tol, v);
}

TEST(LambertTest, lambert_w_exp_long_double) {
  std::mt19937 gen(1);
  auto tol = 4 * static_cast<long double>(
        std::numeric_limits<double>::epsilon());
  std::printf("lambert_w_exp_long_double: tol = %Le\n", tol);

  std::vector<long double> v;
  test_populate(10000, -16, 16, 1.0L, gen, v);
  test_add_0_1_eps_min(1.0L, v); // googletest does not support long double
  test_lambert_w_exp_pos(tol, v);

  v.clear();
  test_populate(10000, -16, 16, -1.0L, gen, v);
  test_add_0_1_eps_min_max(-1.0L, v);
  test_lambert_w_exp_neg(tol, v);
}

template <typename Type>
inline void
test_exp_approx(const Type tol, const std::vector<Type>& v) {
  std::for_each(v.begin(), v.end(), [=](const Type &x){ ASSERT_NEAR(
    std::exp(x), sdca::exp_approx(x),
    tol * std::max(static_cast<Type>(1), std::exp(x))); });
}

TEST(LambertTest, exp_approx_float) {
  std::mt19937 gen(1);
  float tol = 0.001f;
  std::printf("exp_approx_float: tol = %e\n", tol);

  std::vector<float> v;
  test_populate(10000, -8, 3, -1.0f, gen, v);
  test_add_0_1_eps_min(-1.0f, v);
  test_exp_approx(tol, v);

  v.clear();
  test_populate(10000, -8, 0, 1.0f, gen, v);
  test_add_0_1_eps_min(1.0f, v);
  test_exp_approx(tol, v);
}

TEST(LambertTest, exp_approx_double) {
  std::mt19937 gen(1);
  double tol = 0.001;
  std::printf("exp_approx_double: tol = %e\n", tol);

  std::vector<double> v;
  test_populate(10000, -16, 3, -1.0, gen, v);
  test_add_0_1_eps_min(-1.0, v);
  test_exp_approx(tol, v);

  v.clear();
  test_populate(10000, -16, 0, 1.0, gen, v);
  test_add_0_1_eps_min(1.0, v);
  test_exp_approx(tol, v);
}

TEST(LambertTest, exp_approx_long_double) {
  std::mt19937 gen(1);
  long double tol = 0.001L;
  std::printf("exp_approx_long_double: tol = %Le\n", tol);

  std::vector<long double> v;
  test_populate(10000, -16, 3, -1.0L, gen, v);
  test_add_0_1_eps_min(-1.0L, v);
  test_exp_approx(tol, v);

  v.clear();
  test_populate(10000, -16, 0, 1.0L, gen, v);
  test_add_0_1_eps_min(1.0L, v);
  test_exp_approx(tol, v);
}

TEST(LambertTest, omega_const) {
  float tol_f = std::numeric_limits<float>::epsilon();
  float w_f = sdca::lambert_w_exp(0.0f);
  EXPECT_NEAR(w_f, static_cast<float>(sdca::kOmega), tol_f);

  double tol_d = std::numeric_limits<double>::epsilon();
  double w_d = sdca::lambert_w_exp(0.0);
  EXPECT_NEAR(w_d, static_cast<double>(sdca::kOmega), tol_d);

  long double tol_l = std::numeric_limits<long double>::epsilon();
  long double w_l = sdca::lambert_w_exp(0.0L);
  EXPECT_NEAR(w_l, static_cast<long double>(sdca::kOmega), tol_l);

  std::printf("kOmega     : %.16Le\n"
              "long double: %.16Le\n"
              "double     : %.16e\n"
              "float      : %.16e\n"
              , sdca::kOmega, w_l, w_d, w_f);
}
