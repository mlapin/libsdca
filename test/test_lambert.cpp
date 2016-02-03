#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "sdca/math/lambert.h"

template <typename Type>
inline void
populate(const int n, const int pow_from, const int pow_to, const Type coeff,
    std::mt19937 &gen, std::vector<Type>& v) {
  Type base(10);
  for (int p = pow_from; p < pow_to; ++p) {
    Type x = std::pow(base, p);
    std::uniform_real_distribution<Type> d(x, x*base);
    for (int i = 0; i < n; ++i) {
      v.push_back(coeff * d(gen));
    }
  }
}

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
  auto tol = 4 * std::numeric_limits<float>::epsilon();
  std::cout << "lambert_w_exp_float: tol = " << tol << std::endl;

  std::vector<float> v;
  populate(10000, -8, 8, 1.0f, gen, v);
  test_lambert_w_exp_pos(tol, v);

  v.clear();
  populate(10000, -8, 8, -1.0f, gen, v);
  test_lambert_w_exp_neg(tol, v);
}

TEST(LambertTest, lambert_w_exp_double) {
  std::mt19937 gen(1);
  auto tol = 4 * std::numeric_limits<double>::epsilon();
  std::cout << "lambert_w_exp_double: tol = " << tol << std::endl;

  std::vector<double> v;
  populate(10000, -16, 16, 1.0, gen, v);
  test_lambert_w_exp_pos(tol, v);

  v.clear();
  populate(10000, -16, 16, -1.0, gen, v);
  test_lambert_w_exp_neg(tol, v);
}

TEST(LambertTest, lambert_w_exp_long_double) {
  std::mt19937 gen(1);
  auto tol = 4 * static_cast<long double>(
        std::numeric_limits<double>::epsilon());
  std::cout << "lambert_w_exp_long_double: tol = " << tol << std::endl;

  std::vector<long double> v;
  populate(10000, -16, 16, 1.0L, gen, v);
  test_lambert_w_exp_pos(tol, v);

  v.clear();
  populate(10000, -16, 16, -1.0L, gen, v);
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
  std::cout << "exp_approx_float: tol = " << tol << std::endl;

  std::vector<float> v;
  populate(10000, -8, 3, -1.0f, gen, v);
  test_exp_approx(tol, v);

  v.clear();
  populate(10000, -8, 0, 1.0f, gen, v);
  test_exp_approx(tol, v);
}

TEST(LambertTest, exp_approx_double) {
  std::mt19937 gen(1);
  double tol = 0.001;
  std::cout << "exp_approx_double: tol = " << tol << std::endl;

  std::vector<double> v;
  populate(10000, -16, 3, -1.0, gen, v);
  test_exp_approx(tol, v);

  v.clear();
  populate(10000, -16, 0, 1.0, gen, v);
  test_exp_approx(tol, v);
}

TEST(LambertTest, exp_approx_long_double) {
  std::mt19937 gen(1);
  long double tol = 0.001L;
  std::cout << "exp_approx_long_double: tol = " << tol << std::endl;

  std::vector<long double> v;
  populate(10000, -16, 3, -1.0L, gen, v);
  test_exp_approx(tol, v);

  v.clear();
  populate(10000, -16, 0, 1.0L, gen, v);
  test_exp_approx(tol, v);
}
