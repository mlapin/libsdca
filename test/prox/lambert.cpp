#include "sdca/math/lambert.h"
#include "sdca/math/log_exp.h"
#include "test_util.h"

template <typename Type>
inline void
test_lambert_w_exp_pos(const Type eps, const std::vector<Type>& v) {
  std::for_each(v.begin(), v.end(), [=](const Type x){
    ASSERT_TRUE(
      std::abs(x - sdca::lambert_w_exp_inverse(sdca::lambert_w_exp(x)))
      < eps * std::max(static_cast<Type>(1), x)
    ); });
}

template <typename Type>
inline void
test_lambert_w_exp_neg_check(
    const Type x,
    const Type eps
  ) {
  Type d = std::abs(std::exp(x) - sdca::x_exp_x(sdca::lambert_w_exp(x)));
  // Without this, the test fails with -Ofast on long double
  if (d >= eps) {
    std::printf("%.24Le\n\n", static_cast<long double>(d));
  }
  ASSERT_TRUE(d < eps);
}

template <typename Type>
inline void
test_lambert_w_exp_neg(const Type eps, const std::vector<Type>& v) {
  std::for_each(v.begin(), v.end(), [=](const Type x){
#ifdef SDCA_ACCURATE_MATH
    ASSERT_TRUE(
      std::abs(std::exp(x) - sdca::x_exp_x(sdca::lambert_w_exp(x))) < eps
    );
#else
    test_lambert_w_exp_neg_check(x, eps);
#endif
  });
}

TEST(LambertTest, lambert_w_exp_float) {
  std::mt19937 gen(1);
  float eps = 4 * std::numeric_limits<float>::epsilon();

  std::vector<float> v;
  test_populate_real(10000, -8, 8, 1.0f, gen, v);
  test_add_0_1_eps_min_max(1.0f, v);
  test_lambert_w_exp_pos(eps, v);

  v.clear();
  test_populate_real(10000, -8, 8, -1.0f, gen, v);
  test_add_0_1_eps_min_max(-1.0f, v);
  test_lambert_w_exp_neg(eps, v);
}

TEST(LambertTest, lambert_w_exp_double) {
  std::mt19937 gen(1);
  auto eps = 4 * std::numeric_limits<double>::epsilon();

  std::vector<double> v;
  test_populate_real(10000, -16, 16, 1.0, gen, v);
  test_add_0_1_eps_min_max(1.0, v);
  test_lambert_w_exp_pos(eps, v);

  v.clear();
  test_populate_real(10000, -16, 16, -1.0, gen, v);
  test_add_0_1_eps_min_max(-1.0, v);
  test_lambert_w_exp_neg(eps, v);
}

TEST(LambertTest, lambert_w_exp_long_double) {
  std::mt19937 gen(1);
  auto eps = static_cast<long double>(
    4 * std::numeric_limits<double>::epsilon());

  std::vector<long double> v;
  test_populate_real(10000, -16, 16, 1.0L, gen, v);
  test_add_0_1_eps_min(1.0L, v); // googletest does not support long double
  test_lambert_w_exp_pos(eps, v);

  v.clear();
  test_populate_real(10000, -16, 16, -1.0L, gen, v);
  test_add_0_1_eps_min_max(-1.0L, v);
  test_lambert_w_exp_neg(eps, v);
}

template <typename Type>
inline void
test_lambert_exp_approx(const Type eps, const std::vector<Type>& v) {
  std::for_each(v.begin(), v.end(), [=](const Type x){
    ASSERT_TRUE(
      std::abs(std::exp(x) - sdca::exp_approx(x))
      < eps * std::max(static_cast<Type>(1), std::exp(x))
    ); });
}

TEST(LambertTest, exp_approx_float) {
  std::mt19937 gen(1);
  float eps = 0.001f;

  std::vector<float> v;
  test_populate_real(10000, -8, 3, -1.0f, gen, v);
  test_add_0_1_eps_min(-1.0f, v);
  test_lambert_exp_approx(eps, v);

  v.clear();
  test_populate_real(10000, -8, 0, 1.0f, gen, v);
  test_add_0_1_eps_min(1.0f, v);
  test_lambert_exp_approx(eps, v);
}

TEST(LambertTest, exp_approx_double) {
  std::mt19937 gen(1);
  double eps = 0.001;

  std::vector<double> v;
  test_populate_real(10000, -16, 3, -1.0, gen, v);
  test_add_0_1_eps_min(-1.0, v);
  test_lambert_exp_approx(eps, v);

  v.clear();
  test_populate_real(10000, -16, 0, 1.0, gen, v);
  test_add_0_1_eps_min(1.0, v);
  test_lambert_exp_approx(eps, v);
}

TEST(LambertTest, exp_approx_long_double) {
  std::mt19937 gen(1);
  long double eps = 0.001L;

  std::vector<long double> v;
  test_populate_real(10000, -16, 3, -1.0L, gen, v);
  test_add_0_1_eps_min(-1.0L, v);
  test_lambert_exp_approx(eps, v);

  v.clear();
  test_populate_real(10000, -16, 0, 1.0L, gen, v);
  test_add_0_1_eps_min(1.0L, v);
  test_lambert_exp_approx(eps, v);
}

TEST(LambertTest, omega_const) {
  float eps_f = std::numeric_limits<float>::epsilon();
  float w_f = sdca::lambert_w_exp(0.0f);
  EXPECT_NEAR(w_f, static_cast<float>(sdca::kOmega), eps_f);

  double eps_d = std::numeric_limits<double>::epsilon();
  double w_d = sdca::lambert_w_exp(0.0);
  EXPECT_NEAR(w_d, static_cast<double>(sdca::kOmega), eps_d);

  long double eps_l = std::numeric_limits<long double>::epsilon();
  long double w_l = sdca::lambert_w_exp(0.0L);
  long double omega = static_cast<long double>(sdca::kOmega);
  EXPECT_TRUE(std::abs(w_l - omega) < eps_l);
}
