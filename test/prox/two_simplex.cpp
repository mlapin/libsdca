#include "sdca/prox/two_simplex.h"
#include "test_util.h"

template <typename Type>
inline void
test_prox_two_simplex_check_feasible(
    const ptrdiff_t p,
    const Type rhs,
    const Type eps, std::vector<Type>& v) {
  ASSERT_TRUE(p > 0);
  ASSERT_TRUE(static_cast<std::size_t>(p) < v.size());
  std::vector<Type> u(v);
  sdca::prox_two_simplex(v.begin(), v.begin() + p, v.end(), rhs);

  Type lo(0), hi(rhs);
  std::for_each(v.begin(), v.end(), [=](const Type x){
    ASSERT_GE(x, lo); });
  std::for_each(v.begin(), v.end(), [=](const Type x){
    ASSERT_LE(x, hi); });

  Type sum1 = std::accumulate(v.begin(), v.begin() + p, static_cast<Type>(0));
  Type sum2 = std::accumulate(v.begin() + p, v.end(), static_cast<Type>(0));
  ASSERT_LE(sum1, rhs + eps);
  ASSERT_LE(sum2, rhs + eps);
  if (std::abs(sum1 - sum2) > eps &&
      std::abs(sum1 - sum2) > static_cast<Type>(1e-5)) {
    std::printf("sum1 = %.15e\n", sum1);
    std::printf("sum2 = %.15e\n", sum2);
    std::printf("diff = %.15e\n", std::abs(sum1 - sum2));
    sdca::prox_two_simplex(u.begin(), u.begin() + p, u.end(), rhs);
  }
  ASSERT_NEAR(sum1, sum2, eps);
}

template <typename Type>
inline void
test_prox_two_simplex_set_params(
    const std::vector<Type>& v,
    std::mt19937& gen,
    std::uniform_int_distribution<ptrdiff_t>& d_p,
    std::uniform_real_distribution<Type>& d_rhs,
    ptrdiff_t& p, Type& rhs, Type& eps) {
  p = d_p(gen);
  rhs = d_rhs(gen);
  Type max(*std::max_element(v.begin(), v.end()));
  eps = v.size() * std::max(static_cast<Type>(1), std::abs(max))
      * std::numeric_limits<Type>::epsilon();
}

template <typename Type>
inline void
test_prox_two_simplex_feasible(
    const int pow_from, const int pow_to, const int tol) {
  std::mt19937 gen(1);
  std::uniform_int_distribution<ptrdiff_t> d_p(1, 10);
  std::uniform_real_distribution<Type> d_rhs(0, 5);

  ptrdiff_t p;
  Type rhs, eps;
  std::vector<Type> v;

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
