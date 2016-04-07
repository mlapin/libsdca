#include "sdca/prox/two_entropy.h"
#include "test_util.h"

template <typename Type>
inline void
test_prox_two_entropy_check_feasible(
    const ptrdiff_t p,
    const Type alpha,
    const Type eps, std::vector<Type>& v) {
  ASSERT_TRUE(p > 0);
  ASSERT_TRUE(static_cast<std::size_t>(p) < v.size());
  std::vector<Type> u(v);
  sdca::prox_two_entropy(v.begin(), v.begin() + p, v.end(), alpha);

  Type lo(0), hi(1);
  std::for_each(v.begin(), v.end(), [=](const Type x){
    ASSERT_GE(x, lo); });
  std::for_each(v.begin(), v.end(), [=](const Type x){
    ASSERT_LE(x, hi); });

  Type sum1 = std::accumulate(v.begin(), v.begin() + p, static_cast<Type>(0));
  Type sum2 = std::accumulate(v.begin() + p, v.end(), static_cast<Type>(0));
  ASSERT_LE(sum1, hi + eps);
  ASSERT_LE(sum2, hi + eps);

  if (sum1 + sum2 < hi - eps) {
    sdca::prox_two_entropy(u.begin(), u.begin() + p, u.end(), alpha);
  }

  ASSERT_NEAR(sum1 + sum2, hi, eps);
}

template <typename Type>
inline void
test_prox_two_entropy_set_params(
    const std::vector<Type>& v,
    std::mt19937& gen,
    std::uniform_int_distribution<ptrdiff_t>& d_p,
    std::uniform_real_distribution<Type>& d_alpha,
    ptrdiff_t& p, Type& alpha, Type& eps) {
  p = d_p(gen);
  alpha = d_alpha(gen);
  Type max(*std::max_element(v.begin(), v.end()));
  eps = std::numeric_limits<Type>::epsilon()
      * std::max(static_cast<Type>(1), std::abs(max))
      * static_cast<Type>(v.size());
}

template <typename Type>
inline void
test_prox_two_entropy_feasible(
    const int pow_from, const int pow_to, const Type tol) {
  std::mt19937 gen(1);
  std::uniform_int_distribution<ptrdiff_t> d_p(1, 10);
  std::uniform_real_distribution<Type> d_alpha(0, 5);

  ptrdiff_t p;
  Type alpha, eps;
  std::vector<Type> v;

  for (int pow = pow_from; pow < pow_to; ++pow) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, pow, pow + 1, static_cast<Type>(1), gen, v);
      test_prox_two_entropy_set_params(v, gen, d_p, d_alpha, p, alpha, eps);
      test_prox_two_entropy_check_feasible(p, alpha, tol * eps, v);
    }
  }

  for (int pow = pow_from; pow < pow_to; ++pow) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, pow, pow + 1, -static_cast<Type>(1), gen, v);
      test_prox_two_entropy_set_params(v, gen, d_p, d_alpha, p, alpha, eps);
      test_prox_two_entropy_check_feasible(p, alpha, tol * eps, v);
    }
  }

  for (int pow = pow_from; pow < pow_to; ++pow) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, pow, pow + 1, static_cast<Type>(1), gen, v);
      test_populate_real(100, pow, pow + 1, -static_cast<Type>(1), gen, v);
      test_prox_two_entropy_set_params(v, gen, d_p, d_alpha, p, alpha, eps);
      test_prox_two_entropy_check_feasible(p, alpha, tol * eps, v);
    }
  }

  for (int i = 0; i < 100; ++i) {
    for (int pow = pow_from; pow < pow_to; ++pow) {
      test_populate_real(25, pow, pow + 1, static_cast<Type>(1), gen, v);
      test_populate_real(25, pow, pow + 1, -static_cast<Type>(1), gen, v);
      test_prox_two_entropy_set_params(v, gen, d_p, d_alpha, p, alpha, eps);
      test_prox_two_entropy_check_feasible(p, alpha, tol * eps, v);
    }
  }
}

TEST(ProxTwoEntropyTest, test_prox_feasible_float) {
  test_prox_two_entropy_feasible<float>(-3, 3, 1);
}

TEST(ProxTwoEntropyTest, test_prox_feasible_double) {
  test_prox_two_entropy_feasible<double>(-6, 6, 1);
}

