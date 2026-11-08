#include "sdca/prox/entropy.h"
#include "test_util.h"

template <typename Type>
inline void test_prox_entropy_check_feasible(const Type lo, const Type hi, const Type rhs,
                                             const Type eps, std::vector<Type>& v) {
  sdca::prox_entropy(v.begin(), v.end(), hi, rhs);

  std::for_each(v.begin(), v.end(), [=](const Type x) { ASSERT_GE(x, lo); });
  std::for_each(v.begin(), v.end(), [=](const Type x) { ASSERT_LE(x, hi); });
  Type sum = std::accumulate(v.begin(), v.end(), static_cast<Type>(0));
  ASSERT_NEAR(rhs, sum, eps);
}

template <typename Type>
inline void test_prox_entropy_set_params(const std::vector<Type>& v, std::mt19937& gen,
                                         std::uniform_real_distribution<Type>& d_hi,
                                         std::uniform_real_distribution<Type>& d_rhs, Type& lo,
                                         Type& hi, Type& rhs, Type& eps) {
  Type size = static_cast<Type>(v.size());
  for (;;) {
    lo = 0;
    hi = d_hi(gen);
    rhs = d_rhs(gen);
    if (hi * size >= rhs) break;
  }
  Type max(*std::max_element(v.begin(), v.end()));
  eps = std::numeric_limits<Type>::epsilon() * std::max(static_cast<Type>(1), std::abs(max)) *
        static_cast<Type>(v.size());
}

template <typename Type>
inline void test_prox_entropy_feasible(const int pow_from, const int pow_to, const Type tol) {
  std::mt19937 gen(1);
  std::uniform_real_distribution<Type> d_hi(0, 2);
  std::uniform_real_distribution<Type> d_rhs(0, 5);

  Type lo, hi, rhs, eps;
  std::vector<Type> v;

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 50; ++i) {
      test_populate_real(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_prox_entropy_set_params(v, gen, d_hi, d_rhs, lo, hi, rhs, eps);
      test_prox_entropy_check_feasible(lo, hi, rhs, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 50; ++i) {
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_entropy_set_params(v, gen, d_hi, d_rhs, lo, hi, rhs, eps);
      test_prox_entropy_check_feasible(lo, hi, rhs, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 50; ++i) {
      test_populate_real(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_entropy_set_params(v, gen, d_hi, d_rhs, lo, hi, rhs, eps);
      test_prox_entropy_check_feasible(lo, hi, rhs, tol * eps, v);
    }
  }

  for (int i = 0; i < 50; ++i) {
    for (int p = pow_from; p < pow_to; ++p) {
      test_populate_real(25, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(25, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_entropy_set_params(v, gen, d_hi, d_rhs, lo, hi, rhs, eps);
      test_prox_entropy_check_feasible(lo, hi, rhs, tol * eps, v);
    }
  }
}

TEST(ProxEntropyTest, test_prox_feasible_float) { test_prox_entropy_feasible<float>(-3, 3, 1); }

TEST(ProxEntropyTest, test_prox_feasible_double) { test_prox_entropy_feasible<double>(-6, 6, 4); }

template <typename Type>
inline void test_prox_entropy_kkt_sweep() {
  std::mt19937 gen(19);
  std::normal_distribution<Type> d(0, 3);
  const Type tol = 8 * std::numeric_limits<Type>::epsilon();
  for (const std::size_t n : {std::size_t(3), std::size_t(10), std::size_t(30)}) {
    for (int i = 0; i < 20; ++i) {
      std::vector<Type> a(n);
      for (auto& x : a) x = d(gen);
      const Type hi = static_cast<Type>(0.5), rhs = static_cast<Type>(1);
      std::vector<Type> x = a;
      sdca::prox_entropy(x.begin(), x.end(), hi, rhs);
      EXPECT_LE(capped_entropy_kkt_residual(a, x, hi, rhs, false), tol);
    }
  }
}

TEST(ProxEntropyTest, test_prox_kkt_sweep) {
  test_prox_entropy_kkt_sweep<float>();
  test_prox_entropy_kkt_sweep<double>();
}
