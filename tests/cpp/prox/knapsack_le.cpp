#include "sdca/prox/knapsack_le.h"
#include "test_util.h"

template <typename Type>
inline void test_prox_knapsack_le_check_feasible(const Type lo, const Type hi, const Type rhs,
                                                 const Type eps, std::vector<Type>& v) {
  sdca::prox_knapsack_le(v.begin(), v.end(), lo, hi, rhs);

  std::for_each(v.begin(), v.end(), [=](const Type x) { ASSERT_GE(x, lo); });
  std::for_each(v.begin(), v.end(), [=](const Type x) { ASSERT_LE(x, hi); });
  Type sum = std::accumulate(v.begin(), v.end(), static_cast<Type>(0));
  ASSERT_LE(sum, rhs + eps);
}

template <typename Type>
inline void test_prox_knapsack_le_set_params(const std::vector<Type>& v, std::mt19937& gen,
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
inline void test_prox_knapsack_le_feasible(const int pow_from, const int pow_to, const Type tol) {
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
      test_prox_knapsack_le_set_params(v, gen, d_lo, d_hi, d_rhs, lo, hi, rhs, eps);
      test_prox_knapsack_le_check_feasible(lo, hi, rhs, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_knapsack_le_set_params(v, gen, d_lo, d_hi, d_rhs, lo, hi, rhs, eps);
      test_prox_knapsack_le_check_feasible(lo, hi, rhs, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_knapsack_le_set_params(v, gen, d_lo, d_hi, d_rhs, lo, hi, rhs, eps);
      test_prox_knapsack_le_check_feasible(lo, hi, rhs, tol * eps, v);
    }
  }

  for (int i = 0; i < 100; ++i) {
    for (int p = pow_from; p < pow_to; ++p) {
      test_populate_real(25, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(25, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_knapsack_le_set_params(v, gen, d_lo, d_hi, d_rhs, lo, hi, rhs, eps);
      test_prox_knapsack_le_check_feasible(lo, hi, rhs, tol * eps, v);
    }
  }
}

TEST(ProxKnapsackLETest, test_prox_feasible_float) {
  test_prox_knapsack_le_feasible<float>(-3, 3, 128);
}

TEST(ProxKnapsackLETest, test_prox_feasible_double) {
  // Sums accumulate in long double; in double they round up to 4 times more.
  test_prox_knapsack_le_feasible<double>(-6, 6, 128 * (wide_long_double() ? 1 : 4));
}

// Opposite saturated masses are combined before rhs: a bound of -1 binds and
// moves the free entry, a bound of 1 is inactive.
TEST(ProxKnapsackLETest, test_prox_signed_saturated_bounds) {
  for (const double rhs : {-1., 1.}) {
    std::vector<double> x{-1e25, 0, 1e25};
    sdca::prox_knapsack_le(x.begin(), x.end(), -1e20, 1e20, rhs);
    EXPECT_EQ(x, (std::vector<double>{-1e20, std::min(rhs, 0.), 1e20}));
  }
}

template <typename Type>
inline void test_prox_knapsack_le_kkt_sweep() {
  std::mt19937 gen(13);
  std::normal_distribution<Type> d_a(0, 1);
  std::uniform_real_distribution<Type> d_u(0, 1);
  const Type tol = 64 * std::numeric_limits<Type>::epsilon();
  for (int p = -30; p <= 30; p += 2) {
    const Type s = static_cast<Type>(std::pow(10.0L, p));
    for (const std::size_t n : {std::size_t(2), std::size_t(5), std::size_t(20)}) {
      for (int i = 0; i < 15; ++i) {
        std::vector<Type> a(n);
        for (auto& v : a) v = s * d_a(gen);
        const Type lo = -s * d_u(gen), hi = s * (d_u(gen) + Type(0.1));
        const Type rhs = static_cast<Type>(n) * (lo + (hi - lo) * d_u(gen));
        std::vector<Type> x = a;
        sdca::prox_knapsack_le(x.begin(), x.end(), lo, hi, rhs);
        EXPECT_LE(knapsack_kkt_residual(a, x, lo, hi, rhs, Type(0), false), tol)
            << "s = " << s << " n = " << n << " i = " << i;
      }
    }
  }
}

TEST(ProxKnapsackLETest, test_prox_kkt_sweep) {
  test_prox_knapsack_le_kkt_sweep<float>();
  test_prox_knapsack_le_kkt_sweep<double>();
}
