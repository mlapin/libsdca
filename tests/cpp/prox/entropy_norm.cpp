#include "sdca/prox/entropy_norm.h"
#include "test_util.h"

template <typename Type>
inline void test_prox_entropy_norm_check_feasible(const Type lo, const Type hi, const Type rhs,
                                                  const Type eps, std::vector<Type>& v) {
  sdca::prox_entropy_norm(v.begin(), v.end(), hi, rhs);

  std::for_each(v.begin(), v.end(), [=](const Type x) { ASSERT_GE(x, lo); });
  std::for_each(v.begin(), v.end(), [=](const Type x) { ASSERT_LE(x, hi); });
  Type sum = std::accumulate(v.begin(), v.end(), static_cast<Type>(0));
  ASSERT_NEAR(rhs, sum, eps);
}

template <typename Type>
inline void test_prox_entropy_norm_set_params(const std::vector<Type>& v, std::mt19937& gen,
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
inline void test_prox_entropy_norm_feasible(const int pow_from, const int pow_to, const Type tol) {
  std::mt19937 gen(1);
  std::uniform_real_distribution<Type> d_hi(0, 2);
  std::uniform_real_distribution<Type> d_rhs(0, 5);

  Type lo, hi, rhs, eps;
  std::vector<Type> v;

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 25; ++i) {
      test_populate_real(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_prox_entropy_norm_set_params(v, gen, d_hi, d_rhs, lo, hi, rhs, eps);
      test_prox_entropy_norm_check_feasible(lo, hi, rhs, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 25; ++i) {
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_entropy_norm_set_params(v, gen, d_hi, d_rhs, lo, hi, rhs, eps);
      test_prox_entropy_norm_check_feasible(lo, hi, rhs, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 25; ++i) {
      test_populate_real(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_entropy_norm_set_params(v, gen, d_hi, d_rhs, lo, hi, rhs, eps);
      test_prox_entropy_norm_check_feasible(lo, hi, rhs, tol * eps, v);
    }
  }

  for (int i = 0; i < 25; ++i) {
    for (int p = pow_from; p < pow_to; ++p) {
      test_populate_real(25, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(25, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_entropy_norm_set_params(v, gen, d_hi, d_rhs, lo, hi, rhs, eps);
      test_prox_entropy_norm_check_feasible(lo, hi, rhs, tol * eps, v);
    }
  }
}

TEST(ProxEntropyNormTest, test_prox_feasible_float) {
  test_prox_entropy_norm_feasible<float>(-3, 3, 1);
}

TEST(ProxEntropyNormTest, test_prox_feasible_double) {
  test_prox_entropy_norm_feasible<double>(-6, 6, 2);
}

template <typename Type>
inline void test_prox_entropy_norm_kkt_sweep() {
  std::mt19937 gen(23);
  std::normal_distribution<Type> d(0, 3);
  const Type tol = 64 * std::numeric_limits<Type>::epsilon();
  for (const std::size_t n : {std::size_t(3), std::size_t(10), std::size_t(30)}) {
    for (int i = 0; i < 20; ++i) {
      std::vector<Type> a(n);
      for (auto& x : a) x = d(gen);
      const Type hi = static_cast<Type>(0.5), rhs = static_cast<Type>(1);
      std::vector<Type> x = a;
      sdca::prox_entropy_norm(x.begin(), x.end(), hi, rhs);
      EXPECT_LE(capped_entropy_kkt_residual(a, x, hi, rhs, true), tol);
    }
  }
}

TEST(ProxEntropyNormTest, test_prox_kkt_sweep) {
  test_prox_entropy_norm_kkt_sweep<float>();
  test_prox_entropy_norm_kkt_sweep<double>();
}

// x with x + log x = v (Wright omega), by Newton in long double.
inline long double test_omega(const long double v) {
  long double x = v < 1 ? std::exp(v) : v - std::log(v);
  for (int i = 0; i < 100; ++i) {
    long double next = x - (x + std::log(x) - v) / (1 + 1 / x);
    if (!(next > 0)) next = x / 2;
    if (next == x) break;
    x = next;
  }
  return x;
}

// The long double solution: bisection on the uncapped mass less rhs - capped * hi;
// the largest root keeps an entry that rounds to hi uncapped.
inline std::vector<long double> test_entropy_norm_reference(const std::vector<double>& a,
                                                            const long double hi,
                                                            const long double rhs) {
  const auto range = std::minmax_element(a.begin(), a.end());
  long double lo = static_cast<long double>(*range.first) - 1e5L -
                   std::fabs(static_cast<long double>(*range.first)),
              up = static_cast<long double>(*range.second) + 1e5L +
                   std::fabs(static_cast<long double>(*range.second));
  auto excess = [&](long double tau) {
    long double sum = 0, capped = 0;
    for (const double x : a) {
      const long double w = test_omega(static_cast<long double>(x) - tau);
      if (w >= hi) {
        capped += 1;
      } else {
        sum += w;
      }
    }
    return sum - (rhs - capped * hi);
  };
  for (;;) {
    const long double mid = lo + (up - lo) / 2;
    if (mid == lo || mid == up) break;
    (excess(mid) >= 0 ? lo : up) = mid;
  }
  std::vector<long double> x;
  for (const double v : a) x.push_back(std::min(hi, test_omega(static_cast<long double>(v) - lo)));
  return x;
}

// Entries far below the sum tolerance keep their relative accuracy, also when
// one entry rounds to hi = rhs or the free mass rhs - k * hi is a rounding error.
TEST(ProxEntropyNormTest, test_prox_tiny_entries_against_reference) {
  if (std::numeric_limits<long double>::digits < 64) return;
  std::mt19937_64 gen(7);
  const double eps = std::numeric_limits<double>::epsilon();
  for (int id = 0; id < 150; ++id) {
    const std::size_t n = std::uniform_int_distribution<std::size_t>(2, 40)(gen);
    const double scale = id % 3 == 0 ? 100 : 1e4, offset = id % 2 == 0 ? 0 : -700;
    std::vector<double> a(n);
    for (auto& x : a) x = offset + scale * std::uniform_real_distribution<double>(-1, 1)(gen);
    const double hi_rhs[][2] = {{1, 1}, {1e-20, 1e-20}, {0.5, 1}, {1.0 / double(n), 0.5}};
    const double hi = hi_rhs[id % 4][0], rhs = hi_rhs[id % 4][1];
    std::vector<double> x = a;
    sdca::prox_entropy_norm(x.begin(), x.end(), hi, rhs);
    const std::vector<long double> reference =
        test_entropy_norm_reference(a, static_cast<long double>(hi), static_cast<long double>(rhs));
    for (std::size_t i = 0; i < n; ++i) {
      if (reference[i] < 1e-300L) continue;
      const long double error =
          std::fabs(static_cast<long double>(x[i]) - reference[i]) / reference[i];
      EXPECT_LE(error, static_cast<long double>(64 * eps) * (1 + std::fabs(std::log(reference[i]))))
          << "case " << id << " entry " << i << " hi " << hi << " rhs " << rhs;
    }
  }
}

// Scores far apart leave Newton without slope: the search must still reach a
// partially filled entry instead of stopping on the iteration limit.
TEST(ProxEntropyNormTest, test_prox_huge_scores_keep_the_sum) {
  std::mt19937_64 gen(11);
  for (int id = 0; id < 50; ++id) {
    const std::size_t n = 2 * std::uniform_int_distribution<std::size_t>(1, 30)(gen) + 1;
    std::vector<double> x(n);
    for (auto& v : x) v = 1e300 * std::uniform_real_distribution<double>(-1, 1)(gen);
    const double hi = 1.0 / double(n), rhs = 0.5;
    sdca::prox_entropy_norm(x.begin(), x.end(), hi, rhs);
    const double sum = std::accumulate(x.begin(), x.end(), 0.0);
    EXPECT_NEAR(sum, rhs, 16 * std::numeric_limits<double>::epsilon() * rhs) << "case " << id;
  }
}
