#include "sdca/math/entropy.h"
#include "sdca/prox/topk_entropy_biased.h"
#include "test_util.h"

template <typename Type>
inline void test_prox_topk_entropy_biased_check_feasible(const ptrdiff_t k, const Type alpha,
                                                         const Type eps, std::vector<Type>& v) {
  sdca::prox_topk_entropy_biased(v.begin(), v.end(), k, alpha);

  Type sum = std::accumulate(v.begin(), v.end(), static_cast<Type>(0));
  Type lo(0), hi(sum / static_cast<Type>(k)), rhs(1);

  std::for_each(v.begin(), v.end(), [=](const Type x) { ASSERT_GE(x, lo); });
  std::for_each(v.begin(), v.end(), [=](const Type x) { ASSERT_LE(x, hi + eps); });
  ASSERT_LE(sum, rhs + eps);
}

template <typename Type>
inline void test_prox_topk_entropy_biased_set_params(const std::vector<Type>& v, std::mt19937& gen,
                                                     std::uniform_int_distribution<ptrdiff_t>& d_k,
                                                     std::uniform_real_distribution<Type>& d_alpha,
                                                     ptrdiff_t& k, Type& alpha, Type& eps) {
  k = d_k(gen);
  alpha = d_alpha(gen);
  Type max(*std::max_element(v.begin(), v.end()));
  eps = std::numeric_limits<Type>::epsilon() * std::max(static_cast<Type>(1), std::abs(max)) *
        static_cast<Type>(v.size());
}

template <typename Type>
inline void test_prox_topk_entropy_biased_feasible(const int pow_from, const int pow_to,
                                                   const Type tol) {
  std::mt19937 gen(1);
  std::uniform_int_distribution<ptrdiff_t> d_k(1, 10);
  std::uniform_real_distribution<Type> d_alpha(0, 2);

  Type alpha, eps;
  ptrdiff_t k;
  std::vector<Type> v;

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 25; ++i) {
      test_populate_real(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_prox_topk_entropy_biased_set_params(v, gen, d_k, d_alpha, k, alpha, eps);
      test_prox_topk_entropy_biased_check_feasible(k, alpha, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 25; ++i) {
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_entropy_biased_set_params(v, gen, d_k, d_alpha, k, alpha, eps);
      test_prox_topk_entropy_biased_check_feasible(k, alpha, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 25; ++i) {
      test_populate_real(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_entropy_biased_set_params(v, gen, d_k, d_alpha, k, alpha, eps);
      test_prox_topk_entropy_biased_check_feasible(k, alpha, tol * eps, v);
    }
  }

  for (int i = 0; i < 25; ++i) {
    for (int p = pow_from; p < pow_to; ++p) {
      test_populate_real(25, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(25, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_entropy_biased_set_params(v, gen, d_k, d_alpha, k, alpha, eps);
      test_prox_topk_entropy_biased_check_feasible(k, alpha, tol * eps, v);
    }
  }
}

TEST(ProxTopKEntropyBiasedTest, test_prox_feasible_float) {
  test_prox_topk_entropy_biased_feasible<float>(-3, 3, 1);
}

TEST(ProxTopKEntropyBiasedTest, test_prox_feasible_double) {
  test_prox_topk_entropy_biased_feasible<double>(-6, 6, 1);
}

// For k = 1 the problem is the softmax prox with a target score of 0, so a
// single threshold t = alpha * s - log(1 - s) must satisfy
//    a_i - log(x_i) - alpha * x_i = t
// for every output x_i (KKT conditions), including isolated inputs such as
// a = [-11] in float.
template <typename Type>
inline void test_prox_topk_entropy_biased_k1_kkt(std::vector<Type> v, const Type alpha,
                                                 const Type tol) {
  const std::vector<Type> a(v);
  sdca::prox_topk_entropy_biased(v.begin(), v.end(), 1, alpha);
  Type s = std::accumulate(v.begin(), v.end(), static_cast<Type>(0));
  ASSERT_GT(s, static_cast<Type>(0));
  ASSERT_LT(s, static_cast<Type>(1));
  Type t = alpha * s - std::log1p(-s);
  Type scale = std::max(static_cast<Type>(1), std::abs(t));
  // The reference t loses eps / (1 - s) through the rounded sum s.
  Type tol_s = tol * scale + 8 * std::numeric_limits<Type>::epsilon() / (1 - s);
  Type t0 = t;
  bool first = true;
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (v[i] <= 0) continue;  // underflow of exp(a_i - t) is acceptable
    Type ti = a[i] - std::log(v[i]) - alpha * v[i];
    ASSERT_NEAR(ti, t, tol_s) << "i = " << i << ", a_i = " << a[i] << ", alpha = " << alpha
                              << ", x_i = " << v[i];
    if (first) {
      t0 = ti;
      first = false;
    }
    ASSERT_NEAR(ti, t0, tol * scale)
        << "i = " << i << ", a_i = " << a[i] << ", alpha = " << alpha << ", x_i = " << v[i];
  }
}

template <typename Type>
inline void test_prox_topk_entropy_biased_k1_grid(const Type max_margin, const Type tol) {
  for (std::size_t n = 1; n <= 3; ++n) {
    for (Type m = 0; m <= max_margin; m += static_cast<Type>(0.05)) {
      for (Type alpha : {static_cast<Type>(0.1), static_cast<Type>(1), static_cast<Type>(10)}) {
        std::vector<Type> v(n, -m);
        test_prox_topk_entropy_biased_k1_kkt(v, alpha, tol);
      }
    }
  }
  std::mt19937 gen(7);
  std::uniform_real_distribution<Type> d_a(-max_margin, 10);
  std::uniform_real_distribution<Type> d_alpha(-2, 2);
  for (std::size_t n : {2, 5, 20}) {
    for (int trial = 0; trial < 200; ++trial) {
      std::vector<Type> v(n);
      for (auto& x : v) x = d_a(gen);
      Type alpha = std::pow(static_cast<Type>(10), d_alpha(gen));
      test_prox_topk_entropy_biased_k1_kkt(v, alpha, tol);
    }
  }
}

TEST(ProxTopKEntropyBiasedTest, test_prox_k1_kkt_float) {
  test_prox_topk_entropy_biased_k1_grid<float>(30, 1e-4f);
}

TEST(ProxTopKEntropyBiasedTest, test_prox_k1_kkt_double) {
  test_prox_topk_entropy_biased_k1_grid<double>(60, 1e-10);
}

// Solutions for k = 2 checked against an SLSQP solve of the constrained
// problem (agreement to 1e-7); the previous two-variable Newton iteration
// returned points with much larger objective values or x_i > s / k here.
TEST(ProxTopKEntropyBiasedTest, test_prox_k2_validated_cases) {
  struct Case {
    double alpha;
    std::vector<double> a, x;
  };
  const std::vector<Case> cases = {{1.9384585363011542,
                                    {-13.439767169427004, -3.3202137975867387, -19.472542888803879},
                                    {0.000228727220336, 0.00022927613994, 5.4891960387e-07}},
                                   {18.994396812867571,
                                    {1.6944260782712561, -14.320490886807182, -19.558849684874097},
                                    {0.00164915170677, 0.00164017009032, 8.98161645401e-06}},
                                   {0.3174582641691448,
                                    {9.1725520796705169, -20.065989743189114, 3.3323966051011098},
                                    {0.499226717043, 4.03029017833e-11, 0.499226717002}},
                                   {1.8000533198869306,
                                    {-19.372461326062716, -7.4698558076930794, -19.636434877590425},
                                    {1.11562559742e-06, 1.97241959617e-06, 8.5679399875e-07}}};
  for (const auto& c : cases) {
    std::vector<double> x(c.a);
    sdca::prox_topk_entropy_biased(x.begin(), x.end(), 2, c.alpha);
    for (std::size_t i = 0; i < x.size(); ++i) {
      ASSERT_NEAR(x[i], c.x[i], 1e-9 * std::abs(c.x[i])) << "alpha = " << c.alpha << ", i = " << i;
    }
    std::vector<float> xf(c.a.begin(), c.a.end());
    sdca::prox_topk_entropy_biased(xf.begin(), xf.end(), 2, static_cast<float>(c.alpha));
    for (std::size_t i = 0; i < xf.size(); ++i) {
      ASSERT_NEAR(static_cast<double>(xf[i]), c.x[i], 1e-5 * std::abs(c.x[i]))
          << "float, alpha = " << c.alpha << ", i = " << i;
    }
  }
}

// The objective l2_entropy_topk solves the same block with topk_entropy_block
// (sdca/math/entropy.h); the thresholds derived from it must reproduce it.
TEST(ProxTopKEntropyBiasedTest, test_prox_agrees_with_topk_entropy_block) {
  std::mt19937 gen(11);
  std::uniform_real_distribution<double> d_a(-30, 10);
  std::uniform_real_distribution<double> d_alpha(-2, 2);
  for (std::size_t n : {2, 3, 6, 15}) {
    for (int trial = 0; trial < 100; ++trial) {
      std::vector<double> v(n);
      for (auto& x : v) x = d_a(gen);
      double alpha = std::pow(10., d_alpha(gen));
      for (std::size_t k = 1; k < n; ++k) {
        std::vector<double> x(v), block;
        sdca::prox_topk_entropy_biased(x.begin(), x.end(), static_cast<std::ptrdiff_t>(k), alpha);
        sdca::topk_entropy_block(v, k, alpha, block);
        for (std::size_t i = 0; i < n; ++i) {
          ASSERT_NEAR(x[i], block[i], 1e-9 * std::max(1., std::abs(x[i])))
              << "n = " << n << ", k = " << k << ", i = " << i << ", alpha = " << alpha;
        }
      }
    }
  }
}
