// Precision corner cases against independent references: KKT equations or
// closed forms.
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include "sdca/solver.h"
#include "sdca/loss.h"
#include "sdca/utility/logging.h"
#include "test_util.h"

#ifndef __FAST_MATH__
namespace {

typedef long double wide_float;

// A cold or warm 4-class step of entropy (k = 1) or top-k entropy (k = 2).
template <typename Data, typename Result>
std::vector<Data> entropy_step(std::size_t k, double c, std::vector<Data> a, std::vector<Data> z) {
  if (k == 1)
    sdca::l2_entropy<Data, Result>(Result(c)).update_dual_variables(4, Data(1), a.data(), z.data());
  else
    sdca::l2_entropy_topk<Data, Result>(Result(c), k)
        .update_dual_variables(4, Data(1), a.data(), z.data());
  return a;
}

// Largest stationarity residual of an entropy block at norm2 = 1, in units of
// eps (1 + d_j + S): log d_j - log(C - S) + d_j + S = r_j - r_0, d_j = -a_j,
// S = sum d, r = z - old exactly. Also checks sum a = 0 to the same scale.
template <typename Data>
double entropy_kkt(const std::vector<Data>& a, const std::vector<wide_float>& r, double c) {
  const wide_float eps = wide_float(std::numeric_limits<Data>::epsilon());
  wide_float s = 0;
  for (std::size_t j = 1; j < a.size(); ++j) s -= wide_float(a[j]);
  double worst = double(std::abs(wide_float(a[0]) - s) / (eps * (1 + s)));
  for (std::size_t j = 1; j < a.size(); ++j) {
    const wide_float d = -wide_float(a[j]);
    if (!(d > 0)) return std::numeric_limits<double>::infinity();
    const wide_float residual = std::log(d) - std::log(wide_float(c) - s) + d + s - (r[j] - r[0]);
    worst = std::max(worst, double(std::abs(residual) / (eps * (1 + d + s))));
  }
  return worst;
}

// Whether the top-k (k = 2) caps -a_j <= a_0 / 2 are inactive, so the entropy
// equation also holds for top-k entropy.
template <typename Data>
bool caps_inactive(const std::vector<Data>& a) {
  for (std::size_t j = 1; j < a.size(); ++j)
    if (-wide_float(a[j]) >= wide_float(a[0]) / 2 * (1 - 1e-3L)) return false;
  return true;
}

template <typename Data, typename Result>
void warm_cancellation_kkt(const char* precision) {
  // Warm cancellation: old [s, -s/2, -s/2, 0], scores old + [0, 0, 0, 2], C = s; the hardest
  // case is s = 2^80.
  const std::vector<wide_float> r = {0, 0, 0, 2};
  for (int power = 0; power <= 100; power += 2) {
    const Data s = Data(std::ldexp(1., power)), h = s / 2;
    for (std::size_t k : {1, 2}) {
      const auto a =
          entropy_step<Data, Result>(k, double(s), {s, -h, -h, Data(0)}, {s, -h, -h, Data(2)});
      if (k == 2 && !caps_inactive(a)) continue;
      // Top-k blocks at C ~ 2^60 (float) or 2^100 (double) reach ~90 (accuracy, not warm
      // cancellation).
      EXPECT_LE(entropy_kkt(a, r, double(s)), k == 1 ? 64 : 256)
          << precision << " k=" << k << " s=2^" << power << " a=" << a[0] << "," << a[3];
    }
  }
}

template <typename Data, typename Result>
void warm_common_offset_kkt(const char* precision) {
  // Warm cancellation, second family: small duals beside a huge common score keep
  // r_j - r_0 = old_0 - old_j.
  const std::vector<Data> old = {Data(.875), Data(-.375), Data(-.25), Data(-.25)};
  std::vector<wide_float> r(4);
  for (std::size_t j = 0; j < 4; ++j) r[j] = -wide_float(old[j]);
  for (int power = 0; power <= 120; power += 4) {
    const Data offset = Data(std::ldexp(1., power));
    for (std::size_t k : {1, 2}) {
      const auto a = entropy_step<Data, Result>(k, 1, old, std::vector<Data>(4, offset));
      ASSERT_TRUE(caps_inactive(a)) << precision;
      EXPECT_LE(entropy_kkt(a, r, 1), 16) << precision << " k=" << k << " offset=2^" << power;
    }
  }
}

}  // namespace

TEST(Stress, warm_entropy_cancellation_meets_kkt) {
  warm_cancellation_kkt<double, double>("float64");
  warm_cancellation_kkt<float, double>("float32 storage / float64 compute");
  warm_cancellation_kkt<float, float>("float32");
}

TEST(Stress, warm_entropy_common_offset_meets_kkt) {
  warm_common_offset_kkt<double, double>("float64");
  warm_common_offset_kkt<float, double>("float32 storage / float64 compute");
  warm_common_offset_kkt<float, float>("float32");
}

namespace {

// Cold two-class steps at [1, 2^-26], C = norm2 = gamma = 1. The block is
// [s, -s], s = h / (2 + w), h = 2^-26; w = 0 nonsmooth, 1 smooth top-k, 2
// multilabel. The margin is resolved to about an ulp of the scores (absolute).
template <typename Data, typename Result>
void hinge_tiny_margin_steps(const char* precision) {
  const Data z0 = 1, z1 = Data(std::ldexp(1., -26));
  const double h = std::ldexp(1., -26), eps = double(std::numeric_limits<Data>::epsilon());
  auto check = [&](const char* name, const double width, const Data* a) {
    const double s = h / (2 + width), tol = 4 * eps * (s + 2 / (2 + width));
    EXPECT_NEAR(double(a[0]), s, tol) << precision << " " << name;
    EXPECT_NEAR(double(a[1]), -s, tol) << precision << " " << name;
  };
  Data a[2], z[2];
  auto reset = [&]() {
    a[0] = a[1] = 0;
    z[0] = z0;
    z[1] = z1;
  };
  reset();
  sdca::l2_hinge_topk<Data, Result>(1, 1).update_dual_variables(2, Data(1), a, z);
  check("topk_hinge_alpha", 0, a);
  reset();
  sdca::l2_hinge_topk_smooth<Data, Result>(1, 1, 1).update_dual_variables(2, Data(1), a, z);
  check("topk_hinge_alpha", 1, a);
  reset();
  sdca::l2_multilabel_hinge<Data, Result>(1).update_dual_variables(2, 1, Data(1), a, z);
  check("multilabel_hinge", 0, a);
  reset();
  sdca::l2_multilabel_hinge_smooth<Data, Result>(1, 1).update_dual_variables(2, 1, Data(1), a, z);
  check("multilabel_hinge", 2, a);
  reset();
  sdca::l2_topk_hinge<Data, Result>(1, 1).update_dual_variables(2, Data(1), a, z);
  check("topk_hinge_beta", 0, a);
  reset();
  sdca::l2_topk_hinge_smooth<Data, Result>(1, 1, 1).update_dual_variables(2, Data(1), a, z);
  check("topk_hinge_beta", 1, a);
}

}  // namespace

TEST(Stress, hinge_steps_resolve_a_tiny_margin) {
  hinge_tiny_margin_steps<float, float>("float32");
  hinge_tiny_margin_steps<float, double>("float32 storage / float64 compute");
  hinge_tiny_margin_steps<double, double>("float64");
}

namespace {

enum class hinge { topk, topk_smooth, knapsack, knapsack_smooth, multilabel, multilabel_smooth };
const hinge all_hinges[] = {hinge::topk,       hinge::topk_smooth,
                            hinge::knapsack,   hinge::knapsack_smooth,
                            hinge::multilabel, hinge::multilabel_smooth};
const char* const hinge_names[] = {"topk_hinge_alpha", "topk_hinge_alpha", "topk_hinge_beta",
                                   "topk_hinge_beta",  "multilabel_hinge", "multilabel_hinge"};

// One hinge step at norm2 = 1, k = 1 (label 0) or p = 1, gamma = C: rho = 1/2 smooth.
template <typename Data, typename Result>
std::vector<Data> hinge_step(const hinge h, const double c, std::vector<Data> a,
                             std::vector<Data> z) {
  const std::size_t n = a.size();
  const Result C = Result(c);
  switch (h) {
    case hinge::topk:
      sdca::l2_hinge_topk<Data, Result>(C, 1).update_dual_variables(n, Data(1), a.data(), z.data());
      break;
    case hinge::topk_smooth:
      sdca::l2_hinge_topk_smooth<Data, Result>(C, 1, C).update_dual_variables(n, Data(1), a.data(),
                                                                              z.data());
      break;
    case hinge::knapsack:
      sdca::l2_topk_hinge<Data, Result>(C, 1).update_dual_variables(n, Data(1), a.data(), z.data());
      break;
    case hinge::knapsack_smooth:
      sdca::l2_topk_hinge_smooth<Data, Result>(C, 1, C).update_dual_variables(n, Data(1), a.data(),
                                                                              z.data());
      break;
    case hinge::multilabel:
      sdca::l2_multilabel_hinge<Data, Result>(C).update_dual_variables(n, std::size_t(1), Data(1),
                                                                       a.data(), z.data());
      break;
    case hinge::multilabel_smooth:
      sdca::l2_multilabel_hinge_smooth<Data, Result>(C, C).update_dual_variables(
          n, std::size_t(1), Data(1), a.data(), z.data());
      break;
  }
  return a;
}

// Analytic step at residual margins m_j = 1 + r_j - r_0 with active competitors
// 'active' (C inactive): top-k S = rho sum m / (1 + |A| rho), x_j = rho (m_j - S);
// multilabel S = a sum m / (1 + |A|), x_j = a m_j - S; rho = a = 1 or 1/2 smooth.
std::vector<wide_float> hinge_reference(const hinge h, const std::vector<wide_float>& m,
                                        const std::vector<std::size_t>& active) {
  const bool smooth =
      h == hinge::topk_smooth || h == hinge::knapsack_smooth || h == hinge::multilabel_smooth;
  const bool multilabel = h == hinge::multilabel || h == hinge::multilabel_smooth;
  const wide_float f = smooth ? wide_float(.5) : wide_float(1);
  wide_float sum = 0;
  for (std::size_t j : active) sum += m[j];
  const wide_float s = multilabel ? f * sum / wide_float(1 + active.size())
                                  : f * sum / (1 + wide_float(active.size()) * f);
  std::vector<wide_float> block(m.size(), 0);
  block[0] = s;
  for (std::size_t j : active) block[j] = -(multilabel ? f * m[j] - s : f * (m[j] - s));
  return block;
}

// Largest deviation from the reference, in eps(Data) of its mass block[0].
template <typename Data>
double hinge_error(const std::vector<Data>& a, const std::vector<wide_float>& block) {
  double worst = 0;
  for (std::size_t j = 0; j < a.size(); ++j)
    worst = std::max(worst, double(std::abs(wide_float(a[j]) - block[j]) /
                                   (block[0] * wide_float(std::numeric_limits<Data>::epsilon()))));
  return worst;
}

// Warm cancellation in hinge steps: old duals s 2^p beside scores old + r cancel to the
// residual r; the warm step equals the cold one at r. Two classes: old
// [s, -s], r = 0; four: old [s, -t, -u, 0], r = [0, 0, 0, 3]; 100: old
// [s, -s/98 ..., 0], r = 3 at the last class, the only active one.
template <typename Data, typename Result>
void hinge_warm_residual(const char* precision) {
  std::vector<wide_float> m100(100, 1);
  m100[0] = 0;
  m100[99] = 4;
  for (int power = 0; power <= 100; ++power) {
    const Data s = Data(std::ldexp(1., power)), t = Data(.3 * double(s));
    const double c = 4 * std::ldexp(1., power) + 8;
    std::vector<Data> old100(100, Data(-double(s) / 98)), z100;
    old100[0] = s;
    old100[99] = 0;
    z100 = old100;
    z100[99] = 3;
    for (std::size_t h = 0; h < 6; ++h) {
      const hinge k = all_hinges[h];
      const auto two = hinge_reference(k, {0, 1}, {1}),
                 four = hinge_reference(k, {0, 1, 1, 4}, {3});
      const auto many = hinge_reference(k, m100, {99});
      EXPECT_LE(hinge_error(hinge_step<Data, Result>(k, c, {0, 0}, {0, 0}), two), 4) << precision;
      EXPECT_LE(hinge_error(hinge_step<Data, Result>(k, c, {0, 0, 0, 0}, {0, 0, 0, 3}), four), 4)
          << precision;
      const auto warm2 = hinge_step<Data, Result>(k, c, {s, -s}, {s, -s});
      EXPECT_LE(hinge_error(warm2, two), 4)
          << precision << " " << hinge_names[h] << " s=2^" << power << " a=" << warm2[0];
      for (const Data x : {s, t}) {
        const Data y = s - x;
        const auto warm4 = hinge_step<Data, Result>(k, c, {s, -x, -y, 0}, {s, -x, -y, 3});
        EXPECT_LE(hinge_error(warm4, four), 4) << precision << " " << hinge_names[h] << " s=2^"
                                               << power << " t=" << x << " a=" << warm4[0];
      }
      const auto warm100 = hinge_step<Data, Result>(k, c, old100, z100);
      EXPECT_LE(hinge_error(warm100, many), 4)
          << precision << " " << hinge_names[h] << " 100 classes s=2^" << power;
    }
  }
}

template <typename Data, typename Result>
void hinge_warm_common_offset(const char* precision) {
  // Small duals beside a huge common score: r_j - r_0 = old_0 - old_j, all active.
  const std::vector<Data> old = {Data(.875), Data(-.375), Data(-.25), Data(-.25)};
  const std::vector<wide_float> m = {0, 2.25L, 2.125L, 2.125L};
  for (int power = 0; power <= 120; power += 4) {
    const Data offset = Data(std::ldexp(1., power));
    for (std::size_t h = 0; h < 6; ++h) {
      const auto a = hinge_step<Data, Result>(all_hinges[h], 4, old, std::vector<Data>(4, offset));
      EXPECT_LE(hinge_error(a, hinge_reference(all_hinges[h], m, {1, 2, 3})), 4)
          << precision << " " << hinge_names[h] << " offset=2^" << power;
    }
  }
}

}  // namespace

TEST(Stress, hinge_warm_step_keeps_the_residual) {
  hinge_warm_residual<double, double>("float64");
  hinge_warm_residual<float, double>("float32 storage / float64 compute");
  hinge_warm_residual<float, float>("float32");
}

TEST(Stress, hinge_warm_step_keeps_a_common_offset) {
  hinge_warm_common_offset<double, double>("float64");
  hinge_warm_common_offset<float, double>("float32 storage / float64 compute");
  hinge_warm_common_offset<float, float>("float32");
}

TEST(Stress, A1_multilabel_entropy_keeps_the_underflowed_other_mass) {
  // z = [400, 400, -400], labels {0, 1}, C = 1e200, norm2 = 1e-200.
  // C norm2 a is negligible, so -a_2 = C e^-800 / (2 + e^-800), a_0 = a_1 = -a_2 / 2.
  double z[] = {400, 400, -400}, a[] = {0, 0, 0};
  sdca::l2_multilabel_entropy<double, double>(1e200).update_dual_variables(3, 2, 1e-200, a, z);
  const double other = std::exp(std::log(1e200) - 800 - std::log(2.0));
  EXPECT_NEAR(-a[2] / other, 1, 1e-12);
  EXPECT_NEAR(a[0] / (other / 2), 1, 1e-12);
  EXPECT_NEAR(a[1] / (other / 2), 1, 1e-12);
}

TEST(Stress, A6_logging_names_out_of_range_values) {
  // to_string fell off the end of its switch.
  EXPECT_EQ("unknown", sdca::logging::to_string(static_cast<sdca::logging::level>(99)));
  EXPECT_EQ("unknown", sdca::logging::to_string(static_cast<sdca::logging::format>(99)));
}

TEST(Stress, topk_entropy_primal_loss_matches_loss_and_gradient_under_far_cap) {
  // Free scores far below a capped competitor keep their differences in primal_loss.
  double z[] = {0.0, 1.0, 1e16, -0.3, 0.2, 1e16 - 3.0};
  double g[6];
  const double expected =
      sdca::loss_and_gradient(sdca::loss_kind::topk_cross_entropy, 6, 1, z, g, 3, 1.0);
  const double actual = sdca::l2_entropy_topk<double, double>(1.0, 3).primal_loss(6, z);
  EXPECT_DOUBLE_EQ(expected, actual);
}

// Known limits: these pin the documented bounds; they are not correctness claims.

TEST(StressLimit, float32_topk_entropy_at_huge_c_with_zero_scores) {
  // LIMIT: float32 top-k entropy loses accuracy at C ~ 1e35 even at zero scores
  // (measured: 1309 block ulps; float32 storage with float64 compute is within 2, regression.cpp).
  // Oracle a0: log(a) - log(C - a) + (1 + 1/999) a = log(999), 100 digits.
  const std::size_t n = 1000;
  const double a0 = 82.99536980534230771388;
  std::vector<float> a(n, 0.f), z(n, 0.f);
  sdca::l2_entropy_topk<float, float>(float(1e35), 2)
      .update_dual_variables(n, 1.f, a.data(), z.data());
  const double ulp = double(std::numeric_limits<float>::epsilon()) * a0;
  EXPECT_LE(std::abs(double(a[0]) - a0), 2048 * ulp);
  wide_float sum = 0;
  for (float v : a) sum += wide_float(v);
  EXPECT_LE(double(std::abs(sum)), 64 * n * ulp);
}

TEST(StressLimit, topk_entropy_error_grows_with_the_score_spread) {
  // LIMIT: capped log-odds add score differences to the largest competitor, so the error of
  // the loss, the training primal and the projection grows with the spread S. Exact for every
  // S: loss log1p(2 sqrt 2) of [0, S, -S, -S] and projection x0 = r / (1 + r) / 2, r = 2 sqrt 2,
  // x1 = x2 = x0 / 2 of [S, -S, -S], k = 2. The bounds are the documented errors; at DBL_MAX
  // both are 0.
  const double r = 2 * std::sqrt(2.0), loss = std::log1p(r), cap = r / (1 + r) / 2;
  const double spreads[] = {1e4, 1e8, 1e12, 1e16, std::numeric_limits<double>::max()};
  const double bounds[][2] = {{1e-12, 1e-12}, {1e-9, 1e-9}, {2e-5, 1e-5}, {0.2, 0.1}, {1, 1}};
  for (std::size_t i = 0; i < 5; ++i) {
    const double s = spreads[i];
    double z[] = {0, s, -s, -s}, g[4], scores[] = {0, s, -s, -s};
    const double value =
        sdca::loss_and_gradient(sdca::loss_kind::topk_cross_entropy, 4, 1, z, g, 2, 1.0);
    const sdca::l2_entropy_topk<double, double> objective(1.0, 2);
    EXPECT_LE(std::abs(value - loss), bounds[i][0] * loss) << s;
    EXPECT_EQ(value, objective.primal_loss(4, scores)) << s;
    std::vector<double> x = {s, -s, -s};
    sdca::prox_topk_entropy(x.begin(), x.end(), 2);
    EXPECT_LE(std::abs(x[0] - cap), bounds[i][1] * cap) << s;
    EXPECT_LE(std::abs(x[1] - cap / 2), bounds[i][1] * cap / 2) << s;
    EXPECT_EQ(x[1], x[2]) << s;
  }
}

TEST(StressLimit, all_capped_topk_entropy_mean_beyond_double_double) {
  // LIMIT: with k = C - 1 the loss is the mean of the score differences, summed in
  // double-double: differences at 2^200, 2^100 and 1 lose the 1 (exact mean 1/4, loss
  // log1p(4 exp(1/4)) = 1.8142); the documented result is log 5.
  const double h = std::ldexp(1.0, 200), m = std::ldexp(1.0, 100);
  double z[] = {m / 4, h, m, 1, -h}, g[5];
  const double value =
      sdca::loss_and_gradient(sdca::loss_kind::topk_cross_entropy, 5, 1, z, g, 4, 1.0);
  const double exact = std::log1p(4 * std::exp(0.25));
  EXPECT_LE(std::abs(value - exact), exact - std::log(5.0) + 1e-15);
}

TEST(StressLimit, hinge_margins_relative_to_a_far_label) {
  // LIMIT: margins are formed relative to the label score, so other scores closer
  // than the resolution of a huge gap tie; the value stays accurate to its own size.
  double z[] = {-1e20, 0, 1e-3}, g[3];
  const double value = sdca::loss_and_gradient(sdca::loss_kind::topk_hinge_alpha, 3, 1, z, g, 1, 0);
  EXPECT_NEAR(value / (1e20 + 1 + 1e-3), 1, 2 * std::numeric_limits<double>::epsilon());
  EXPECT_EQ(-1, g[0]);
  EXPECT_EQ(1, g[1] + g[2]);
}
#endif
