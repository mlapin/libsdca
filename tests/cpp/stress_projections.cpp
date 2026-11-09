// Projections at extreme scales: tiny entries, overflow near 1e308 in the cone and the
// biased projections, and saturated bounds.
// The expected values are analytic (stationarity x_i + rho sum x = a_i, or
// feasibility); errors are relative to the size of the exact answer.
#include <cmath>
#include <limits>
#include <vector>
#include "sdca/prox.h"
#include "sdca/solver.h"
#include "test_util.h"

namespace {

template <typename T>
void expect_relative(const std::vector<T>& x, const std::vector<double>& expected, double tol,
                     const char* what) {
  double scale = 0;
  for (double v : expected) scale = std::max(scale, std::abs(v));
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_LE(std::abs(double(x[i]) - expected[i]), tol * scale) << what << " " << i;
}

}  // namespace

TEST(StressProjections, tiny_biased_knapsack_is_interior) {
  // Tiny entries: [1e-20, 1e-20], lo 0, hi 1, rhs 1, rho 1 -> a / 3 (sum bound inactive).
  // The current code returns the input (an absolute tolerance floor of 1).
  for (double s : {1e-20, 1e-30, 1e-10, 1e-3, 1.0}) {
    std::vector<double> x{s, s};
    sdca::prox_knapsack_le_biased(x.begin(), x.end(), 0., 1., 1., 1.);
    expect_relative(x, {s / 3, s / 3}, 16 * std::numeric_limits<double>::epsilon(),
                    "knapsack_le_biased");
  }
}

TEST(StressProjections, topk_hinge_steps_at_huge_norm2) {
  // Huge norm2 through the coordinate step: zero scores, 3 classes, C = 1, k = 1. The
  // block is [2, -1, -1] / (3 norm2) (smooth gamma = 1: 3 norm2 + 1, the same here).
  const double norm2s[] = {1e20, 1e30};
  for (double norm2 : norm2s) {
    const double s = 1 / (3 * norm2);
    std::vector<double> a(3, 0.), z(3, 0.);
    sdca::l2_topk_hinge<double, double>(1, 1).update_dual_variables(3, norm2, a.data(), z.data());
    expect_relative(a, {2 * s, -s, -s}, 1e-14, "topk_hinge float64");
    std::vector<double> b(3, 0.), y(3, 0.);
    sdca::l2_topk_hinge_smooth<double, double>(1, 1, 1).update_dual_variables(3, norm2, b.data(),
                                                                              y.data());
    expect_relative(b, {2 * s, -s, -s}, 1e-14, "topk_hinge_smooth float64");
    std::vector<float> c(3, 0.f), w(3, 0.f);
    sdca::l2_topk_hinge<float, double>(1, 1).update_dual_variables(3, float(norm2), c.data(),
                                                                   w.data());
    expect_relative(c, {2 * s, -s, -s}, 1e-6, "topk_hinge float32 storage / float64 compute");
  }
  // Float32 reaches the floor near norm2 = 1e8.
  std::vector<float> a(3, 0.f), z(3, 0.f);
  sdca::l2_topk_hinge<float, float>(1, 1).update_dual_variables(3, 1e8f, a.data(), z.data());
  const double s = 1 / (3 * double(1e8f));
  expect_relative(a, {2 * s, -s, -s}, 1e-6, "topk_hinge float32");
}

TEST(StressProjections, float32_topk_hinge_resolves_a_tiny_margin) {
  // The float32 topk_hinge case of the tiny-margin steps: [1, 2^-26], C = norm2 = gamma = 1,
  // s = 2^-26 / 2 (nonsmooth) and 2^-26 / 3 (smooth), to an ulp of the scores (absolute).
  const double h = std::ldexp(1., -26), eps = double(std::numeric_limits<float>::epsilon());
  float a[] = {0, 0}, z[] = {1, float(h)};
  sdca::l2_topk_hinge<float, float>(1, 1).update_dual_variables(2, 1.f, a, z);
  for (int i = 0; i < 2; ++i) EXPECT_NEAR(a[i], (i ? -h : h) / 2, 4 * eps) << i;
  float b[] = {0, 0}, y[] = {1, float(h)};
  sdca::l2_topk_hinge_smooth<float, float>(1, 1, 1).update_dual_variables(2, 1.f, b, y);
  for (int i = 0; i < 2; ++i) EXPECT_NEAR(b[i], (i ? -h : h) / 3, 4 * eps) << i;
}

TEST(StressProjections, signed_saturated_bounds_keep_the_free_mass) {
  // Saturated bounds: [-1e25, 0, 1e25], lo -1e20, hi 1e20, rhs -1 -> [-1e20, -1, 1e20] (t = 1).
  // The current code returns [-1e20, 0, 1e20]: the middle coordinate is off by 1.
  const std::vector<double> expected = {-1e20, -1, 1e20};
  std::vector<double> x{-1e25, 0, 1e25};
  sdca::prox_knapsack_eq(x.begin(), x.end(), -1e20, 1e20, -1.);
  EXPECT_EQ(expected, x);
  x = {-1e25, 0, 1e25};
  sdca::prox_knapsack_le(x.begin(), x.end(), -1e20, 1e20, -1.);
  EXPECT_EQ(expected, x);
  x = {-1e25, 0, 1e25};
  sdca::prox_knapsack_le_biased(x.begin(), x.end(), -1e20, 1e20, -1., 1.);
  EXPECT_EQ(expected, x);
}

TEST(StressProjections, huge_entries_do_not_overflow) {
  // Overflow: [1e308, 1e308], k = 1, rho = 1, hi = rhs = 1e308. The cone keeps the
  // feasible input; the biased projections are interior at a / 3.
  const double big = 1e308, third = big / 3, tol = 16 * std::numeric_limits<double>::epsilon();
  std::vector<double> x{big, big};
  sdca::prox_topk_cone(x.begin(), x.end(), 1);
  expect_relative(x, {big, big}, tol, "topk_cone");
  x = {big, big};
  sdca::prox_topk_cone_biased(x.begin(), x.end(), 1, 1.);
  expect_relative(x, {third, third}, tol, "topk_cone_biased");
  x = {big, big};
  sdca::prox_topk_simplex_biased(x.begin(), x.end(), 1, big, 1.);
  expect_relative(x, {third, third}, tol, "topk_simplex_biased");
  x = {big, big};
  sdca::prox_knapsack_le_biased(x.begin(), x.end(), 0., big, big, 1.);
  expect_relative(x, {third, third}, tol, "knapsack_le_biased");
}

TEST(StressProjections, tiny_two_simplex_balances_the_sums) {
  // a = s [2, -3], b = s [-1, -3], rhs = 3 s: <1, x> = <1, y> at l = 1.5 s,
  // x = [s / 2, 0], y = [s / 2, 0]. A tolerance floor of 1 kept every entry.
  for (double s : {1e-300, 1e-30, 1e-20, 1.0, 1e20}) {
    std::vector<double> x{2 * s, -3 * s, -s, -3 * s};
    sdca::prox_two_simplex(x.begin(), x.begin() + 2, x.end(), 3 * s);
    expect_relative(x, {s / 2, 0, s / 2, 0}, 4 * std::numeric_limits<double>::epsilon(),
                    "two_simplex double");
  }
  for (float s : {1e-30f, 1e-20f, 1.0f, 1e20f}) {
    std::vector<float> x{2 * s, -3 * s, -s, -3 * s};
    sdca::prox_two_simplex(x.begin(), x.begin() + 2, x.end(), 3 * s);
    expect_relative(x, {double(s / 2.f), 0, double(s / 2.f), 0},
                    double(4 * std::numeric_limits<float>::epsilon()), "two_simplex float");
  }
}

TEST(StressProjections, multilabel_hinge_steps_at_huge_norm2) {
  // One label of four, scores [0, 0, 0, -3], C = 1: the last class leaves the
  // support, and the block is [2, -1, -1, 0] / (3 norm2).
  for (double norm2 : {1e20, 1e30}) {
    const double s = 1 / (3 * norm2);
    std::vector<double> a(4, 0.), z{0, 0, 0, -3};
    sdca::l2_multilabel_hinge<double, double>(1).update_dual_variables(4, 1, norm2, a.data(),
                                                                       z.data());
    expect_relative(a, {2 * s, -s, -s, 0}, 1e-14, "multilabel_hinge float64");
    std::vector<float> b(4, 0.f), y{0, 0, 0, -3};
    sdca::l2_multilabel_hinge<float, float>(1).update_dual_variables(4, 1, float(norm2), b.data(),
                                                                     y.data());
    expect_relative(b, {2 * s, -s, -s, 0}, 1e-6, "multilabel_hinge float32");
  }
}
