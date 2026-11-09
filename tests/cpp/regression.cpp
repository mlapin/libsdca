#include <algorithm>
#include <deque>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "sdca/solver.h"
#include "sdca/loss.h"
#include "sdca/prox.h"
#include "test_util.h"

// The losses by index 0-8: each hinge nonsmooth (gamma = 0), then smoothed.
static std::pair<sdca::loss_kind, bool> loss_case(int i) {
  using kind = sdca::loss_kind;
  const std::pair<kind, bool> cases[] = {
      {kind::cross_entropy, false},         {kind::topk_cross_entropy, false},
      {kind::topk_hinge_alpha, false},      {kind::topk_hinge_beta, false},
      {kind::topk_hinge_alpha, true},       {kind::topk_hinge_beta, true},
      {kind::multilabel_hinge, false},      {kind::multilabel_hinge, true},
      {kind::proximal_cross_entropy, false}};
  return cases[i];
}

// Tests guarded by __FAST_MATH__ rely on IEEE semantics (nonfinite inputs,
// exact cancellation, fixed evaluation order) that -Ofast does not keep.

#ifndef __FAST_MATH__
TEST(NumericalRegression, uniform_entropy_keeps_representable_extreme_mean_gaps) {
  const double largest = std::numeric_limits<double>::max();
  const std::vector<std::vector<double>> scores = {
      {0, -largest, -largest}, {largest, largest, -largest}, {largest, -largest}};
  const std::vector<std::vector<double>> expected_gradients = {
      {2.0 / 3, -1.0 / 3, -1.0 / 3}, {1.0 / 6, 1.0 / 6, -1.0 / 3}, {0.5, -0.5}};
  for (std::size_t row = 0; row < scores.size(); ++row) {
    const auto& z = scores[row];
    std::vector<double> gradient(z.size());
    const double expected = row < 2 ? (largest / 3) * 2 : largest;
    const double loss = sdca::loss_and_gradient<double>(sdca::loss_kind::cross_entropy, z.size(),
                                                        z.size(), z.data(), gradient.data());
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_NEAR(loss / expected, 1, 4 * std::numeric_limits<double>::epsilon());
    const double value = sdca::entropy_loss<double>(z.size(), z.size(), z.data());
    EXPECT_TRUE(std::isfinite(value));
    EXPECT_NEAR(value / expected, 1, 4 * std::numeric_limits<double>::epsilon());
    for (std::size_t j = 0; j < z.size(); ++j)
      EXPECT_NEAR(gradient[j], expected_gradients[row][j], 4e-16);
  }
}
#endif

#ifndef __FAST_MATH__
TEST(NumericalRegression, topk_entropy_blocks_keep_free_scores_below_a_far_capped_one) {
  // The coordinate step and the biased prox took the scores relative to the largest one,
  // which kept their distance from it to 53 bits: from a gap of about 1e4 on the free
  // scores lost their differences (by up to 0.17 at 1e16). Beyond 1e4 the results are
  // those at 1e4, where the far scores' exponentials vanish.
  auto row = [](int shape, double gap, double* z) {
    const double rows[3][6] = {{-gap, 1., gap, -.3, .2, -1.5},
                               {0., 1., gap, -.3, .2, gap - 3.},
                               {0., gap, gap, 1., -.3, .2}};
    std::copy(rows[shape], rows[shape] + 6, z);
  };
  for (std::size_t k : {2, 3, 4}) {
    for (double c : {1e-3, 1., 1e3}) {
      const sdca::l2_entropy_topk<double, double> objective(c, k);
      for (int shape = 0; shape < 3; ++shape) {
        for (bool warm : {false, true}) {
          // A feasible previous block, or a cold one.
          const double start[6] = {warm ? .4 * c : 0.,  warm ? -.1 * c : 0., warm ? -.1 * c : 0.,
                                   warm ? -.1 * c : 0., warm ? -.1 * c : 0., 0.};
          double z[6], near[6];
          row(shape, 1e4, z);
          std::copy(start, start + 6, near);
          objective.update_dual_variables(6, 1., near, z);
          for (double gap : {1e8, 1e16, 1e30}) {
            double far[6];
            row(shape, gap, z);
            std::copy(start, start + 6, far);
            objective.update_dual_variables(6, 1., far, z);
            for (int j = 0; j < 6; ++j)
              EXPECT_NEAR(far[j], near[j], 4e-15 * c)
                  << k << " " << c << " " << shape << " " << gap;
          }
        }
      }
      // The biased prox of the competitors (alpha = c) is the cold step of a target at 0.
      for (double gap : {1e4, 1e8, 1e16, 1e30}) {
        std::vector<double> x = {1., gap, -.3, .2, -1.5};
        double z[6] = {0., 1., gap, -.3, .2, -1.5}, a[6] = {0};
        sdca::prox_topk_entropy_biased(x.begin(), x.end(), static_cast<std::ptrdiff_t>(k), c);
        objective.update_dual_variables(6, 1., a, z);
        for (int j = 0; j < 5; ++j) EXPECT_NEAR(x[j], -a[j + 1] / c, 1e-15) << k << " " << gap;
      }
    }
  }
}
#endif

#ifndef __FAST_MATH__
TEST(NumericalRegression, topk_entropy_free_scores_far_below_a_capped_competitor) {
  // Beyond a gap of 700 the far scores' exponentials vanish, so the gradient and the
  // prox are those at 700. Scores relative to the maximum lost the free ones'
  // differences from a gap of about 1e4 on (by 0.28 at 1e16).
  auto row = [](int shape, double gap, double* z) {
    const double rows[3][6] = {{-gap, 1., gap, -.3, .2, -1.5},
                               {0., 1., gap, -.3, .2, gap - 3.},
                               {0., gap, gap, 1., -.3, .2}};
    std::copy(rows[shape], rows[shape] + 6, z);
  };
  const auto kind = sdca::loss_kind::topk_cross_entropy;
  for (std::size_t k : {2, 3, 4}) {
    for (int shape = 0; shape < 3; ++shape) {
      double z[6], near[6];
      row(shape, 700., z);
      sdca::loss_and_gradient(kind, 6, 1, z, near, k);
      for (double gap : {1e4, 1e8, 1e16, 1e30}) {
        double far[6];
        row(shape, gap, z);
        sdca::loss_and_gradient(kind, 6, 1, z, far, k);
        for (int j = 0; j < 6; ++j) EXPECT_NEAR(far[j], near[j], 1e-13) << shape << " " << gap;
      }
    }
    // The projection keeps them too: its map takes scores relative to the first free one.
    std::vector<double> near = {700., 1., -.3, .2, -1.5};
    sdca::prox_topk_entropy(near.begin(), near.end(), static_cast<std::ptrdiff_t>(k));
    for (double gap : {1e4, 1e8, 1e16, 1e30}) {
      std::vector<double> far = {gap, 1., -.3, .2, -1.5};
      sdca::prox_topk_entropy(far.begin(), far.end(), static_cast<std::ptrdiff_t>(k));
      for (int j = 0; j < 5; ++j) EXPECT_NEAR(far[j], near[j], 1e-13) << gap;
    }
  }
}
#endif

#ifndef __FAST_MATH__
TEST(NumericalRegression, single_entropy_aggregate_subnormal_tail) {
  // The aggregate is representable even though each competitor rounds to zero.
  // Evaluate the analytic reference in wider arithmetic independently of the
  // implementation's factored double exponential.
  std::vector<double> z(1000, -750.), gradient(1000);
  z[0] = 0;
  // log1p(999 exp(-750)) = 3.845 denorm_min, which rounds to 4 denorm_min.
  const double expected = 4 * std::numeric_limits<double>::denorm_min();
  const double loss = sdca::entropy_loss_gradient<double>(1000, 1, z.data(), gradient.data());
  EXPECT_GT(loss, 0);
  EXPECT_EQ(loss, expected);
  EXPECT_EQ(gradient[0], -expected);
  for (std::size_t j = 1; j < gradient.size(); ++j) EXPECT_EQ(gradient[j], 0);

  std::vector<float> small(1000, -110.f), small_gradient(1000);
  small[0] = 0;
  const float expected_float = static_cast<float>(std::log1p(999.L * std::exp(-110.L)));
  const float small_loss =
      sdca::entropy_loss_gradient<float>(1000, 1, small.data(), small_gradient.data());
  EXPECT_GT(small_loss, 0);
  EXPECT_EQ(small_loss, expected_float);
  EXPECT_EQ(small_gradient[0], -expected_float);
}
#endif

TEST(NumericalRegression, entropy_losses_and_residuals) {
  float z[] = {12, -6, -6}, a[] = {0, 0, 0};
  sdca::l2_entropy<float, float> objective(1);
  EXPECT_GT(objective.primal_loss(3, z), 0);
  objective.update_dual_variables(3, 1, a, z);
  EXPECT_GT(a[0], 0);
  EXPECT_EQ(a[0], -a[1] - a[2]);
  EXPECT_NEAR(double(a[0]) / (2 * std::exp(-18.)), 1., 1e-6);
  float infeasible[] = {0, -.1f, -.1f};
  EXPECT_EQ(objective.dual_loss(3, infeasible), -std::numeric_limits<float>::infinity());

  float multi_z[] = {20, 20, -20}, multi_a[] = {0, 0, 0};
  sdca::l2_multilabel_entropy<float, double> multi(1);
  multi.update_dual_variables(3, 2, 1, multi_a, multi_z);
  EXPECT_GT(multi_a[0], 0);
  EXPECT_EQ(multi_a[0], multi_a[1]);
  EXPECT_EQ(multi_a[0] + multi_a[1], -multi_a[2]);
  EXPECT_NEAR(double(multi_a[0]) / (std::exp(-40.) / 4), 1., 1e-6);
}

TEST(NumericalRegression, topk_entropy_tail) {
  for (std::size_t k = 1; k <= 2; ++k) {
    sdca::l2_entropy_topk<double, double> obj(1, k);
    double z[] = {0, -50, -60};
    double expected =
        k == 1 ? std::log1p(std::exp(-50.) + std::exp(-60.)) : std::log1p(2 * std::exp(-55.));
    EXPECT_NEAR(obj.primal_loss(3, z) / expected, 1., 1e-14);
  }
  double x[] = {-700, -700};
  EXPECT_GT(sdca::log_1_sum_exp(x, x + 2), 0);
  EXPECT_NEAR(sdca::log_1_sum_exp(x, x + 2) / (2 * std::exp(-700.)), 1., 1e-12);
  for (double c : {1., 1e8, 1e18})
    for (std::size_t k : {2UL, 3UL, 4UL}) {
      double z[] = {0, 100, 0, -20, -40}, old_z[] = {0, 100, 0, -20, -40}, a[5] = {0}, g[5];
      sdca::l2_entropy_topk<double, double>(c, k).update_dual_variables(5, 1., a, z);
      for (int j = 0; j < 5; ++j) old_z[j] += a[j];
      sdca::loss_and_gradient(sdca::loss_kind::topk_cross_entropy, 5, 1, old_z, g, k);
      for (int j = 0; j < 5; ++j) EXPECT_NEAR(a[j], -c * g[j], 1e-8 * std::max(1., std::abs(a[j])));
    }
}

template <typename Objective>
void check_hinge_shift(Objective obj) {
  float shifted[] = {1e10f, 1e10f, 1e10f}, centered[] = {0, 0, 0};
  auto ref = obj.primal_loss(3, centered);
  EXPECT_GT(ref, 0);
  EXPECT_NEAR(obj.primal_loss(3, shifted), ref, 1e-6);
  float a[] = {0, 0, 0}, b[] = {0, 0, 0};
  float s[] = {1e10f, 1e10f, 1e10f}, t[] = {0, 0, 0};
  obj.update_dual_variables(3, 1, a, s);
  obj.update_dual_variables(3, 1, b, t);
  for (int j = 0; j < 3; ++j) EXPECT_NEAR(a[j], b[j], 1e-6);
}

#ifndef __FAST_MATH__
TEST(NumericalRegression, hinge_common_offsets) {
  check_hinge_shift(sdca::l2_hinge_topk<float, double>(1, 2));
  check_hinge_shift(sdca::l2_topk_hinge<float, double>(1, 2));
  check_hinge_shift(sdca::l2_hinge_topk_smooth<float, double>(1, 2, .7));
  check_hinge_shift(sdca::l2_topk_hinge_smooth<float, double>(1, 2, .7));
  sdca::l2_multilabel_hinge_smooth<float, double> obj(1, .7);
  float a[] = {1e10f, 1e10f, 1e10f}, b[] = {0, 0, 0};
  EXPECT_NEAR(obj.primal_loss(3, 2, a), obj.primal_loss(3, 2, b), 1e-6);
}
#endif

// Zero scores, three classes, C = 1, k = 1: the non-label entries x solve
// x + rho <1, x> = rho / norm2, so the block is [2, -1, -1] / (3 norm2) and,
// smoothed with gamma = 1, [2, -1, -1] / (3 norm2 + 1). Large norm2 makes the
// projected scores tiny; no absolute tolerance may absorb them.
template <typename Data, typename Result>
void check_hinge_tiny_projection() {
  const double tol = double(8 * std::numeric_limits<Data>::epsilon());
  for (const double norm2 : {1., 1e8, 1e20}) {
    const double plain = 1 / (3 * norm2), smooth = 1 / (3 * norm2 + 1);
    auto check = [&](const Data* a, const double e, const char* name) {
      EXPECT_NEAR(a[0], 2 * e, tol * 2 * e) << name << " norm2 = " << norm2;
      EXPECT_NEAR(a[1], -e, tol * e) << name << " norm2 = " << norm2;
      EXPECT_NEAR(a[2], -e, tol * e) << name << " norm2 = " << norm2;
    };
    Data a[3] = {0, 0, 0}, z[3] = {0, 0, 0};
    sdca::l2_topk_hinge<Data, Result>(1, 1).update_dual_variables(3, Data(norm2), a, z);
    check(a, plain, "topk_hinge_beta");
    Data b[3] = {0, 0, 0}, y[3] = {0, 0, 0};
    sdca::l2_topk_hinge_smooth<Data, Result>(1, 1, 1).update_dual_variables(3, Data(norm2), b, y);
    check(b, smooth, "topk_hinge_beta");
    Data c[3] = {0, 0, 0}, w[3] = {0, 0, 0};
    sdca::l2_hinge_topk<Data, Result>(1, 1).update_dual_variables(3, Data(norm2), c, w);
    check(c, plain, "topk_hinge_alpha");
  }
}

TEST(NumericalRegression, hinge_coordinate_steps_at_tiny_projected_scores) {
  check_hinge_tiny_projection<float, float>();
  check_hinge_tiny_projection<float, double>();
  check_hinge_tiny_projection<double, double>();
}

TEST(NumericalRegression, invalid_labels_and_parameters) {
  std::vector<int> empty, negative = {0, -2};
  std::vector<double> fractional = {0, 1.5};
  EXPECT_THROW(sdca::make_output_multiclass(empty.begin(), empty.end()), std::invalid_argument);
  EXPECT_THROW(sdca::make_output_multiclass(negative.begin(), negative.end()),
               std::invalid_argument);
  EXPECT_THROW(sdca::make_output_multiclass(fractional.begin(), fractional.end()),
               std::invalid_argument);
  std::vector<sdca::size_type> labels = {0, 1}, missing, oversized = {0, 10},
                               descending = {0, 2, 1, 2}, extra = {0, 1};
  EXPECT_THROW(sdca::validate_labels_and_offsets(3, labels, missing), std::invalid_argument);
  EXPECT_THROW(sdca::validate_labels_and_offsets(3, labels, oversized), std::invalid_argument);
  EXPECT_THROW(sdca::validate_labels_and_offsets(3, labels, descending), std::invalid_argument);
  EXPECT_THROW(sdca::validate_labels_and_offsets(3, labels, extra), std::invalid_argument);
  using Smooth = sdca::l2_hinge_topk_smooth<float, double>;
  EXPECT_THROW(Smooth(-1, 1, -1), std::invalid_argument);
  sdca::l2_hinge_topk<float, double> obj(1, 3);
  float scores[] = {0, 0, 0};
  EXPECT_THROW(obj.primal_loss(3, scores), std::invalid_argument);
}

// An objective whose coordinate step always fails.
template <typename Data, typename Result>
struct failing_entropy : sdca::l2_entropy<Data, Result> {
  failing_entropy() : sdca::l2_entropy<Data, Result>(1) {}
  template <typename Int>
  void update_dual_variables(Int, Data, Data*, Data*) const {
    throw std::runtime_error("step failed");
  }
};

TEST(NumericalRegression, failed_step_keeps_the_dual_block_and_weights) {
  // The label is last, so the step permutes the block before it fails.
  double x[] = {1, 2}, a[] = {-0.1, -0.2, 0.3}, w[6];
  for (int c = 0; c < 3; ++c)
    for (int j = 0; j < 2; ++j) w[j + 2 * c] = x[j] * a[c];
  const std::vector<double> a0(a, a + 3), w0(w, w + 6);
  std::vector<sdca::size_type> labels = {2};
  auto ctx =
      sdca::make_context(sdca::make_input_feature(2, 1, x), sdca::multiclass_output(3, labels),
                         failing_entropy<double, double>(), a, w);
  sdca::solver_scratch<double, sdca::feature_input> scratch;
  scratch.init(ctx);
  EXPECT_THROW(sdca::update_variables(0, ctx, scratch), std::runtime_error);
  EXPECT_EQ(a0, std::vector<double>(a, a + 3));
  EXPECT_EQ(w0, std::vector<double>(w, w + 6));

  double gram[] = {5}, b[] = {-0.1, -0.2, 0.3};
  std::vector<sdca::size_type> kernel_labels = {2};
  auto kernel_ctx = sdca::make_context(sdca::make_input_kernel(1, gram),
                                       sdca::multiclass_output(3, kernel_labels),
                                       failing_entropy<double, double>(), b);
  sdca::solver_scratch<double, sdca::kernel_input> kernel_scratch;
  kernel_scratch.init(kernel_ctx);
  EXPECT_THROW(sdca::update_variables(0, kernel_ctx, kernel_scratch), std::runtime_error);
  EXPECT_EQ(a0, std::vector<double>(b, b + 3));
}

#ifndef __FAST_MATH__
TEST(NumericalRegression, centered_hinge_near_zero_uses_full_objective_scale) {
  for (double deficit : {0.0, std::ldexp(1.0, -30)}) {
    double x[] = {1}, a[] = {0, 0}, w0[] = {1 - deficit, 0}, w[] = {1 - deficit, 0};
    std::vector<sdca::size_type> labels = {0};
    auto ctx =
        sdca::make_context(sdca::make_input_feature(1, 1, x), sdca::multiclass_output(2, labels),
                           sdca::l2_topk_hinge<double, double>(1, 1), a, w);
    ctx.primal_initial = w0;
    ctx.criteria.eval_on_start = true;
    ctx.criteria.max_epoch = 0;
    ctx.criteria.epsilon = 1e-8;
    sdca::make_solver(ctx).solve();
    const auto& e = ctx.train.evals.back();
    EXPECT_EQ(deficit, e.primal);
    EXPECT_EQ(0, e.dual);
    EXPECT_NE(sdca::solver_status::solved, ctx.status);
    EXPECT_GT(e.primal - e.dual + e.gap_error,
              ctx.criteria.epsilon * std::max(std::abs(e.primal), std::abs(e.dual)));
    // At exactly zero objective the conservative roundoff bound still
    // prevents a relative certificate; no absolute tolerance is substituted.
    if (deficit == 0) {
      EXPECT_GT(e.gap_error, 0);
    }

    // The exact optimum is A=(deficit/2,-deficit/2), W=W0+A,
    // with P=D=deficit^2/4. All values are dyadic here. Even this optimum
    // must not certify a relative target below the evaluation error.
    a[0] = deficit / 2;
    a[1] = -deficit / 2;
    w[0] = w0[0] + a[0];
    w[1] = a[1];
    sdca::make_solver(ctx).solve();
    const auto& optimum = ctx.train.evals.back();
    EXPECT_EQ(deficit * deficit / 4, optimum.primal);
    EXPECT_EQ(deficit * deficit / 4, optimum.dual);
    EXPECT_NE(sdca::solver_status::solved, ctx.status);
    EXPECT_GT(optimum.gap_error,
              ctx.criteria.epsilon * std::max(std::abs(optimum.primal), std::abs(optimum.dual)));
  }
}

TEST(NumericalRegression, stopping_and_test_evaluation) {
  double x[] = {1}, a[] = {0, 0}, w[] = {0, 0};
  std::vector<sdca::size_type> labels = {0};
  auto ctx =
      sdca::make_context(sdca::make_input_feature(1, 1, x), sdca::multiclass_output(2, labels),
                         sdca::l2_entropy<double, double>(1), a, w);
  ctx.status = sdca::solver_status::solving;
  ctx.criteria.epsilon = 0;
  ctx.train.evals.resize(1);
  ctx.train.evals[0].primal = 1;
  ctx.train.evals[0].dual = 1 + 1e-15;
  ctx.train.evals[0].gap_error = 4e-15;
  sdca::check_stopping_criteria<double, double>(ctx);
  EXPECT_EQ(sdca::solver_status::solving, ctx.status);
  // A tolerance is met only with the rounding bound added to the gap.
  ctx.criteria.epsilon = 2e-15;
  ctx.train.evals[0].dual = 1;
  sdca::check_stopping_criteria<double, double>(ctx);
  EXPECT_EQ(sdca::solver_status::solving, ctx.status);
  ctx.criteria.epsilon = 4e-15;
  sdca::check_stopping_criteria<double, double>(ctx);
  EXPECT_EQ(sdca::solver_status::solved, ctx.status);
  // A negative gap beyond the rounding bound fails.
  ctx.status = sdca::solver_status::solving;
  ctx.train.evals[0].dual = 1 + 1e-14;
  sdca::check_stopping_criteria<double, double>(ctx);
  EXPECT_EQ(sdca::solver_status::failed, ctx.status);
  ctx.status = sdca::solver_status::solving;
  ctx.criteria.epsilon = 0;
  ctx.train.evals[0].primal = std::numeric_limits<double>::quiet_NaN();
  sdca::check_stopping_criteria<double, double>(ctx);
  EXPECT_EQ(sdca::solver_status::failed, ctx.status);
  std::vector<sdca::size_type> wrong_y = {0};
  EXPECT_THROW(ctx.add_test(sdca::make_input_feature(2, 1, x), sdca::multiclass_output(2, wrong_y)),
               std::invalid_argument);
  std::vector<double> test_x(100, 1);
  std::vector<sdca::size_type> test_y(100, 0);
  auto test = sdca::make_dataset_test<double>(sdca::make_input_feature(1, 100, test_x.data()),
                                              sdca::multiclass_output(2, test_y));
  sdca::solver_scratch<double, sdca::feature_input> scratch;
  scratch.init(ctx);
  sdca::evaluate_dataset(ctx, test, scratch);
  EXPECT_NEAR(test.evals.back().primal_loss, 100 * std::log(2.), 1e-12);
}
#endif

// An entropy objective whose conjugate loss is overstated by 1e-9 per example.
template <typename Data, typename Result>
struct overstated_entropy : sdca::l2_entropy<Data, Result> {
  overstated_entropy() : sdca::l2_entropy<Data, Result>(1) {}
  template <typename Int>
  Result dual_loss(const Int n, const Data* a) const {
    return sdca::l2_entropy<Data, Result>::dual_loss(n, a) + Result(1e-9);
  }
};

#ifndef __FAST_MATH__
TEST(NumericalRegression, negative_gap_beyond_rounding_fails) {
  double x[] = {1, 0, 0, 1, 1, 1, -1, 2}, a[12] = {}, w[6] = {};
  std::vector<sdca::size_type> labels = {0, 1, 2, 1};
  auto ctx =
      sdca::make_context(sdca::make_input_feature(2, 4, x), sdca::multiclass_output(3, labels),
                         overstated_entropy<double, double>(), a, w);
  ctx.criteria.epsilon = 0;
  ctx.criteria.max_epoch = 500;
  sdca::make_solver(ctx).solve();
  EXPECT_EQ(sdca::solver_status::failed, ctx.status);
  const auto& e = ctx.train.evals.back();
  EXPECT_LT(e.primal - e.dual, -e.gap_error);
  EXPECT_GT(e.gap_error, 0);
  EXPECT_LT(e.gap_error, 1e-12);
}
#endif

// recompute_primal only replaces W by W0 + X A' (rounded from double for float
// storage): the duals and the dual objective are the same either way.
template <typename Data>
void check_recompute_primal(const bool prox) {
  const sdca::size_type d = 3, n = 6, m = 3;
  const Data x[] = {1, 0, 0.5, 0, 1, -0.5, 1, 1, 0.25, -1, 0.5, 1, 0.3, -2, 1, 2, 0.1, -0.7};
  const Data w0[] = {0.1, -0.2, 0.3, 0, 0.5, -0.1, 0.2, 0.2, -0.3};
  std::vector<Data> a[2], w[2];
  double dual[2], primal[2];
  for (int recompute = 0; recompute < 2; ++recompute) {
    std::vector<sdca::size_type> labels = {0, 1, 2, 1, 0, 2};
    a[recompute].assign(n * m, 0);
    w[recompute] = prox ? std::vector<Data>(w0, w0 + d * m) : std::vector<Data>(d * m, 0);
    auto ctx = sdca::make_context(
        sdca::make_input_feature(d, n, x), sdca::multiclass_output(m, labels),
        sdca::l2_entropy<Data, double>(1), a[recompute].data(), w[recompute].data());
    if (prox) ctx.primal_initial = w0;
    ctx.criteria.recompute_primal = recompute == 1;
    ctx.criteria.max_epoch = ctx.criteria.eval_epoch = 7;
    ctx.criteria.epsilon = 0;
    sdca::make_solver(ctx).solve();
    dual[recompute] = ctx.train.evals.back().dual;
    primal[recompute] = ctx.train.evals.back().primal;
  }
  EXPECT_EQ(a[0], a[1]);
  EXPECT_EQ(dual[0], dual[1]);
  EXPECT_NEAR(primal[0], primal[1], 1e-5 * std::abs(primal[1]));
  for (sdca::size_type c = 0; c < m; ++c)
    for (sdca::size_type j = 0; j < d; ++j) {
      double v = prox ? double(w0[d * c + j]) : 0;
      for (sdca::size_type i = 0; i < n; ++i) v += double(x[d * i + j]) * double(a[1][m * i + c]);
      EXPECT_NEAR(double(w[1][d * c + j]), v, 4 * std::numeric_limits<Data>::epsilon())
          << prox << " " << c << " " << j;
    }
}

TEST(NumericalRegression, recompute_primal_is_a_runtime_choice) {
  for (bool prox : {false, true}) {
    check_recompute_primal<float>(prox);
    check_recompute_primal<double>(prox);
  }
}

// Stored blocks a rounding put outside their sets come back feasible.
TEST(NumericalRegression, evaluation_repairs_rounded_dual_blocks) {
  const double d = 1e-7, tiny = 1e-15;
  auto expect_balanced = [&](const std::vector<double>& a) {
    double sum = 0, size = 0;
    for (double v : a) {
      sum += v;
      size += std::abs(v);
    }
    EXPECT_LE(std::abs(sum), tiny * size);
  };
  // Top-k simplex (k = 2, c = 1): b_j <= sum b / 2, sum b <= 1.
  std::vector<double> a = {0, -(0.5 + d), -0.5, 0};
  sdca::l2_hinge_topk<double, double>(1, 2).feasible_dual(4, a.data());
  expect_balanced(a);
  EXPECT_LE(a[0], 1 + tiny);
  for (int j = 1; j < 4; ++j) EXPECT_LE(-a[j], a[0] / 2 * (1 + tiny));
  // Fixed caps (k = 2, c = 1): b_j <= 1/2, sum b <= 1.
  a = {0, -(0.5 + d), -0.3, -0.3};
  sdca::l2_topk_hinge<double, double>(1, 2).feasible_dual(4, a.data());
  expect_balanced(a);
  EXPECT_LE(a[0], 1 + tiny);
  for (int j = 1; j < 4; ++j) EXPECT_LE(-a[j], 0.5);
  // Two labels of a distribution (c = 1): a_j <= 1/2 on the labels, <= 0 elsewhere.
  a = {0.5 + d, 0.3, -0.5, -0.3};
  sdca::l2_multilabel_entropy<double, double>(1).feasible_dual(4, 2, a.data());
  expect_balanced(a);
  EXPECT_LE(a[0], 0.5);
  // Two simplices (c = 1): equal sums of at most 1.
  a = {0.6, 0.5, -0.7, -(0.4 + d)};
  sdca::l2_multilabel_hinge<double, double>(1).feasible_dual(4, 2, a.data());
  expect_balanced(a);
  EXPECT_LE(a[0] + a[1], 1 + tiny);
  // A feasible block keeps its entries.
  a = {0.375, -0.125, -0.25};
  sdca::l2_entropy<double, double>(1).feasible_dual(3, a.data());
  EXPECT_EQ(std::vector<double>({0.375, -0.125, -0.25}), a);
}

// A distillation block's rounding residual goes to its most probable class, not to a class of
// probability 0 ahead of it (pi = q - a / c; -pi log pi has an infinite slope at 0).
TEST(NumericalRegression, distillation_residual_leaves_zero_probability_classes) {
  const double q[] = {0, 0.25, 0.75}, d = 1e-7;
  const sdca::l2_distillation<double, double> objective(1);
  std::vector<double> a = {0, 0.25, -0.25 + d};  // pi = q - a = {0, 0, 1 - d}: an excess d
  objective.feasible_dual(3, q, a.data());
  EXPECT_EQ(std::vector<double>({0, 0.25, -0.25}), a);
  a = {0, 0.25 - d, -0.25};  // pi = {0, d, 1}: a deficit d, not taken from the class at d
  objective.feasible_dual(3, q, a.data());
  EXPECT_EQ(std::vector<double>({0, 0.25 - d, -0.25 + d}), a);
}

// Warm log-odds that overflow raise instead of writing NaN into the block.
TEST(NumericalRegression, distillation_rejects_overflowing_warm_log_odds) {
  const double q[] = {0.5, 0.5};
  const sdca::l2_distillation<double, double> objective(1);
  std::vector<double> a = {0, 0}, scores = {1e308, -1e308};
  EXPECT_THROW(objective.update_dual_variables(2, q, 1e10, a.data(), scores.data()),
               std::invalid_argument);
}

TEST(NumericalRegression, native_loss_values_match_objectives) {
  double input[] = {.4, -1., 2., .1}, g[4];
  for (int code = 0; code < 8; ++code) {
    std::vector<double> z(input, input + 4);
    double expected = 0;
    switch (code) {
      case 0:
        expected = sdca::l2_entropy<double, double>(1).primal_loss(4, z.data());
        break;
      case 1:
        expected = sdca::l2_entropy_topk<double, double>(1, 2).primal_loss(4, z.data());
        break;
      case 2:
        expected = sdca::l2_hinge_topk<double, double>(1, 2).primal_loss(4, z.data()) / 2;
        break;
      case 3:
        expected = sdca::l2_topk_hinge<double, double>(1, 2).primal_loss(4, z.data()) / 2;
        break;
      case 4:
        expected = sdca::l2_hinge_topk_smooth<double, double>(1, 2, .7).primal_loss(4, z.data());
        break;
      case 5:
        expected = sdca::l2_topk_hinge_smooth<double, double>(1, 2, .7).primal_loss(4, z.data());
        break;
      case 6:
        expected = sdca::l2_multilabel_hinge<double, double>(1).primal_loss(4, 2, z.data());
        break;
      case 7:
        expected =
            sdca::l2_multilabel_hinge_smooth<double, double>(1, .7).primal_loss(4, 2, z.data());
        break;
    }
    EXPECT_NEAR(expected,
                sdca::loss_and_gradient(loss_case(code).first, 4, code < 6 ? 1 : 2, input, g, 2,
                                        loss_case(code).second ? .7 : 0),
                1e-13);
  }
}

TEST(NumericalRegression, topk_coordinate_kkt_randomized) {
  std::mt19937 generator(91);
  std::normal_distribution<double> normal(0, 1);
  for (int trial = 0; trial < 80; ++trial) {
    const std::size_t n = 5 + trial % 6, k = 1 + trial % (n - 1);
    const double c = std::pow(10., trial % 13 - 6);
    SCOPED_TRACE(::testing::Message() << "trial=" << trial << " k=" << k << " C=" << c);
    std::vector<double> z(n), original(n), a(n), g(n);
    for (auto& value : z) value = 10 * normal(generator);
    original = z;
    sdca::l2_entropy_topk<double, double>(c, k).update_dual_variables(n, 1., a.data(), z.data());
    for (std::size_t j = 0; j < n; ++j) original[j] += a[j];
    sdca::loss_and_gradient(sdca::loss_kind::topk_cross_entropy, n, 1, original.data(), g.data(),
                            k);
    for (std::size_t j = 0; j < n; ++j)
      EXPECT_NEAR(a[j], -c * g[j], 1e-8 * std::max(1., std::abs(a[0])));
  }
}

TEST(NumericalRegression, raw_biased_entropy_nonzero_and_kkt) {
  // The old unbracketed Householder iteration occasionally returned zero
  // for perfectly ordinary confident inputs. Feasibility alone missed this.
  for (int i = 0; i < 400; ++i) {
    const double score = -3. - 27. * i / 399., alpha = std::pow(10., i % 9 - 4);
    double lo = -800, hi = score;
    for (int it = 0; it < 100; ++it) {
      double r = lo + (hi - lo) / 2, p = 1 / (1 + std::exp(-r));
      if (r + 2 * alpha * p < score)
        lo = r;
      else
        hi = r;
    }
    double ref = std::exp(-sdca::softplus(-(lo + hi) / 2));
    double d = score;
    float f = static_cast<float>(score);
    sdca::prox_topk_entropy_biased(&d, &d + 1, 1, alpha);
    sdca::prox_topk_entropy_biased(&f, &f + 1, 1, alpha);
    EXPECT_GT(d, 0);
    EXPECT_GT(f, 0);
    EXPECT_NEAR(d / ref, 1., 1e-12);
    EXPECT_NEAR(double(f) / ref, 1., 3e-6);
  }
  double x[] = {0, 0}, scratch[2];
  sdca::prox_topk_entropy_biased(1, x, x + 2, scratch, 1, 1.);
  EXPECT_NEAR(x[0], x[1], 1e-15);
  EXPECT_THROW(sdca::prox_topk_entropy_biased(0, x, x + 2, scratch), std::invalid_argument);
  EXPECT_THROW(sdca::prox_topk_entropy_biased(x, x + 2, 3), std::invalid_argument);
}

TEST(NumericalRegression, public_entropy_probability_prox) {
  for (double alpha : {1e-300, 1., 1e18}) {
    double z[] = {0, -40, -40};
    double original[] = {0, -40, -40}, a[3] = {};
    sdca::l2_entropy<double, double>(1).update_dual_variables(3, alpha, a, original);
    sdca::prox_two_entropy(z, z + 1, z + 3, alpha);
    EXPECT_NEAR(z[0] + z[1] + z[2], 1., 1e-14);
    EXPECT_GT(z[1], 0);
    EXPECT_NEAR(z[1] / (-a[1]), 1., 1e-13);
    double raw[] = {0, -40, -40};
    auto thresholds = sdca::thresholds_two_entropy(raw, raw + 1, raw + 1, raw + 3, alpha);
    sdca::prox(thresholds.first, raw, raw + 1);
    sdca::prox(thresholds.second, raw + 1, raw + 3);
    for (int j = 0; j < 3; ++j) EXPECT_NEAR(raw[j] / z[j], 1., 2e-13);
  }
  for (std::ptrdiff_t k : {1, 2, 3}) {
    double raw[] = {10, -30, -50}, direct[] = {10, -30, -50};
    auto thresholds = sdca::thresholds_topk_entropy_biased(raw, raw + 3, k, 2.);
    sdca::prox(thresholds, raw, raw + 3);
    sdca::prox_topk_entropy_biased(direct, direct + 3, k, 2.);
    for (int j = 0; j < 3; ++j) EXPECT_NEAR(raw[j] / direct[j], 1., 2e-13);
  }
}

#ifndef __FAST_MATH__
TEST(NumericalRegression, lambert_extreme_and_extended_precision) {
  for (double x : {1e7, 1e8, 536870912., 1e10, 1e100, 1e300}) {
    double w = sdca::lambert_w_exp(x);
    EXPECT_TRUE(std::isfinite(w));
    EXPECT_NEAR((w - x) + std::log(w), 0., 2 * std::numeric_limits<double>::epsilon() * x);
    if (x < 1e30) {
      float xf = static_cast<float>(x), wf = sdca::lambert_w_exp(xf);
      EXPECT_TRUE(std::isfinite(wf));
      EXPECT_EQ(wf, static_cast<float>(sdca::lambert_w_exp(static_cast<double>(xf))));
    }
  }
  long double w = sdca::lambert_w_exp(1e18L);
  EXPECT_NEAR(w, 1e18L - std::log(1e18L), .125L);
  if (std::numeric_limits<long double>::min_exponent < -2000) {
    w = sdca::lambert_w_exp(-1000.L);
    EXPECT_GT(w, 0);
    EXPECT_NEAR(w / std::exp(-1000.L), 1.L, 1e-17L);
  }
  EXPECT_TRUE(std::isnan(sdca::lambert_w_exp(std::numeric_limits<double>::quiet_NaN())));
}
#endif

TEST(NumericalRegression, projection_moments_and_batch_validation) {
  double z[] = {1e8, 1e8};
  auto t = sdca::thresholds_knapsack_eq(z, z + 2, 0., 1., 1.);
  EXPECT_NEAR(sdca::dot_prox_prox(t, z, z + 2), .5, 1e-14);
  EXPECT_NEAR(sdca::dot_x_prox(t, z, z + 2), 1e8, 1e-7);
  double batch[] = {2, 0, 0, 2}, aux[2];
  sdca::prox_knapsack_eq(2, batch, batch + 4, aux, 0., 1., 1.);
  EXPECT_EQ(batch[0], 1.);
  EXPECT_EQ(batch[1], 0.);
  EXPECT_EQ(batch[2], 0.);
  EXPECT_EQ(batch[3], 1.);
  EXPECT_THROW(sdca::prox_knapsack_eq(0, batch, batch + 4, aux), std::invalid_argument);
  EXPECT_THROW(sdca::prox_knapsack_eq(3, batch, batch + 4, aux), std::invalid_argument);
  EXPECT_THROW(sdca::prox_topk_simplex(batch, batch + 4, 0), std::invalid_argument);
  EXPECT_THROW(sdca::prox_topk_entropy(batch, batch + 4, 5), std::invalid_argument);
  double pair[] = {2, 0};
  sdca::prox_knapsack_eq(pair, pair + 2, 0., 1., 1.);
  EXPECT_EQ(pair[0], 1.);
  EXPECT_EQ(pair[1], 0.);
  sdca::prox_two_simplex(pair, pair + 1, pair + 1, pair + 2);
  EXPECT_NEAR(pair[0], pair[1], 1e-14);
  std::vector<double> many(30000, 0.);
  many[0] = 3.;
  sdca::prox_knapsack_eq(many.begin(), many.end(), 0., 10., 3.1);
  EXPECT_NEAR(std::accumulate(many.begin(), many.end(), 0.L), 3.1L, 1e-12L);
  EXPECT_NEAR(many[1], (3.1 - 3.) / 30000, 1e-18);
}

#ifndef __FAST_MATH__
TEST(NumericalRegression, capped_entropy_offsets_and_boundary_sums) {
  for (bool quadratic : {false, true})
    for (double rhs : {0., .3, 1., 3.}) {
      double z[] = {1e16, 1e16, 1e16};
      if (quadratic)
        sdca::prox_entropy_norm(z, z + 3, 1., rhs);
      else
        sdca::prox_entropy(z, z + 3, 1., rhs);
      for (auto x : z) EXPECT_NEAR(x, rhs / 3, 4e-16);
    }
  double z[] = {2, 0, -3};
  EXPECT_THROW(sdca::prox_entropy(z, z + 3, 1., 4.), std::invalid_argument);
  EXPECT_THROW(sdca::prox_entropy_norm(z, z + 3, -1., 1.), std::invalid_argument);
}
#endif

#ifndef __FAST_MATH__
TEST(NumericalRegression, topk_entropy_large_wrong_equal_scores) {
  for (std::size_t k : {1UL, 2UL, 3UL}) {
    double z[] = {0, 1e16, 1e16, 1e16}, g[4];
    sdca::loss_and_gradient(sdca::loss_kind::topk_cross_entropy, 4, 1, z, g, k);
    EXPECT_NEAR(g[0], -1., 4e-16);
    for (int j = 1; j < 4; ++j) EXPECT_NEAR(g[j], 1. / 3., 4e-16);
  }
  for (auto kind : {sdca::loss_kind::topk_hinge_alpha, sdca::loss_kind::topk_hinge_beta,
                    sdca::loss_kind::multilabel_hinge}) {
    double z[] = {0, 1e16, 1e16}, g[3];
    sdca::loss_and_gradient(kind, 3, 1, z, g, 1, .7);
    EXPECT_NEAR(g[0], -1., 1e-14);
    EXPECT_NEAR(g[1], .5, 1e-14);
    EXPECT_NEAR(g[2], .5, 1e-14);
  }
}
#endif

#ifndef __FAST_MATH__
TEST(NumericalRegression, two_simplex_boundary_and_large_scores) {
  for (bool sort : {false, true})
    for (double rhs : {0., 1.}) {
      double a[] = {.5}, b[] = {1e16, 1e16};
      if (sort)
        sdca::prox_two_simplex_sort(a, a + 1, b, b + 2, rhs);
      else
        sdca::prox_two_simplex(a, a + 1, b, b + 2, rhs);
      EXPECT_EQ(a[0], rhs);
      EXPECT_EQ(b[0], rhs / 2);
      EXPECT_EQ(b[1], rhs / 2);
    }
  // The single active positive must retain a unique sum multiplier: treating
  // its redundant per-entry cap as active selected the wrong second phase.
  double a[] = {.38461538, 4.17240876}, b[] = {-1.9227493, -2.29980888, -1.56413696};
  sdca::prox_two_simplex(a, a + 2, b, b + 3, .7);
  EXPECT_NEAR(a[0], 0., 1e-14);
  EXPECT_NEAR(a[1], .7, 1e-14);
  EXPECT_NEAR(b[0], .17069383, 1e-8);
  EXPECT_NEAR(b[1], 0., 1e-14);
  EXPECT_NEAR(b[2], .52930617, 1e-8);
}
#endif

TEST(NumericalRegression, entropy_scalar_inverse_across_switches) {
  // Construct the right hand side from a known residual, independently of
  // the inverse's Lambert/Newton implementation. Include both fast-path
  // boundaries, clipping, tiny residuals, and extreme proximal scales.
  for (double q : {1., .5, .1})
    for (double alpha : {1e-300, 1e-10, 1., 1e10, 1e18})
      for (double fraction : {-.999, -.501, -.5, -.499, -1e-18, 0., 1e-18, .499, .5, .501, .99}) {
        double r = q * fraction;
        if (r < q - 1) continue;
        long double ref = static_cast<long double>(r), Q = static_cast<long double>(q),
                    A = static_cast<long double>(alpha);
        double t = static_cast<double>(A * ref - std::log1p(-ref / Q));
        double actual = sdca::detail::entropy_residual(t, alpha, q);
        EXPECT_NEAR(actual, r, 16 * std::numeric_limits<double>::epsilon() * std::abs(r))
            << "alpha=" << alpha << " q=" << q << " r=" << r;
      }
}

TEST(NumericalRegression, entropy_probabilities_at_large_scale_wrong_scores) {
  for (double fraction : {.5, .75}) {
    double scores[] = {0, 2e18 * fraction};
    sdca::prox_two_entropy(scores, scores + 1, scores + 2, 1e18);
    EXPECT_NEAR(scores[0], 1 - fraction, 2e-14);
    EXPECT_NEAR(scores[1], fraction, 2e-14);
  }
  for (std::size_t n : {2UL, 64UL, 65UL, 200UL}) {
    std::vector<double> z(n, 0.), a(n, 0.);
    sdca::l2_entropy<double, double>(1).update_dual_variables(n, 1., a.data(), z.data());
    double denominator = std::exp(a[0]) + double(n - 1) * std::exp(a[1]);
    for (std::size_t j = 1; j < n; ++j) EXPECT_NEAR(a[j], -std::exp(a[j]) / denominator, 1e-15);
    EXPECT_NEAR(std::accumulate(a.begin(), a.end(), 0.), 0., 1e-15);
  }
}

// Randomized independent KKT checks.
// Log-domain KKT check of an entropy block: for the updated scores z + a - a0
// the stored block must be C (q - softmax) to relative precision, including
// residuals far below the resolution of the probabilities themselves.
template <typename Data>
void check_entropy_block_kkt(std::size_t n, std::size_t p, double c, const std::vector<Data>& z,
                             const std::vector<Data>& old, const std::vector<Data>& a, double tol) {
  typedef long double L;
  std::vector<L> s(n);
  for (std::size_t j = 0; j < n; ++j) s[j] = L(z[j]) + (L(a[j]) - L(old[j]));
  std::size_t im = std::size_t(std::max_element(s.begin(), s.end()) - s.begin());
  L tail(0), total(0);
  for (std::size_t j = 0; j < n; ++j) {
    if (j != im) tail += std::exp(s[j] - s[im]);
    total += L(a[j]);
  }
  // z_j - lse as (z_j - max) - log1p(tail): the subtraction first, so that
  // the reference itself does not lose the small correction.
  L correction = std::log1p(tail), size(0), q = L(1) / L(p);
  for (std::size_t j = 0; j < n; ++j) size += std::abs(L(a[j]));
  EXPECT_LE(std::abs(total), L(n) * L(std::numeric_limits<Data>::epsilon()) * size);
  // Entries below the normal range of Data may flush to zero in storage.
  const double floor = 2 * double(std::numeric_limits<Data>::min()) / c;
  for (std::size_t j = 0; j < n; ++j) {
    L centered = (s[j] - s[im]) - correction;
    L expected = j < p ? -q * std::expm1(centered - std::log(q)) : -std::exp(centered);
    EXPECT_NEAR(double(L(a[j]) / L(c)), double(expected), tol * double(std::abs(expected)) + floor)
        << "n=" << n << " p=" << p << " C=" << c << " j=" << j;
  }
}

template <typename Data, typename Result>
void entropy_block_kkt_randomized(double max_log_alpha, double tol, int trials) {
  std::mt19937 generator(2026);
  std::normal_distribution<double> normal(0, 1);
  std::uniform_real_distribution<double> uniform(0, 1);
  for (int trial = 0; trial < trials; ++trial) {
    const std::size_t n = trial % 5 == 0 ? 40 : 2 + std::size_t(trial % 7);
    const std::size_t p = trial % 3 == 2 ? 1 + std::size_t(trial % (n - 1)) : 1;
    const double c = std::pow(10., -4 + (max_log_alpha + 4) * uniform(generator));
    const double scale = trial % 4 == 3 ? 40 : trial % 2 ? 10 : 1;
    SCOPED_TRACE(::testing::Message() << "trial=" << trial << " n=" << n << " p=" << p << " C=" << c
                                      << " scale=" << scale);
    std::vector<Data> z(n), old(n), a(n);
    std::vector<double> prob(n);
    double den(0);
    for (std::size_t j = 0; j < n; ++j) {
      z[j] = Data(scale * normal(generator));
      prob[j] = std::exp(2 * normal(generator));
      den += prob[j];
    }
    // A feasible old block C (q - p_old), zero-sum by construction.
    double sum(0);
    for (std::size_t j = 0; j < n; ++j) {
      old[j] = Data(c * ((j < p ? 1. / double(p) : 0.) - prob[j] / den));
      if (j) sum += double(old[j]);
    }
    old[0] = Data(-sum);
    a = old;
    if (p == 1)
      sdca::l2_entropy<Data, Result>(Result(c)).update_dual_variables(n, Data(1), a.data(),
                                                                      z.data());
    else
      sdca::l2_multilabel_entropy<Data, Result>(Result(c)).update_dual_variables(
          n, p, Data(1), a.data(), z.data());
    check_entropy_block_kkt(n, p, c, z, old, a, tol);
  }
}

#ifndef __FAST_MATH__
TEST(NumericalRegression, entropy_block_kkt_randomized_double) {
  entropy_block_kkt_randomized<double, double>(6, 1e-8, 300);
}
#endif

#ifndef __FAST_MATH__
TEST(NumericalRegression, entropy_block_kkt_randomized_float32_storage_float64_compute) {
  entropy_block_kkt_randomized<float, double>(1, 2e-4, 150);
}
#endif

#ifndef __FAST_MATH__
// <float, float> solves in float: residuals keep float relative precision.
TEST(NumericalRegression, entropy_block_kkt_randomized_float) {
  entropy_block_kkt_randomized<float, float>(1, 2e-4, 150);
  // The symmetric large-C optimum, in float arithmetic.
  sdca::l2_entropy<float, float> large(1e18f);
  float a[] = {0, 0, 0}, z[] = {0, 0, 0};
  large.update_dual_variables(3, 1.f, a, z);
  EXPECT_NEAR(double(a[0]), 25.923031359473278, 2e-5);
  EXPECT_EQ(a[0], -a[1] - a[2]);
}
#endif

TEST(NumericalRegression, entropy_preserves_wrong_label_score_differences) {
  // With negligible target probability, x solves log(x/(1-x))+2*x=0.
  long double lo = 0, hi = 1;
  for (int i = 0; i < 100; ++i) {
    const long double x = (lo + hi) / 2;
    if (std::log(x) - std::log1p(-x) + 2 * x < 0)
      lo = x;
    else
      hi = x;
  }
  const double x = static_cast<double>((lo + hi) / 2);
  for (double gap : {700., 1e16, 1e30}) {
    double z[] = {-gap, 0, 1}, a[3] = {0}, g[3];
    sdca::entropy_update(3, 1, 1., 1., a, z);
    EXPECT_NEAR(a[0], 1., 2e-15);
    EXPECT_NEAR(a[1], -x, 2e-15);
    EXPECT_NEAR(a[2], x - 1, 2e-15);
    sdca::loss_and_gradient(sdca::loss_kind::proximal_cross_entropy, 3, 1, z, g);
    for (int j = 0; j < 3; ++j) EXPECT_NEAR(g[j], -a[j], 2e-15);
    sdca::prox_two_entropy(z, z + 1, z + 3, 1.);
    EXPECT_NEAR(z[1], x, 2e-15);
    EXPECT_NEAR(z[2], 1 - x, 2e-15);
  }
}

TEST(NumericalRegression, capped_entropy_small_mass_and_multiple_caps) {
  for (double scale : {1., 1e-20, 1e-100, 1e-300}) {
    double x[] = {2, 0, 0};
    sdca::prox_entropy(x, x + 3, scale, 1.5 * scale);
    EXPECT_NEAR(x[0] / scale, 1., 2e-13);
    EXPECT_NEAR(x[1] / scale, .25, 2e-13);
    EXPECT_NEAR(x[2] / scale, .25, 2e-13);
    double y[] = {100, 90, 0, 0};
    sdca::prox_entropy(y, y + 4, scale, 2.5 * scale);
    EXPECT_NEAR(y[0] / scale, 1., 2e-13);
    EXPECT_NEAR(y[1] / scale, 1., 2e-13);
    EXPECT_NEAR(y[2] / scale, .25, 2e-13);
    EXPECT_NEAR(y[3] / scale, .25, 2e-13);
  }
}

TEST(NumericalRegression, knapsack_mass_with_widely_separated_bound_sets) {
  for (double gap : {1e16, 1e30, 1e100, 1e300}) {
    std::vector<double> z = {gap, 0, 0, gap, 0, gap};
    std::vector<double> expected = {1, 1. / 6, 1. / 6, 1, 1. / 6, 1};
    for (int pass = 0; pass < 6; ++pass) {
      auto x = z;
      sdca::prox_knapsack_eq(x.begin(), x.end(), 0., 1., 3.5);
      for (std::size_t j = 0; j < x.size(); ++j) EXPECT_NEAR(x[j], expected[j], 2e-15);
      // Also check the lower-bound counterpart and storage order changes.
      for (auto& value : x) value = 0;
      std::transform(z.begin(), z.end(), x.begin(), [](double value) { return -value; });
      sdca::prox_knapsack_eq(x.begin(), x.end(), 0., 1., 2.5);
      for (std::size_t j = 0; j < x.size(); ++j) EXPECT_NEAR(x[j], 1 - expected[j], 2e-15);
      std::rotate(z.begin(), z.begin() + 1, z.end());
      std::rotate(expected.begin(), expected.begin() + 1, expected.end());
    }
  }
}

TEST(NumericalRegression, topk_entropy_optional_outputs_and_scratch_boundaries) {
  for (std::size_t n : {10UL, 64UL, 65UL, 200UL})
    for (std::size_t k : {1UL, 2UL, n})
      for (double alpha : {1e-100, 1., 1e18}) {
        std::vector<double> v(n), x, direct(n);
        for (std::size_t j = 0; j < n; ++j) v[j] = std::sin(static_cast<double>(j)) * 20;
        double normal, cap;
        sdca::topk_entropy_block(v, k, alpha, x, &normal, &cap);
        sdca::topk_entropy_block(n, v.data(), k, alpha, direct.data());
        EXPECT_LE(std::accumulate(x.begin(), x.end(), 0.), 1. + 1e-12);
        for (std::size_t j = 0; j < n; ++j) {
          EXPECT_EQ(x[j], direct[j]);
          EXPECT_GE(x[j], 0.);
          EXPECT_LE(x[j], cap * (1 + 1e-12));
          const double score = v[j] - normal, arg = std::log(alpha) + score;
          const double w = sdca::lambert_w_exp(arg);
          const double mapped = std::min(cap, arg < -18 ? std::exp(score - w) : w / alpha);
          EXPECT_NEAR(mapped / x[j], 1., 1e-13);
        }
      }
}

#ifndef __FAST_MATH__
TEST(NumericalRegression, topk_separated_offset_and_pointer_projection) {
  // At a saturated total mass, unconstrained coordinates have a common
  // log(x)+alpha*x-v. Check this without the production loss implementation.
  const double v[] = {0, 1, 2, -1};
  for (double gap : {700., 1e16, 1e30})
    for (std::size_t k : {1UL, 2UL, 3UL, 4UL}) {
      double z[] = {-gap, 0, 1, 2, -1}, a[5] = {0};
      sdca::l2_entropy_topk<double, double>(1., k).update_dual_variables(5, 1., a, z);
      EXPECT_NEAR(a[0], 1., 2e-14);
      double reference = 0;
      bool have_reference = false;
      for (std::size_t j = 0; j < 4; ++j) {
        double x = -a[j + 1];
        EXPECT_GT(x, 0.);
        EXPECT_LE(x, 1. / static_cast<double>(k) + 2e-14);
        if (x < 1. / static_cast<double>(k) - 1e-12) {
          double stationarity = std::log(x) + x - v[j];
          if (have_reference) {
            EXPECT_NEAR(stationarity, reference, 3e-13);
          }
          reference = stationarity;
          have_reference = true;
        }
      }
      if (have_reference)
        for (std::size_t j = 0; j < 4; ++j) {
          if (-a[j + 1] >= 1. / static_cast<double>(k) - 1e-12) {
            EXPECT_LE(std::log(-a[j + 1]) - a[j + 1] - v[j], reference + 3e-13);
          }
        }
    }
  // The no-cap fast path and sorted path must both support raw pointers.
  for (std::ptrdiff_t k : {1, 2, 4}) {
    const std::vector<double> scores = {0, 1, 2, -1};
    double a[] = {0, 1, 2, -1};
    std::vector<double> b(scores), c(scores), aux(scores);
    sdca::prox_topk_entropy(a, a + 4, k);
    sdca::prox_topk_entropy(b.begin(), b.end(), k);
    sdca::prox(sdca::thresholds_topk_entropy_offset(aux.begin(), aux.end(), k, 0.), c.begin(),
               c.end());
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_EQ(a[j], b[j]);
      EXPECT_EQ(a[j], c[j]);
    }
    std::vector<double> d(scores);
    aux = scores;
    sdca::prox(sdca::thresholds_topk_entropy_offset(aux.begin(), aux.end(), k, 1e16), d.begin(),
               d.end());
    double total = 0, reference = 0;
    bool have_reference = false;
    for (std::size_t j = 0; j < 4; ++j) {
      total += d[j];
      if (d[j] < 1. / static_cast<double>(k) - 1e-12) {
        if (have_reference) {
          EXPECT_NEAR(std::log(d[j]) - scores[j], reference, 1e-13);
        }
        reference = std::log(d[j]) - scores[j];
        have_reference = true;
      } else {
        EXPECT_NEAR(d[j], 1. / static_cast<double>(k), 1e-15);
      }
    }
    EXPECT_NEAR(total, 1., 1e-14);
  }
}

TEST(NumericalRegression, entropy_rescales_underflowed_probabilities) {
  for (double margin : {710., 745., 750., 1000.})
    for (double c : {1e100, 1e200}) {
      double z[] = {margin, 0}, a[] = {0., 0.};
      sdca::entropy_update(2, 1, c, 1., a, z);
      const double expected = std::exp(std::log(c) - margin);
      if (expected == 0) {
        EXPECT_EQ(a[0], 0.);
      } else {
        EXPECT_NEAR(a[0] / expected, 1., 3e-13);
      }
      EXPECT_EQ(a[0], -a[1]);
    }
  for (double margin : {60., 100.}) {
    float z[] = {static_cast<float>(margin), 0, 0, 0}, a[4] = {0, 0, 0, 0}, b[4] = {0, 0, 0, 0};
    sdca::l2_entropy_topk<float, float>(1e10f, 2).update_dual_variables(4, 1.f, a, z);
    sdca::l2_entropy<float, float>(1e10f).update_dual_variables(4, 1.f, b, z);
    for (int j = 0; j < 4; ++j)
      EXPECT_NEAR(static_cast<double>(a[j]) / static_cast<double>(b[j]), 1., 1e-6);
  }
  for (double margin : {720., 740.}) {
    double z[] = {margin, 0, 0, 0}, a[4] = {0, 0, 0, 0}, b[4] = {0, 0, 0, 0};
    sdca::l2_entropy_topk<double, double>(1e10, 2).update_dual_variables(4, 1., a, z);
    sdca::l2_entropy<double, double>(1e10).update_dual_variables(4, 1., b, z);
    for (int j = 0; j < 4; ++j) EXPECT_NEAR(a[j] / b[j], 1., 1e-13);
  }
}
#endif

TEST(NumericalRegression, topk_scaled_outputs_keep_probability_metadata) {
  const std::vector<double> v = {0., 1., 2., -1.};
  for (std::size_t k : {1UL, 2UL, 4UL}) {
    std::vector<double> x, y;
    double t, cap, scaled_t, scaled_cap;
    sdca::topk_entropy_block(v, k, 1., x, &t, &cap, 2.);
    sdca::topk_entropy_block(v, k, 1., y, &scaled_t, &scaled_cap, 2., 17.);
    EXPECT_EQ(t, scaled_t);
    EXPECT_EQ(cap, scaled_cap);
    for (std::size_t j = 0; j < v.size(); ++j) EXPECT_EQ(y[j], 17 * x[j]);
  }
  for (std::size_t k : {1UL, 2UL}) {
    std::vector<double> v2(2, 0.), x;
    sdca::topk_entropy_block(v2, k, 1e200, x, static_cast<double*>(nullptr),
                             static_cast<double*>(nullptr), -750., 1e200);
    const double expected = std::exp(std::log(1e200) - 750.);
    for (double value : x) EXPECT_NEAR(value / expected, 1., 3e-13);
  }
}

TEST(NumericalRegression, topk_negligible_tail_at_cap_boundary) {
  for (std::size_t k : {2UL, 3UL, 4UL})
    for (double c : {1., 100., 1e10, 1e18}) {
      const double margin = k == 2 ? 20. : (k == 3 ? 10. : 0.);
      long double lo = -1000, hi = 1000;
      const long double C = static_cast<long double>(c), K = static_cast<long double>(k);
      for (int i = 0; i < 300; ++i) {
        const long double t = (lo + hi) / 2, s = 1 / (1 + std::exp(-t));
        if (t + C * (1 + 1 / K) * s < static_cast<long double>(margin) + std::log(K))
          lo = t;
        else
          hi = t;
      }
      const double expected = static_cast<double>(C / (1 + std::exp(-(lo + hi) / 2)));
      std::vector<double> z(k + 2, margin), a(k + 2, 0.);
      z[0] = 0;
      z.back() = -100;
      sdca::l2_entropy_topk<double, double>(c, k).update_dual_variables(z.size(), 1., a.data(),
                                                                        z.data());
      EXPECT_NEAR(a[0] / expected, 1., 2e-12);
      for (std::size_t j = 1; j <= k; ++j)
        EXPECT_NEAR(-a[j] * static_cast<double>(k) / expected, 1., 2e-12);
      EXPECT_TRUE(std::isfinite(sdca::entropy_dual_loss(z.size(), std::size_t(1), c, a.data())));
    }
}

TEST(NumericalRegression, topk_cycle_exit_preserves_mass) {
  double z[] = {-1e16, 0, 1, 0, 2, 0, 0, 1}, a[8] = {0};
  sdca::l2_entropy_topk<double, double>(1e18, 5).update_dual_variables(8, 1., a, z);
  for (int j = 1; j < 8; ++j) EXPECT_LE(-a[j], (a[0] / 5) * (1 + 2e-13));
  // Quadratic-dominated solution; logarithmic corrections are below 1e-13
  // relative. Feasibility alone would also accept the all-zero block.
  EXPECT_NEAR(a[0] / 8.75e15, 1., 2e-13);
  for (int j = 1; j < 8; ++j) EXPECT_NEAR(-a[j] / 1.25e15, 1., 2e-13);
}

TEST(NumericalRegression, topk_all_capped_balanced_large_offset) {
  double z[] = {-1e16, 0, 0, 0}, a[4] = {0};
  sdca::l2_entropy_topk<double, double>(1e18, 3).update_dual_variables(4, 1., a, z);
  EXPECT_NEAR(a[0] / 7.5e15, 1., 2e-13);
  for (int j = 1; j < 4; ++j) EXPECT_NEAR(-a[j] / 2.5e15, 1., 2e-13);
}

#ifndef __FAST_MATH__
TEST(NumericalRegression, topk_all_capped_retains_mean_of_cancelling_scores) {
  for (double scale : {1e16, 1e30}) {
    double z[] = {0, scale, 1, -scale}, a[4] = {0}, base[] = {0, 0, 1, 0}, expected[4] = {0};
    sdca::l2_entropy_topk<double, double> objective(1, 3);
    objective.update_dual_variables(4, 1., a, z);
    objective.update_dual_variables(4, 1., expected, base);
    for (int j = 0; j < 4; ++j) EXPECT_NEAR(a[j], expected[j], 1e-14);
    EXPECT_NEAR(objective.primal_loss(4, z), objective.primal_loss(4, base), 1e-14);
    double v[] = {scale, 1, -scale};
    sdca::prox_topk_entropy(v, v + 3, 3);
    const double cap = 1 / (3 + std::exp(-1. / 3));
    for (double x : v) EXPECT_NEAR(x, cap, 1e-14);
  }
}
#endif

TEST(NumericalRegression, topk_free_threshold_beside_saturated_scores) {
  double z[] = {-1e16, 0, -1e15, -2e15, -3e15, -4e15, -5e15, -6e15}, a[8] = {0};
  sdca::l2_entropy_topk<double, double>(.001, 3).update_dual_variables(8, 1., a, z);
  EXPECT_NEAR(a[0] / .001, 1., 2e-13);
  for (int j = 1; j <= 3; ++j) EXPECT_NEAR(-a[j] * 3 / .001, 1., 2e-13);
  for (int j = 4; j < 8; ++j) EXPECT_EQ(a[j], 0.);
}

TEST(NumericalRegression, knapsack_saturated_scores_keep_small_free_mass) {
  double x[] = {1e16, 1e16, 1e16, .5, .3};
  sdca::prox_knapsack_eq(x, x + 5, 0., .3, 1.);
  EXPECT_NEAR(x[3], .1, 1e-15);
  EXPECT_EQ(x[4], 0.);
  EXPECT_NEAR(std::accumulate(x, x + 5, 0.), 1., 1e-15);
}

TEST(NumericalRegression, topk_clipping_does_not_rescale_free_mass_again) {
  double z[] = {0, 1e12, 0, -1e12}, a[4] = {0};
  sdca::l2_entropy_topk<double, double>(1., 2).update_dual_variables(4, 1., a, z);
  EXPECT_NEAR(a[0], 1., 1e-14);
  EXPECT_NEAR(a[1], -.5, 1e-14);
  EXPECT_NEAR(a[2], -.5, 1e-14);
  EXPECT_EQ(a[3], 0.);
}

TEST(NumericalRegression, two_simplex_small_mass_beside_huge_pair) {
  for (double gap : {1e8, 1e12, 1e15, 1e16}) {
    // Beside 1e16, 1 needs more than the 53 bits of double.
    if (gap > 1e15 && !wide_long_double()) continue;
    double a[] = {gap, 1, 0}, b[] = {gap, 0, 0};
    sdca::prox_two_simplex(a, a + 3, b, b + 3, 2 * gap);
    // Shared multiplier is 1/5: the huge pair cancels from its equation.
    EXPECT_NEAR(a[1], .8, 1e-15);
    EXPECT_EQ(a[2], 0.);
    EXPECT_NEAR(b[1], .2, 1e-15);
    EXPECT_NEAR(b[2], .2, 1e-15);
  }
}

#ifndef __FAST_MATH__
TEST(NumericalRegression, knapsack_inequality_retains_tiny_active_mass) {
  for (double scale : {1., 1e-20, 1e-100, 1e-300}) {
    double x[] = {1, 1, 1}, scratch[3];
    sdca::prox_knapsack_le(x, x + 3, scratch, 0., scale, scale);
    for (double value : x) EXPECT_NEAR(value / scale, 1. / 3, 2e-15);
    double inactive[] = {scale / 8, scale / 4, -scale};
    sdca::prox_knapsack_le(inactive, inactive + 3, 0., scale, scale);
    EXPECT_EQ(inactive[0], scale / 8);
    EXPECT_EQ(inactive[1], scale / 4);
    EXPECT_EQ(inactive[2], 0.);
    // The equality projection must not snap a small positive mass to zero.
    double equal[] = {1, 0};
    sdca::prox_knapsack_eq(equal, equal + 2, 0., 1., scale);
    EXPECT_NEAR(equal[0] / scale, 1., 2e-15);
    EXPECT_EQ(equal[1], 0.);
  }
  // Below zero the inequality still constrains the sum, and cancellation
  // in the box mass must not erase a small positive residual.
  for (double big : {1e15, 1e16}) {
    // Beside 1e16, 1 needs more than the 53 bits of double.
    if (big > 1e15 && !wide_long_double()) continue;
    double signed_x[] = {big, 1, -big};
    sdca::prox_knapsack_le(signed_x, signed_x + 3, -big, big, 0.);
    // The lower bound fixes the third entry; the free pair has multiplier .5.
    EXPECT_NEAR(signed_x[1], .5, 1e-15);
  }
}
#endif

#ifndef __FAST_MATH__
TEST(NumericalRegression, knapsack_validates_inactive_and_endpoint_inputs) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (double bad : {nan, inf, -inf})
    for (double rhs : {0., 1., 2.}) {
      double x[] = {bad, 0};
      EXPECT_THROW(sdca::prox_knapsack_eq(x, x + 2, 0., 1., rhs), std::invalid_argument);
      EXPECT_THROW(sdca::prox_knapsack_le(x, x + 2, 0., 1., rhs), std::invalid_argument);
    }
  double x[] = {0., 0.};
  EXPECT_THROW(sdca::prox_knapsack_le(x, x + 2, 1., 0., 1.), std::invalid_argument);
  EXPECT_THROW(sdca::prox_knapsack_le(x, x + 2, 1., 2., 1.), std::invalid_argument);
  EXPECT_THROW(sdca::prox_knapsack_le(x, x + 2, 0., 1., nan), std::invalid_argument);
  EXPECT_THROW(sdca::prox_knapsack_le(x, x, 0., 1., 1.), std::invalid_argument);
  // Unlike equality, a bound above the box's maximum sum is feasible.
  sdca::prox_knapsack_le(x, x + 2, 0., 1., 3.);
  EXPECT_EQ(x[0], 0.);
}
#endif

#ifndef __FAST_MATH__
TEST(NumericalRegression, tiny_smoothing_topk_hinge_loss_and_gradient) {
  for (double gamma : {1., 1e-10, 1e-20, 1e-100, 1e-300})
    for (std::size_t k : {1UL, 2UL, 3UL}) {
      for (auto kind : {sdca::loss_kind::topk_hinge_beta, sdca::loss_kind::topk_hinge_alpha}) {
        double z[] = {0, 0, 0, 0}, g[4];
        const double value = sdca::loss_and_gradient(kind, 4, 1, z, g, k, gamma);
        EXPECT_NEAR(value, 1. - gamma / 6, 2e-15);
        EXPECT_NEAR(g[0], -1., 2e-15);
        for (int j = 1; j < 4; ++j) EXPECT_NEAR(g[j], 1. / 3, 2e-15);
      }
    }
}

template <typename T>
bool same_bits(T a, T b) {
  return (a == b && std::signbit(a) == std::signbit(b)) || (std::isnan(a) && std::isnan(b));
}

// The top-1 hinge as single_target_loss computed it before hinge_label: every competitor's
// compensated margin in T, topk_hinge_loss at k = 1, the target entry a compensated sum.
template <typename T, typename Data>
double full_hinge_row(std::size_t n, std::size_t label, const Data* z, Data* gradient) {
  std::vector<T> h, a(n - 1, T(0));
  for (std::size_t j = 0; j < n; ++j)
    if (j != label) h.push_back(sdca::detail::margin(T(z[j]), T(z[label])));
  const double value = sdca::detail::topk_hinge_loss(false, n - 1, h.data(), 1, a.data());
  sdca::detail::compensated_sum<double> sum;
  for (std::size_t j = 0, q = 0; j < n; ++j) {
    if (j == label) continue;
    sum.add(double(a[q]));
    gradient[j] = Data(a[q++]);
  }
  gradient[label] = Data(-sum.value());
  return value;
}

template <typename T>
void expect_same_row(double value, double reference, const std::vector<T>& g,
                     const std::vector<T>& expected) {
  EXPECT_TRUE(same_bits(value, reference)) << value << " " << reference;
  for (std::size_t j = 0; j < g.size(); ++j)
    EXPECT_TRUE(same_bits(g[j], expected[j])) << "j=" << j << " " << g[j] << " " << expected[j];
}

// hinge_label_loss in place (and loss_and_gradient with the label first) equals the full
// row, bit for bit (signed zeros included), on tied, offset, near-tied, cancelling,
// nonfinite and float-overflowing scores, in double and (float data) float arithmetic.
template <typename Data>
void check_hinge_label_rows() {
  std::mt19937 rng(7);
  const double big = double(std::numeric_limits<float>::max()) / 2;
  const double eps = double(std::numeric_limits<Data>::epsilon());
  const double specials[] = {std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity()};
  for (std::size_t n : {2UL, 3UL, 17UL, 100UL})
    for (double offset : {0., 1e8, -1e30, 1e16, big})
      for (int trial = 0; trial < 40; ++trial) {
        std::vector<Data> z(n), first(n), g(n), expected(n);
        const std::size_t label = std::size_t(trial) % n;
        for (std::size_t j = 0; j < n; ++j)
          z[j] = Data(offset + double(std::uniform_int_distribution<int>(-3, 3)(rng)) / 2);
        if (trial % 8 == 3) z[(label + 1) % n] = Data(-2 * big);
        if (trial % 8 == 5)
          for (std::size_t j = 0; j < n; ++j)
            z[j] = Data(double(z[label]) - 1 + std::ldexp(double(j % 3), -45));
        if (trial % 8 == 6)
          for (std::size_t j = 0; j < n; ++j) z[j] = Data(offset * (1 + eps * double(j % 4)));
        if (trial % 8 == 7) z[(label + std::size_t(trial)) % n] = Data(specials[trial / 8 % 3]);
        SCOPED_TRACE("n=" + std::to_string(n) + " offset=" + std::to_string(offset) +
                     " trial=" + std::to_string(trial));
        const double reference = full_hinge_row<double>(n, label, z.data(), expected.data());
        EXPECT_TRUE(same_bits(sdca::hinge_label_loss(n, label, z.data()), reference));
        expect_same_row(sdca::hinge_label_loss(n, label, z.data(), g.data()), reference, g,
                        expected);
        first[0] = z[label];
        for (std::size_t j = 0, q = 1; j < n; ++j)
          if (j != label) first[q++] = z[j];
        std::vector<Data> permuted(n);
        for (auto kind : {sdca::loss_kind::multiclass_hinge, sdca::loss_kind::topk_hinge_beta}) {
          const double value = sdca::loss_and_gradient(kind, n, 1, first.data(), permuted.data());
          for (std::size_t j = 0, q = 1; j < n; ++j)
            g[j] = j == label ? permuted[0] : permuted[q++];
          expect_same_row(value, reference, g, expected);
        }
        if (!std::is_same<Data, float>::value) continue;
        const bool fits = sdca::detail::loss_fits<float>(n, z.data(), 0, 1);
        const double narrow = fits ? full_hinge_row<float>(n, label, z.data(), expected.data())
                                   : full_hinge_row<double>(n, label, z.data(), expected.data());
        expect_same_row(sdca::hinge_label_loss<Data, float>(n, label, z.data(), g.data()), narrow,
                        g, expected);
      }
}

TEST(NumericalRegression, hinge_label_rows_match_label_first_losses) {
  check_hinge_label_rows<double>();
  check_hinge_label_rows<float>();
}

TEST(NumericalRegression, hinge_label_keeps_wider_scores_in_pruning_bound) {
  using L = long double;
  if (std::numeric_limits<L>::digits <= std::numeric_limits<double>::digits) return;
  // The largest score rounds upward in double. A double pruning floor can then exceed
  // every score and leave the first (smaller) competitor selected.
  const std::vector<L> z = {0.L, 0.L, 1.L + 3 * std::ldexp(1.L, -54)};
  std::vector<L> g(z.size()), expected(z.size());
  const double reference = full_hinge_row<L>(z.size(), 0, z.data(), expected.data());
  EXPECT_EQ(expected[1], 0.L);
  EXPECT_EQ(expected[2], 1.L);
  EXPECT_EQ(sdca::hinge_label_loss(z.size(), 0, z.data()), reference);
  expect_same_row(sdca::hinge_label_loss(z.size(), 0, z.data(), g.data()), reference, g, expected);
  const double value =
      sdca::loss_and_gradient(sdca::loss_kind::multiclass_hinge, z.size(), 1, z.data(), g.data());
  expect_same_row(value, reference, g, expected);
}
#endif

TEST(NumericalRegression, knapsack_multistep_tiny_equality) {
  for (double scale : {1e-20, 1e-100, 1e-290, 1e-300}) {
    for (bool equality : {false, true}) {
      std::vector<double> input = {0., scale, 2 * scale};
      do {
        auto x = input;
        if (equality)
          sdca::prox_knapsack_eq(x.begin(), x.end(), 0., 1., scale / 2);
        else
          sdca::prox_knapsack_le(x.begin(), x.end(), 0., 1., scale / 2);
        for (std::size_t j = 0; j < x.size(); ++j)
          EXPECT_NEAR(x[j] / scale, input[j] == 2 * scale ? .5 : 0., 2e-15);
      } while (std::next_permutation(input.begin(), input.end()));
    }
  }
}

TEST(NumericalRegression, knapsack_rounded_endpoints_preserve_interior) {
  for (int n : {3, 7, 10}) {
    const double hi = 1. / n;
    std::vector<double> x(n, 1.);
    sdca::prox_knapsack_eq(x.begin(), x.end(), 0., hi, 1.);
    for (double value : x) EXPECT_NEAR(value, hi, 2e-16);
    std::fill(x.begin(), x.end(), -1.);
    sdca::prox_knapsack_eq(x.begin(), x.end(), -hi, 0., -1.);
    for (double value : x) EXPECT_NEAR(value, -hi, 2e-16);
    std::fill(x.begin(), x.end(), -1.);
    sdca::prox_knapsack_le(x.begin(), x.end(), -hi, 0., -1.);
    for (double value : x) EXPECT_NEAR(value, -hi, 2e-16);
  }
  // Values inside the box, even one representable step from a bound,
  // are valid equality targets, not rounding mistakes to clamp away.
  const double below = std::nextafter(1., 0.);
  const double above = std::nextafter(1., 2.);
  double upper[] = {1., 1.}, lower[] = {1., 1.};
  sdca::prox_knapsack_eq(upper, upper + 2, 0., 1., 2 * below);
  sdca::prox_knapsack_eq(lower, lower + 2, 1., 2., 2 * above);
  for (double value : upper) EXPECT_EQ(value, below);
  for (double value : lower) EXPECT_EQ(value, above);
  double bad[] = {0., 0.};
  EXPECT_THROW(sdca::prox_knapsack_eq(bad, bad + 2, 0., 1., 2. + 1e-12), std::invalid_argument);
  EXPECT_THROW(sdca::prox_knapsack_le(bad, bad + 2, 1., 2., 2. - 1e-12), std::invalid_argument);
}

TEST(NumericalRegression, knapsack_extreme_scores_and_float_storage) {
  for (double gap : {1e16, 1e30, 1e300, 1e308}) {
    double x[] = {gap, 0., -gap};
    sdca::prox_knapsack_eq(x, x + 3, 0., 1., 1.5);
    EXPECT_EQ(x[0], 1.);
    EXPECT_EQ(x[1], .5);
    EXPECT_EQ(x[2], 0.);
  }
  float x[] = {2e-20f, 1e-20f, 0.f};
  sdca::prox_knapsack_eq<float>(x, x + 3, 0.f, 1.f, .5e-20f);
  EXPECT_EQ(x[0], .5e-20f);
  EXPECT_EQ(x[1], 0.f);
  EXPECT_EQ(x[2], 0.f);
  // Equal box bounds are valid only at the corresponding total mass.
  double fixed[] = {1., -1.};
  sdca::prox_knapsack_eq(fixed, fixed + 2, .25, .25, .5);
  EXPECT_EQ(fixed[0], .25);
  EXPECT_EQ(fixed[1], .25);
}

TEST(NumericalRegression, topk_threshold_preserves_logmass_separation) {
  const double v[] = {0., 1., 2.};
  double out[3], normalizer = 0., upper = 0.;
  // Even when all probabilities underflow, the limiting threshold must
  // retain its O(1) part rather than acquiring a false cap from log(s).
  sdca::topk_entropy_block(3, v, 2, 1., out, &normalizer, &upper, -1e16);
  for (double value : out) EXPECT_EQ(value, 0.);
  EXPECT_EQ(upper, 0.);
  EXPECT_TRUE(std::isfinite(normalizer));
  // In the underflow limit, logit(s) = offset + log(2) +
  // (2 + log(1+exp(1)))/2, with one capped competitor.
  const long double r = std::log(2.L) + (2.L + std::log1p(std::exp(1.L))) / 2;
  for (double offset : {-710., -740., -750.}) {
    try {
      sdca::topk_entropy_block(3, v, 2, 1., out, &normalizer, &upper, offset, 1e200);
    } catch (const std::runtime_error&) {
      // A subnormal total mass needs the exponent range of the wider retry.
      EXPECT_FALSE(wide_long_double()) << "offset " << offset;
      continue;
    }
    const long double mass = std::exp(std::log(1e200L) + static_cast<long double>(offset) + r);
    const long double free0 = mass / (2 * (1 + std::exp(1.L)));
    EXPECT_NEAR(out[0] / static_cast<double>(free0), 1., 3e-13);
    EXPECT_NEAR(out[1] / static_cast<double>(free0 * std::exp(1.L)), 1., 3e-13);
    EXPECT_NEAR(out[2] / static_cast<double>(mass / 2), 1., 3e-13);
  }
}

#include "regression_solver.h"
