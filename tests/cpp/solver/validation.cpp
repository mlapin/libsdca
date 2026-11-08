// Regression tests of the solver's input checks, step failures and evaluation.
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#include "sdca/solver.h"
#include "test_util.h"

namespace {

// An objective whose coordinate step throws an invalid_argument.
template <typename Data, typename Result>
struct throwing_entropy : sdca::l2_entropy<Data, Result> {
  throwing_entropy() : sdca::l2_entropy<Data, Result>(1) {}
  template <typename Int>
  void update_dual_variables(Int, Data, Data*, Data*) const {
    throw std::invalid_argument("step rejected");
  }
};

}  // namespace

TEST(SolverValidation, any_failed_step_ends_with_failed_and_an_evaluation) {
  sdca::logging::set_level(sdca::logging::level::none);
  double x[] = {1, 2, 0, 1}, a[] = {0, 0, 0, 0}, w[4] = {};
  std::vector<sdca::size_type> labels = {0, 1};
  auto ctx =
      sdca::make_context(sdca::make_input_feature(2, 2, x), sdca::multiclass_output(2, labels),
                         throwing_entropy<double, double>(), a, w);
  ctx.criteria.max_epoch = 3;
  EXPECT_NO_THROW(sdca::make_solver(ctx).solve());
  EXPECT_EQ(sdca::solver_status::failed, ctx.status);
  ASSERT_EQ(1u, ctx.train.evals.size());
  EXPECT_EQ(0u, ctx.train.evals.back().epoch);
  for (double v : a) EXPECT_EQ(0, v);
}

TEST(SolverValidation, step_scale_is_checked_before_solving) {
  // C * norm2 = 1e10 * 1e36 overflows float storage.
  float x[] = {1e18f}, a[2] = {}, w[2] = {};
  std::vector<sdca::size_type> labels = {0};
  auto ctx =
      sdca::make_context(sdca::make_input_feature(1, 1, x), sdca::multiclass_output(2, labels),
                         sdca::l2_entropy<float, double>(1e10), a, w);
  EXPECT_THROW(sdca::make_solver(ctx).solve(), std::invalid_argument);
  // C itself must fit the data type: a zero row takes a block of size C.
  float zero[] = {0};
  std::vector<sdca::size_type> zero_labels = {0};
  auto big = sdca::make_context(sdca::make_input_feature(1, 1, zero),
                                sdca::multiclass_output(2, zero_labels),
                                sdca::l2_entropy<float, double>(1e300), a, w);
  EXPECT_THROW(sdca::make_solver(big).solve(), std::invalid_argument);
}

TEST(SolverValidation, top_k_must_suit_the_classes_at_construction) {
  double x[] = {1}, a[2] = {}, w[2] = {};
  std::vector<sdca::size_type> labels = {0};
  EXPECT_THROW(
      sdca::make_context(sdca::make_input_feature(1, 1, x), sdca::multiclass_output(2, labels),
                         sdca::l2_topk_hinge<double, double>(1, 2), a, w),
      std::invalid_argument);
}

TEST(SolverValidation, training_kernels_must_be_symmetric_with_a_valid_diagonal) {
  std::vector<sdca::size_type> labels = {0, 1};
  auto solve = [&](std::vector<double> K) {
    std::vector<double> a(4, 0);
    std::vector<sdca::size_type> y = labels;
    auto ctx =
        sdca::make_context(sdca::make_input_kernel(2, K.data()), sdca::multiclass_output(2, y),
                           sdca::l2_entropy<double, double>(1), a.data());
    ctx.criteria.max_epoch = 2;
    sdca::make_solver(ctx).solve();
    return ctx.status;
  };
  EXPECT_THROW(solve({1, 0.5, 0.4, 1}), std::invalid_argument);  // asymmetric
  EXPECT_THROW(solve({-1, 0, 0, 1}), std::invalid_argument);     // negative diagonal
  EXPECT_THROW(solve({1, 2, 2, 1}), std::invalid_argument);      // |K_01| > 1
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(solve({1, nan, nan, 1}), std::invalid_argument);               // not finite
  EXPECT_THROW(solve({0, 1e-300, 1e-300, 1}), std::invalid_argument);         // zero row, K_01 != 0
  EXPECT_NE(sdca::solver_status::failed, solve({1, 1 + 1e-9, 1 + 1e-9, 1}));  // rounding
  EXPECT_NE(sdca::solver_status::failed, solve({0, 0, 0, 1}));
  // Larger than a block of the check, with one asymmetric pair far from the diagonal.
  const std::size_t n = 150;
  std::vector<double> K(n * n, 0);
  for (std::size_t i = 0; i < n; ++i) K[n * i + i] = 1;
  K[n * 140 + 3] = 0.5;
  std::vector<double> a(2 * n, 0);
  std::vector<sdca::size_type> y(n, 0);
  y[1] = 1;
  auto ctx = sdca::make_context(sdca::make_input_kernel(n, K.data()), sdca::multiclass_output(2, y),
                                sdca::l2_entropy<double, double>(1), a.data());
  for (std::size_t threads : {1, 4}) {
    ctx.criteria.num_threads = threads;
    EXPECT_THROW(sdca::make_solver(ctx).solve(), std::invalid_argument);
  }
}

TEST(SolverValidation, kernel_primal_is_at_the_stored_duals) {
  // The stored block [0.5, -0.4] does not sum to zero; the dual repairs it
  // to [0.4, -0.4], the primal (the model a kernel fit returns) keeps it.
  double K[] = {1}, a[] = {0.5, -0.4};
  std::vector<sdca::size_type> labels = {0};
  auto ctx = sdca::make_context(sdca::make_input_kernel(1, K), sdca::multiclass_output(2, labels),
                                sdca::l2_entropy<double, double>(1), a);
  sdca::solver_scratch<double, sdca::kernel_input> scratch;
  sdca::evaluate_dataset(ctx, ctx.train, scratch);
  const auto& e = ctx.train.evals.back();
  EXPECT_DOUBLE_EQ(0.205, e.primal_regularizer);
  EXPECT_DOUBLE_EQ(0.16, e.dual_regularizer);
  EXPECT_DOUBLE_EQ(std::log1p(std::exp(-0.9)), e.primal_loss);
}

TEST(SolverValidation, evaluations_are_identical_for_every_thread_count) {
  const std::size_t n = 3000, d = 200, m = 7;
  std::mt19937 gen(5);
  std::normal_distribution<double> normal(0, 1);
  std::vector<float> X(n * d);
  for (auto& v : X) v = float(normal(gen));
  std::vector<sdca::size_type> y(n);
  for (std::size_t i = 0; i < n; ++i) y[i] = i % m;
  std::vector<float> K(n * n / 9);  // a kernel on the first n / 3 examples
  const std::size_t nk = n / 3;
  for (std::size_t i = 0; i < nk; ++i)
    for (std::size_t l = 0; l < nk; ++l) {
      double s = 0;
      for (std::size_t j = 0; j < d; ++j) s += double(X[i * d + j]) * double(X[l * d + j]);
      K[i * nk + l] = float(s);
    }
  for (int prox = 0; prox < 2; ++prox) {
    std::vector<std::vector<double>> runs;
    for (std::size_t threads : {1, 3, 0}) {
      std::vector<float> A(n * m, 0), W(d * m, 0), W0(d * m, 0.25f), B(nk * m, 0);
      if (prox) W = W0;
      std::vector<sdca::size_type> labels = y, test_labels = y,
                                   kernel_labels(y.begin(), y.begin() + nk);
      auto ctx = sdca::make_context(sdca::make_input_feature(d, n, X.data()),
                                    sdca::multiclass_output(m, labels),
                                    sdca::l2_hinge_topk<float, double>(1, 2), A.data(), W.data());
      if (prox) ctx.primal_initial = W0.data();
      ctx.criteria.max_epoch = 2;
      ctx.criteria.eval_epoch = 1;
      ctx.criteria.num_threads = threads;
      ctx.add_test(sdca::make_input_feature(d, n, X.data()),
                   sdca::multiclass_output(m, test_labels));
      sdca::make_solver(ctx).solve();
      auto kctx = sdca::make_context(sdca::make_input_kernel(nk, K.data()),
                                     sdca::multiclass_output(m, kernel_labels),
                                     sdca::l2_entropy<float, double>(1), B.data());
      kctx.criteria.max_epoch = 2;
      kctx.criteria.eval_epoch = 1;
      kctx.criteria.num_threads = threads;
      sdca::make_solver(kctx).solve();
      std::vector<double> run(W.begin(), W.end());
      for (const auto* evals : {&ctx.train.evals, &kctx.train.evals})
        for (const auto& e : *evals) {
          run.insert(run.end(), {e.primal, e.dual, e.primal_loss, e.dual_loss, e.primal_regularizer,
                                 e.dual_regularizer, e.gap_error});
          run.insert(run.end(), e.accuracy.begin(), e.accuracy.end());
        }
      for (const auto& e : ctx.test[0].evals) run.push_back(e.primal_loss);
      runs.push_back(run);
    }
    EXPECT_EQ(runs[0], runs[1]);
    EXPECT_EQ(runs[0], runs[2]);
  }
}
