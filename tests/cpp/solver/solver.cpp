#include "sdca/solver.h"
#include "test_util.h"

template <typename Data, typename Result, typename Context>
inline void test_solver_check_converged(Context& ctx) {
  // The certificate adds the evaluation's rounding bound (about 1e-13 here).
  ctx.criteria.epsilon =
      std::max(1e-11, 64 * std::max(static_cast<double>(std::numeric_limits<Data>::epsilon()),
                                    static_cast<double>(std::numeric_limits<Result>::epsilon())));

  ctx.criteria.eval_epoch = 2;
  auto solver = sdca::make_solver(ctx);
  solver.solve();

  EXPECT_TRUE(ctx.status == sdca::solver_status::solved);
  ASSERT_FALSE(ctx.train.evals.empty());
  const auto& eval = ctx.train.evals.back();
  EXPECT_TRUE(std::isfinite(eval.primal) && std::isfinite(eval.dual));
  EXPECT_GE(eval.primal + eval.gap_error, eval.dual);
  const double scale = std::max(std::abs(double(eval.primal)), std::abs(double(eval.dual)));
  EXPECT_LE(double(eval.primal - eval.dual) + eval.gap_error, ctx.criteria.epsilon * scale);
}

template <typename Data, typename Result, template <typename> class Input,
          template <typename, typename> class Objective>
inline void test_solver_check_performance(
    const sdca::solver_context<Data, Result, Input, sdca::multiclass_output, Objective>& ctx,
    const Result expected_accuracy) {
  const Result accuracy = ctx.train.evals.back().accuracy[0];
  if (accuracy != expected_accuracy) {
    std::printf("%s\n", ctx.to_string().c_str());
    std::printf("%s\n", ctx.status_string().c_str());
  }
  EXPECT_EQ(expected_accuracy, accuracy);
}

template <typename Data, typename Result, template <typename> class Input,
          template <typename, typename> class Objective>
inline void test_solver_check_performance(
    const sdca::solver_context<Data, Result, Input, sdca::multilabel_output, Objective>& ctx,
    const Result expected_accuracy) {
  const Result accuracy = 1 - ctx.train.evals.back().rank_loss;
  if (accuracy != expected_accuracy) {
    std::printf("%s\n", ctx.to_string().c_str());
    std::printf("%s\n", ctx.status_string().c_str());
  }
  EXPECT_EQ(expected_accuracy, accuracy);
}

template <typename Data, typename Result, typename Output,
          template <typename, typename> class Objective>
inline void test_solver_basic_tests_feature_in(Output out, Objective<Data, Result> objective,
                                               const sdca::size_type n, const std::vector<Data>& X,
                                               const Result expected_performance) {
  sdca::size_type dn = X.size(), m = out.num_classes;
  sdca::size_type d = dn / n;
  std::vector<Data> W(d * m);  // primal vars
  std::vector<Data> A(m * n);  // dual vars

  // Use features
  auto ctx_feat = sdca::make_context(sdca::make_input_feature(d, n, &X[0]), std::move(out),
                                     std::move(objective), &A[0], &W[0]);

  test_solver_check_converged<Data, Result>(ctx_feat);
  test_solver_check_performance(ctx_feat, expected_performance);

  // Kernel (Gram) matrix
  std::vector<Data> K(n * n);
  sdca::blas_int D = static_cast<sdca::blas_int>(d);
  sdca::blas_int N = static_cast<sdca::blas_int>(n);
  sdca::sdca_blas_gemm(N, N, D, &X[0], D, &X[0], D, &K[0], CblasTrans);

  // Warm restart - use the same dual variables as above
  auto ctx_warm =
      sdca::make_context(sdca::make_input_kernel(n, &K[0]), std::move(ctx_feat.train.out),
                         std::move(ctx_feat.objective), &A[0]);

  ctx_warm.criteria.eval_on_start = true;
  test_solver_check_converged<Data, Result>(ctx_warm);
  test_solver_check_performance(ctx_warm, expected_performance);
  EXPECT_TRUE(ctx_warm.epoch <= ctx_warm.criteria.eval_epoch);

  // Prox restart around half the previous solution. Centering at the full
  // hinge solution can make the true centered optimum zero, where a relative
  // certificate is below evaluation rounding; this tests actual convergence
  // at a nonzero objective instead.
  std::vector<Data> W0(W);  // initial primal vars
  for (sdca::size_type j = 0; j < d * m; ++j) W[j] = W0[j] *= Data(0.5);
  auto ctx_prox =
      sdca::make_context(sdca::make_input_feature(d, n, &X[0]), std::move(ctx_warm.train.out),
                         std::move(ctx_warm.objective), &A[0], &W[0]);

  // Note: the solver expects that W = W0 before the first iteration;
  // therefore, the W is not reset to 0
  std::fill(A.begin(), A.end(), static_cast<Data>(0));  // reset to 0
  ctx_prox.primal_initial = &W0[0];
  ctx_prox.criteria.eval_on_start = true;
  test_solver_check_converged<Data, Result>(ctx_prox);
  test_solver_check_performance(ctx_prox, expected_performance);

  // Zero the dual variables and train from scratch
  std::fill(A.begin(), A.end(), static_cast<Data>(0));
  auto ctx_ker =
      sdca::make_context(sdca::make_input_kernel(n, &K[0]), std::move(ctx_prox.train.out),
                         std::move(ctx_prox.objective), &A[0]);

  ctx_ker.criteria.eval_on_start = true;
  test_solver_check_converged<Data, Result>(ctx_ker);
  test_solver_check_performance(ctx_ker, expected_performance);
  EXPECT_TRUE(ctx_ker.epoch > 0UL);
}

template <typename Data, typename Result, typename OutputMaker,
          template <typename, typename> class Objective>
inline void test_solver_multiclass_basic(OutputMaker make_output,
                                         Objective<Data, Result> objective) {
  sdca::size_type d = 3, n = 4;
  std::vector<Data> X(d * n);         // features
  std::vector<sdca::size_type> Y(n);  // labels
  Result accuracy = 1;

  // Last column is the offset (bias) feature
  X = {0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1};

  // OR
  Y = {0, 1, 1, 1};
  test_solver_basic_tests_feature_in(make_output(Y), objective, n, X, accuracy);

  // AND
  Y = {0, 0, 0, 1};
  test_solver_basic_tests_feature_in(make_output(Y), objective, n, X, accuracy);

  // XOR
  // Perturb the last point to break symmetry
  // (otherwise test fails due to (?) regularized bias term)
  Data eps = static_cast<Data>(0.15);
  X = {0, 0, 1, 0, 1, 1, 1, 0, 1, 1 - eps, 1 - eps, 1};
  Y = {0, 1, 1, 0};
  accuracy = static_cast<Result>(0.75);
  test_solver_basic_tests_feature_in(make_output(Y), objective, n, X, accuracy);
}

template <typename Data, typename Result, template <typename, typename> class Objective>
inline void test_solver_multilabel_basic(Objective<Data, Result> objective) {
  sdca::size_type d = 3, n = 5;
  std::vector<Data> X(d * n);                   // features
  std::vector<std::vector<sdca::size_type>> Y;  // labels
  Result accuracy = 1;

  // Last column is the offset (bias) feature
  X = {0, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 0, 1};

  Y = {{0}, {0, 1}, {1}, {1, 2}, {2}};
  test_solver_basic_tests_feature_in(sdca::make_output_multilabel(Y), objective, n, X, accuracy);
}

template <typename Data, typename Result>
inline void test_solver_multiclass_basic_all() {
  Result C = 4;
  auto multiclass_output_maker = [](std::vector<sdca::size_type>& Y) {
    return sdca::make_output_multiclass(Y.begin(), Y.end());
  };

  test_solver_multiclass_basic(multiclass_output_maker,
                               sdca::make_objective_l2_entropy<Data, Result>(C));

  test_solver_multiclass_basic(multiclass_output_maker,
                               sdca::make_objective_l2_entropy_topk<Data, Result>(C));

  test_solver_multiclass_basic(multiclass_output_maker,
                               sdca::make_objective_l2_hinge_topk<Data, Result>(C));

  test_solver_multiclass_basic(multiclass_output_maker,
                               sdca::make_objective_l2_hinge_topk_smooth<Data, Result>(C));

  test_solver_multiclass_basic(multiclass_output_maker,
                               sdca::make_objective_l2_topk_hinge<Data, Result>(C));

  test_solver_multiclass_basic(multiclass_output_maker,
                               sdca::make_objective_l2_topk_hinge_smooth<Data, Result>(C));

  auto multilabel_output_maker = [](std::vector<sdca::size_type>& Y) {
    return sdca::make_output_multilabel(Y.begin(), Y.end());
  };

  test_solver_multiclass_basic(multilabel_output_maker,
                               sdca::make_objective_l2_multilabel_entropy<Data, Result>(C));

  test_solver_multiclass_basic(multilabel_output_maker,
                               sdca::make_objective_l2_multilabel_hinge<Data, Result>(C));

  test_solver_multiclass_basic(multilabel_output_maker,
                               sdca::make_objective_l2_multilabel_hinge_smooth<Data, Result>(C));
}

template <typename Data, typename Result>
inline void test_solver_multilabel_basic_all() {
  Result C = 4;
  test_solver_multilabel_basic(sdca::make_objective_l2_multilabel_entropy<Data, Result>(C));

  test_solver_multilabel_basic(sdca::make_objective_l2_multilabel_hinge<Data, Result>(C));

  test_solver_multilabel_basic(sdca::make_objective_l2_multilabel_hinge_smooth<Data, Result>(C));
}

TEST(SolverTest, multiclass_basic_problems_all_objectives) {
  sdca::logging::set_level(sdca::logging::level::warning);
  sdca::logging::set_format(sdca::logging::format::short_e);
  test_solver_multiclass_basic_all<float, float>();
  test_solver_multiclass_basic_all<float, double>();
  test_solver_multiclass_basic_all<double, float>();
  test_solver_multiclass_basic_all<double, double>();
}

// Evaluations off the eval_epoch schedule: at most two early ones (one when
// max_epoch is not a multiple of eval_epoch, one less with eval_on_start),
// plus an end evaluation at such a max_epoch; with the start, at most
// max_epoch / eval_epoch + 2 rows. warm: the duals of three epochs start it.
template <typename Data>
static void test_early_evaluations(const bool on_start, const sdca::size_type max_epoch,
                                   const double tol, const bool warm) {
  const sdca::size_type n = 400, d = 12, m = 5, every = 10;
  std::mt19937 gen(11);
  std::normal_distribution<double> noise(0, 1);
  std::vector<Data> X(d * n), W(d * m), A(m * n);
  std::vector<sdca::size_type> Y(n);
  for (sdca::size_type i = 0; i < n; ++i) {
    Y[i] = i % m;
    for (sdca::size_type j = 0; j < d; ++j) X[d * i + j] = Data(noise(gen) + (j == Y[i] ? 1.5 : 0));
  }
  auto make = [&]() {
    return sdca::make_context(sdca::make_input_feature(d, n, &X[0]),
                              sdca::make_output_multiclass(Y.begin(), Y.end()),
                              sdca::make_objective_l2_entropy<Data, Data>(Data(1)), &A[0], &W[0]);
  };
  if (warm) {
    auto start = make();
    start.criteria.max_epoch = 3;
    start.criteria.epsilon = 0;
    sdca::make_solver(start).solve();
  }
  auto ctx = make();
  ctx.criteria.eval_epoch = every;
  ctx.criteria.max_epoch = max_epoch;
  ctx.criteria.epsilon = tol;
  ctx.criteria.eval_on_start = on_start;
  sdca::make_solver(ctx).solve();
  sdca::size_type off = 0;
  for (const auto& e : ctx.train.evals) off += e.epoch % every != 0;
  const sdca::size_type early = tol > 0 ? (max_epoch % every == 0 ? 2 : 1) - (on_start ? 1 : 0) : 0,
                        end = max_epoch % every != 0 ? 1 : 0;
  const std::string where = "eval_on_start " + std::to_string(on_start) + " max_epoch " +
                            std::to_string(max_epoch) + " tol " + std::to_string(tol) + " warm " +
                            std::to_string(warm);
  EXPECT_LE(off, early + end) << where;
  EXPECT_LE(ctx.train.evals.size(), max_epoch / every + 2) << where;
}

TEST(SolverTest, early_evaluations_stay_within_the_history) {
  sdca::logging::set_level(sdca::logging::level::warning);
  for (bool warm : {false, true})
    for (bool on_start : {false, true})
      for (sdca::size_type max_epoch : {0UL, 5UL, 25UL, 30UL, 1000UL})
        for (double tol : {0., 1e-2, 1e-3, 1e-4, 1e-6}) {
          test_early_evaluations<double>(on_start, max_epoch, tol, warm);
          test_early_evaluations<float>(on_start, max_epoch, tol, warm);
        }
}

TEST(SolverTest, early_evaluation_stops_before_the_schedule) {
  sdca::logging::set_level(sdca::logging::level::warning);
  // The gap's trend from the zero start: the stop comes before the next multiple of 10.
  const sdca::size_type n = 400, d = 12, m = 5;
  std::mt19937 gen(11);
  std::normal_distribution<double> noise(0, 1);
  std::vector<double> X(d * n), W(d * m), A(m * n);
  std::vector<sdca::size_type> Y(n);
  for (sdca::size_type i = 0; i < n; ++i) {
    Y[i] = i % m;
    for (sdca::size_type j = 0; j < d; ++j) X[d * i + j] = noise(gen) + (j == Y[i] ? 1.5 : 0);
  }
  std::vector<double> gaps;
  for (sdca::size_type every : {1UL, 10UL}) {
    std::fill(W.begin(), W.end(), 0.);
    std::fill(A.begin(), A.end(), 0.);
    auto ctx = sdca::make_context(
        sdca::make_input_feature(d, n, &X[0]), sdca::make_output_multiclass(Y.begin(), Y.end()),
        sdca::make_objective_l2_entropy<double, double>(1.), &A[0], &W[0]);
    ctx.criteria.eval_epoch = every;
    ctx.criteria.epsilon = every == 1 ? 0. : 1.01 * gaps.at(14);
    ctx.criteria.max_epoch = 40;
    sdca::make_solver(ctx).solve();
    if (every == 1) {
      for (const auto& e : ctx.train.evals)
        gaps.push_back((e.primal - e.dual + e.gap_error) /
                       std::max(std::abs(e.primal), std::abs(e.dual)));
      continue;
    }
    EXPECT_EQ(ctx.status, sdca::solver_status::solved);
    EXPECT_LT(ctx.epoch, 20UL);
    EXPECT_GE(ctx.epoch, 14UL);
  }
}

// Shrinking leaves out steps; the gap is still evaluated over all examples,
// so every mode certifies the same tolerance as one that updates every class
// of every example in every epoch, and gives the same bits run after run.
// skips: whether the objective must skip steps (1), must not (0), or either
// (-1) with shrinking on.
template <template <typename, typename> class Objective>
static void test_shrinking(const Objective<double, double>& objective, const int skips) {
  const sdca::size_type n = 400, d = 12, m = 5;
  std::mt19937 gen(11);
  std::normal_distribution<double> noise(0, 1);
  std::vector<double> X(d * n);
  std::vector<sdca::size_type> Y(n);
  for (sdca::size_type i = 0; i < n; ++i) {
    Y[i] = i % m;
    for (sdca::size_type j = 0; j < d; ++j) X[d * i + j] = noise(gen) + (j == Y[i] ? 3 : 0);
  }
  const std::string where = objective.to_string();
  std::vector<double> primal;
  for (const auto mode :
       {sdca::shrinking_mode::off, sdca::shrinking_mode::heuristic, sdca::shrinking_mode::safe,
        sdca::shrinking_mode::both, sdca::shrinking_mode::automatic}) {
    const std::string what = where + " " + sdca::shrinking_mode_name(mode);
    std::vector<double> first_A;
    for (int run = 0; run < 2; ++run) {
      std::vector<double> W(d * m), A(m * n);
      auto ctx = sdca::make_context(sdca::make_input_feature(d, n, &X[0]),
                                    sdca::make_output_multiclass(Y.begin(), Y.end()),
                                    Objective<double, double>(objective), &A[0], &W[0]);
      ctx.criteria.shrinking = mode;
      ctx.criteria.epsilon = 1e-6;
      ctx.criteria.max_epoch = 20000;
      ctx.criteria.num_threads = run == 0 ? 1 : 3;
      auto solver = sdca::make_solver(ctx);
      solver.solve();
      if (run == 1) {
        EXPECT_TRUE(A == first_A) << what << ": not deterministic over thread counts";
        continue;
      }
      first_A = A;
      EXPECT_EQ(ctx.status, sdca::solver_status::solved) << what;
      ASSERT_FALSE(ctx.train.evals.empty());
      const auto& last = ctx.train.evals.back();
      if (mode == sdca::shrinking_mode::off || skips == 0) {
        EXPECT_EQ(solver.num_skipped(), 0UL) << what;
        EXPECT_EQ(solver.num_screened_classes(), 0UL) << what;
      }
      if (mode != sdca::shrinking_mode::off && mode != sdca::shrinking_mode::safe && skips == 1) {
        EXPECT_GT(solver.num_skipped(), 0UL) << what;
      }
      EXPECT_GT(solver.num_updates(), 0UL) << what;

      // A fresh evaluation of the final iterate gives the reported objectives.
      auto check = sdca::make_context(sdca::make_input_feature(d, n, &X[0]),
                                      sdca::make_output_multiclass(Y.begin(), Y.end()),
                                      Objective<double, double>(objective), &A[0], &W[0]);
      check.criteria.max_epoch = 0;
      check.criteria.eval_on_start = true;
      sdca::make_solver(check).solve();
      ASSERT_EQ(check.train.evals.size(), 1UL);
      EXPECT_EQ(check.train.evals[0].primal, last.primal) << what;
      EXPECT_EQ(check.train.evals[0].dual, last.dual) << what;
      primal.push_back(double(last.primal));
    }
  }
  for (const double p : primal) EXPECT_NEAR(p, primal[0], 2e-6 * std::abs(primal[0])) << where;
}

// Every dual that safe screening fixes at zero is zero at the optimum: here,
// in a solve without shrinking to a relative gap of 1e-10.
template <template <typename, typename> class Objective>
static void test_screening_is_safe(const Objective<double, double>& objective) {
  const sdca::size_type n = 300, d = 8, m = 10;
  std::mt19937 gen(3);
  std::normal_distribution<double> noise(0, 1);
  std::vector<double> X(d * n);
  std::vector<sdca::size_type> Y(n);
  for (sdca::size_type i = 0; i < n; ++i) {
    Y[i] = i % m;
    for (sdca::size_type j = 0; j < d; ++j)
      X[d * i + j] = noise(gen) + 2 * std::cos(double(Y[i] * (j + 1)));
  }
  const std::string where = objective.to_string();
  auto solve = [&](const sdca::shrinking_mode mode, const double tol, std::vector<double>& A,
                   std::vector<char>* examples, std::vector<char>* classes) {
    std::vector<double> W(d * m);
    A.assign(m * n, 0);
    auto ctx = sdca::make_context(sdca::make_input_feature(d, n, &X[0]),
                                  sdca::make_output_multiclass(Y.begin(), Y.end()),
                                  Objective<double, double>(objective), &A[0], &W[0]);
    ctx.criteria.shrinking = mode;
    ctx.criteria.epsilon = tol;
    ctx.criteria.max_epoch = 100000;
    ctx.criteria.eval_epoch = 5;
    auto solver = sdca::make_solver(ctx);
    solver.solve();
    EXPECT_EQ(ctx.status, sdca::solver_status::solved)
        << where << " " << sdca::shrinking_mode_name(mode) << " " << ctx.status_string();
    if (examples) *examples = solver.screened_examples();
    if (classes) *classes = solver.screened_classes();
  };
  std::vector<double> best, A;
  std::vector<char> examples, classes;
  solve(sdca::shrinking_mode::off, 1e-10, best, nullptr, nullptr);
  solve(sdca::shrinking_mode::safe, 1e-7, A, &examples, &classes);
  sdca::size_type marked = 0;
  for (sdca::size_type i = 0; i < n; ++i)
    for (sdca::size_type c = 0; c < m; ++c) {
      if (!classes[m * i + c]) continue;
      ++marked;
      EXPECT_LE(std::abs(best[m * i + c]), 1e-7) << where << " example " << i << " class " << c;
    }
  EXPECT_GT(marked, 0UL) << where;
  EXPECT_GT(std::count(examples.begin(), examples.end(), char(1)), 0) << where;
}

TEST(SolverTest, screening_is_safe) {
  sdca::logging::set_level(sdca::logging::level::warning);
  test_screening_is_safe(sdca::make_objective_l2_hinge_topk<double, double>(1.));
  test_screening_is_safe(sdca::make_objective_l2_hinge_topk<double, double>(1., 3));
  test_screening_is_safe(sdca::make_objective_l2_topk_hinge<double, double>(1., 3));
  test_screening_is_safe(sdca::make_objective_l2_hinge_topk_smooth<double, double>(1.));
  test_screening_is_safe(sdca::make_objective_l2_topk_hinge_smooth<double, double>(1., 3));
}
TEST(SolverTest, shrinking_keeps_the_certificate) {
  sdca::logging::set_level(sdca::logging::level::warning);
  test_shrinking(sdca::make_objective_l2_hinge_topk<double, double>(1.), 1);
  test_shrinking(sdca::make_objective_l2_hinge_topk<double, double>(1., 2), 1);
  test_shrinking(sdca::make_objective_l2_topk_hinge<double, double>(1., 2), 1);
  test_shrinking(sdca::make_objective_l2_hinge_topk_smooth<double, double>(1.), -1);
  test_shrinking(sdca::make_objective_l2_topk_hinge_smooth<double, double>(1.), -1);
  test_shrinking(sdca::make_objective_l2_entropy<double, double>(1.), 0);
  test_shrinking(sdca::make_objective_l2_entropy_topk<double, double>(1., 2), 0);
}

// A step on the label and the classes of nonzero dual (whole = false) holds
// the other classes at zero and, where the whole step keeps them at zero too,
// gives the block of the whole step.
template <template <typename, typename> class Objective>
static void test_part_steps(const Objective<double, double>& objective) {
  const sdca::size_type n = 300, d = 10, m = 12;
  std::mt19937 gen(5);
  std::normal_distribution<double> noise(0, 1);
  std::vector<double> X(d * n), W(d * m), A(m * n);
  std::vector<sdca::size_type> Y(n);
  for (sdca::size_type i = 0; i < n; ++i) {
    Y[i] = i % m;
    for (sdca::size_type j = 0; j < d; ++j) X[d * i + j] = noise(gen) + (j == Y[i] % d ? 1.5 : 0);
  }
  auto ctx = sdca::make_context(sdca::make_input_feature(d, n, &X[0]),
                                sdca::make_output_multiclass(Y.begin(), Y.end()),
                                Objective<double, double>(objective), &A[0], &W[0]);
  ctx.criteria.max_epoch = 5;
  ctx.criteria.epsilon = 0;
  ctx.criteria.shrinking = sdca::shrinking_mode::off;
  sdca::make_solver(ctx).solve();

  const std::string where = objective.to_string();
  sdca::solver_scratch<double, sdca::feature_input> scratch;
  scratch.init(ctx);
  sdca::size_type parts = 0, matches = 0;
  for (sdca::size_type i = 0; i < n; ++i) {
    const std::vector<double> a0(&A[m * i], &A[m * i] + m), w0(W);
    sdca::update_variables(i, ctx, scratch, true);
    const std::vector<double> whole(&A[m * i], &A[m * i] + m), w_whole(W);
    std::copy(a0.begin(), a0.end(), &A[m * i]);
    W = w0;
    sdca::update_variables(i, ctx, scratch, false);
    const std::vector<double> part(&A[m * i], &A[m * i] + m), w_part(W);
    std::copy(a0.begin(), a0.end(), &A[m * i]);
    W = w0;

    sdca::size_type support = 0;
    bool off_zero = true;
    for (sdca::size_type c = 0; c < m; ++c) {
      if (c == Y[i] || a0[c] != 0) {
        ++support;
        continue;
      }
      off_zero = off_zero && whole[c] == 0;
    }
    // A support of k classes or fewer takes zero classes in; one of every class is whole.
    if (support <= objective.k || support == m) continue;
    ++parts;
    for (sdca::size_type c = 0; c < m; ++c) {
      if (c != Y[i] && a0[c] == 0) {
        EXPECT_EQ(part[c], 0.) << where << " example " << i;
      }
    }
    if (!off_zero) continue;
    ++matches;
    for (sdca::size_type c = 0; c < m; ++c)
      EXPECT_NEAR(part[c], whole[c], 1e-12) << where << " example " << i << " class " << c;
    for (sdca::size_type j = 0; j < d * m; ++j)
      EXPECT_NEAR(w_part[j], w_whole[j], 1e-12) << where << " example " << i;
  }
  EXPECT_GT(parts, 0UL) << where;
  EXPECT_GT(matches, 0UL) << where;
}

TEST(SolverTest, part_steps_match_whole_steps) {
  sdca::logging::set_level(sdca::logging::level::warning);
  test_part_steps(sdca::make_objective_l2_hinge_topk<double, double>(1.));
  test_part_steps(sdca::make_objective_l2_hinge_topk<double, double>(1., 3));
  test_part_steps(sdca::make_objective_l2_topk_hinge<double, double>(1., 2));
  test_part_steps(sdca::make_objective_l2_hinge_topk_smooth<double, double>(1., 2));
  test_part_steps(sdca::make_objective_l2_topk_hinge_smooth<double, double>(1., 2));
}

// Float data with double results: the solve keeps W and the duals in double
// and returns them rounded to float, and its certificate is that of the
// returned values: a fresh evaluation of the returned buffers gives the same
// objectives, bit for bit, for every thread count.
template <typename Context>
static std::pair<sdca::solver_status, typename Context::train_eval_type> wide_solve(
    Context&& ctx, const sdca::size_type max_epoch, const double epsilon, const bool on_start,
    const std::size_t threads) {
  ctx.criteria.epsilon = epsilon;
  ctx.criteria.max_epoch = max_epoch;
  ctx.criteria.eval_on_start = on_start;
  ctx.criteria.num_threads = threads;
  sdca::make_solver(ctx).solve();
  return std::make_pair(ctx.status, ctx.train.evals.back());
}

template <template <typename, typename> class Objective, typename Output>
static void test_wide_iterate(const Objective<float, double>& objective, const Output& output,
                              const bool kernel, const double tol) {
  const sdca::size_type n = 300, d = 12, m = output.num_classes;
  std::mt19937 gen(13);
  std::normal_distribution<double> noise(0, 1);
  std::vector<float> X(d * n), K(n * n);
  for (sdca::size_type i = 0; i < n; ++i)
    for (sdca::size_type j = 0; j < d; ++j) X[d * i + j] = float(noise(gen) + (j == i % m ? 2 : 0));
  for (sdca::size_type i = 0; i < n; ++i)
    for (sdca::size_type l = 0; l < n; ++l) {
      double s = 0;
      for (sdca::size_type j = 0; j < d; ++j) s += double(X[d * i + j]) * double(X[d * l + j]);
      K[n * i + l] = float(s / double(d));
    }
  const std::string where = objective.to_string() + (kernel ? " kernel" : " features");
  std::vector<float> first_A;
  for (const std::size_t threads : {1UL, 3UL}) {
    std::vector<float> W(d * m), A(m * n);
    typedef sdca::eval_train<sdca::eval_type<double>, Output> eval_type;
    auto solve = [&](const sdca::size_type max_epoch, const double epsilon,
                     const bool on_start) -> std::pair<sdca::solver_status, eval_type> {
      if (kernel)
        return wide_solve(sdca::make_context(sdca::make_input_kernel(n, &K[0]), Output(output),
                                             Objective<float, double>(objective), &A[0]),
                          max_epoch, epsilon, on_start, threads);
      return wide_solve(sdca::make_context(sdca::make_input_feature(d, n, &X[0]), Output(output),
                                           Objective<float, double>(objective), &A[0], &W[0]),
                        max_epoch, epsilon, on_start, threads);
    };
    const auto fit = solve(100000, tol, false);
    EXPECT_EQ(fit.first, sdca::solver_status::solved) << where;
    const auto check = solve(0, 0, true);
    EXPECT_EQ(check.second.primal, fit.second.primal) << where;
    EXPECT_EQ(check.second.dual, fit.second.dual) << where;
    if (threads == 1)
      first_A = A;
    else
      EXPECT_TRUE(A == first_A) << where << ": not deterministic over thread counts";
  }
}

TEST(SolverTest, wide_iterate_certifies_the_returned_values) {
  sdca::logging::set_level(sdca::logging::level::warning);
  std::vector<sdca::size_type> y(300);
  for (sdca::size_type i = 0; i < y.size(); ++i) y[i] = i % 5;
  const auto multiclass = sdca::make_output_multiclass(y.begin(), y.end());
  std::vector<std::vector<sdca::size_type>> labels(300);
  for (sdca::size_type i = 0; i < labels.size(); ++i)
    labels[i] = {std::min(i % 5, (i + 1) % 5), std::max(i % 5, (i + 1) % 5)};
  const auto multilabel = sdca::make_output_multilabel(labels);
  for (const bool kernel : {false, true}) {
    test_wide_iterate(sdca::make_objective_l2_hinge_topk<float, double>(1.), multiclass, kernel,
                      1e-6);
    test_wide_iterate(sdca::make_objective_l2_topk_hinge<float, double>(1., 2), multiclass, kernel,
                      1e-6);
    test_wide_iterate(sdca::make_objective_l2_entropy<float, double>(1.), multiclass, kernel, 1e-6);
    test_wide_iterate(sdca::make_objective_l2_multilabel_hinge<float, double>(1.), multilabel,
                      kernel, 1e-6);
    test_wide_iterate(sdca::make_objective_l2_multilabel_entropy<float, double>(1.), multilabel,
                      kernel, 1e-6);
  }
}

TEST(SolverTest, multilabel_basic_problems_all_objectives) {
  sdca::logging::set_level(sdca::logging::level::warning);
  sdca::logging::set_format(sdca::logging::format::short_e);
  test_solver_multilabel_basic_all<float, float>();
  test_solver_multilabel_basic_all<float, double>();
  test_solver_multilabel_basic_all<double, float>();
  test_solver_multilabel_basic_all<double, double>();
}
