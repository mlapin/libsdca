#include "sdca/solver.h"
#include "test_util.h"

#include "sdca/utility/logging.cpp"


template <typename Data,
          typename Result,
          typename Context>
inline void
test_solver_check_converged(
    Context& ctx
  ) {
  ctx.criteria.epsilon = 64 * std::max(
    static_cast<double>(std::numeric_limits<Data>::epsilon()),
    static_cast<double>(std::numeric_limits<Result>::epsilon()));

  ctx.criteria.eval_epoch = 2;
  auto solver = sdca::make_solver(ctx);
  solver.solve();

  EXPECT_TRUE(ctx.status == sdca::solver_status::solved);
}


template <typename Data,
          typename Result,
          template <typename> class Input,
          template <typename, typename> class Objective>
inline void
test_solver_check_performance(
    const sdca::solver_context<Data, Result, Input,
                               sdca::multiclass_output, Objective>& ctx,
    const Result expected_accuracy
  ) {
  const Result accuracy = ctx.train.evals.back().accuracy[0];
  if (accuracy != expected_accuracy) {
    std::printf("%s\n", ctx.to_string().c_str());
    std::printf("%s\n", ctx.status_string().c_str());
  }
  EXPECT_EQ(expected_accuracy, accuracy);
}


template <typename Data,
          typename Result,
          template <typename> class Input,
          template <typename, typename> class Objective>
inline void
test_solver_check_performance(
    const sdca::solver_context<Data, Result, Input,
                               sdca::multilabel_output, Objective>& ctx,
    const Result expected_accuracy
  ) {
  const Result accuracy = 1 - ctx.train.evals.back().rank_loss;
  if (accuracy != expected_accuracy) {
    std::printf("%s\n", ctx.to_string().c_str());
    std::printf("%s\n", ctx.status_string().c_str());
  }
  EXPECT_EQ(expected_accuracy, accuracy);
}


template <typename Data,
          typename Result,
          typename Output,
          template <typename, typename> class Objective>
inline void
test_solver_basic_tests_feature_in(
    Output out,
    Objective<Data, Result> objective,
    const sdca::size_type n,
    const std::vector<Data>& X,
    const Result expected_performance
  ) {
  sdca::size_type dn = X.size(), m = out.num_classes;
  sdca::size_type d = dn / n;
  std::vector<Data> W(d * m); // primal vars
  std::vector<Data> A(m * n); // dual vars

  // Use features
  auto ctx_feat = sdca::make_context(
    sdca::make_input_feature(d, n, &X[0]),
    std::move(out), std::move(objective), &A[0], &W[0]);

  test_solver_check_converged<Data, Result>(ctx_feat);
  test_solver_check_performance(ctx_feat, expected_performance);

  // Kernel (Gram) matrix
  std::vector<Data> K(n * n);
  sdca::blas_int D = static_cast<sdca::blas_int>(d);
  sdca::blas_int N = static_cast<sdca::blas_int>(n);
  sdca::sdca_blas_gemm(N, N, D, &X[0], D, &X[0], D, &K[0], CblasTrans);

  // Warm restart - use the same dual variables as above
  auto ctx_warm = sdca::make_context(
    sdca::make_input_kernel(n, &K[0]),
    std::move(ctx_feat.train.out), std::move(ctx_feat.objective), &A[0]);

  ctx_warm.criteria.eval_on_start = true;
  test_solver_check_converged<Data, Result>(ctx_warm);
  test_solver_check_performance(ctx_warm, expected_performance);
  EXPECT_TRUE(ctx_warm.epoch <= ctx_warm.criteria.eval_epoch);

  // Zero the dual variables and train from scratch
  std::fill(A.begin(), A.end(), static_cast<Data>(0));
  auto ctx_ker = sdca::make_context(
    sdca::make_input_kernel(n, &K[0]),
    std::move(ctx_warm.train.out), std::move(ctx_warm.objective), &A[0]);

  ctx_ker.criteria.eval_on_start = true;
  test_solver_check_converged<Data, Result>(ctx_ker);
  test_solver_check_performance(ctx_ker, expected_performance);
  EXPECT_TRUE(ctx_ker.epoch > 0UL);
}


template <typename Data,
          typename Result,
          typename OutputMaker,
          template <typename, typename> class Objective>
inline void
test_solver_multiclass_basic(
    OutputMaker make_output,
    Objective<Data, Result> objective
  ) {
  sdca::size_type d = 3, n = 4;
  std::vector<Data> X(d * n); // features
  std::vector<sdca::size_type> Y(n); // labels
  Result accuracy = 1;

  // Last column is the offset (bias) feature
  X = {0, 0, 1,
       0, 1, 1,
       1, 0, 1,
       1, 1, 1};

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
  X = {0, 0, 1,
       0, 1, 1,
       1, 0, 1,
       1 - eps, 1 - eps, 1};
  Y = {0, 1, 1, 0};
  accuracy = static_cast<Result>(0.75);
  test_solver_basic_tests_feature_in(make_output(Y), objective, n, X, accuracy);
}


template <typename Data,
          typename Result,
          template <typename, typename> class Objective>
inline void
test_solver_multilabel_basic(
    Objective<Data, Result> objective
  ) {
  sdca::size_type d = 3, n = 5;
  std::vector<Data> X(d * n); // features
  std::vector<std::vector<sdca::size_type>> Y; // labels
  Result accuracy = 1;

  // Last column is the offset (bias) feature
  X = {0, 2, 1,
       1, 2, 1,
       2, 2, 1,
       2, 1, 1,
       2, 0, 1};

  Y = {{0}, {0, 1}, {1}, {1, 2}, {2}};
  test_solver_basic_tests_feature_in(
    sdca::make_output_multilabel(Y), objective, n, X, accuracy);
}


template <typename Data,
          typename Result>
inline void
test_solver_multiclass_basic_all() {
  Result C = 4;
  auto multiclass_output_maker = [](std::vector<sdca::size_type>& Y) {
    return sdca::make_output_multiclass(Y.begin(), Y.end());
  };

  test_solver_multiclass_basic(
    multiclass_output_maker,
    sdca::make_objective_l2_entropy_topk<Data, Result>(C));

  test_solver_multiclass_basic(
    multiclass_output_maker,
    sdca::make_objective_l2_hinge_topk<Data, Result>(C));

  test_solver_multiclass_basic(
    multiclass_output_maker,
    sdca::make_objective_l2_hinge_topk_smooth<Data, Result>(C));

  test_solver_multiclass_basic(
    multiclass_output_maker,
    sdca::make_objective_l2_topk_hinge<Data, Result>(C));

  test_solver_multiclass_basic(
    multiclass_output_maker,
    sdca::make_objective_l2_topk_hinge_smooth<Data, Result>(C));

  auto multilabel_output_maker = [](std::vector<sdca::size_type>& Y) {
    return sdca::make_output_multilabel(Y.begin(), Y.end());
  };

  test_solver_multiclass_basic(
    multilabel_output_maker,
    sdca::make_objective_l2_multilabel_hinge<Data, Result>(C));

  test_solver_multiclass_basic(
    multilabel_output_maker,
    sdca::make_objective_l2_multilabel_hinge_smooth<Data, Result>(C));
}


template <typename Data,
          typename Result>
inline void
test_solver_multilabel_basic_all() {
  Result C = 4;
  test_solver_multilabel_basic(
    sdca::make_objective_l2_multilabel_hinge<Data, Result>(C));

  test_solver_multilabel_basic(
    sdca::make_objective_l2_multilabel_hinge_smooth<Data, Result>(C));
}


TEST(SolverTest, multiclass_basic_problems_all_objectives) {
  sdca::logging::set_level(sdca::logging::level::warning);
  sdca::logging::set_format(sdca::logging::format::short_e);
  test_solver_multiclass_basic_all<float, float>();
  test_solver_multiclass_basic_all<float, double>();
  test_solver_multiclass_basic_all<double, float>();
  test_solver_multiclass_basic_all<double, double>();
}


TEST(SolverTest, multilabel_basic_problems_all_objectives) {
  sdca::logging::set_level(sdca::logging::level::warning);
  sdca::logging::set_format(sdca::logging::format::short_e);
  test_solver_multilabel_basic_all<float, float>();
  test_solver_multilabel_basic_all<float, double>();
  test_solver_multilabel_basic_all<double, float>();
  test_solver_multilabel_basic_all<double, double>();
}
