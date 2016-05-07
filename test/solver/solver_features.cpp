#include "sdca/solver.h"
#include "test_util.h"

#include "sdca/utility/logging.cpp"


template <typename Data,
          typename Result,
          typename Context>
inline void
test_solver_features_check_converged(
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
test_solver_features_check_performance(
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
          typename Output,
          template <typename, typename> class Objective>
inline void
test_solver_features_basic_tests(
    Output out,
    Objective<Data, Result> objective,
    const sdca::size_type n,
    const std::vector<Data>& W,
    const std::vector<Data>& X0,
    const Result expected_performance
  ) {
  sdca::size_type dn = X0.size(), m = out.num_classes;
  sdca::size_type d = dn / n;
  std::vector<Data> X(dn); // primal vars
  std::vector<Data> A(m * n); // dual vars

  // Create context
  auto ctx_init = sdca::make_context(
    sdca::make_input_model(d, n, m, &W[0]),
    std::move(out), std::move(objective), &A[0], &X[0], &X0[0]);

  test_solver_features_check_converged<Data, Result>(ctx_init);
  test_solver_features_check_performance(ctx_init, expected_performance);
}


template <typename Data,
          typename Result,
          typename OutputMaker,
          template <typename, typename> class Objective>
inline void
test_solver_features_multiclass_basic(
    OutputMaker make_output,
    Objective<Data, Result> obj
  ) {
  sdca::size_type d = 3, m = 3, n = 6;
  std::vector<Data> W(d * m); // model weights
  std::vector<Data> X0(d * n); // initial features
  std::vector<sdca::size_type> Y(n); // labels
  Result accuracy = 1;

  // Identity
  W = {1, 0, 0,
       0, 1, 0,
       0, 0, 1};

  // Features are row-wise
  X0 = {10, 3, 2,
        10, -6, 1,
        10, -5, 5,
        4, 10, -7,
        3, 10, 3,
        9, 9, 10};

  // Labels
  Y = {0, 0, 0, 1, 1, 2};

  test_solver_features_basic_tests(make_output(Y), obj, n, W, X0, accuracy);
}


template <typename Data,
          typename Result>
inline void
test_solver_features_multiclass_basic_all() {
  Result C = 4;
  auto multiclass_output_maker = [](std::vector<sdca::size_type>& Y) {
    return sdca::make_output_multiclass(Y.begin(), Y.end());
  };

  test_solver_features_multiclass_basic(
    multiclass_output_maker,
    sdca::make_objective_l2_entropy_nn_features<Data, Result>(C));
}


TEST(SolverFeaturesTest, multiclass_basic_problems_all_objectives) {
  sdca::logging::set_level(sdca::logging::level::verbose);
  sdca::logging::set_format(sdca::logging::format::short_e);
  test_solver_features_multiclass_basic_all<float, float>();
  test_solver_features_multiclass_basic_all<float, double>();
  test_solver_features_multiclass_basic_all<double, float>();
  test_solver_features_multiclass_basic_all<double, double>();
}
