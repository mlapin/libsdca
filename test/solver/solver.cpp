#include "sdca/solver.h"
#include "test_util.h"

#include "sdca/utility/logging.cpp"


template <typename Data,
          typename Result,
          template <typename, typename> class Objective>
inline void
test_solver_basic_feature_in(
    Objective<Data, Result>& objective,
    const std::vector<Data>& X,
    const std::vector<sdca::size_type>& Y,
    const Result expected_accuracy
  ) {
  sdca::size_type dn = X.size(), n = Y.size(), m = 2;
  sdca::size_type d = dn / n;
  std::vector<Data> W(d * m); // primal vars
  std::vector<Data> A(m * n); // dual vars

  auto ctx = sdca::make_context_multiclass(
    std::move(objective),
    d, n, &X[0], Y.begin(), &A[0], &W[0]);

//  ctx.criteria.eval_epoch = 1;
  ctx.criteria.epsilon = 64 * std::max(
    static_cast<double>(std::numeric_limits<Data>::epsilon()),
    static_cast<double>(std::numeric_limits<Result>::epsilon()));
  auto solver = sdca::make_solver(ctx);
  solver.solve();

  EXPECT_TRUE(ctx.status == sdca::solver_status::solved);

  const Result accuracy = ctx.train.evals.back().accuracy[0];
  if (accuracy != expected_accuracy) {
    std::printf("%s\n", ctx.to_string().c_str());
    std::printf("%s\n", ctx.status_string().c_str());
  }
  EXPECT_EQ(expected_accuracy, accuracy);
}


template <typename Data,
          typename Result,
          template <typename, typename> class Objective>
inline void
test_solver_basic(
    Objective<Data, Result> objective
  ) {
  sdca::size_type d = 3, n = 4;
  std::vector<Data> X(d * n); // features
  std::vector<sdca::size_type> Y(n); // labels

  // Last column is the offset (bias) feature
  X = {0, 0, 1,
       0, 1, 1,
       1, 0, 1,
       1, 1, 1};

  // OR
  Y = {0, 1, 1, 1};
  test_solver_basic_feature_in(objective, X, Y, static_cast<Result>(1));

  // AND
  Y = {0, 0, 0, 1};
  test_solver_basic_feature_in(objective, X, Y, static_cast<Result>(1));

  // XOR
  // Perturb the last point to break symmetry
  // (otherwise test fails due to (?) regularized bias term)
  Data eps = static_cast<Data>(0.15);
  X = {0, 0, 1,
       0, 1, 1,
       1, 0, 1,
       1 - eps, 1 - eps, 1};
  Y = {0, 1, 1, 0};
  test_solver_basic_feature_in(objective, X, Y, static_cast<Result>(0.75));
}


template <typename Data,
          typename Result>
inline void
test_solver_basic_all() {
  Result C = 4;
  test_solver_basic(
    sdca::make_objective_l2_entropy_topk<Data, Result>(C));

  test_solver_basic(
    sdca::make_objective_l2_hinge_topk<Data, Result>(C));
  test_solver_basic(
    sdca::make_objective_l2_hinge_topk_smooth<Data, Result>(C));

  test_solver_basic(
    sdca::make_objective_l2_topk_hinge<Data, Result>(C));
  test_solver_basic(
    sdca::make_objective_l2_topk_hinge_smooth<Data, Result>(C));
}


TEST(SolverTest, basic_multiclass_problems_all_objectives) {
  sdca::logging::set_level(sdca::logging::level::warning);
  sdca::logging::set_format(sdca::logging::format::short_e);
  test_solver_basic_all<float, float>();
  test_solver_basic_all<float, double>();
  test_solver_basic_all<double, float>();
  test_solver_basic_all<double, double>();
}
