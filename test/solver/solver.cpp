#include "sdca/solver.h"
#include "test_util.h"

#include "sdca/utility/logging.cpp"

TEST(SolverSolverTest, feature_in_multiclass_out) {
  typedef float Data;
//  typedef double Result;

  sdca::size_type n = 50, m = 3, d = 5, pow_from = 0, pow_to = 1;
  std::vector<Data> features;
  std::vector<sdca::size_type> labels;
  std::vector<Data> primal(d * m);
  std::vector<Data> dual(m * n);
  primal[2] = 1;
  dual[3] = 2;

  std::mt19937 gen(1);
  test_populate_real(n * d, pow_from, pow_to, 1.0f, gen, features);
  test_populate_int<sdca::size_type>(n, 1, m, gen, labels);

  auto ctx = sdca::make_context_multiclass(
    sdca::make_objective_l2_topk_hinge<Data>(),
    d, n, &features[0], labels.begin(), &dual[0], &primal[0]);

  EXPECT_EQ(d, ctx.train.num_dimensions());
  EXPECT_EQ(n, ctx.train.num_examples());
  EXPECT_EQ(m, ctx.train.num_classes());

  sdca::logging::set_level(sdca::logging::level::info);
  sdca::logging::set_format(sdca::logging::format::short_e);

  auto solver = sdca::make_solver(ctx);
  solver.solve();
}
