#include <cstdio>
#include <type_traits>

#include "gtest/gtest.h"
#include "sdca/solver.h"

#include "test_util.h"

TEST(SolverDualSolverTest, feature_in_multiclass_out) {
  sdca::size_type n = 50, n_tst = 100, m = 3, pow_from = 0, pow_to = 1;
  std::vector<double> kernel;
  std::vector<sdca::size_type> labels;
  std::vector<double> dual(m * n);
  dual[3] = 2;

  std::mt19937 gen(1);
  test_populate_real(n * n_tst, pow_from, pow_to, 1.0, gen, kernel);
  test_populate_int<sdca::size_type>(n_tst, 1, m, gen, labels);

  auto ctx = sdca::make_multiclass_context(
    n, n_tst, &kernel[0], labels.begin(), &dual[0]);

  auto obj = sdca::make_objective_l2_entropy_topk<double>();

  auto solver = sdca::make_dual_solver(std::move(ctx), std::move(obj));

  std::printf("primal = %g\n", solver.primal());
}
