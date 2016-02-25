#include <cstdio>

#include "gtest/gtest.h"
#include "sdca/solver/context.h"

#include "test_util.h"

TEST(SolverContextTest, feature_in_multiclass_out) {
  sdca::size_type n = 50, m = 3, d = 5, pow_from = 0, pow_to = 1;
  std::vector<float> features;
  std::vector<sdca::size_type> labels;
  std::vector<float> primal(d * m);
  std::vector<float> dual(m * n);
  primal[2] = 1;
  dual[3] = 2;

  std::mt19937 gen(1);
  test_populate_real(n * d, pow_from, pow_to, 1.0f, gen, features);
  test_populate_int<sdca::size_type>(n, 1, m, gen, labels);

  auto ctx = sdca::make_context_multiclass(
    d, n, &features[0], labels.begin(), &dual[0], &primal[0]);

  EXPECT_EQ(d, ctx.train.num_dimensions());
  EXPECT_EQ(n, ctx.train.num_examples());
  EXPECT_EQ(m, ctx.train.num_classes());
  EXPECT_EQ(n, ctx.train.out.labels.size());
  EXPECT_EQ(primal[2], ctx.primal_variables[2]);
  EXPECT_EQ(dual[3], ctx.dual_variables[3]);
  EXPECT_FALSE(ctx.is_dual());
}


TEST(SolverContextTest, kernel_in_multiclass_out) {
  sdca::size_type n = 50, m = 3, pow_from = 0, pow_to = 1;
  std::vector<double> kernel;
  std::vector<sdca::size_type> labels;
  std::vector<double> dual(m * n);
  dual[3] = 2;

  std::mt19937 gen(1);
  test_populate_real(n * n, pow_from, pow_to, 1.0, gen, kernel);
  test_populate_int<sdca::size_type>(n, 1, m, gen, labels);

  auto ctx = sdca::make_context_multiclass(
    n, &kernel[0], labels.begin(), &dual[0]);

  EXPECT_EQ(n, ctx.train.num_examples());
  EXPECT_EQ(m, ctx.train.num_classes());
  EXPECT_EQ(n, ctx.train.out.labels.size());
  EXPECT_EQ(dual[3], ctx.dual_variables[3]);
  EXPECT_TRUE(ctx.is_dual());
}
