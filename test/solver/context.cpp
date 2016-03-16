#include "sdca/solver/context.h"
#include "test_util.h"

TEST(SolverContextTest, feature_in_multiclass_out) {
  typedef float Data;
  typedef double Result;

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

  auto ctx = sdca::make_context(
    sdca::make_input_feature(d, n, &features[0]),
    sdca::make_output_multiclass(labels.begin(), labels.end()),
    sdca::make_objective_l2_topk_hinge<Data>(),
    &dual[0], &primal[0]);

  EXPECT_EQ(d, ctx.train.num_dimensions());
  EXPECT_EQ(n, ctx.train.num_examples());
  EXPECT_EQ(m, ctx.train.num_classes());
  EXPECT_EQ(n, ctx.train.out.labels.size());
  EXPECT_EQ(primal[2], ctx.primal_variables[2]);
  EXPECT_EQ(dual[3], ctx.dual_variables[3]);
  EXPECT_FALSE(ctx.is_dual());

  sdca::size_type n_tst = n - 5;
  labels.resize(n_tst);
  ctx.add_test(sdca::make_input_feature(d, n_tst, &features[0]),
               sdca::make_output_multiclass(labels.begin(), labels.end()));

  EXPECT_EQ(static_cast<std::size_t>(1), ctx.test.size());
  EXPECT_EQ(d, ctx.test[0].num_dimensions());
  EXPECT_EQ(n_tst, ctx.test[0].num_examples());
  EXPECT_EQ(m, ctx.test[0].num_classes());
  EXPECT_EQ(n_tst, ctx.test[0].out.labels.size());

  typedef decltype(ctx)::data_type data_type;
  typedef decltype(ctx)::result_type result_type;
  EXPECT_TRUE((std::is_same<Data, data_type>::value));
  EXPECT_TRUE((std::is_same<Result, result_type>::value));
}


TEST(SolverContextTest, kernel_in_multiclass_out) {
  typedef double Data;
  typedef double Result;

  sdca::size_type n = 50, m = 3, pow_from = 0, pow_to = 1;
  std::vector<Data> kernel;
  std::vector<sdca::size_type> labels;
  std::vector<Data> dual(m * n);
  dual[3] = 2;

  std::mt19937 gen(1);
  test_populate_real(n * n, pow_from, pow_to, 1.0, gen, kernel);
  test_populate_int<sdca::size_type>(n, 1, m, gen, labels);

  auto ctx = sdca::make_context(
    sdca::make_input_kernel(n, &kernel[0]),
    sdca::make_output_multiclass(labels.begin(), labels.end()),
    sdca::make_objective_l2_topk_hinge<Data>(),
    &dual[0]);

  EXPECT_EQ(n, ctx.train.num_examples());
  EXPECT_EQ(m, ctx.train.num_classes());
  EXPECT_EQ(n, ctx.train.out.labels.size());
  EXPECT_EQ(dual[3], ctx.dual_variables[3]);
  EXPECT_TRUE(ctx.is_dual());

  sdca::size_type n_tst = n - 5;
  labels.resize(n_tst);
  ctx.add_test(sdca::make_input_kernel(n, n_tst, &kernel[0]),
               sdca::make_output_multiclass(labels.begin(), labels.end()));

  EXPECT_EQ(n_tst, ctx.test[0].num_examples());
  EXPECT_EQ(m, ctx.test[0].num_classes());
  EXPECT_EQ(n_tst, ctx.test[0].out.labels.size());

  typedef decltype(ctx)::data_type data_type;
  typedef decltype(ctx)::result_type result_type;
  EXPECT_TRUE((std::is_same<Data, data_type>::value));
  EXPECT_TRUE((std::is_same<Result, result_type>::value));
}
