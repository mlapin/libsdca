#include <cstdio>
#include <typeinfo>

#include "gtest/gtest.h"
#include "sdca/solver/dataset.h"

#include "test_util.h"


TEST(SolverDatasetTest, multiclass_output) {
  sdca::size_type n = 50, m = 3;
  std::vector<sdca::size_type> labels;
  std::mt19937 gen(1);
  test_populate_int<sdca::size_type>(n, 1, m, gen, labels);

  auto out = sdca::make_output_multiclass(labels.begin(), labels.end());

  EXPECT_EQ(m, out.num_classes);
  EXPECT_EQ(n, out.labels.size());
}


TEST(SolverDatasetTest, multiclass_output_invalid_argument) {
  sdca::size_type n = 50, m = 3;
  std::vector<sdca::size_type> labels;
  std::mt19937 gen(1);
  test_populate_int<sdca::size_type>(n, 2, m, gen, labels);

  EXPECT_THROW(
        sdca::make_output_multiclass(labels.begin(), labels.end()),
        std::invalid_argument);
}


TEST(SolverDatasetTest, feature_in_multiclass_out) {
  sdca::size_type n = 50, m = 3, d = 5, pow_from = 0, pow_to = 1;
  std::vector<float> features;
  std::vector<sdca::size_type> labels;

  std::mt19937 gen(1);
  test_populate_real(n * d, pow_from, pow_to, 1.0f, gen, features);
  test_populate_int<sdca::size_type>(n, 1, m, gen, labels);

  auto dataset = sdca::make_dataset_feature_in_multiclass_out(
    d, n, &features[0], labels.begin());

  EXPECT_EQ(d, dataset.num_dimensions());
  EXPECT_EQ(n, dataset.num_examples());
  EXPECT_EQ(m, dataset.num_classes());
  EXPECT_EQ(n, dataset.out.labels.size());

  auto trn_dataset = sdca::make_dataset_train_feature_in_multiclass_out(
    d, n, &features[0], labels.begin());

  EXPECT_EQ(d, trn_dataset.num_dimensions());
  EXPECT_EQ(n, trn_dataset.num_examples());
  EXPECT_EQ(m, trn_dataset.num_classes());
  EXPECT_EQ(n, trn_dataset.out.labels.size());

  typedef decltype(trn_dataset)::eval_type trn_eval_type;
  EXPECT_TRUE((std::is_same<sdca::eval_train<double>, trn_eval_type>::value));

  auto tst_dataset = sdca::make_dataset_test_feature_in_multiclass_out(
    d, n, &features[0], labels.begin());

  EXPECT_EQ(d, tst_dataset.num_dimensions());
  EXPECT_EQ(n, tst_dataset.num_examples());
  EXPECT_EQ(m, tst_dataset.num_classes());
  EXPECT_EQ(n, tst_dataset.out.labels.size());

  typedef decltype(tst_dataset)::eval_type tst_eval_type;
  EXPECT_TRUE((std::is_same<sdca::eval_test<double>, tst_eval_type>::value));
}


TEST(SolverDatasetTest, kernel_in_multiclass_out) {
  sdca::size_type n = 50, n_tst = 100, m = 3, pow_from = 0, pow_to = 1;
  std::vector<double> kernel;
  std::vector<sdca::size_type> labels;

  std::mt19937 gen(1);
  test_populate_real(n * n_tst, pow_from, pow_to, 1.0, gen, kernel);
  test_populate_int<sdca::size_type>(n_tst, 1, m, gen, labels);

  auto dataset = sdca::make_dataset_kernel_in_multiclass_out(
    n, n_tst, &kernel[0], labels.begin());

  EXPECT_EQ(n_tst, dataset.num_examples());
  EXPECT_EQ(m, dataset.num_classes());
  EXPECT_EQ(n, dataset.in.num_train_examples);
  EXPECT_EQ(n_tst, dataset.out.labels.size());

  auto trn_dataset = sdca::make_dataset_train_kernel_in_multiclass_out<float>(
    n, &kernel[0], labels.begin());

  EXPECT_EQ(n, trn_dataset.num_examples());
  EXPECT_EQ(m, trn_dataset.num_classes());
  EXPECT_EQ(n, trn_dataset.in.num_train_examples);
  EXPECT_EQ(n, trn_dataset.out.labels.size());

  typedef decltype(trn_dataset)::eval_type trn_eval_type;
  EXPECT_TRUE((std::is_same<sdca::eval_train<float>, trn_eval_type>::value));

  auto tst_dataset = sdca::make_dataset_test_kernel_in_multiclass_out<float>(
    n, n_tst, &kernel[0], labels.begin());

  EXPECT_EQ(n_tst, tst_dataset.num_examples());
  EXPECT_EQ(m, tst_dataset.num_classes());
  EXPECT_EQ(n, tst_dataset.in.num_train_examples);
  EXPECT_EQ(n_tst, tst_dataset.out.labels.size());

  typedef decltype(tst_dataset)::eval_type tst_eval_type;
  EXPECT_TRUE((std::is_same<sdca::eval_test<float>, tst_eval_type>::value));
}
