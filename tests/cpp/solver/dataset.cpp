#include "sdca/solver/data.h"
#include "test_util.h"

TEST(SolverDatasetTest, multiclass_output) {
  sdca::size_type n = 50, m = 3;
  std::vector<sdca::size_type> labels;
  std::mt19937 gen(1);
  test_populate_int<sdca::size_type>(n, 0, m - 1, gen, labels);

  auto out = sdca::make_output_multiclass(labels.begin(), labels.end(), m);

  EXPECT_EQ(m, out.num_classes);
  EXPECT_EQ(n, out.labels.size());
}

TEST(SolverDatasetTest, multiclass_output_invalid_argument) {
  sdca::size_type n = 50, m = 3;
  std::vector<sdca::size_type> labels;
  std::mt19937 gen(1);

  // Labels are 0-based: m is out of range (no 1-based shift).
  test_populate_int<sdca::size_type>(n, 1, m, gen, labels);
  labels.back() = m;
  EXPECT_THROW(sdca::make_output_multiclass(labels.begin(), labels.end(), m),
               std::invalid_argument);
  EXPECT_NO_THROW(sdca::make_output_multiclass(labels.begin(), labels.end(), m + 1));
  // No labels, negative and fractional labels.
  std::vector<double> none, negative = {0, -1}, fractional = {0, 0.5};
  EXPECT_THROW(sdca::make_output_multiclass(none.begin(), none.end(), m), std::invalid_argument);
  EXPECT_THROW(sdca::make_output_multiclass(negative.begin(), negative.end(), m),
               std::invalid_argument);
  EXPECT_THROW(sdca::make_output_multiclass(fractional.begin(), fractional.end(), m),
               std::invalid_argument);
}

TEST(SolverDatasetTest, multilabel_output) {
  std::vector<std::vector<sdca::size_type>> labels;

  sdca::size_type m = 4;
  labels = {{0, 1}, {2}, {0, 1}, {0, 2}, {1, 2}, {1, 2, 3}};

  auto out = sdca::make_output_multilabel(labels, m);
  EXPECT_EQ(m, out.num_classes);
  EXPECT_EQ(labels.size(), out.offsets.size() - 1);
}

TEST(SolverDatasetTest, multilabel_output_invalid_argument) {
  std::vector<std::vector<sdca::size_type>> labels;
  const sdca::size_type m = 4;

  // A label out of [0, m) (no 1-based shift)
  labels = {{1}, {2}, {1, 2}, {4}};
  EXPECT_THROW(sdca::make_output_multilabel(labels, m), std::invalid_argument);

  // Too many labels for the last example (all classes)
  labels = {{0, 1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2, 3}};
  EXPECT_THROW(sdca::make_output_multilabel(labels, m), std::invalid_argument);

  // An empty set
  labels = {{0, 1}, {}};
  EXPECT_THROW(sdca::make_output_multilabel(labels, m), std::invalid_argument);

  // Non-distinct labels
  labels = {{0, 1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 1}};
  EXPECT_THROW(sdca::make_output_multilabel(labels, m), std::invalid_argument);

  // Not correctly sorted labels
  labels = {{0, 1}, {2}, {0, 1}, {0, 2}, {1, 2}, {2, 1, 0}};
  EXPECT_THROW(sdca::make_output_multilabel(labels, m), std::invalid_argument);

  // Offsets that do not span the labels
  std::vector<sdca::size_type> flat = {0, 1, 2}, offsets = {0, 1, 2};
  EXPECT_THROW(
      sdca::make_output_multilabel(flat.begin(), flat.end(), offsets.begin(), offsets.end(), m),
      std::invalid_argument);
}

TEST(SolverDatasetTest, feature_in_multiclass_out) {
  sdca::size_type n = 50, m = 3, d = 5, pow_from = 0, pow_to = 1;
  std::vector<float> features;
  std::vector<sdca::size_type> labels;

  std::mt19937 gen(1);
  test_populate_real(n * d, pow_from, pow_to, 1.0f, gen, features);
  test_populate_int<sdca::size_type>(n, 0, m - 1, gen, labels);

  auto trn_dataset =
      sdca::make_dataset_train(sdca::make_input_feature(n, d, &features[0]),
                               sdca::make_output_multiclass(labels.begin(), labels.end(), m));

  EXPECT_EQ(d, trn_dataset.num_dimensions());
  EXPECT_EQ(n, trn_dataset.num_examples());
  EXPECT_EQ(m, trn_dataset.num_classes());
  EXPECT_EQ(n, trn_dataset.out.labels.size());

  typedef decltype(trn_dataset)::eval_type trn_eval_type;
  EXPECT_TRUE(
      (std::is_same<sdca::eval_train<double, sdca::multiclass_output>, trn_eval_type>::value));

  auto tst_dataset =
      sdca::make_dataset_test(sdca::make_input_feature(n, d, &features[0]),
                              sdca::make_output_multiclass(labels.begin(), labels.end(), m));

  EXPECT_EQ(d, tst_dataset.num_dimensions());
  EXPECT_EQ(n, tst_dataset.num_examples());
  EXPECT_EQ(m, tst_dataset.num_classes());
  EXPECT_EQ(n, tst_dataset.out.labels.size());

  typedef decltype(tst_dataset)::eval_type tst_eval_type;
  EXPECT_TRUE(
      (std::is_same<sdca::eval_test<double, sdca::multiclass_output>, tst_eval_type>::value));
}

TEST(SolverDatasetTest, feature_in_multilabel_out) {
  std::vector<float> features;
  std::vector<std::vector<sdca::size_type>> labels;

  sdca::size_type n = 6, m = 4, d = 5, pow_from = 0, pow_to = 1;
  labels = {{0, 1}, {2}, {0, 1}, {0, 2}, {1, 2}, {1, 2, 3}};

  sdca::size_type nnz = 0;
  std::for_each(labels.begin(), labels.end(),
                [&](std::vector<sdca::size_type>& yi) { nnz += yi.size(); });

  std::mt19937 gen(1);
  test_populate_real(n * d, pow_from, pow_to, 1.0f, gen, features);

  auto trn_dataset = sdca::make_dataset_train(sdca::make_input_feature(n, d, &features[0]),
                                              sdca::make_output_multilabel(labels, m));

  EXPECT_EQ(d, trn_dataset.num_dimensions());
  EXPECT_EQ(n, trn_dataset.num_examples());
  EXPECT_EQ(m, trn_dataset.num_classes());
  EXPECT_EQ(nnz, trn_dataset.out.labels.size());
  EXPECT_EQ(n + 1, trn_dataset.out.offsets.size());

  typedef decltype(trn_dataset)::eval_type trn_eval_type;
  EXPECT_TRUE(
      (std::is_same<sdca::eval_train<double, sdca::multilabel_output>, trn_eval_type>::value));

  auto tst_dataset = sdca::make_dataset_test(sdca::make_input_feature(n, d, &features[0]),
                                             sdca::make_output_multilabel(labels, m));

  EXPECT_EQ(d, tst_dataset.num_dimensions());
  EXPECT_EQ(n, tst_dataset.num_examples());
  EXPECT_EQ(m, tst_dataset.num_classes());
  EXPECT_EQ(nnz, tst_dataset.out.labels.size());
  EXPECT_EQ(n + 1, tst_dataset.out.offsets.size());

  typedef decltype(tst_dataset)::eval_type tst_eval_type;
  EXPECT_TRUE(
      (std::is_same<sdca::eval_test<double, sdca::multilabel_output>, tst_eval_type>::value));
}

TEST(SolverDatasetTest, kernel_in_multiclass_out) {
  sdca::size_type n = 50, n_tst = 100, m = 3, pow_from = 0, pow_to = 1;
  std::vector<double> kernel;
  std::vector<sdca::size_type> trn_labels;
  std::vector<sdca::size_type> tst_labels;

  std::mt19937 gen(1);
  test_populate_real(n * n_tst, pow_from, pow_to, 1.0, gen, kernel);
  test_populate_int<sdca::size_type>(n, 0, m - 1, gen, trn_labels);
  test_populate_int<sdca::size_type>(n_tst, 0, m - 1, gen, tst_labels);

  auto trn_dataset = sdca::make_dataset_train<float>(
      sdca::make_input_kernel(n, &kernel[0]),
      sdca::make_output_multiclass(trn_labels.begin(), trn_labels.end(), m));

  EXPECT_EQ(n, trn_dataset.num_examples());
  EXPECT_EQ(m, trn_dataset.num_classes());
  EXPECT_EQ(n, trn_dataset.in.num_train_examples);
  EXPECT_EQ(n, trn_dataset.out.labels.size());

  typedef decltype(trn_dataset)::eval_type trn_eval_type;
  EXPECT_TRUE(
      (std::is_same<sdca::eval_train<float, sdca::multiclass_output>, trn_eval_type>::value));

  auto tst_dataset = sdca::make_dataset_test<float>(
      sdca::make_input_kernel(n_tst, n, &kernel[0]),
      sdca::make_output_multiclass(tst_labels.begin(), tst_labels.end(), m));

  EXPECT_EQ(n_tst, tst_dataset.num_examples());
  EXPECT_EQ(m, tst_dataset.num_classes());
  EXPECT_EQ(n, tst_dataset.in.num_train_examples);
  EXPECT_EQ(n_tst, tst_dataset.out.labels.size());

  typedef decltype(tst_dataset)::eval_type tst_eval_type;
  EXPECT_TRUE(
      (std::is_same<sdca::eval_test<float, sdca::multiclass_output>, tst_eval_type>::value));
}

TEST(SolverDatasetTest, kernel_in_multilabel_out) {
  std::vector<double> kernel;
  std::vector<std::vector<sdca::size_type>> trn_labels;
  std::vector<std::vector<sdca::size_type>> tst_labels;

  sdca::size_type n = 6, n_tst = 7, m = 4, pow_from = 0, pow_to = 1;
  trn_labels = {{0, 1}, {2}, {0, 1}, {0, 2}, {1, 2}, {1, 2, 3}};
  tst_labels = {{0, 1}, {2}, {0, 1}, {0, 2}, {1, 2}, {1, 2, 3}, {0, 3}};

  sdca::size_type nnz_trn = 0;
  std::for_each(trn_labels.begin(), trn_labels.end(),
                [&](std::vector<sdca::size_type>& yi) { nnz_trn += yi.size(); });

  sdca::size_type nnz_tst = 0;
  std::for_each(tst_labels.begin(), tst_labels.end(),
                [&](std::vector<sdca::size_type>& yi) { nnz_tst += yi.size(); });

  std::mt19937 gen(1);
  test_populate_real(n * n_tst, pow_from, pow_to, 1.0, gen, kernel);

  auto trn_dataset = sdca::make_dataset_train<float>(sdca::make_input_kernel(n, &kernel[0]),
                                                     sdca::make_output_multilabel(trn_labels, m));

  EXPECT_EQ(n, trn_dataset.num_examples());
  EXPECT_EQ(m, trn_dataset.num_classes());
  EXPECT_EQ(n, trn_dataset.in.num_train_examples);
  EXPECT_EQ(nnz_trn, trn_dataset.out.labels.size());
  EXPECT_EQ(n + 1, trn_dataset.out.offsets.size());

  typedef decltype(trn_dataset)::eval_type trn_eval_type;
  EXPECT_TRUE(
      (std::is_same<sdca::eval_train<float, sdca::multilabel_output>, trn_eval_type>::value));

  auto tst_dataset = sdca::make_dataset_test<float>(sdca::make_input_kernel(n_tst, n, &kernel[0]),
                                                    sdca::make_output_multilabel(tst_labels, m));

  EXPECT_EQ(n_tst, tst_dataset.num_examples());
  EXPECT_EQ(m, tst_dataset.num_classes());
  EXPECT_EQ(n, tst_dataset.in.num_train_examples);
  EXPECT_EQ(nnz_tst, tst_dataset.out.labels.size());
  EXPECT_EQ(n_tst + 1, tst_dataset.out.offsets.size());

  typedef decltype(tst_dataset)::eval_type tst_eval_type;
  EXPECT_TRUE(
      (std::is_same<sdca::eval_test<float, sdca::multilabel_output>, tst_eval_type>::value));
}
