#include "sdca/solver/data.h"
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

  // Labels do not start from 0 or 1
  test_populate_int<sdca::size_type>(n, 2, m, gen, labels);
  EXPECT_THROW(sdca::make_output_multiclass(labels.begin(), labels.end()),
               std::invalid_argument);
}


TEST(SolverDatasetTest, multilabel_output) {
  std::vector<std::vector<sdca::size_type>> labels;

  sdca::size_type m = 4;
  labels = { {1, 2}, {3}, {1, 2}, {1, 3}, {2, 3}, {2, 3, 4} };

  auto out = sdca::make_output_multilabel(labels);
  EXPECT_EQ(m, out.num_classes);
  EXPECT_EQ(labels.size(), out.offsets.size() - 1);
}


TEST(SolverDatasetTest, multilabel_output_invalid_argument) {
  std::vector<std::vector<sdca::size_type>> labels;

  // Labels do not start from 0 or 1
  labels = { {2}, {3}, {2, 3}, {4}};
  EXPECT_THROW(sdca::make_output_multilabel(labels), std::invalid_argument);

  // Too many labels for the last example (all classes)
  labels = { {1, 2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3, 4} };
  EXPECT_THROW(sdca::make_output_multilabel(labels), std::invalid_argument);

  // Non-distinct labels
  labels = { {1, 2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 2} };
  EXPECT_THROW(sdca::make_output_multilabel(labels), std::invalid_argument);

  // Not correctly sorted labels
  labels = { {1, 2}, {3}, {1, 2}, {1, 3}, {2, 3}, {3, 2, 1} };
  EXPECT_THROW(sdca::make_output_multilabel(labels), std::invalid_argument);
}


TEST(SolverDatasetTest, feature_in_multiclass_out) {
  sdca::size_type n = 50, m = 3, d = 5, pow_from = 0, pow_to = 1;
  std::vector<float> features;
  std::vector<sdca::size_type> labels;

  std::mt19937 gen(1);
  test_populate_real(n * d, pow_from, pow_to, 1.0f, gen, features);
  test_populate_int<sdca::size_type>(n, 1, m, gen, labels);

  auto trn_dataset = sdca::make_dataset_train(
    sdca::make_input_feature(d, n, &features[0]),
    sdca::make_output_multiclass(labels.begin(), labels.end())
    );

  EXPECT_EQ(d, trn_dataset.num_dimensions());
  EXPECT_EQ(n, trn_dataset.num_examples());
  EXPECT_EQ(m, trn_dataset.num_classes());
  EXPECT_EQ(n, trn_dataset.out.labels.size());

  typedef decltype(trn_dataset)::eval_type trn_eval_type;
  EXPECT_TRUE((std::is_same<sdca::eval_train<double, sdca::multiclass_output>,
               trn_eval_type>::value));

  auto tst_dataset = sdca::make_dataset_test(
    sdca::make_input_feature(d, n, &features[0]),
    sdca::make_output_multiclass(labels.begin(), labels.end())
    );

  EXPECT_EQ(d, tst_dataset.num_dimensions());
  EXPECT_EQ(n, tst_dataset.num_examples());
  EXPECT_EQ(m, tst_dataset.num_classes());
  EXPECT_EQ(n, tst_dataset.out.labels.size());

  typedef decltype(tst_dataset)::eval_type tst_eval_type;
  EXPECT_TRUE((std::is_same<sdca::eval_test<double, sdca::multiclass_output>,
               tst_eval_type>::value));
}


TEST(SolverDatasetTest, feature_in_multilabel_out) {
  std::vector<float> features;
  std::vector<std::vector<sdca::size_type>> labels;

  sdca::size_type n = 6, m = 4, d = 5, pow_from = 0, pow_to = 1;
  labels = { {1, 2}, {3}, {1, 2}, {1, 3}, {2, 3}, {2, 3, 4} };

  sdca::size_type nnz = 0;
  std::for_each(labels.begin(), labels.end(),
                [&](std::vector<sdca::size_type>& yi){ nnz += yi.size(); });

  std::mt19937 gen(1);
  test_populate_real(n * d, pow_from, pow_to, 1.0f, gen, features);

  auto trn_dataset = sdca::make_dataset_train(
    sdca::make_input_feature(d, n, &features[0]),
    sdca::make_output_multilabel(labels)
    );

  EXPECT_EQ(d, trn_dataset.num_dimensions());
  EXPECT_EQ(n, trn_dataset.num_examples());
  EXPECT_EQ(m, trn_dataset.num_classes());
  EXPECT_EQ(nnz, trn_dataset.out.labels.size());
  EXPECT_EQ(n + 1, trn_dataset.out.offsets.size());

  typedef decltype(trn_dataset)::eval_type trn_eval_type;
  EXPECT_TRUE((std::is_same<sdca::eval_train<double, sdca::multilabel_output>,
               trn_eval_type>::value));

  auto tst_dataset = sdca::make_dataset_test(
    sdca::make_input_feature(d, n, &features[0]),
    sdca::make_output_multilabel(labels)
    );

  EXPECT_EQ(d, tst_dataset.num_dimensions());
  EXPECT_EQ(n, tst_dataset.num_examples());
  EXPECT_EQ(m, tst_dataset.num_classes());
  EXPECT_EQ(nnz, tst_dataset.out.labels.size());
  EXPECT_EQ(n + 1, tst_dataset.out.offsets.size());

  typedef decltype(tst_dataset)::eval_type tst_eval_type;
  EXPECT_TRUE((std::is_same<sdca::eval_test<double, sdca::multilabel_output>,
               tst_eval_type>::value));
}


TEST(SolverDatasetTest, kernel_in_multiclass_out) {
  sdca::size_type n = 50, n_tst = 100, m = 3, pow_from = 0, pow_to = 1;
  std::vector<double> kernel;
  std::vector<sdca::size_type> trn_labels;
  std::vector<sdca::size_type> tst_labels;

  std::mt19937 gen(1);
  test_populate_real(n * n_tst, pow_from, pow_to, 1.0, gen, kernel);
  test_populate_int<sdca::size_type>(n, 1, m, gen, trn_labels);
  test_populate_int<sdca::size_type>(n_tst, 1, m, gen, tst_labels);

  auto trn_dataset = sdca::make_dataset_train<float>(
    sdca::make_input_kernel(n, &kernel[0]),
    sdca::make_output_multiclass(trn_labels.begin(), trn_labels.end())
    );

  EXPECT_EQ(n, trn_dataset.num_examples());
  EXPECT_EQ(m, trn_dataset.num_classes());
  EXPECT_EQ(n, trn_dataset.in.num_train_examples);
  EXPECT_EQ(n, trn_dataset.out.labels.size());

  typedef decltype(trn_dataset)::eval_type trn_eval_type;
  EXPECT_TRUE((std::is_same<sdca::eval_train<float, sdca::multiclass_output>,
               trn_eval_type>::value));

  auto tst_dataset = sdca::make_dataset_test<float>(
    sdca::make_input_kernel(n, n_tst, &kernel[0]),
    sdca::make_output_multiclass(tst_labels.begin(), tst_labels.end())
    );

  EXPECT_EQ(n_tst, tst_dataset.num_examples());
  EXPECT_EQ(m, tst_dataset.num_classes());
  EXPECT_EQ(n, tst_dataset.in.num_train_examples);
  EXPECT_EQ(n_tst, tst_dataset.out.labels.size());

  typedef decltype(tst_dataset)::eval_type tst_eval_type;
  EXPECT_TRUE((std::is_same<sdca::eval_test<float, sdca::multiclass_output>,
               tst_eval_type>::value));
}


TEST(SolverDatasetTest, kernel_in_multilabel_out) {
  std::vector<double> kernel;
  std::vector<std::vector<sdca::size_type>> trn_labels;
  std::vector<std::vector<sdca::size_type>> tst_labels;

  sdca::size_type n = 6, n_tst = 7, m = 4, pow_from = 0, pow_to = 1;
  trn_labels = { {1, 2}, {3}, {1, 2}, {1, 3}, {2, 3}, {2, 3, 4} };
  tst_labels = { {1, 2}, {3}, {1, 2}, {1, 3}, {2, 3}, {2, 3, 4}, {1, 4} };

  sdca::size_type nnz_trn = 0;
  std::for_each(trn_labels.begin(), trn_labels.end(),
                [&](std::vector<sdca::size_type>& yi){ nnz_trn += yi.size(); });

  sdca::size_type nnz_tst = 0;
  std::for_each(tst_labels.begin(), tst_labels.end(),
                [&](std::vector<sdca::size_type>& yi){ nnz_tst += yi.size(); });

  std::mt19937 gen(1);
  test_populate_real(n * n_tst, pow_from, pow_to, 1.0, gen, kernel);

  auto trn_dataset = sdca::make_dataset_train<float>(
    sdca::make_input_kernel(n, &kernel[0]),
    sdca::make_output_multilabel(trn_labels)
    );

  EXPECT_EQ(n, trn_dataset.num_examples());
  EXPECT_EQ(m, trn_dataset.num_classes());
  EXPECT_EQ(n, trn_dataset.in.num_train_examples);
  EXPECT_EQ(nnz_trn, trn_dataset.out.labels.size());
  EXPECT_EQ(n + 1, trn_dataset.out.offsets.size());

  typedef decltype(trn_dataset)::eval_type trn_eval_type;
  EXPECT_TRUE((std::is_same<sdca::eval_train<float, sdca::multilabel_output>,
               trn_eval_type>::value));

  auto tst_dataset = sdca::make_dataset_test<float>(
    sdca::make_input_kernel(n, n_tst, &kernel[0]),
    sdca::make_output_multilabel(tst_labels)
    );

  EXPECT_EQ(n_tst, tst_dataset.num_examples());
  EXPECT_EQ(m, tst_dataset.num_classes());
  EXPECT_EQ(n, tst_dataset.in.num_train_examples);
  EXPECT_EQ(nnz_tst, tst_dataset.out.labels.size());
  EXPECT_EQ(n_tst + 1, tst_dataset.out.offsets.size());

  typedef decltype(tst_dataset)::eval_type tst_eval_type;
  EXPECT_TRUE((std::is_same<sdca::eval_test<float, sdca::multilabel_output>,
               tst_eval_type>::value));
}
