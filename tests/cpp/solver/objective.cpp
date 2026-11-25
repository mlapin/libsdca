#include "sdca/solver.h"
#include "test_util.h"

template <typename Data, typename Result, typename Maker>
inline void test_objective_default(Maker maker) {
  auto obj = maker();

  EXPECT_EQ(1.0, obj.c);
  EXPECT_EQ(1UL, obj.k);

  typedef typename decltype(obj)::data_type data_type;
  typedef typename decltype(obj)::result_type result_type;
  EXPECT_TRUE((std::is_same<Data, data_type>::value));
  EXPECT_TRUE((std::is_same<Result, result_type>::value));
}

template <typename Data, typename Result, typename Maker>
inline void test_objective_simple(Result c, std::size_t k, Maker maker) {
  auto obj = maker();

  EXPECT_EQ(c, obj.c);
  EXPECT_EQ(k, obj.k);

  typedef typename decltype(obj)::data_type data_type;
  typedef typename decltype(obj)::result_type result_type;
  EXPECT_TRUE((std::is_same<Data, data_type>::value));
  EXPECT_TRUE((std::is_same<Result, result_type>::value));
}

template <typename Data, typename Result, typename Maker>
inline void test_objective_simple(Result c, std::size_t k, Result gamma, Maker maker) {
  auto obj = maker();

  EXPECT_EQ(c, obj.c);
  EXPECT_EQ(gamma, obj.gamma);
  EXPECT_EQ(k, obj.k);

  typedef typename decltype(obj)::data_type data_type;
  typedef typename decltype(obj)::result_type result_type;
  EXPECT_TRUE((std::is_same<Data, data_type>::value));
  EXPECT_TRUE((std::is_same<Result, result_type>::value));
}

template <typename Data, typename Result>
inline void test_objective_l2_entropy_topk() {
  auto maker_default = []() -> sdca::l2_topk_cross_entropy<Data, Data> {
    return sdca::make_objective_l2_topk_cross_entropy<Data>(Data(1), 1);
  };
  test_objective_default<Data, Data>(maker_default);

  Result c = static_cast<Result>(3.14);
  std::size_t k = 2;
  auto maker_simple = [=]() -> sdca::l2_topk_cross_entropy<Data, Result> {
    return sdca::make_objective_l2_topk_cross_entropy<Data>(c, k);
  };
  test_objective_simple<Data, Result>(c, k, maker_simple);
}

template <typename Data, typename Result>
inline void test_objective_l2_hinge_topk() {
  auto maker_default = []() -> sdca::l2_topk_hinge_alpha<Data, Data> {
    return sdca::make_objective_l2_topk_hinge_alpha<Data>(Data(1), 1);
  };
  test_objective_default<Data, Data>(maker_default);

  Result c = static_cast<Result>(3.14);
  std::size_t k = 2;
  auto maker_simple = [=]() -> sdca::l2_topk_hinge_alpha<Data, Result> {
    return sdca::make_objective_l2_topk_hinge_alpha<Data>(c, k);
  };
  test_objective_simple<Data, Result>(c, k, maker_simple);
}

template <typename Data, typename Result>
inline void test_objective_l2_topk_hinge() {
  auto maker_default = []() -> sdca::l2_topk_hinge_beta<Data, Data> {
    return sdca::make_objective_l2_topk_hinge_beta<Data>(Data(1), 1);
  };
  test_objective_default<Data, Data>(maker_default);

  Result c = static_cast<Result>(3.14);
  std::size_t k = 2;
  auto maker_simple = [=]() -> sdca::l2_topk_hinge_beta<Data, Result> {
    return sdca::make_objective_l2_topk_hinge_beta<Data>(c, k);
  };
  test_objective_simple<Data, Result>(c, k, maker_simple);
}

template <typename Data, typename Result>
inline void test_objective_l2_hinge_topk_smooth() {
  auto maker_default = []() -> sdca::l2_topk_hinge_alpha_smooth<Data, Data> {
    return sdca::make_objective_l2_topk_hinge_alpha_smooth<Data>(Data(1), 1, Data(1));
  };
  test_objective_default<Data, Data>(maker_default);

  Result c = static_cast<Result>(3.14), gamma = static_cast<Result>(2.72);
  std::size_t k = 2;
  auto maker_simple = [=]() -> sdca::l2_topk_hinge_alpha_smooth<Data, Result> {
    return sdca::make_objective_l2_topk_hinge_alpha_smooth<Data>(c, k, gamma);
  };
  test_objective_simple<Data, Result>(c, k, gamma, maker_simple);
}

template <typename Data, typename Result>
inline void test_objective_l2_topk_hinge_smooth() {
  auto maker_default = []() -> sdca::l2_topk_hinge_beta_smooth<Data, Data> {
    return sdca::make_objective_l2_topk_hinge_beta_smooth<Data>(Data(1), 1, Data(1));
  };
  test_objective_default<Data, Data>(maker_default);

  Result c = static_cast<Result>(3.14), gamma = static_cast<Result>(2.72);
  std::size_t k = 2;
  auto maker_simple = [=]() -> sdca::l2_topk_hinge_beta_smooth<Data, Result> {
    return sdca::make_objective_l2_topk_hinge_beta_smooth<Data>(c, k, gamma);
  };
  test_objective_simple<Data, Result>(c, k, gamma, maker_simple);
}

TEST(SolverObjectiveTest, l2_topk_cross_entropy) {
  test_objective_l2_entropy_topk<float, float>();
  test_objective_l2_entropy_topk<float, double>();
  test_objective_l2_entropy_topk<double, float>();
  test_objective_l2_entropy_topk<double, double>();
}

TEST(SolverObjectiveTest, l2_topk_hinge_alpha) {
  test_objective_l2_hinge_topk<float, float>();
  test_objective_l2_hinge_topk<float, double>();
  test_objective_l2_hinge_topk<double, float>();
  test_objective_l2_hinge_topk<double, double>();
}

TEST(SolverObjectiveTest, l2_topk_hinge_beta) {
  test_objective_l2_topk_hinge<float, float>();
  test_objective_l2_topk_hinge<float, double>();
  test_objective_l2_topk_hinge<double, float>();
  test_objective_l2_topk_hinge<double, double>();
}

TEST(SolverObjectiveTest, l2_topk_hinge_alpha_smooth) {
  test_objective_l2_hinge_topk_smooth<float, float>();
  test_objective_l2_hinge_topk_smooth<float, double>();
  test_objective_l2_hinge_topk_smooth<double, float>();
  test_objective_l2_hinge_topk_smooth<double, double>();
}

TEST(SolverObjectiveTest, l2_topk_hinge_beta_smooth) {
  test_objective_l2_topk_hinge_smooth<float, float>();
  test_objective_l2_topk_hinge_smooth<float, double>();
  test_objective_l2_topk_hinge_smooth<double, float>();
  test_objective_l2_topk_hinge_smooth<double, double>();
}
