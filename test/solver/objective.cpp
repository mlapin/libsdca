#include <cstdio>
#include <typeinfo>

#include "gtest/gtest.h"
#include "sdca/solver/objective.h"

#include "test_util.h"


template <typename Data,
          typename Result,
          typename Maker>
inline void
test_objective_default(Maker maker) {
  auto obj = maker();

  EXPECT_EQ(1.0, obj.c);
  EXPECT_EQ(1UL, obj.k);

  typedef typename decltype(obj)::data_type data_type;
  typedef typename decltype(obj)::result_type result_type;
  EXPECT_TRUE((std::is_same<Data, data_type>::value));
  EXPECT_TRUE((std::is_same<Result, result_type>::value));
}


template <typename Data,
          typename Result,
          typename Maker>
inline void
test_objective_simple(Result c, std::size_t k, Maker maker) {
  auto obj = maker();

  EXPECT_EQ(c, obj.c);
  EXPECT_EQ(k, obj.k);

  typedef typename decltype(obj)::data_type data_type;
  typedef typename decltype(obj)::result_type result_type;
  EXPECT_TRUE((std::is_same<Data, data_type>::value));
  EXPECT_TRUE((std::is_same<Result, result_type>::value));
}


template <typename Data,
          typename Result,
          typename Maker>
inline void
test_objective_simple(Result c, Result gamma, std::size_t k, Maker maker) {
  auto obj = maker();

  EXPECT_EQ(c, obj.c);
  EXPECT_EQ(gamma, obj.gamma);
  EXPECT_EQ(k, obj.k);

  typedef typename decltype(obj)::data_type data_type;
  typedef typename decltype(obj)::result_type result_type;
  EXPECT_TRUE((std::is_same<Data, data_type>::value));
  EXPECT_TRUE((std::is_same<Result, result_type>::value));
}


template <typename Data,
          typename Result>
inline void
test_objective_l2_entropy_topk() {
  auto maker_default = []() -> sdca::l2_entropy_topk<Data, double> {
    return sdca::make_objective_l2_entropy_topk<Data>(); };
  test_objective_default<Data, double>(maker_default);

  Result c = 3.14f;
  std::size_t k = 2;
  auto maker_simple = [=]() -> sdca::l2_entropy_topk<Data, Result> {
    return sdca::make_objective_l2_entropy_topk<Data>(c, k); };
  test_objective_simple<Data, Result>(c, k, maker_simple);
}


template <typename Data,
          typename Result>
inline void
test_objective_l2_hinge_topk() {
  auto maker_default = []() -> sdca::l2_hinge_topk<Data, double> {
    return sdca::make_objective_l2_hinge_topk<Data>(); };
  test_objective_default<Data, double>(maker_default);

  Result c = 3.14f;
  std::size_t k = 2;
  auto maker_simple = [=]() -> sdca::l2_hinge_topk<Data, Result> {
    return sdca::make_objective_l2_hinge_topk<Data>(c, k); };
  test_objective_simple<Data, Result>(c, k, maker_simple);
}


template <typename Data,
          typename Result>
inline void
test_objective_l2_topk_hinge() {
  auto maker_default = []() -> sdca::l2_topk_hinge<Data, double> {
    return sdca::make_objective_l2_topk_hinge<Data>(); };
  test_objective_default<Data, double>(maker_default);

  Result c = 3.14f;
  std::size_t k = 2;
  auto maker_simple = [=]() -> sdca::l2_topk_hinge<Data, Result> {
    return sdca::make_objective_l2_topk_hinge<Data>(c, k); };
  test_objective_simple<Data, Result>(c, k, maker_simple);
}


template <typename Data,
          typename Result>
inline void
test_objective_l2_hinge_topk_smooth() {
  auto maker_default = []() -> sdca::l2_hinge_topk_smooth<Data, double> {
    return sdca::make_objective_l2_hinge_topk_smooth<Data>(); };
  test_objective_default<Data, double>(maker_default);

  Result c = 3.14f, gamma = 2.72f;
  std::size_t k = 2;
  auto maker_simple = [=]() -> sdca::l2_hinge_topk_smooth<Data, Result> {
    return sdca::make_objective_l2_hinge_topk_smooth<Data>(c, gamma, k); };
  test_objective_simple<Data, Result>(c, gamma, k, maker_simple);
}


template <typename Data,
          typename Result>
inline void
test_objective_l2_topk_hinge_smooth() {
  auto maker_default = []() -> sdca::l2_topk_hinge_smooth<Data, double> {
    return sdca::make_objective_l2_topk_hinge_smooth<Data>(); };
  test_objective_default<Data, double>(maker_default);

  Result c = 3.14f, gamma = 2.72f;
  std::size_t k = 2;
  auto maker_simple = [=]() -> sdca::l2_topk_hinge_smooth<Data, Result> {
    return sdca::make_objective_l2_topk_hinge_smooth<Data>(c, gamma, k); };
  test_objective_simple<Data, Result>(c, gamma, k, maker_simple);
}


TEST(SolverObjectiveTest, l2_entropy_topk) {
  test_objective_l2_entropy_topk<float, float>();
  test_objective_l2_entropy_topk<float, double>();
  test_objective_l2_entropy_topk<double, float>();
  test_objective_l2_entropy_topk<double, double>();
}


TEST(SolverObjectiveTest, l2_hinge_topk) {
  test_objective_l2_hinge_topk<float, float>();
  test_objective_l2_hinge_topk<float, double>();
  test_objective_l2_hinge_topk<double, float>();
  test_objective_l2_hinge_topk<double, double>();
}


TEST(SolverObjectiveTest, l2_topk_hinge) {
  test_objective_l2_topk_hinge<float, float>();
  test_objective_l2_topk_hinge<float, double>();
  test_objective_l2_topk_hinge<double, float>();
  test_objective_l2_topk_hinge<double, double>();
}


TEST(SolverObjectiveTest, l2_hinge_topk_smooth) {
  test_objective_l2_hinge_topk_smooth<float, float>();
  test_objective_l2_hinge_topk_smooth<float, double>();
  test_objective_l2_hinge_topk_smooth<double, float>();
  test_objective_l2_hinge_topk_smooth<double, double>();
}


TEST(SolverObjectiveTest, l2_topk_hinge_smooth) {
  test_objective_l2_topk_hinge_smooth<float, float>();
  test_objective_l2_topk_hinge_smooth<float, double>();
  test_objective_l2_topk_hinge_smooth<double, float>();
  test_objective_l2_topk_hinge_smooth<double, double>();
}
