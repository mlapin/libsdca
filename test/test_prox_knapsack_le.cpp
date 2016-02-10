#include <cstdio>

#include "gtest/gtest.h"
#include "sdca/prox/knapsack_le.h"

#include "test_util.h"

template <typename Type>
inline void
test_prox_knapsack_le_check_feasible(
    const Type lo, const Type hi, const Type rhs,
    const Type eps, std::vector<Type>& v) {
  sdca::prox_knapsack_le(v.begin(), v.end(), lo, hi, rhs);

  std::for_each(v.begin(), v.end(), [=](const Type x){
    ASSERT_GE(x, lo); });
  std::for_each(v.begin(), v.end(), [=](const Type x){
    ASSERT_LE(x, hi); });
  Type sum = std::accumulate(v.begin(), v.end(), static_cast<Type>(0));
  ASSERT_LE(sum, rhs + eps);
}

template <typename Type>
inline void
test_prox_knapsack_le_set_params(
    const std::vector<Type>& v,
    std::mt19937& gen,
    std::uniform_real_distribution<Type>& d_lo,
    std::uniform_real_distribution<Type>& d_hi,
    std::uniform_real_distribution<Type>& d_rhs,
    Type& lo, Type& hi, Type& rhs, Type& eps) {
  for (;;) {
    lo = d_lo(gen);
    hi = d_hi(gen);
    rhs = d_rhs(gen);
    if (lo <= hi && lo * v.size() <= rhs && hi * v.size() >= rhs) break;
  }
  Type max(*std::max_element(v.begin(), v.end()));
  eps = v.size() * std::max(static_cast<Type>(1), std::abs(max))
      * std::numeric_limits<Type>::epsilon();
}

template <typename Type>
inline void
test_prox_knapsack_le_feasible(
    const int pow_from, const int pow_to, const int tol) {
  std::mt19937 gen(1);
  std::uniform_real_distribution<Type> d_lo(-2, 0.5);
  std::uniform_real_distribution<Type> d_hi(-0.5, 2);
  std::uniform_real_distribution<Type> d_rhs(-5, 5);

  Type lo, hi, rhs, eps;
  std::vector<Type> v;

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_prox_knapsack_le_set_params(
        v, gen, d_lo, d_hi, d_rhs, lo, hi, rhs, eps);
      test_prox_knapsack_le_check_feasible(lo, hi, rhs, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_knapsack_le_set_params(
        v, gen, d_lo, d_hi, d_rhs, lo, hi, rhs, eps);
      test_prox_knapsack_le_check_feasible(lo, hi, rhs, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_knapsack_le_set_params(
        v, gen, d_lo, d_hi, d_rhs, lo, hi, rhs, eps);
      test_prox_knapsack_le_check_feasible(lo, hi, rhs, tol * eps, v);
    }
  }

  for (int i = 0; i < 100; ++i) {
    for (int p = pow_from; p < pow_to; ++p) {
      test_populate(25, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate(25, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_knapsack_le_set_params(
        v, gen, d_lo, d_hi, d_rhs, lo, hi, rhs, eps);
      test_prox_knapsack_le_check_feasible(lo, hi, rhs, tol * eps, v);
    }
  }
}

TEST(ProxKnapsackLETest, test_prox_feasible_float) {
  test_prox_knapsack_le_feasible<float>(-3, 3, 128);
}

TEST(ProxKnapsackLETest, test_prox_feasible_double) {
  test_prox_knapsack_le_feasible<double>(-6, 6, 128);
}
