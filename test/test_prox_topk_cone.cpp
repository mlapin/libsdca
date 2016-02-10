#include <cstdio>

#include "gtest/gtest.h"
#include "sdca/prox/topk_cone.h"

#include "test_util.h"

template <typename Type>
inline void
test_prox_topk_cone_check_feasible(const ptrdiff_t k,
    const Type eps, std::vector<Type>& v) {
  sdca::prox_topk_cone(v.begin(), v.end(), k);

  Type sum = std::accumulate(v.begin(), v.end(), static_cast<Type>(0));

  Type lo(0), hi(sum/k);
  std::for_each(v.begin(), v.end(), [=](const Type x){
    ASSERT_GE(x, lo); });
  std::for_each(v.begin(), v.end(), [=](const Type x){
    ASSERT_LE(x, hi + eps); });
}

template <typename Type>
inline void
test_prox_topk_cone_set_params(
    const std::vector<Type>& v,
    std::mt19937& gen,
    std::uniform_int_distribution<ptrdiff_t>& d_k,
    ptrdiff_t& k, Type& eps) {
  k = d_k(gen);
  Type max(*std::max_element(v.begin(), v.end()));
  eps = v.size() * std::max(static_cast<Type>(1), std::abs(max))
      * std::numeric_limits<Type>::epsilon();
}

template <typename Type>
inline void
test_prox_topk_cone_feasible(
    const int pow_from, const int pow_to, const int tol) {
  std::mt19937 gen(1);
  std::uniform_int_distribution<ptrdiff_t> d_k(1, 10);

  Type eps;
  ptrdiff_t k;
  std::vector<Type> v;

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_prox_topk_cone_set_params(v, gen, d_k, k, eps);
      test_prox_topk_cone_check_feasible(k, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_cone_set_params(v, gen, d_k, k, eps);
      test_prox_topk_cone_check_feasible(k, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_cone_set_params(v, gen, d_k, k, eps);
      test_prox_topk_cone_check_feasible(k, tol * eps, v);
    }
  }

  for (int i = 0; i < 100; ++i) {
    for (int p = pow_from; p < pow_to; ++p) {
      test_populate(25, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate(25, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_cone_set_params(v, gen, d_k, k, eps);
      test_prox_topk_cone_check_feasible(k, tol * eps, v);
    }
  }
}

TEST(ProxTopKConeTest, test_prox_feasible_float) {
  test_prox_topk_cone_feasible<float>(-3, 3, 1);
}

TEST(ProxTopKConeTest, test_prox_feasible_double) {
  test_prox_topk_cone_feasible<double>(-6, 6, 1);
}
