#include "sdca/prox/topk_entropy.h"
#include "test_util.h"

template <typename Type>
inline void
test_prox_topk_entropy_check_feasible(const ptrdiff_t k,
    const Type eps, std::vector<Type>& v) {
  sdca::prox_topk_entropy(v.begin(), v.end(), k);

  Type sum = std::accumulate(v.begin(), v.end(), static_cast<Type>(0));

  Type lo(0), hi(sum / static_cast<Type>(k)), rhs(1);
  std::for_each(v.begin(), v.end(), [=](const Type x){
    ASSERT_GE(x, lo); });
  std::for_each(v.begin(), v.end(), [=](const Type x){
    ASSERT_LE(x, hi + eps); });
  ASSERT_LE(sum, rhs + eps);
}

template <typename Type>
inline void
test_prox_topk_entropy_set_params(
    const std::vector<Type>& v,
    std::mt19937& gen,
    std::uniform_int_distribution<ptrdiff_t>& d_k,
    ptrdiff_t& k, Type& eps) {
  k = d_k(gen);
  Type max(*std::max_element(v.begin(), v.end()));
  eps = std::numeric_limits<Type>::epsilon()
      * std::max(static_cast<Type>(1), std::abs(max))
      * static_cast<Type>(v.size());
}

template <typename Type>
inline void
test_prox_topk_entropy_feasible(
    const int pow_from, const int pow_to, const Type tol) {
  std::mt19937 gen(1);
  std::uniform_int_distribution<ptrdiff_t> d_k(1, 10);

  Type eps;
  ptrdiff_t k;
  std::vector<Type> v;

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_prox_topk_entropy_set_params(v, gen, d_k, k, eps);
      test_prox_topk_entropy_check_feasible(k, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_entropy_set_params(v, gen, d_k, k, eps);
      test_prox_topk_entropy_check_feasible(k, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_entropy_set_params(v, gen, d_k, k, eps);
      test_prox_topk_entropy_check_feasible(k, tol * eps, v);
    }
  }

  for (int i = 0; i < 100; ++i) {
    for (int p = pow_from; p < pow_to; ++p) {
      test_populate_real(25, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(25, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_entropy_set_params(v, gen, d_k, k, eps);
      test_prox_topk_entropy_check_feasible(k, tol * eps, v);
    }
  }
}

TEST(ProxTopKEntropyTest, test_prox_feasible_float) {
  test_prox_topk_entropy_feasible<float>(-3, 3, 1);
}

TEST(ProxTopKEntropyTest, test_prox_feasible_double) {
  test_prox_topk_entropy_feasible<double>(-6, 6, 1);
}
