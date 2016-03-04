#include "sdca/prox/topk_cone_biased.h"
#include "test_util.h"

template <typename Type>
inline void
test_prox_topk_cone_biased_check_feasible(const ptrdiff_t k, const Type rho,
    const Type eps, std::vector<Type>& v) {
  sdca::prox_topk_cone_biased(v.begin(), v.end(), k, rho);

  Type sum = std::accumulate(v.begin(), v.end(), static_cast<Type>(0));

  Type lo(0), hi(sum / static_cast<Type>(k));
  std::for_each(v.begin(), v.end(), [=](const Type x){
    ASSERT_GE(x, lo); });
  std::for_each(v.begin(), v.end(), [=](const Type x){
    ASSERT_LE(x, hi + eps); });
}

template <typename Type>
inline void
test_prox_topk_cone_biased_set_params(
    const std::vector<Type>& v,
    std::mt19937& gen,
    std::uniform_int_distribution<ptrdiff_t>& d_k,
    std::uniform_real_distribution<Type>& d_rho,
    ptrdiff_t& k, Type& rho, Type& eps) {
  k = d_k(gen);
  rho = d_rho(gen);
  Type max(*std::max_element(v.begin(), v.end()));
  eps = std::numeric_limits<Type>::epsilon()
      * std::max(static_cast<Type>(1), std::abs(max))
      * static_cast<Type>(v.size());
}

template <typename Type>
inline void
test_prox_topk_cone_biased_feasible(
    const int pow_from, const int pow_to, const Type tol) {
  std::mt19937 gen(1);
  std::uniform_int_distribution<ptrdiff_t> d_k(1, 10);
  std::uniform_real_distribution<Type> d_rho(0, 2);

  Type rho, eps;
  ptrdiff_t k;
  std::vector<Type> v;

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_prox_topk_cone_biased_set_params(v, gen, d_k, d_rho, k, rho, eps);
      test_prox_topk_cone_biased_check_feasible(k, rho, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_cone_biased_set_params(v, gen, d_k, d_rho, k, rho, eps);
      test_prox_topk_cone_biased_check_feasible(k, rho, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_cone_biased_set_params(v, gen, d_k, d_rho, k, rho, eps);
      test_prox_topk_cone_biased_check_feasible(k, rho, tol * eps, v);
    }
  }

  for (int i = 0; i < 100; ++i) {
    for (int p = pow_from; p < pow_to; ++p) {
      test_populate_real(25, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(25, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_cone_biased_set_params(v, gen, d_k, d_rho, k, rho, eps);
      test_prox_topk_cone_biased_check_feasible(k, rho, tol * eps, v);
    }
  }
}

TEST(ProxTopKConeBiasedTest, test_prox_feasible_float) {
  test_prox_topk_cone_biased_feasible<float>(-3, 3, 1);
}

TEST(ProxTopKConeBiasedTest, test_prox_feasible_double) {
  test_prox_topk_cone_biased_feasible<double>(-6, 6, 1);
}

