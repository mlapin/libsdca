#include "sdca/prox/topk_simplex.h"
#include "test_util.h"

template <typename Type>
inline void
test_prox_topk_simplex_check_feasible(const ptrdiff_t k, const Type rhs,
    const Type eps, std::vector<Type>& v) {
  sdca::prox_topk_simplex(v.begin(), v.end(), k, rhs);

  Type sum = std::accumulate(v.begin(), v.end(), static_cast<Type>(0));

  Type lo(0), hi(sum / static_cast<Type>(k));
  std::for_each(v.begin(), v.end(), [=](const Type x){
    ASSERT_GE(x, lo); });
  std::for_each(v.begin(), v.end(), [=](const Type x){
    ASSERT_LE(x, hi + eps); });
  ASSERT_LE(sum, rhs + eps);
}

template <typename Type>
inline void
test_prox_topk_simplex_set_params(
    const std::vector<Type>& v,
    std::mt19937& gen,
    std::uniform_int_distribution<ptrdiff_t>& d_k,
    std::uniform_real_distribution<Type>& d_rhs,
    ptrdiff_t& k, Type& rhs, Type& eps) {
  k = d_k(gen);
  rhs = d_rhs(gen);
  Type max(*std::max_element(v.begin(), v.end()));
  eps = std::numeric_limits<Type>::epsilon()
      * std::max(static_cast<Type>(1), std::abs(max))
      * static_cast<Type>(v.size());
}

template <typename Type>
inline void
test_prox_topk_simplex_feasible(
    const int pow_from, const int pow_to, const Type tol) {
  std::mt19937 gen(1);
  std::uniform_int_distribution<ptrdiff_t> d_k(1, 10);
  std::uniform_real_distribution<Type> d_rhs(0, 10);

  Type rhs, eps;
  ptrdiff_t k;
  std::vector<Type> v;

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_prox_topk_simplex_set_params(v, gen, d_k, d_rhs, k, rhs, eps);
      test_prox_topk_simplex_check_feasible(k, rhs, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_simplex_set_params(v, gen, d_k, d_rhs, k, rhs, eps);
      test_prox_topk_simplex_check_feasible(k, rhs, tol * eps, v);
    }
  }

  for (int p = pow_from; p < pow_to; ++p) {
    v.clear();
    for (int i = 0; i < 100; ++i) {
      test_populate_real(100, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(100, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_simplex_set_params(v, gen, d_k, d_rhs, k, rhs, eps);
      test_prox_topk_simplex_check_feasible(k, rhs, tol * eps, v);
    }
  }

  for (int i = 0; i < 100; ++i) {
    for (int p = pow_from; p < pow_to; ++p) {
      test_populate_real(25, p, p + 1, static_cast<Type>(1), gen, v);
      test_populate_real(25, p, p + 1, -static_cast<Type>(1), gen, v);
      test_prox_topk_simplex_set_params(v, gen, d_k, d_rhs, k, rhs, eps);
      test_prox_topk_simplex_check_feasible(k, rhs, tol * eps, v);
    }
  }
}

TEST(ProxTopKSimplexTest, test_prox_feasible_float) {
#ifdef SDCA_ACCURATE_MATH
  test_prox_topk_simplex_feasible<float>(-3, 3, 5);
#else
  test_prox_topk_simplex_feasible<float>(-3, 3, 8);
#endif
}

TEST(ProxTopKSimplexTest, test_prox_feasible_double) {
#ifdef SDCA_ACCURATE_MATH
  test_prox_topk_simplex_feasible<double>(-6, 6, 4);
#else
  test_prox_topk_simplex_feasible<double>(-6, 6, 8);
#endif
}
