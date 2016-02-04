#ifndef SDCA_TEST_UTIL_H
#define SDCA_TEST_UTIL_H

#include <cmath>
#include <limits>
#include <random>
#include <vector>

template <typename Type>
inline void
test_populate(
    const int n,
    const int pow_from,
    const int pow_to,
    const Type coeff,
    std::mt19937 &gen,
    std::vector<Type>& v
  ) {
  Type base(10);
  for (int p = pow_from; p < pow_to; ++p) {
    Type x = std::pow(base, p);
    std::uniform_real_distribution<Type> d(x, x*base);
    for (int i = 0; i < n; ++i) {
      v.push_back(coeff * d(gen));
    }
  }
}

template <typename Type>
inline void
test_add_0_1_eps_min(const Type coeff, std::vector<Type>& v) {
  v.push_back(0);
  v.push_back(coeff);
  v.push_back(coeff * std::numeric_limits<Type>::epsilon());
  v.push_back(coeff * std::numeric_limits<Type>::min());
}

template <typename Type>
inline void
test_add_0_1_eps_min_max(const Type coeff, std::vector<Type>& v) {
  v.push_back(0);
  v.push_back(coeff);
  v.push_back(coeff * std::numeric_limits<Type>::epsilon());
  v.push_back(coeff * std::numeric_limits<Type>::min());
  v.push_back(coeff * std::numeric_limits<Type>::max());
}

#endif
