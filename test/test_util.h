#ifndef SDCA_TEST_UTIL_H
#define SDCA_TEST_UTIL_H

#include <cmath>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <limits>
#include <random>
#include <typeinfo>
#include <vector>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundef"
#pragma clang diagnostic ignored "-Wc++98-compat-pedantic"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#pragma clang diagnostic ignored "-Wpadded"
#pragma clang diagnostic ignored "-Wshift-sign-overflow"
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
#pragma clang diagnostic ignored "-Wweak-vtables"
#pragma clang diagnostic ignored "-Wdeprecated"
#endif

#include "gtest/gtest.h"

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

template <typename IntType,
          typename UIntType,
          typename RealType>
inline void
test_populate_real(
    const UIntType n,
    const IntType pow_from,
    const IntType pow_to,
    const RealType coeff,
    std::mt19937& gen,
    std::vector<RealType>& v
  ) {
  RealType base(10);
  for (IntType p = pow_from; p < pow_to; ++p) {
    RealType x = static_cast<RealType>(std::pow(base, p));
    std::uniform_real_distribution<RealType> d(x, x*base);
    for (UIntType i = 0; i < n; ++i) {
      v.push_back(coeff * d(gen));
    }
  }
}


template <typename Type,
          typename UIntType>
inline void
test_populate_int(
    const UIntType n,
    const Type a,
    const Type b,
    std::mt19937& gen,
    std::vector<Type>& v
  ) {
  std::uniform_int_distribution<Type> d(a, b);
  for (UIntType i = 0; i < n; ++i) {
    v.push_back(d(gen));
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



inline float next_float(float x) {
  return nextafterf(x, std::numeric_limits<float>::infinity());
}

inline double next_float(double x) {
  return nextafter(x, std::numeric_limits<double>::infinity());
}

inline long double next_float(long double x) {
  return nextafterl(x, std::numeric_limits<long double>::infinity());
}


inline float prev_float(float x) {
  return nextafterf(x, -std::numeric_limits<float>::infinity());
}

inline double prev_float(double x) {
  return nextafter(x, -std::numeric_limits<double>::infinity());
}

inline long double prev_float(long double x) {
  return nextafterl(x, -std::numeric_limits<long double>::infinity());
}

#endif
