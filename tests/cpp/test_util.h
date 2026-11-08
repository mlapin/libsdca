#ifndef SDCA_TEST_UTIL_H
#define SDCA_TEST_UTIL_H

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
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

// Whether long double extends the precision and exponent range of double
// (x87, binary128). Where it is double (Apple arm64, MSVC), the extended
// retries have the range of double: tests beyond it expect an exception or
// the documented tolerance instead (docs/numerics.md, platform ranges).
constexpr bool wide_long_double() {
  return std::numeric_limits<long double>::digits >= 64 &&
         std::numeric_limits<long double>::max_exponent > std::numeric_limits<double>::max_exponent;
}

template <typename IntType, typename UIntType, typename RealType>
inline void test_populate_real(const UIntType n, const IntType pow_from, const IntType pow_to,
                               const RealType coeff, std::mt19937& gen, std::vector<RealType>& v) {
  RealType base(10);
  for (IntType p = pow_from; p < pow_to; ++p) {
    RealType x = static_cast<RealType>(std::pow(base, p));
    std::uniform_real_distribution<RealType> d(x, x * base);
    for (UIntType i = 0; i < n; ++i) {
      v.push_back(coeff * d(gen));
    }
  }
}

template <typename Type, typename UIntType>
inline void test_populate_int(const UIntType n, const Type a, const Type b, std::mt19937& gen,
                              std::vector<Type>& v) {
  std::uniform_int_distribution<Type> d(a, b);
  for (UIntType i = 0; i < n; ++i) {
    v.push_back(d(gen));
  }
}

template <typename Type>
inline void test_add_0_1_eps_min(const Type coeff, std::vector<Type>& v) {
  v.push_back(0);
  v.push_back(coeff);
  v.push_back(coeff * std::numeric_limits<Type>::epsilon());
  v.push_back(coeff * std::numeric_limits<Type>::min());
}

template <typename Type>
inline void test_add_0_1_eps_min_max(const Type coeff, std::vector<Type>& v) {
  v.push_back(0);
  v.push_back(coeff);
  v.push_back(coeff * std::numeric_limits<Type>::epsilon());
  v.push_back(coeff * std::numeric_limits<Type>::min());
  v.push_back(coeff * std::numeric_limits<Type>::max());
}

// The KKT residual of x for min 0.5 (<x, x> + rho <1, x>^2) - <a, x> over
// lo <= x <= hi, <1, x> <= rhs (= rhs if equality): each condition relative
// to the magnitude of its own terms, minimized over the candidate thresholds.
template <typename T>
inline double knapsack_kkt_residual(const std::vector<T>& a, const std::vector<T>& x, const T lo,
                                    const T hi, const T rhs, const T rho, const bool equality) {
  typedef long double L;
  auto rel = [](L v, L m) { return v <= 0 ? L(0) : (m > 0 ? v / m : v); };
  L s = 0, abs_s = 0;
  for (const T v : x) {
    if (!(v >= lo && v <= hi)) return HUGE_VAL;
    s += L(v);
    abs_s += std::abs(L(v));
  }
  const L mass = abs_s + std::abs(L(rhs));
  const L sum_res = equality ? rel(std::abs(s - L(rhs)), mass) : rel(s - L(rhs), mass);
  const bool active = equality || s - L(rhs) >= -64 * L(std::numeric_limits<T>::epsilon()) * mass;
  std::vector<L> ts{L(rho) * s};
  for (std::size_t i = 0; i < a.size(); ++i)
    if (lo < x[i] && x[i] < hi) ts.push_back(L(a[i]) - L(x[i]));
  L best = std::numeric_limits<L>::infinity();
  for (const L t : ts) {
    L r = sum_res, terms = 0;  // terms: the magnitudes in <1, x> = sum (a_i - t) + bounds
    for (std::size_t i = 0; i < a.size(); ++i) {
      const L ai = L(a[i]), xi = L(x[i]);
      const L m = std::max(std::max(std::abs(ai), std::abs(xi)), std::abs(t));
      const bool is_free = L(lo) < xi && xi < L(hi);
      terms += is_free ? std::abs(ai) + std::abs(t) : std::abs(xi);
      if (lo == hi) continue;
      if (xi == L(lo)) r = std::max(r, rel((ai - L(lo)) - t, m));
      if (xi == L(hi)) r = std::max(r, rel(t - (ai - L(hi)), m));
      if (is_free) r = std::max(r, rel(std::abs((ai - xi) - t), m));
    }
    if (!equality) {
      const L m = std::max(std::abs(t), L(rho) * terms);
      r = std::max(r, rel(L(rho) * s - t, m));
      if (!active) r = std::max(r, rel(t - L(rho) * s, m));
    }
    best = std::min(best, r);
  }
  return static_cast<double>(best);
}

// KKT residual for min 0.5 (<x, x> + rho <1, x>^2) - <a, x> over
// 0 <= x_i <= <1, x> / k, <1, x> <= rhs (rhs = +inf for the cone).
template <typename T>
inline double topk_cone_simplex_kkt_residual(const std::vector<T>& a, const std::vector<T>& x,
                                             const std::ptrdiff_t k, const T rhs, const T rho) {
  typedef long double L;
  auto rel = [](L v, L m) { return v <= 0 ? L(0) : (m > 0 ? v / m : v); };
  const std::size_t uk = static_cast<std::size_t>(k);
  std::vector<L> sorted_a(a.begin(), a.end());
  std::sort(sorted_a.begin(), sorted_a.end(), std::greater<L>());
  L sum_top_k = 0, scale_top_k = 0;
  for (std::size_t i = 0; i < uk; ++i) {
    sum_top_k += sorted_a[i];
    scale_top_k += std::abs(sorted_a[i]);
  }
  L s = 0;
  for (const T v : x) {
    if (!(v >= 0) || !std::isfinite(v)) return HUGE_VAL;
    s += L(v);
  }
  if (s == 0) {
    if (rhs == 0) return 0;
    return static_cast<double>(rel(sum_top_k, scale_top_k));
  }
  const L K = L(k), u = s / K, eps = L(std::numeric_limits<T>::epsilon());
  L feas = std::isfinite(rhs) ? rel(s - L(rhs), s + std::abs(L(rhs))) : L(0);
  for (const T v : x) feas = std::max(feas, rel(L(v) - u, u));
  const bool sum_active = std::isfinite(rhs) && s - L(rhs) >= -64 * eps * (s + std::abs(L(rhs)));
  std::vector<L> ts{sorted_a[uk - 1] - u};
  if (uk < sorted_a.size()) ts.push_back(sorted_a[uk]);
  for (std::size_t i = 0; i < a.size(); ++i)
    if (x[i] > 0 && L(x[i]) < u * (1 - 16 * eps)) ts.push_back(L(a[i]) - L(x[i]));
  for (std::size_t nu = 0; nu <= uk; ++nu) {
    L sum_u = 0;
    for (std::size_t j = 0; j < nu; ++j) sum_u += sorted_a[j];
    if (nu < uk) ts.push_back((sum_u - L(nu) * u - K * L(rho) * s) / (K - L(nu)));
  }
  L best = std::numeric_limits<L>::infinity();
  for (const L t : ts) {
    L r = feas, cap_mult = 0, terms = scale_top_k + s + std::abs(t) * K;
    for (std::size_t i = 0; i < a.size(); ++i) {
      const L ai = L(a[i]), xi = L(x[i]);
      const L clipped = std::max(L(0), std::min(ai - t, u));
      const L m = std::max(std::max(std::abs(ai), u), std::abs(t));
      r = std::max(r, rel(std::abs(xi - clipped), m));
      if (ai - t > u) cap_mult += (ai - t) - u;
    }
    const L dual_s = t + cap_mult / K - L(rho) * s;
    const L ms = std::max(terms / K, std::abs(L(rho) * s));
    r = std::max(r, rel(-dual_s, ms));
    if (!sum_active) r = std::max(r, rel(dual_s, ms));
    best = std::min(best, r);
  }
  return static_cast<double>(best);
}

// KKT residual for min 0.5 (||x - a||^2 + ||y - b||^2) s.t. x, y >= 0, <1, x> = <1, y> <= rhs.
template <typename T>
inline double two_simplex_kkt_residual(const std::vector<T>& a, const std::vector<T>& b,
                                       const std::vector<T>& x, const std::vector<T>& y,
                                       const T rhs) {
  typedef long double L;
  auto rel = [](L v, L m) { return v <= 0 ? L(0) : (m > 0 ? v / m : v); };
  L sx = 0, sy = 0, max_a = -std::numeric_limits<L>::infinity(), max_b = max_a,
    scale = std::abs(L(rhs));
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (!(x[i] >= 0)) return HUGE_VAL;
    sx += L(x[i]);
    max_a = std::max(max_a, L(a[i]));
    scale = std::max(scale, std::abs(L(a[i])));
  }
  for (std::size_t j = 0; j < y.size(); ++j) {
    if (!(y[j] >= 0)) return HUGE_VAL;
    sy += L(y[j]);
    max_b = std::max(max_b, L(b[j]));
    scale = std::max(scale, std::abs(L(b[j])));
  }
  if (rhs == 0) return (sx == 0 && sy == 0) ? 0 : HUGE_VAL;
  const L eps = L(std::numeric_limits<T>::epsilon());
  const L mass = std::max(sx + sy + std::abs(L(rhs)), scale);
  L feas = std::max(rel(std::abs(sx - sy), mass), rel(std::max(sx, sy) - L(rhs), mass));
  const bool active = std::max(sx, sy) >= L(rhs) * (1 - 64 * eps);
  std::vector<L> tas{max_a}, tbs{max_b};
  for (std::size_t i = 0; i < x.size(); ++i)
    if (x[i] > 0) tas.push_back(L(a[i]) - L(x[i]));
  for (std::size_t j = 0; j < y.size(); ++j)
    if (y[j] > 0) tbs.push_back(L(b[j]) - L(y[j]));
  L best = std::numeric_limits<L>::infinity();
  for (const L ta : tas) {
    L ra = feas;
    for (std::size_t i = 0; i < x.size(); ++i) {
      const L m = std::max(std::max(std::abs(L(a[i])), std::abs(L(x[i]))), std::abs(ta));
      ra = std::max(ra, rel(std::abs(L(x[i]) - std::max(L(0), L(a[i]) - ta)), m));
    }
    for (const L tb0 : tbs) {
      for (const L tb : {tb0, -ta}) {
        L r = ra;
        for (std::size_t j = 0; j < y.size(); ++j) {
          const L m = std::max(std::max(std::abs(L(b[j])), std::abs(L(y[j]))), std::abs(tb));
          r = std::max(r, rel(std::abs(L(y[j]) - std::max(L(0), L(b[j]) - tb)), m));
        }
        const L mt = std::max(std::max(std::abs(ta), std::abs(tb)), scale);
        r = std::max(r, rel(-(ta + tb), mt));
        if (!active) r = std::max(r, rel(ta + tb, mt));
        best = std::min(best, r);
      }
    }
  }
  return static_cast<double>(best);
}

// KKT residual for min <x, log x> [+ 0.5 <x, x>] - <a, x> over 0 <= x_i <= hi, <1, x> = rhs.
template <typename T>
inline double capped_entropy_kkt_residual(const std::vector<T>& a, const std::vector<T>& x,
                                          const T hi, const T rhs, const bool quadratic) {
  typedef long double L;
  L s = 0;
  for (const T v : x) {
    if (!(v >= 0 && v <= hi)) return HUGE_VAL;
    s += L(v);
  }
  if (rhs == 0) return s == 0 ? 0 : HUGE_VAL;
  const L eps = L(std::numeric_limits<T>::epsilon());
  L r = std::abs(s - L(rhs)) / (s + L(rhs));
  const L floor = L(std::numeric_limits<T>::min()) / eps;
  L t_min = -std::numeric_limits<L>::infinity(), t_max = -t_min, scale = 1;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (L(x[i]) <= floor) continue;
    const L eta = L(a[i]) - std::log(L(x[i])) - (quadratic ? L(x[i]) : L(0));
    scale = std::max(scale, std::abs(L(a[i])) + std::abs(std::log(L(x[i]))) + L(x[i]));
    if (L(x[i]) >= L(hi) * (1 - 16 * eps)) {
      t_max = std::min(t_max, eta);
    } else {
      t_min = std::max(t_min, eta);
      t_max = std::min(t_max, eta);
    }
  }
  if (t_min > t_max) r = std::max(r, (t_min - t_max) / scale);
  return static_cast<double>(r);
}

inline float next_float(float x) { return nextafterf(x, std::numeric_limits<float>::infinity()); }

inline double next_float(double x) { return nextafter(x, std::numeric_limits<double>::infinity()); }

inline long double next_float(long double x) {
  return nextafterl(x, std::numeric_limits<long double>::infinity());
}

inline float prev_float(float x) { return nextafterf(x, -std::numeric_limits<float>::infinity()); }

inline double prev_float(double x) {
  return nextafter(x, -std::numeric_limits<double>::infinity());
}

inline long double prev_float(long double x) {
  return nextafterl(x, -std::numeric_limits<long double>::infinity());
}

#endif
