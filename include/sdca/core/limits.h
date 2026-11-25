#ifndef SDCA_CORE_LIMITS_H
#define SDCA_CORE_LIMITS_H

// The limits of a floating-point work type T that thresholds, tolerances and iteration counts
// derive from, each from std::numeric_limits<T> alone (shared by every instruction set).
// C++11 has no constexpr logarithm: the logarithm of a power of two 2^n is n ln 2, with ln 2
// in two parts (fdlibm: n kLn2Hi is exact for |n| < 2^20), rounded once to T.
// python/libsdca/_limits.py defines the same quantities; tests/cpp/rewrite_smoke_limits.cpp
// and tests/python/test_limits.py check both against std::log, numpy.finfo and mpmath.
#include <limits>

namespace sdca {
namespace limits {

constexpr long double kLn2Hi = 6.93147180369123816490e-01L;
constexpr long double kLn2Lo = 1.90821492927058770002e-10L;

// n ln 2 rounded to T: log(2^n).
template <typename T>
constexpr T log_pow2(const int n) {
  return static_cast<T>(n * kLn2Hi + n * kLn2Lo);
}

template <typename T>
constexpr int digits() {
  return std::numeric_limits<T>::digits;
}

// eps = 2^(1 - digits), the spacing of T at one.
template <typename T>
constexpr T eps() {
  return std::numeric_limits<T>::epsilon();
}

// u = eps / 2 = 2^-digits, the unit roundoff: |fl(x) - x| <= u |x| for normal x.
template <typename T>
constexpr T roundoff() {
  return std::numeric_limits<T>::epsilon() / 2;
}

// 1 - u, the largest T below one.
template <typename T>
constexpr T one_below() {
  return 1 - std::numeric_limits<T>::epsilon() / 2;
}

// 1 / eps = 2^(digits - 1): from there on T holds no fractions, and x - log(x) rounds to x
// within (log(x) + 1) ulp.
template <typename T>
constexpr T inv_eps() {
  return 1 / std::numeric_limits<T>::epsilon();
}

// min / eps = 2^(min_exponent + digits - 2): the least x whose eps x is normal, so that a
// tolerance c eps x (c >= 1) of a normal x does not flush to zero.
template <typename T>
constexpr T min_over_eps() {
  return std::numeric_limits<T>::min() / std::numeric_limits<T>::epsilon();
}

// log(eps) = (1 - digits) ln 2.
template <typename T>
constexpr T log_eps() {
  return log_pow2<T>(1 - std::numeric_limits<T>::digits);
}

// log(u) = -digits ln 2: below it exp(x) < u, so 1 + exp(x) rounds to one and W(exp(x)) =
// exp(x) (1 - exp(x) + ...) rounds to exp(x).
template <typename T>
constexpr T log_roundoff() {
  return log_pow2<T>(-std::numeric_limits<T>::digits);
}

// log(min) = (min_exponent - 1) ln 2: below it exp(x) is subnormal (or zero).
template <typename T>
constexpr T log_min() {
  return log_pow2<T>(std::numeric_limits<T>::min_exponent - 1);
}

// log(denorm_min) = (min_exponent - digits) ln 2: below it less ln 2 exp(x) rounds to zero.
template <typename T>
constexpr T log_denorm_min() {
  return log_pow2<T>(std::numeric_limits<T>::min_exponent - std::numeric_limits<T>::digits);
}

// log(max) = max_exponent ln 2 + log1p(-2^-digits), which rounds to max_exponent ln 2 for
// float and double: above it exp(x) overflows.
template <typename T>
constexpr T log_max() {
  return log_pow2<T>(std::numeric_limits<T>::max_exponent);
}

// log(max / 4) = (max_exponent - 2) ln 2 (to 2^-digits): below it exp(x) <= max / 4 (times
// 1 + |x| u, the rounding of the logarithm), a product with a factor in [0, 2] stays finite,
// and the range reduction 2^k exp(r) of an inline exp (k <= max_exponent - 2) has a normal
// 2^k.
template <typename T>
constexpr T log_quarter_max() {
  return log_pow2<T>(std::numeric_limits<T>::max_exponent - 2);
}

// (m / 2) ln 2 rounded to T: log(2^(m / 2)).
template <typename T>
constexpr T log_pow2_half(const int m) {
  return static_cast<T>((m * kLn2Hi + m * kLn2Lo) / 2);
}

// The range of an inline exp 2^k exp(r), k = round(x / ln 2), exp(r) in [2^-1/2, 2^1/2] (as
// core::exp): from (min_exponent - 1/2) ln 2 on, k >= min_exponent keeps 2^k exp(r) normal;
// below (max_exponent - 1/2) ln 2, k <= max_exponent - 1 keeps it finite.
template <typename T>
constexpr T exp_reduction_min() {
  return log_pow2_half<T>(2 * std::numeric_limits<T>::min_exponent - 1);
}
template <typename T>
constexpr T exp_reduction_max() {
  return log_pow2_half<T>(2 * std::numeric_limits<T>::max_exponent - 1);
}

namespace detail {
template <typename T>
constexpr T square(const T x) {
  return x * x;
}
// 2^n by squaring (exact for normal results).
template <typename T>
constexpr T pow2(const int n) {
  return n == 0       ? T(1)
         : n % 2 != 0 ? (n > 0 ? T(2) : T(0.5)) * pow2<T>(n - (n > 0 ? 1 : -1))
                      : square(pow2<T>(n / 2));
}
}  // namespace detail

// 2^((1 - digits) / 3), the power of two at or above cbrt(eps) within a factor 2^(2/3).
template <typename T>
constexpr T cbrt_eps() {
  return detail::pow2<T>((1 - std::numeric_limits<T>::digits) / 3);
}

namespace detail {
constexpr int ceil_log2(const long long n, const int bits = 0) {
  return (1LL << bits) >= n ? bits : ceil_log2(n, bits + 1);
}
}  // namespace detail

// The halvings that bring any bracket of finite T to two adjacent values, bisecting the bits
// (the ordered keys of float and double, sdca/core/project_base.h): T has 2^digits
// (max_exponent - min_exponent + 2) finite values, -0 included (32 for float, 64 for double).
template <typename T>
constexpr int key_bisection_steps() {
  return std::numeric_limits<T>::digits +
         detail::ceil_log2(std::numeric_limits<T>::max_exponent -
                           std::numeric_limits<T>::min_exponent + 2);
}

// The halvings that bring any bracket of finite T to two adjacent values, bisecting the values
// (lo + (hi - lo) / 2): from a width below 2^(max_exponent + 1) to the least spacing
// 2^(min_exponent - digits), max_exponent - min_exponent + digits + 1 (278 for float, 2099
// for double).
template <typename T>
constexpr int value_bisection_steps() {
  return std::numeric_limits<T>::max_exponent - std::numeric_limits<T>::min_exponent +
         std::numeric_limits<T>::digits + 1;
}

namespace detail {
constexpr int order_steps(const int order, const int bits, const int goal, const int steps = 0) {
  return bits >= goal ? steps : order_steps(order, bits * order, goal, steps + 1);
}
}  // namespace detail

// The steps of an iteration of order p (Newton 2, Halley 3, Householder 5) from a start whose
// relative error is below 2^-bits until it is below the unit roundoff 2^-digits: the error
// 2^-b goes to (C 2^-b)^p, so the bits correct grow as b p^n (C <= 1 in the basin, where the
// start is taken), and n = ceil(log_p(digits / bits)).
template <typename T>
constexpr int steps_of_order(const int order, const int bits) {
  return detail::order_steps(order, bits, std::numeric_limits<T>::digits);
}

}  // namespace limits
}  // namespace sdca

#endif  // SDCA_CORE_LIMITS_H
