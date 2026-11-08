#ifndef SDCA_MATH_COMPENSATED_H
#define SDCA_MATH_COMPENSATED_H

#include <cmath>
#include <limits>

namespace sdca {

namespace detail {

// s + e == a + b exactly (TwoSum), unless a + b overflows.
template <typename T>
inline T two_sum(const T a, const T b, T& e) {
  const T s = a + b, bb = s - a;
  e = (a - (s - bb)) + (b - bb);
  return s;
}

// p + e == a * b exactly (TwoProduct), unless a * b overflows or underflows.
template <typename T>
inline T two_product(const T a, const T b, T& e) {
  const T p = a * b;
  e = std::fma(a, b, -p);
  return p;
}

// A product of floats is exact in double.
template <>
inline float two_product(const float a, const float b, float& e) {
  const double exact = static_cast<double>(a) * static_cast<double>(b);
  const float p = static_cast<float>(exact);
  e = static_cast<float>(exact - static_cast<double>(p));
  return p;
}

// Veltkamp split: x = hi + lo, each with half the significand.
template <typename T>
inline void veltkamp_split(const T x, T& hi, T& lo) {
  const T factor = T((1ull << ((std::numeric_limits<T>::digits + 1) / 2)) + 1);
  const T scaled = factor * x;
  hi = scaled - (scaled - x);
  lo = x - hi;
}

// Dekker product: p + e == a * b exactly without fma, unless the split
// (|a|, |b| near the overflow threshold) or the product over/underflows.
template <typename T>
inline T split_product(const T a, const T b, T& e) {
  T a_hi, a_lo, b_hi, b_lo;
  veltkamp_split(a, a_hi, a_lo);
  veltkamp_split(b, b_hi, b_lo);
  const T p = a * b;
  e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
  return p;
}

// A sum kept as hi + lo: lo collects the rounding errors of the additions.
template <typename T>
struct compensated_sum {
  T hi = T(0), lo = T(0);
  void add(const T x) {
    T e;
    hi = two_sum(hi, x, e);
    lo += e;
  }
  void add_product(const T x, const T y) {
    T e;
    add(two_product(x, y, e));
    lo += e;
  }
  T value() const { return hi + lo; }
};

// A hinge margin below this fraction of its terms cancels and is refined;
// results that lose no bits keep them.
template <typename T>
constexpr T margin_cancellation() {
  return T(1) / T(16);
}

// The hinge margin 1 + x - y; when it cancels, with the low parts of both
// roundings: a tiny margin survives a large x - y or a large common offset.
template <typename T>
inline T margin(const T x, const T y) {
  T e1, e2;
  const T d = two_sum(x, -y, e1), hi = two_sum(d, T(1), e2), lo = e1 + e2;
  const bool cancels = std::abs(hi) < margin_cancellation<T>() * std::abs(d);
  return cancels && std::isfinite(lo) ? hi + lo : hi;
}

// beta v + a (z - c) with the rounding errors kept: the low part of z - c is
// the margin a large common offset of z and c would absorb.
template <typename T>
inline compensated_sum<T> scaled_difference(const T a, const T beta, const T v, const T z,
                                            const T c) {
  compensated_sum<T> s;
  T low;
  const T d = two_sum(z, -c, low);
  s.add_product(beta, v);
  s.add_product(a, d);
  s.lo += a * low;
  return s;
}

// Whether ha - aa (ha = <h, a>, aa = <a, a> / 2) is finite and far enough
// above underflow; zero takes the scaled form, with the same signed zero.
template <typename T>
inline bool smooth_hinge_normal(const T ha, const T aa) {
  const T value = std::abs(ha - aa);
  return value <= std::numeric_limits<T>::max() &&
         value >= std::numeric_limits<T>::min() / std::numeric_limits<T>::epsilon();
}

// (h a - a^2 / 2) / gamma; scales by a / gamma before multiplying where the
// products would overflow or lose bits to underflow.
template <typename T>
inline T smooth_hinge_term(const T h, const T a, const T gamma) {
  const T ha = h * a, aa = a * a / 2;
  return smooth_hinge_normal(ha, aa) ? (ha - aa) / gamma : (h - a / 2) * (a / gamma);
}

}  // namespace detail

}  // namespace sdca

#endif
