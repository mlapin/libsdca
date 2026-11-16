#ifndef SDCA_MATH_STORAGE_H
#define SDCA_MATH_STORAGE_H

// Exact storage conversions shared by the native C ABI and framework adapters.
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__GNUC__) || defined(__clang__)
#define SDCA_STORAGE_INLINE inline __attribute__((always_inline))
#else
#define SDCA_STORAGE_INLINE inline
#endif
#ifdef __has_builtin
#if __has_builtin(__builtin_bit_cast)
#define SDCA_STORAGE_BIT_CAST(To, x) __builtin_bit_cast(To, x)
#endif
#endif

namespace sdca {
namespace tensor {
// Storage formats: the stored type, 16-bit ones as raw bits.
struct float16 {
  typedef std::uint16_t type;
};
struct bfloat16 {
  typedef std::uint16_t type;
};
struct float32 {
  typedef float type;
};
struct float64 {
  typedef double type;
};

SDCA_STORAGE_INLINE std::uint32_t float_bits(float x) {
#ifdef SDCA_STORAGE_BIT_CAST
  return SDCA_STORAGE_BIT_CAST(std::uint32_t, x);
#else
  std::uint32_t bits;
  std::memcpy(&bits, &x, sizeof bits);
  return bits;
#endif
}

SDCA_STORAGE_INLINE float bits_float(std::uint32_t bits) {
#ifdef SDCA_STORAGE_BIT_CAST
  return SDCA_STORAGE_BIT_CAST(float, bits);
#else
  float x;
  std::memcpy(&x, &bits, sizeof x);
  return x;
#endif
}

// a if c, else b, through a bit mask: GCC does not if-convert the selects of
// floating-point results (they may trap), which would keep loops scalar.
SDCA_STORAGE_INLINE std::uint32_t select(bool c, std::uint32_t a, std::uint32_t b) {
  const std::uint32_t mask = 0u - static_cast<std::uint32_t>(c);
  return (a & mask) | (b & ~mask);
}

// The stored values as float (exact) or double.
SDCA_STORAGE_INLINE float widen(float16, std::uint16_t h) {
  const std::uint32_t sign = (h & 0x8000u) << 16, em = h & 0x7fffu;
  // Subnormal: m 2^-24; normal: the exponent rebiased (15 -> 127); inf and NaN keep the payload.
  const float subnormal = float(static_cast<std::int32_t>(em)) * 5.9604644775390625e-8f;
  const std::uint32_t normal = (em << 13) + (112u << 23), special = (em << 13) | 0x7f800000u;
  const std::uint32_t bits =
      select(em < 0x400u, float_bits(subnormal), select(em >= 0x7c00u, special, normal));
  return bits_float(bits | sign);
}
SDCA_STORAGE_INLINE float widen(bfloat16, std::uint16_t h) {
  return bits_float(std::uint32_t(h) << 16);
}
SDCA_STORAGE_INLINE float widen(float32, float x) { return x; }
SDCA_STORAGE_INLINE double widen(float64, double x) { return x; }

// x rounded to nearest even into the format: overflow to inf, NaN quiet.
SDCA_STORAGE_INLINE std::uint16_t narrow(float16, float x) {
  const std::uint32_t bits = float_bits(x), sign = (bits >> 16) & 0x8000u, a = bits & 0x7fffffffu;
  const std::uint32_t big = select(a > 0x7f800000u, 0x7e00u | ((a >> 13) & 0x3ffu), 0x7c00u);
  // Below 2^-14 the float sum with 0.5 rounds to a multiple of 2^-24 (the subnormal unit).
  const std::uint32_t small = float_bits(bits_float(a) + 0.5f) - 0x3f000000u;
  const std::uint32_t normal = (a - (112u << 23) + 0xfffu + ((a >> 13) & 1u)) >> 13;
  const std::uint32_t h = select(a >= 0x47800000u, big, select(a < 0x38800000u, small, normal));
  return static_cast<std::uint16_t>(h | sign);
}
SDCA_STORAGE_INLINE std::uint16_t narrow(bfloat16, float x) {
  const std::uint32_t bits = float_bits(x);
  const std::uint32_t rounded = (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
  const std::uint32_t nan = (bits >> 16) | 0x40u;
  return static_cast<std::uint16_t>(select((bits & 0x7fffffffu) > 0x7f800000u, nan, rounded));
}
SDCA_STORAGE_INLINE float narrow(float32, float x) { return x; }

// x rounded to odd into float: truncated, its last bit set if inexact. A
// 16-bit format rounds that to nearest even as it would round x itself (float
// has at least two more digits at every magnitude), so x rounds once.
SDCA_STORAGE_INLINE float odd_float(double x) {
  const float nearest = float(x);
  const std::uint32_t bits = float_bits(nearest) - (std::fabs(double(nearest)) > std::fabs(x));
  return bits_float(bits | (double(bits_float(bits)) != x));
}
SDCA_STORAGE_INLINE std::uint16_t narrow(float16 f, double x) { return narrow(f, odd_float(x)); }
SDCA_STORAGE_INLINE std::uint16_t narrow(bfloat16 f, double x) { return narrow(f, odd_float(x)); }
SDCA_STORAGE_INLINE float narrow(float32, double x) { return float(x); }
SDCA_STORAGE_INLINE double narrow(float64, double x) { return x; }

}  // namespace tensor
}  // namespace sdca

#undef SDCA_STORAGE_INLINE
#ifdef SDCA_STORAGE_BIT_CAST
#undef SDCA_STORAGE_BIT_CAST
#endif
#endif
