#ifndef SDCA_CPU_LANES_H
#define SDCA_CPU_LANES_H

// Shared portable SIMD row arithmetic; API-specific storage adapters live outside it.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#if __has_include("_cpu_storage.h")
#include "_cpu_storage.h"
#else
#include "../../include/sdca/math/storage.h"
#endif

// The lanes' exponentials inline into the row loops: GCC otherwise calls them out of line,
// and every vector live across the call goes through the stack.
// Row kernels flatten: their loops' lambdas and accumulators inline, so the accumulators stay
// in registers instead of being stored and reloaded through the lambdas' captures each step.
#if defined(__GNUC__) || defined(__clang__)
#define SDCA_LANE inline __attribute__((always_inline))
#define SDCA_FLAT __attribute__((flatten))
#else
#define SDCA_LANE inline
#define SDCA_FLAT
#endif
#define SDCA_ROW inline SDCA_FLAT

namespace {
namespace sdca_cpu {
// Framework-independent 16-bit storage, using the native ABI's exact conversions.
template <typename Format>
struct Storage16 {
  uint16_t bits;
  Storage16() = default;
  explicit Storage16(float x) : bits(sdca::tensor::narrow(Format{}, x)) {}
  explicit Storage16(double x) : bits(sdca::tensor::narrow(Format{}, x)) {}
  explicit operator float() const {
#if defined(__ARM_FP16_FORMAT_IEEE)
    if constexpr (std::is_same_v<Format, sdca::tensor::float16>) {
      __fp16 value;
      std::memcpy(&value, &bits, sizeof value);
      return static_cast<float>(value);
    }
#endif
    return sdca::tensor::widen(Format{}, bits);
  }
  explicit operator double() const { return static_cast<float>(*this); }
};
using Half = Storage16<sdca::tensor::float16>;
using BFloat16 = Storage16<sdca::tensor::bfloat16>;
static_assert(sizeof(Half) == 2 && sizeof(BFloat16) == 2, "16-bit storage");

// Row sums for float64 logits: plain partial sums of 16 terms, combined with TwoSum, so C - 1
// equal terms drift by a few epsilons instead of C / 2. Other dtypes sum plainly.
constexpr int kBlock = 16;

template <typename T>
inline void two_sum(T& hi, T& lo, const T& x) {
  const T t = hi + x, b = t - hi;
  lo = lo + ((hi - (t - b)) + (x - b));
  hi = t;
}

template <bool Compensated>
struct Sum {
  double hi = 0.0, lo = 0.0, block = 0.0;
  int count = 0;
  void add(double x) {
    if constexpr (Compensated) {
      block += x;
      if (++count == kBlock) {
        two_sum(hi, lo, block);
        block = 0.0;
        count = 0;
      }
    } else {
      hi += x;
    }
  }
  double value() const {
    if constexpr (!Compensated) {
      return hi;
    } else {
      double h = hi, l = lo;
      two_sum(h, l, block);
      return h + l;
    }
  }
};

// Fixed-width float lanes in GCC/Clang vector extensions, compiled to NEON on arm64 and to
// SSE/AVX2/AVX-512 by the build flags of the ISA (no libm calls). Their polynomials take
// fused multiply-adds where the ISA has them (the double code keeps -ffp-contract=off).
namespace lanes {
// 16 lanes where the compiler splits wide vectors well (Clang: four NEON registers); GCC
// scalarizes comparisons wider than the ISA's registers, so it takes their width.
#if defined(__clang__) || defined(__AVX512F__)
constexpr int L = 16;
#elif defined(__AVX__)
constexpr int L = 8;
#else
constexpr int L = 4;
#endif
typedef float vf __attribute__((vector_size(4 * L)));
typedef int32_t vi __attribute__((vector_size(4 * L)));
typedef uint32_t vu __attribute__((vector_size(4 * L)));
typedef uint16_t vh __attribute__((vector_size(2 * L)));
typedef double vd __attribute__((vector_size(8 * L)));

inline vf load(const float* p) {
  vf v;
  std::memcpy(&v, p, sizeof v);
  return v;
}
// Lanes of 16-bit logits widened (exactly) to float.

inline void store(float* p, vf v) { std::memcpy(p, &v, sizeof v); }
inline vf splat(float x) { return vf{} + x; }
inline vf select(vi mask, vf a, vf b) { return (vf)((mask & (vi)a) | (~mask & (vi)b)); }
// Lanes rounded to nearest (even) into 16-bit logits.

// Storage conversion is supplied by the API adapter; float rows use the overload above.
template <typename T>
struct Storage;
template <typename T>
struct BFloatStorage {
  static inline vf load(const T* p) {
    vh h;
    std::memcpy(&h, p, sizeof h);
    return (vf)(__builtin_convertvector(h, vu) << 16);
  }
  static inline void store(T* p, vf v) {
    const vu u = (vu)v;
    const vu rounded = (u + (0x7fffu + ((u >> 16) & 1u))) >> 16;
    const vu bits = (vu)select(v != v, (vf)(vu{} + 0x7fc0u), (vf)rounded);
    const vh h = __builtin_convertvector(bits, vh);
    std::memcpy(p, &h, sizeof h);
  }
};
template <typename T>
struct HalfStorage {
  static inline vf load(const T* p) {
    vf v;
#if defined(__AVX2__) && defined(__F16C__)
    // Eight lanes at a time (L is 8 or 16).
    for (int k = 0; k < L; k += 8) {
      const __m256 x = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + k)));
      std::memcpy(reinterpret_cast<float*>(&v) + k, &x, sizeof x);
    }
#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    typedef __fp16 half_vector __attribute__((vector_size(2 * L)));
    half_vector h;
    std::memcpy(&h, p, sizeof h);
    v = __builtin_convertvector(h, vf);
#else
    for (int i = 0; i < L; ++i) v[i] = static_cast<float>(p[i]);
#endif
    return v;
  }
  static inline void store(T* p, vf v) {
#if defined(__AVX2__) && defined(__F16C__)
    for (int k = 0; k < L; k += 8) {
      __m256 x;
      std::memcpy(&x, reinterpret_cast<const float*>(&v) + k, sizeof x);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(p + k),
                       _mm256_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT));
    }
#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    typedef __fp16 half_vector __attribute__((vector_size(2 * L)));
    const half_vector h = __builtin_convertvector(v, half_vector);
    std::memcpy(p, &h, sizeof h);
#else
    for (int i = 0; i < L; ++i) p[i] = static_cast<T>(v[i]);
#endif
  }
};
template <>
struct Storage<BFloat16> : BFloatStorage<BFloat16> {};
template <>
struct Storage<Half> : HalfStorage<Half> {};
template <typename T>
inline vf load(const T* p) {
  return Storage<T>::load(p);
}
template <typename T>
inline void store(T* p, vf v) {
  Storage<T>::store(p, v);
}
// select(a > b, a, b) and select(a < b, a, b) (b where either is NaN or both are zeros): the
// x86 max and min, one instruction where the select takes a compare and a blend. GCC's lanes
// are the ISA's registers (Clang compiles the conditionals to max and min itself).
inline vf max(vf a, vf b) {
#if defined(__AVX512F__) && !defined(__clang__)
  return (vf)_mm512_max_ps((__m512)a, (__m512)b);
#elif defined(__AVX2__) && !defined(__clang__)
  return (vf)_mm256_max_ps((__m256)a, (__m256)b);
#else
  return a > b ? a : b;
#endif
}
inline vf min(vf a, vf b) {
#if defined(__AVX512F__) && !defined(__clang__)
  return (vf)_mm512_min_ps((__m512)a, (__m512)b);
#elif defined(__AVX2__) && !defined(__clang__)
  return (vf)_mm256_min_ps((__m256)a, (__m256)b);
#else
  return a < b ? a : b;
#endif
}
inline vf abs(vf a) { return (vf)((vi)a & 0x7fffffff); }
inline double sum(vd v) {
  double s = 0;
  for (int i = 0; i < L; ++i) s += v[i];
  return s;
}
// a * b + c, fused where the ISA has it.
inline vf fma(vf a, vf b, vf c) {
#if defined(__clang__)
#pragma clang fp contract(fast)
  return a * b + c;
#elif defined(__AVX512F__)
  return (vf)_mm512_fmadd_ps((__m512)a, (__m512)b, (__m512)c);
#elif defined(__AVX2__) && defined(__FMA__)
  return (vf)_mm256_fmadd_ps((__m256)a, (__m256)b, (__m256)c);
#else
  return a * b + c;
#endif
}
inline vf fma(vf a, vf b, float c) { return fma(a, b, splat(c)); }
inline vf fma(vf a, float b, vf c) { return fma(a, splat(b), c); }

// x rounded to an integer (|x| < 2^22): its float value and its int value.
SDCA_LANE vf round(vf x, vi& k) {
  const vf magic = splat(12582912.0f);  // 1.5 * 2^23
  const vf t = x + magic;
  k = (vi)t - (vi)magic;
  return t - magic;
}

// exp(x) within about 1.5 ulp (Cephes expf); 0 below -86.6 (and for -inf), x clamped at 88.
SDCA_LANE vf exp(vf x) {
  const vi tiny = x < -86.6f;
  x = min(max(x, splat(-86.6f)), splat(88.0f));
  vi k;
  const vf n = round(x * 1.44269504088896341f, k);
  const vf r = fma(n, 2.12194440e-4f, fma(n, -0.693359375f, x));
  vf p = fma(splat(1.9875691500e-4f), r, 1.3981999507e-3f);
  p = fma(p, r, 8.3334519073e-3f);
  p = fma(p, r, 4.1665795894e-2f);
  p = fma(p, r, 1.6666665459e-1f);
  p = fma(p, r, 5.0000001201e-1f);
  const vf y = fma(p * r, r, r) + 1.0f;
  return (vf)(((vi)y + (k << 23)) & ~tiny);
}

// exp(x) - 1 in the lanes of small (|x| <= 1 there, within about 2 ulp), exp(x) elsewhere
// (0 below -86.6, x clamped at 88): one range reduction, 2^k expm1(r) + (2^k - 1) or + 2^k.
SDCA_LANE vf exp_or_expm1(vf x, vi small) {
  const vi tiny = x < -86.6f;
  x = min(max(x, splat(-86.6f)), splat(88.0f));
  vi k;
  const vf n = round(x * 1.44269504088896341f, k);
  const vf r = fma(n, 2.12194440e-4f, fma(n, -0.693359375f, x));
  vf p = fma(splat(1.0f / 40320), r, 1.0f / 5040);
  p = fma(p, r, 1.0f / 720);
  p = fma(p, r, 1.0f / 120);
  p = fma(p, r, 1.0f / 24);
  p = fma(p, r, 1.0f / 6);
  p = fma(p, r, 0.5f);
  const vf m = fma(p * r, r, r);
  const vf scale = (vf)((k + 127) << 23);
  return select(small, fma(scale, m, scale - 1.0f), (vf)((vi)fma(scale, m, scale) & ~tiny));
}

inline vi iota() {
  vi index{};
  for (int i = 0; i < L; ++i) index[i] = i;
  return index;
}

// A float64 sum of float lanes, each lane adding up to four terms in float first. The double
// totals of the lower and upper halves of the lanes are vectors of the ISA's width (GCC keeps
// a vd wider than its registers in memory).
struct Accumulator {
  typedef float hf __attribute__((vector_size(2 * L)));
  typedef double hd __attribute__((vector_size(4 * L)));
  hd low{}, high{};
  vf block{};
  int count = 0;
  static void widen(vf v, hd& lo, hd& hi) {
    hf a, b;
    std::memcpy(&a, &v, sizeof a);
    std::memcpy(&b, reinterpret_cast<const char*>(&v) + sizeof a, sizeof b);
    lo = __builtin_convertvector(a, hd), hi = __builtin_convertvector(b, hd);
  }
  void add(vf v) {
    block = block + v;
    if (++count == 4) {
      hd lo, hi;
      widen(block, lo, hi);
      low = low + lo, high = high + hi;
      block = vf{};
      count = 0;
    }
  }
  double value() const {
    hd lo, hi;
    widen(block, lo, hi);
    lo = low + lo, hi = high + hi;
    double s = 0;
    for (int i = 0; i < L / 2; ++i) s += lo[i];
    for (int i = 0; i < L / 2; ++i) s += hi[i];
    return s;
  }
};

// Vectorized loops over [0, n) with a padded tail: body(v, j) on full vectors of the row.
// x is float or 16-bit (widened as loaded).
template <typename T, typename F>
inline void each(const T* x, int64_t n, float pad, F&& body) {
  int64_t j = 0;
  for (; j + L <= n; j += L) body(lanes::load(x + j), j, L);
  if (j < n) {
    float tail[L];
    for (int i = 0; i < L; ++i) tail[i] = i < n - j ? static_cast<float>(x[j + i]) : pad;
    body(load(tail), j, static_cast<int>(n - j));
  }
}

// The largest of x[0, n) (-inf if empty); finite turns false on any nonfinite entry.
SDCA_ROW float max_finite(const float* x, int64_t n, bool& finite) {
  vf top = splat(-std::numeric_limits<float>::infinity()), bad = splat(0.0f);
  // Padding with an entry of the row changes neither result.
  each(x, n, n > 0 ? x[0] : 0.0f, [&](vf v, int64_t, int) {
    top = max(top, v);
    bad = bad + (v * 0.0f);  // NaN for inf and NaN entries
  });
  float result = top[0];
  for (int i = 0; i < L; ++i) {
    result = std::max(result, top[i]);
    finite &= bad[i] == 0.0f;
  }
  return result;
}

// The extrema of x[0, n) (+-inf if empty); finite turns false on any nonfinite entry.
template <typename T>
SDCA_ROW void extrema(const T* x, int64_t n, float& lo, float& hi, bool& finite) {
  vf top = splat(-std::numeric_limits<float>::infinity());
  vf bottom = splat(std::numeric_limits<float>::infinity()), bad = splat(0.0f);
  each(x, n, n > 0 ? static_cast<float>(x[0]) : 0.0f, [&](vf v, int64_t, int) {
    top = max(top, v);
    bottom = min(bottom, v);
    bad = bad + (v * 0.0f);
  });
  lo = bottom[0], hi = top[0];
  for (int i = 0; i < L; ++i) {
    lo = std::min(lo, bottom[i]), hi = std::max(hi, top[i]);
    finite &= bad[i] == 0.0f;
  }
}

// Sum of exp(x[j] - shift) over [0, n) in double: float exponentials, float lanes of up to
// four terms each.
SDCA_ROW double exp_sum(const float* x, int64_t n, float shift) {
  Accumulator total;
  each(x, n, -std::numeric_limits<float>::infinity(),
       [&](vf v, int64_t, int) { total.add(exp(v - shift)); });
  return total.value();
}

// exp_sum over j != skip of float or 16-bit x (widened as loaded): one pass, the skipped lane
// at -inf.
template <typename T>
SDCA_ROW double exp_sum_except(const T* x, int64_t n, int64_t skip, float shift) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  Accumulator total;
  auto body = [&](vf v, int64_t j) {
    if (skip >= j && skip < j + L)
      v = select(iota() == static_cast<int32_t>(skip - j), splat(-kInf), v);
    total.add(exp(v - shift));
  };
  int64_t j = 0;
  for (; j + L <= n; j += L) body(lanes::load(x + j), j);
  if (j < n) {
    float tail[L];
    for (int i = 0; i < L; ++i) tail[i] = i < n - j ? static_cast<float>(x[j + i]) : -kInf;
    body(load(tail), j);
  }
  return total.value();
}

// out[j] = sign * (exp((x[j] - center) / 2) * root)^2: squared halves keep entries whose
// exponential alone would underflow (to about 175 below center).
SDCA_ROW void halves(const float* x, int64_t n, float center, float root, float sign, float* out) {
  each(x, n, center, [&](vf v, int64_t j, int count) {
    const vf half = exp((v - center) * 0.5f) * root;
    const vf g = (half * half) * sign;
    if (count == L) {
      store(out + j, g);
    } else {
      for (int i = 0; i < count; ++i) out[j + i] = g[i];
    }
  });
}
}  // namespace lanes

// Fixed-width double lanes in GCC/Clang vector extensions (NEON, SSE, AVX2 or AVX-512 by the
// build flags), for the float64 row kernels on every ISA. Their exponentials keep the double
// rounding of the polynomial (fused where the ISA has it) and need no libm call.
namespace dlanes {
#if defined(__AVX512F__)
constexpr int L = 8;
#elif defined(__AVX__) || defined(__clang__)
constexpr int L = 4;
#else
constexpr int L = 2;
#endif
typedef double vd __attribute__((vector_size(8 * L)));
typedef int64_t vl __attribute__((vector_size(8 * L)));

inline vd load(const double* p) {
  vd v;
  std::memcpy(&v, p, sizeof v);
  return v;
}
inline void store(double* p, vd v) { std::memcpy(p, &v, sizeof v); }
inline vd splat(double x) { return vd{} + x; }
inline vd select(vl mask, vd a, vd b) { return (vd)((mask & (vl)a) | (~mask & (vl)b)); }
// As lanes::max and lanes::min.
inline vd max(vd a, vd b) {
#if defined(__AVX512F__) && !defined(__clang__)
  return (vd)_mm512_max_pd((__m512d)a, (__m512d)b);
#elif defined(__AVX2__) && !defined(__clang__)
  return (vd)_mm256_max_pd((__m256d)a, (__m256d)b);
#else
  return a > b ? a : b;
#endif
}
inline vd min(vd a, vd b) {
#if defined(__AVX512F__) && !defined(__clang__)
  return (vd)_mm512_min_pd((__m512d)a, (__m512d)b);
#elif defined(__AVX2__) && !defined(__clang__)
  return (vd)_mm256_min_pd((__m256d)a, (__m256d)b);
#else
  return a < b ? a : b;
#endif
}
inline vd abs(vd a) { return (vd)((vl)a & std::numeric_limits<int64_t>::max()); }
inline vl iota() {
  vl index{};
  for (int i = 0; i < L; ++i) index[i] = i;
  return index;
}
// Lanes i < count of v, zeros elsewhere.
inline vd head(vd v, int count) {
  if (count == L) return v;
  return select(iota() < count, v, splat(0.0));
}
inline void put(double* p, vd v, int count) {
  if (count == L) {
    store(p, v);
  } else {
    for (int i = 0; i < count; ++i) p[i] = v[i];
  }
}
// The lane of x at j (count entries, the rest pad).
inline vd part(const double* x, int64_t j, int count, double pad) {
  if (count == L) return load(x + j);
  double tail[L];
  for (int i = 0; i < L; ++i) tail[i] = i < count ? x[j + i] : pad;
  return load(tail);
}
typedef float vf __attribute__((vector_size(4 * L)));
// The lane of float x at j as doubles (count entries, the rest pad).
inline vd part(const float* x, int64_t j, int count, double pad) {
  if (count == L) {
    vf v;
    std::memcpy(&v, x + j, sizeof v);
    return __builtin_convertvector(v, vd);
  }
  vd v = splat(pad);
  for (int i = 0; i < count; ++i) v[i] = x[j + i];
  return v;
}
inline double hmax(vd v) {
  double r = v[0];
  for (int i = 1; i < L; ++i) r = std::max(r, v[i]);
  return r;
}

// a * b + c, fused where the ISA has it.
inline vd fma(vd a, vd b, vd c) {
#if defined(__clang__)
#pragma clang fp contract(fast)
  return a * b + c;
#elif defined(__AVX512F__)
  return (vd)_mm512_fmadd_pd((__m512d)a, (__m512d)b, (__m512d)c);
#elif defined(__AVX2__) && defined(__FMA__)
  return (vd)_mm256_fmadd_pd((__m256d)a, (__m256d)b, (__m256d)c);
#else
  return a * b + c;
#endif
}
inline vd fma(vd a, double b, double c) { return fma(a, splat(b), splat(c)); }

// x rounded to an integer (|x| < 2^51): its double value and its int64 value.
SDCA_LANE vd round(vd x, vl& k) {
  const vd magic = splat(6755399441055744.0);  // 1.5 * 2^52
  const vd t = x + magic;
  k = (vl)t - (vl)magic;
  return t - magic;
}

// expm1(r) for |r| <= log(2) / 2: degree 13, within 1e-17 of r before rounding.
SDCA_LANE vd expm1_reduced(vd r) {
  const vd r2 = r * r, r4 = r2 * r2;
  const vd p12 = fma(r, 1.0 / 6227020800.0, 1.0 / 479001600.0);
  const vd p10 = fma(r, 1.0 / 39916800.0, 1.0 / 3628800.0);
  const vd p8 = fma(r, 1.0 / 362880.0, 1.0 / 40320.0);
  const vd p6 = fma(r, 1.0 / 5040.0, 1.0 / 720.0);
  const vd p4 = fma(r, 1.0 / 120.0, 1.0 / 24.0);
  const vd p2 = fma(r, 1.0 / 6.0, 0.5);
  const vd q10 = fma(p12, r2, p10), q6 = fma(p8, r2, p6), q2 = fma(p4, r2, p2);
  return fma(fma(fma(q10, r4, q6), r4, q2), r2, r);
}

// exp(x) - 1 in the lanes of small (|x| <= 1 there), exp(x) elsewhere: within about 2 ulp;
// 0 below -708 (results below double's normal range), x clamped at 709. One range
// reduction: 2^k expm1(r) + (2^k - 1) or + 2^k.
SDCA_LANE vd exp_or_expm1(vd x, vl small) {
  const vl tiny = x < -708.0;
  x = min(max(x, splat(-708.0)), splat(709.0));
  vl k;
  const vd n = round(x * 1.44269504088896338700, k);
  // log(2) in two parts (fdlibm): n log2_hi is exact for |n| < 2^20.
  const vd r = (x - n * 6.93147180369123816490e-01) - n * 1.90821492927058770002e-10;
  const vd m = expm1_reduced(r);
  const vd scale = (vd)((k + 1023) << 52);
  return select(small, fma(scale, m, scale - 1.0), (vd)((vl)fma(scale, m, scale) & ~tiny));
}
SDCA_LANE vd exp(vd x) { return exp_or_expm1(x, vl{}); }

// log1p(x) for finite x > -1 within an ulp (musl's): 1 + x = 2^k (1 + f) with 1 + f in
// [sqrt(2) / 2, sqrt(2)), c the rounding error of 1 + x over it, and log(1 + f) by the
// polynomial in s = f / (2 + f). Where 1 + x is already in that range, f = x exactly.
inline vd log1p(vd x) {
  const vd u = 1.0 + x;
  const vl bits = (vl)u;
  const vl high = (bits >> 32) + (0x3ff00000 - 0x3fe6a09e);
  const vl k = (high >> 20) - 0x3ff;
  const vl reduced = (((high & 0x000fffff) + 0x3fe6a09e) << 32) | (bits & 0xffffffff);
  const vl direct = k == 0;
  const vd dk = __builtin_convertvector(k, vd);
  vd c = select(k >= 2, 1.0 - (u - x), x - (u - 1.0)) / u;
  c = select(direct | (k >= 54), splat(0.0), c);
  const vd f = select(direct, x, (vd)reduced - 1.0);
  const vd hfsq = 0.5 * f * f, s = f / (2.0 + f), z = s * s, w = z * z;
  const vd t1 = w * (3.999999999940941908e-01 +
                     w * (2.222219843214978396e-01 + w * 1.531383769920937332e-01));
  const vd t2 = z * (6.666666666666735130e-01 +
                     w * (2.857142874366239149e-01 +
                          w * (1.818357216161805012e-01 + w * 1.479819860511658591e-01)));
  const vd r = t2 + t1;
  return s * (hfsq + r) + (dk * 1.90821492927058770002e-10 + c) - hfsq + f +
         dk * 6.93147180369123816490e-01;
}

// The extrema of x[0, n); finite turns false on any nonfinite entry.
inline void extrema(const double* x, int64_t n, double& lo, double& hi, bool& finite) {
  vd top = splat(-std::numeric_limits<double>::infinity());
  vd bottom = splat(std::numeric_limits<double>::infinity()), bad = splat(0.0);
  int64_t j = 0;
  for (; j + L <= n; j += L) {
    const vd v = load(x + j);
    top = max(top, v), bottom = min(bottom, v), bad = bad + v * 0.0;
  }
  lo = bottom[0], hi = top[0];
  for (int i = 0; i < L; ++i) {
    lo = std::min(lo, bottom[i]), hi = std::max(hi, top[i]);
    finite &= bad[i] == 0.0;
  }
  for (; j < n; ++j) {
    lo = std::min(lo, x[j]), hi = std::max(hi, x[j]);
    finite &= std::isfinite(x[j]);
  }
}

// Lane sums of doubles: plain, or (Compensated) partial sums of kBlock terms per lane
// combined with TwoSum, as Sum.
template <bool Compensated>
struct Accumulator {
  vd hi{}, lo{}, block{};
  int count = 0;
  void add(vd x) {
    if constexpr (Compensated) {
      block = block + x;
      if (++count == kBlock) {
        two_sum(hi, lo, block);
        block = vd{};
        count = 0;
      }
    } else {
      hi = hi + x;
    }
  }
  double value() const {
    Sum<true> total;
    vd h = hi, l = lo;
    if constexpr (Compensated) two_sum(h, l, block);
    for (int i = 0; i < L; ++i) two_sum(total.hi, total.lo, h[i]);
    for (int i = 0; i < L; ++i) total.lo += l[i];
    return total.value();
  }
};
}  // namespace dlanes

// Whether any lane of a mask is set.
template <typename V>
inline bool any_lane(V mask) {
  constexpr int kWords = sizeof(V) / sizeof(uint64_t);
  uint64_t words[kWords];
  std::memcpy(words, &mask, sizeof mask);
  uint64_t any = 0;
  for (int i = 0; i < kWords; ++i) any |= words[i];
  return any != 0;
}

// The first j in [from, n) with x[j] >= floor (n if none), a vector at a time.
// x is float or 16-bit (widened as loaded).
template <typename T>
inline int64_t find_at_least(const T* x, int64_t from, int64_t n, float floor) {
  int64_t j = from;
  for (; j + lanes::L <= n; j += lanes::L)
    if (any_lane(lanes::load(x + j) >= floor)) break;
  for (; j < n; ++j)
    if (static_cast<float>(x[j]) >= floor) return j;
  return n;
}
inline int64_t find_at_least(const double* x, int64_t from, int64_t n, double floor) {
  int64_t j = from;
  for (; j + dlanes::L <= n; j += dlanes::L)
    if (any_lane(dlanes::load(x + j) >= floor)) break;
  for (; j < n; ++j)
    if (x[j] >= floor) return j;
  return n;
}

// The largest of x[j] over j != skip (-inf if none) of float or 16-bit x; finite turns false on
// any nonfinite x[j].
template <typename T>
inline float max_except(const T* x, int64_t n, int64_t skip, bool& finite) {
  using namespace lanes;
  constexpr float kInf = std::numeric_limits<float>::infinity();
  vf top = splat(-kInf), bad = splat(0.0f);
  int64_t j = 0;
  for (; j + L <= n; j += L) {
    vf v = lanes::load(x + j);
    bad = bad + v * 0.0f;  // NaN for inf and NaN entries
    if (skip >= j && skip < j + L)
      v = select(iota() == static_cast<int32_t>(skip - j), splat(-kInf), v);
    top = max(top, v);
  }
  float result = -kInf;
  for (int i = 0; i < L; ++i) result = std::max(result, top[i]), finite &= bad[i] == 0.0f;
  for (; j < n; ++j) {
    const float v = static_cast<float>(x[j]);
    finite &= std::isfinite(v);
    if (j != skip) result = std::max(result, v);
  }
  return result;
}
inline double max_except(const double* x, int64_t n, int64_t skip, bool& finite) {
  using namespace dlanes;
  constexpr double kInf = std::numeric_limits<double>::infinity();
  vd top = splat(-kInf), bad = splat(0.0);
  int64_t j = 0;
  for (; j + L <= n; j += L) {
    vd v = load(x + j);
    bad = bad + v * 0.0;
    if (skip >= j && skip < j + L) v = select(iota() == skip - j, splat(-kInf), v);
    top = max(top, v);
  }
  double result = -kInf;
  for (int i = 0; i < L; ++i) result = std::max(result, top[i]), finite &= bad[i] == 0.0;
  for (; j < n; ++j) {
    finite &= std::isfinite(x[j]);
    if (j != skip) result = std::max(result, x[j]);
  }
  return result;
}

}  // namespace sdca_cpu
}  // namespace
#endif
