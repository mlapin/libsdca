#ifndef SDCA_SRC_TENSOR_LOSSES_H
#define SDCA_SRC_TENSOR_LOSSES_H

// Rows of the tensor entry points of c_api.h (sdca_cross_entropy_forward,
// sdca_cross_entropy_backward, sdca_loss_gradient_rows and the distillation
// losses): IEEE binary16 and bfloat16 storage, float32 exponentials with
// double row statistics, ignored rows and class weights. The kernels use separate IEEE operations over fixed
// lanes, as those of sdca/math/exp_rows.h: every ISA clone gives the same bits.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#include "sdca/loss.h"
#include "sdca/math/distillation.h"
#include "sdca/math/entropy.h"
#include "sdca/math/exp_rows.h"
#include "sdca/math/lambert.h"
#include "sdca/math/log_exp.h"
#include "sdca/math/prox_rows.h"
#include "sdca/utility/parallel.h"

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

SDCA_ROW_INLINE std::uint32_t float_bits(float x) {
#ifdef SDCA_BIT_CAST
  return SDCA_BIT_CAST(std::uint32_t, x);
#else
  std::uint32_t bits;
  std::memcpy(&bits, &x, sizeof bits);
  return bits;
#endif
}

SDCA_ROW_INLINE float bits_float(std::uint32_t bits) {
#ifdef SDCA_BIT_CAST
  return SDCA_BIT_CAST(float, bits);
#else
  float x;
  std::memcpy(&x, &bits, sizeof x);
  return x;
#endif
}

// a if c, else b, through a bit mask: GCC does not if-convert the selects of
// floating-point results (they may trap), which would keep loops scalar.
SDCA_ROW_INLINE std::uint32_t select(bool c, std::uint32_t a, std::uint32_t b) {
  const std::uint32_t mask = 0u - static_cast<std::uint32_t>(c);
  return (a & mask) | (b & ~mask);
}

// The stored values as float (exact) or double.
SDCA_ROW_INLINE float widen(float16, std::uint16_t h) {
  const std::uint32_t sign = (h & 0x8000u) << 16, em = h & 0x7fffu;
  // Subnormal: m 2^-24; normal: the exponent rebiased (15 -> 127); inf and NaN keep the payload.
  const float subnormal = float(static_cast<std::int32_t>(em)) * 5.9604644775390625e-8f;
  const std::uint32_t normal = (em << 13) + (112u << 23), special = (em << 13) | 0x7f800000u;
  const std::uint32_t bits =
      select(em < 0x400u, float_bits(subnormal), select(em >= 0x7c00u, special, normal));
  return bits_float(bits | sign);
}
SDCA_ROW_INLINE float widen(bfloat16, std::uint16_t h) {
  return bits_float(std::uint32_t(h) << 16);
}
SDCA_ROW_INLINE float widen(float32, float x) { return x; }
SDCA_ROW_INLINE double widen(float64, double x) { return x; }

// x rounded to nearest even into the format: overflow to inf, NaN quiet.
SDCA_ROW_INLINE std::uint16_t narrow(float16, float x) {
  const std::uint32_t bits = float_bits(x), sign = (bits >> 16) & 0x8000u, a = bits & 0x7fffffffu;
  const std::uint32_t big = select(a > 0x7f800000u, 0x7e00u | ((a >> 13) & 0x3ffu), 0x7c00u);
  // Below 2^-14 the float sum with 0.5 rounds to a multiple of 2^-24 (the subnormal unit).
  const std::uint32_t small = float_bits(bits_float(a) + 0.5f) - 0x3f000000u;
  const std::uint32_t normal = (a - (112u << 23) + 0xfffu + ((a >> 13) & 1u)) >> 13;
  const std::uint32_t h = select(a >= 0x47800000u, big, select(a < 0x38800000u, small, normal));
  return static_cast<std::uint16_t>(h | sign);
}
SDCA_ROW_INLINE std::uint16_t narrow(bfloat16, float x) {
  const std::uint32_t bits = float_bits(x);
  const std::uint32_t rounded = (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
  const std::uint32_t nan = (bits >> 16) | 0x40u;
  return static_cast<std::uint16_t>(select((bits & 0x7fffffffu) > 0x7f800000u, nan, rounded));
}
SDCA_ROW_INLINE float narrow(float32, float x) { return x; }

// x rounded to odd into float: truncated, its last bit set if inexact. A
// 16-bit format rounds that to nearest even as it would round x itself (float
// has at least two more digits at every magnitude), so x rounds once.
SDCA_ROW_INLINE float odd_float(double x) {
  const float nearest = float(x);
  const std::uint32_t bits = float_bits(nearest) - (std::fabs(double(nearest)) > std::fabs(x));
  return bits_float(bits | (double(bits_float(bits)) != x));
}
SDCA_ROW_INLINE std::uint16_t narrow(float16 f, double x) { return narrow(f, odd_float(x)); }
SDCA_ROW_INLINE std::uint16_t narrow(bfloat16 f, double x) { return narrow(f, odd_float(x)); }
SDCA_ROW_INLINE float narrow(float32, double x) { return float(x); }
SDCA_ROW_INLINE double narrow(float64, double x) { return x; }

// exp(x) in float within about an ulp for x <= 88 (Cody-Waite reduction and
// the polynomial of Cephes' expf); 0 below -87.6, where 2^n leaves the normals.
SDCA_ROW_INLINE float exp_float(float x) {
  const float shift = 12582912.0f;  // 1.5 * 2^23: adding it rounds to an integer
  const float t = x * 1.44269504088896341f + shift, n = t - shift;
  const float r = (x - n * 0.693359375f) - n * -2.12194440e-4f;
  float p = 1.9875691500e-4f;
  p = p * r + 1.3981999507e-3f;
  p = p * r + 8.3334519073e-3f;
  p = p * r + 4.1665795894e-2f;
  p = p * r + 1.6666665459e-1f;
  p = p * r + 5.0000001201e-1f;
  p = (p * r) * r + r + 1.0f;
  // The low bits of t hold n; 2^n is its biased exponent.
  const float power = bits_float((float_bits(t) - float_bits(shift) + 127u) << 23);
  return bits_float(select(x < -87.6f, 0u, float_bits(p * power)));
}

// Lanes of the float row kernels; each lane adds at most float_chunk /
// float_lanes float terms before a double lane takes their sum.
const std::size_t float_lanes = 32, float_chunk = 512;

// Updates the lanes of the extrema of z[0, n); bad turns NaN at a nonfinite score.
template <typename Format>
SDCA_ROW_INLINE void extrema_lanes(std::size_t n, const typename Format::type* z, float* high,
                                   float* low, float* bad) {
  std::size_t j = 0;
  for (; j + float_lanes <= n; j += float_lanes) {
    for (std::size_t l = 0; l < float_lanes; ++l) {
      const float v = widen(Format(), z[j + l]);
      high[l] = v > high[l] ? v : high[l];
      low[l] = v < low[l] ? v : low[l];
      bad[l] += v - v;
    }
  }
  for (std::size_t l = 0; j + l < n; ++l) {
    const float v = widen(Format(), z[j + l]);
    high[l] = v > high[l] ? v : high[l];
    low[l] = v < low[l] ? v : low[l];
    bad[l] += v - v;
  }
}

// Adds exp(z[j] - shift), j < n, in float to the double lanes of sum.
template <typename Format>
SDCA_ROW_INLINE void exp_sum_lanes(std::size_t n, const typename Format::type* z, float shift,
                                   double* sum) {
  for (std::size_t begin = 0; begin < n; begin += float_chunk) {
    const std::size_t count = std::min(n - begin, float_chunk);
    const typename Format::type* x = z + begin;
    float part[float_lanes] = {};
    std::size_t j = 0;
    for (; j + float_lanes <= count; j += float_lanes) {
      for (std::size_t l = 0; l < float_lanes; ++l)
        part[l] += exp_float(widen(Format(), x[j + l]) - shift);
    }
    for (std::size_t l = 0; j + l < count; ++l)
      part[l] += exp_float(widen(Format(), x[j + l]) - shift);
    for (std::size_t l = 0; l < float_lanes; ++l) sum[l] += double(part[l]);
  }
}

// The competitors' (j != label) largest and smallest score and the sum of
// their float exp(z_j - maximum) in double; finite whether every score is.
struct float_row {
  float maximum, lowest;
  double sum;
  bool finite;
};

// float_row_stats of a short row (n <= float_lanes): one term at a time.
template <typename Format>
SDCA_ROW_INLINE float_row short_row_stats(std::size_t n, const typename Format::type* z,
                                          std::size_t label) {
  const float target = widen(Format(), z[label]);
  float_row row = {-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                   0.0, true};
  float bad = target - target;
  for (std::size_t j = 0; j < n; ++j) {
    const float v = widen(Format(), z[j]);
    if (j == label) continue;
    row.maximum = v > row.maximum ? v : row.maximum;
    row.lowest = v < row.lowest ? v : row.lowest;
    bad += v - v;
  }
  row.finite = bad == 0;
  if (!row.finite) return row;
  float e[float_lanes];
  for (std::size_t j = 0; j < n; ++j) e[j] = exp_float(widen(Format(), z[j]) - row.maximum);
  e[label] = 0;
  for (std::size_t j = 0; j < n; ++j) row.sum += double(e[j]);
  return row;
}

template <typename Format>
SDCA_ROW_INLINE float_row float_row_stats(std::size_t n, const typename Format::type* z,
                                          std::size_t label) {
  if (n <= float_lanes) return short_row_stats<Format>(n, z, label);
  const float infinity = std::numeric_limits<float>::infinity();
  float high[float_lanes], low[float_lanes], bad[float_lanes];
  for (std::size_t l = 0; l < float_lanes; ++l) {
    high[l] = -infinity;
    low[l] = infinity;
    bad[l] = 0;
  }
  extrema_lanes<Format>(label, z, high, low, bad);
  extrema_lanes<Format>(n - label - 1, z + label + 1, high, low, bad);
  const float target = widen(Format(), z[label]);
  float_row row = {high[0], low[0], 0.0, target - target == 0};
  for (std::size_t l = 0; l < float_lanes; ++l) {
    row.maximum = high[l] > row.maximum ? high[l] : row.maximum;
    row.lowest = low[l] < row.lowest ? low[l] : row.lowest;
    row.finite = row.finite && bad[l] == 0;
  }
  if (!row.finite) return row;
  double sum[float_lanes] = {};
  exp_sum_lanes<Format>(label, z, row.maximum, sum);
  exp_sum_lanes<Format>(n - label - 1, z + label + 1, row.maximum, sum);
  for (std::size_t l = 0; l < float_lanes; ++l) row.sum += sum[l];
  return row;
}

// out[j] = sign (exp((z[j] - maximum) / 2) root)^2 rounded into the format:
// squared halves keep the entries whose exp(z[j] - maximum) underflows float.
template <typename Format>
SDCA_ROW_INLINE void float_gradient_lanes(std::size_t n, const typename Format::type* z,
                                          float maximum, float root, float sign,
                                          typename Format::type* out) {
  for (std::size_t j = 0; j < n; ++j) {
    const float half = exp_float((widen(Format(), z[j]) - maximum) * 0.5f) * root;
    out[j] = narrow(Format(), (half * half) * sign);
  }
}

// Whether a stored value is infinite or NaN: its exponent bits all set.
SDCA_ROW_INLINE bool nonfinite(float16, std::uint16_t h) { return (h & 0x7c00u) == 0x7c00u; }
SDCA_ROW_INLINE bool nonfinite(bfloat16, std::uint16_t h) { return (h & 0x7f80u) == 0x7f80u; }

// out[j] = Work(z[j]) of 16-bit z; whether every z[j] is finite.
template <typename Format, typename Work>
SDCA_ROW_INLINE bool widen_lanes(std::size_t n, const std::uint16_t* z, Work* out) {
  unsigned bad = 0;  // an integer reduction: a float one would keep the loop scalar
  for (std::size_t j = 0; j < n; ++j) {
    bad |= nonfinite(Format(), z[j]);
    out[j] = Work(widen(Format(), z[j]));
  }
  return bad == 0;
}

// out[j] = g[j] rounded once into the format.
template <typename Format>
SDCA_ROW_INLINE void narrow_lanes(std::size_t n, const double* g, typename Format::type* out) {
  for (std::size_t j = 0; j < n; ++j) out[j] = narrow(Format(), g[j]);
}

// The kernels of each format, cloned per ISA where available. Code inlined
// into the clones avoids library templates (std::min, std::find): Clang's
// clones can reference their out-of-line instances without emitting them.
#define SDCA_TENSOR_FLOAT_KERNELS(F)                                                            \
  SDCA_ROW_KERNEL float_row float_row_kernel(F, std::size_t n, const F::type* z,                \
                                             std::size_t label) {                               \
    return float_row_stats<F>(n, z, label);                                                     \
  }                                                                                             \
  SDCA_ROW_KERNEL void float_gradient_kernel(F, std::size_t n, const F::type* z, float maximum, \
                                             float root, float sign, F::type* out) {            \
    float_gradient_lanes<F>(n, z, maximum, root, sign, out);                                    \
  }
#define SDCA_TENSOR_HALF_KERNELS(F)                                                     \
  SDCA_ROW_KERNEL bool widen_kernel(F, std::size_t n, const F::type* z, float* out) {   \
    return widen_lanes<F>(n, z, out);                                                   \
  }                                                                                     \
  SDCA_ROW_KERNEL bool widen_kernel(F, std::size_t n, const F::type* z, double* out) {  \
    return widen_lanes<F>(n, z, out);                                                   \
  }                                                                                     \
  SDCA_ROW_KERNEL void narrow_kernel(F, std::size_t n, const double* g, F::type* out) { \
    narrow_lanes<F>(n, g, out);                                                         \
  }
SDCA_TENSOR_FLOAT_KERNELS(float16)
SDCA_TENSOR_FLOAT_KERNELS(bfloat16)
SDCA_TENSOR_FLOAT_KERNELS(float32)
SDCA_TENSOR_HALF_KERNELS(float16)
SDCA_TENSOR_HALF_KERNELS(bfloat16)
#undef SDCA_TENSOR_FLOAT_KERNELS
#undef SDCA_TENSOR_HALF_KERNELS

// The float exp(z[j] - top) sums of a row whose largest score is top: over the entries
// other than one at top (rest; the others at top add exactly 1) and over the negatives
// (mask zero), in float lanes of float_chunk / float_lanes terms and double sums.
struct mask_sums {
  double rest, negatives;
};
// Adds the terms of x[l], b[l] (l < lanes) to the lanes of mask_sums_kernel.
SDCA_ROW_INLINE void mask_terms(std::size_t lanes, const float* x, const std::uint8_t* b, float top,
                                std::uint32_t* ties, float* r, float* q) {
  for (std::size_t l = 0; l < lanes; ++l) {
    const float v = x[l];
    const std::uint32_t e = float_bits(exp_float(v - top));
    ties[l] += v == top;
    r[l] += bits_float(select(v == top, 0u, e));
    q[l] += bits_float(select(b[l] != 0, 0u, e));
  }
}
SDCA_ROW_KERNEL mask_sums mask_sums_kernel(std::size_t n, const float* z, const std::uint8_t* m,
                                           float top) {
  double rest[float_lanes] = {}, negatives[float_lanes] = {};
  std::uint32_t ties[float_lanes] = {};
  for (std::size_t begin = 0; begin < n; begin += float_chunk) {
    const std::size_t count = n - begin < float_chunk ? n - begin : float_chunk;
    const float* x = z + begin;
    const std::uint8_t* b = m + begin;
    float r[float_lanes] = {}, q[float_lanes] = {};
    std::size_t j = 0;
    for (; j + float_lanes <= count; j += float_lanes)
      mask_terms(float_lanes, x + j, b + j, top, ties, r, q);
    mask_terms(count - j, x + j, b + j, top, ties, r, q);
    for (std::size_t l = 0; l < float_lanes; ++l) {
      rest[l] += double(r[l]);
      negatives[l] += double(q[l]);
    }
  }
  mask_sums sums = {0.0, 0.0};
  std::uint32_t tied = 0;
  for (std::size_t l = 0; l < float_lanes; ++l) {
    sums.rest += rest[l];
    sums.negatives += negatives[l];
    tied += ties[l];
  }
  sums.rest += double(tied) - 1.0;
  return sums;
}

// float64 storage has no float path (rejected by the entry points).
inline float_row float_row_kernel(float64, std::size_t, const double*, std::size_t) {
  throw std::logic_error("float64 storage takes double arithmetic");
}
inline void float_gradient_kernel(float64, std::size_t, const double*, float, float, float,
                                  double*) {
  throw std::logic_error("float64 storage takes double arithmetic");
}

// The statistics of a row as sdca::single_entropy_row from its competitors'
// maximum, lowest score and sum of exp(z_j - maximum), and the target z[label];
// the loss softplus(relative) and the mass sigmoid(relative) share an exponential.
inline entropy_row<double> entropy_statistics(double maximum, double lowest, double sum,
                                              double target) {
  entropy_row<double> row = {maximum, lowest, sum, 0.0, 0.0};
  const double gap = maximum - target, relative = gap + std::log(sum);
  if (relative < -36) {
    const double half = std::exp(gap / 2), tail = half * (half * sum);
    row.loss = std::log1p(tail);
    row.mass = tail / (1 + tail);
  } else {
    const double e = std::exp(-std::fabs(relative)), log = std::log1p(e);
    row.loss = relative > 0 ? relative + log : log;
    row.mass = (relative > 0 ? 1.0 : e) / (1 + e);
  }
  return row;
}

// entropy_row_tiny, without its exponential for rows far from the tiny
// derivatives: e^-600 times 1e-30 is above the smallest normal double.
inline bool row_tiny(const entropy_row<double>& row) {
  if (row.lowest - row.maximum > -600 && row.mass / row.sum > 1e-30) return false;
  return entropy_row_tiny(row);
}

// Columns of the row statistics: maximum, lowest, sum, mass, tiny (as
// sdca_entropy_forward).
const std::size_t entropy_stats = 5;

// Row scratch of a range of rows.
template <typename T>
struct row_buffer {
  detail::entropy_buffer<T> buffer;
  T* get(std::size_t size) { return buffer.get(size); }
};

// out[j] = Work(z[j]) of 16-bit z; whether every z[j] is finite.
inline bool widen_row(float16 f, std::size_t n, const std::uint16_t* z, float* out) {
  return widen_kernel(f, n, z, out);
}
inline bool widen_row(bfloat16 f, std::size_t n, const std::uint16_t* z, float* out) {
  return widen_kernel(f, n, z, out);
}
inline bool widen_row(float16 f, std::size_t n, const std::uint16_t* z, double* out) {
  return widen_kernel(f, n, z, out);
}
inline bool widen_row(bfloat16 f, std::size_t n, const std::uint16_t* z, double* out) {
  return widen_kernel(f, n, z, out);
}

// The statistics of a row as sdca_entropy_forward (double exponentials);
// 16-bit rows pass through float scratch. finite: whether every score is.
template <typename Format>
entropy_row<double> double_row(Format, std::size_t k, std::size_t label,
                               const typename Format::type* z, row_buffer<float>&, bool& finite) {
  const entropy_row<double> r =
      single_entropy_row<double>(k, label, z, static_cast<double*>(nullptr));
  finite = std::isfinite(r.lowest) && std::isfinite(r.sum) && std::isfinite(double(z[label]));
  return r;
}
template <typename Format>
entropy_row<double> half_double_row(std::size_t k, std::size_t label, const std::uint16_t* z,
                                    row_buffer<float>& scratch, bool& finite) {
  float* row = scratch.get(k);
  finite = widen_row(Format(), k, z, row);
  if (!finite) return entropy_row<double>();
  return double_row(float32(), k, label, row, scratch, finite);
}
inline entropy_row<double> double_row(float16, std::size_t k, std::size_t label,
                                      const std::uint16_t* z, row_buffer<float>& scratch,
                                      bool& finite) {
  return half_double_row<float16>(k, label, z, scratch, finite);
}
inline entropy_row<double> double_row(bfloat16, std::size_t k, std::size_t label,
                                      const std::uint16_t* z, row_buffer<float>& scratch,
                                      bool& finite) {
  return half_double_row<bfloat16>(k, label, z, scratch, finite);
}

// The statistics of a short row (n <= float_lanes) in double, one term at a
// time: as double_row to rounding, without its fixed cost per row.
template <typename Format>
entropy_row<double> short_double_row(std::size_t n, std::size_t label,
                                     const typename Format::type* z, bool& finite) {
  const double target = double(widen(Format(), z[label]));
  double high = -std::numeric_limits<double>::infinity(), low = -high, bad = target - target;
  for (std::size_t j = 0; j < n; ++j) {
    const double v = double(widen(Format(), z[j]));
    if (j == label) continue;
    high = v > high ? v : high;
    low = v < low ? v : low;
    bad += v - v;
  }
  finite = bad == 0;
  if (!finite) return entropy_row<double>();
  const bool normal = low - high >= -708;
  double sum = 0, carry = 0;
  for (std::size_t j = 0; j < n; ++j) {
    if (j == label) continue;
    const double x = double(widen(Format(), z[j])) - high;
    const double e = (normal ? detail::exp_normal(x) : std::exp(x)) - carry, t = sum + e;
    carry = (t - sum) - e;
    sum = t;
  }
  return entropy_statistics(high, low, sum, target);
}

// The gradient of a row times scale, rounded once into the format, as
// sdca_entropy_gradient; 16-bit rows pass through double scratch.
template <typename Format>
void double_gradient(Format, std::size_t k, std::size_t label, const typename Format::type* z,
                     const entropy_row<double>& r, double scale, row_buffer<double>&,
                     typename Format::type* g) {
  single_entropy_gradient(k, label, z, r, scale, static_cast<const double*>(nullptr), g);
}
template <typename Format>
void half_double_gradient(std::size_t k, std::size_t label, const std::uint16_t* z,
                          const entropy_row<double>& r, double scale, row_buffer<double>& scratch,
                          std::uint16_t* g) {
  double* row = scratch.get(2 * k);
  widen_row(Format(), k, z, row);
  single_entropy_gradient(k, label, row, r, scale, static_cast<const double*>(nullptr), row + k);
  narrow_kernel(Format(), k, row + k, g);
}
inline void double_gradient(float16, std::size_t k, std::size_t label, const std::uint16_t* z,
                            const entropy_row<double>& r, double scale, row_buffer<double>& scratch,
                            std::uint16_t* g) {
  half_double_gradient<float16>(k, label, z, r, scale, scratch, g);
}
inline void double_gradient(bfloat16, std::size_t k, std::size_t label, const std::uint16_t* z,
                            const entropy_row<double>& r, double scale, row_buffer<double>& scratch,
                            std::uint16_t* g) {
  half_double_gradient<bfloat16>(k, label, z, r, scale, scratch, g);
}

// The gradient of a row times scale in float exponentials: squared halves
// where the row factor fits float, else double; the target entry in double.
template <typename Format>
void float_gradient(std::size_t k, std::size_t label, const typename Format::type* z,
                    const entropy_row<double>& r, double scale, typename Format::type* g) {
  const double factor = (r.mass / r.sum) * std::fabs(scale);
  const float sign = scale < 0 ? -1.0f : 1.0f;
  if (factor <= double(std::numeric_limits<float>::max())) {
    float_gradient_kernel(Format(), k, z, float(r.maximum), float(std::sqrt(factor)), sign, g);
  } else {
    for (std::size_t j = 0; j < k; ++j) {
      const double e = std::exp(double(widen(Format(), z[j])) - r.maximum);
      g[j] = narrow(Format(), (e * factor) * double(sign));
    }
  }
  g[label] = narrow(Format(), -(r.mass * scale));
}

const char* const label_message = "Labels must lie in [0, num_classes)";
const char* const finite_message = "Logits must be finite";

// The label of a row that is not ignored; throws outside [0, k).
inline std::size_t checked_label(std::int64_t label, std::size_t k) {
  if (label < 0 || static_cast<std::uint64_t>(label) >= k)
    throw std::invalid_argument(label_message);
  return static_cast<std::size_t>(label);
}

// Cross-entropy of the rows [begin, end): values (n) and stats (n, entropy_stats).
template <typename Format>
struct cross_entropy_forward_rows {
  std::size_t k;
  const typename Format::type* z;
  const std::int64_t* labels;
  std::int64_t ignore;
  const double* weights;
  bool fp32;
  double* values;
  double* stats;

  void operator()(std::size_t begin, std::size_t end) const {
    row_buffer<float> scratch;
    for (std::size_t i = begin; i < end; ++i) {
      double* s = stats + i * entropy_stats;
      if (labels[i] == ignore) {
        values[i] = 0;
        std::fill(s, s + entropy_stats, 0.0);
        continue;
      }
      const std::size_t label = checked_label(labels[i], k);
      const typename Format::type* row = z + i * k;
      bool finite = false;
      entropy_row<double> r;
      if (fp32) {
        const float_row f = float_row_kernel(Format(), k, row, label);
        finite = f.finite;
        if (finite) {
          const double target = double(widen(Format(), row[label]));
          r = entropy_statistics(double(f.maximum), double(f.lowest), f.sum, target);
        }
      } else if (k <= float_lanes) {
        r = short_double_row<Format>(k, label, row, finite);
      } else {
        r = double_row(Format(), k, label, row, scratch, finite);
      }
      if (!finite) throw std::invalid_argument(finite_message);
      values[i] = (weights ? weights[label] : 1.0) * r.loss;
      s[0] = r.maximum;
      s[1] = r.lowest;
      s[2] = r.sum;
      s[3] = r.mass;
      s[4] = row_tiny(r) ? 1 : 0;
    }
  }
};

// Gradients of the rows [begin, end) from their stats, times scale[i * stride]
// and the class weight; zero rows where ignored.
template <typename Format>
struct cross_entropy_backward_rows {
  std::size_t k;
  const typename Format::type* z;
  const std::int64_t* labels;
  std::int64_t ignore;
  const double* weights;
  const double* stats;
  const double* scale;
  std::size_t stride;
  bool fp32;
  typename Format::type* gradients;

  void operator()(std::size_t begin, std::size_t end) const {
    row_buffer<double> scratch;
    for (std::size_t i = begin; i < end; ++i) {
      typename Format::type* g = gradients + i * k;
      if (labels[i] == ignore) {
        std::memset(static_cast<void*>(g), 0, k * sizeof *g);
        continue;
      }
      const std::size_t label = static_cast<std::size_t>(labels[i]);
      const double factor = scale[i * stride] * (weights ? weights[label] : 1.0);
      const double* s = stats + i * entropy_stats;
      const entropy_row<double> r = {s[0], s[1], s[2], s[3], 0.0};
      if (fp32)
        float_gradient<Format>(k, label, z + i * k, r, factor, g);
      else
        double_gradient(Format(), k, label, z + i * k, r, factor, scratch, g);
    }
  }
};

// Mask rows of cross-entropy (sdca_loss_gradient and sdca_loss_gradient_rows): the
// uniform target on the positives of a 0/1 mask, in place (no permutation of the row).

// The positions of the nonzero entries of m[0, k) into index, skipping zero words of the
// mask; returns their count.
inline std::size_t mask_positions(std::size_t k, const std::uint8_t* m, std::size_t* index) {
  std::size_t p = 0, j = 0;
  for (; j + 8 <= k; j += 8) {
    std::uint64_t word;
    std::memcpy(&word, m + j, sizeof word);
    if (word == 0) continue;
    for (std::size_t i = j; i < j + 8; ++i)
      if (m[i]) index[p++] = i;
  }
  for (; j < k; ++j)
    if (m[j]) index[p++] = j;
  return p;
}

// The exponential sums of a mask row in float32 (mask_sums_kernel) and the unscaled
// gradient exp(z - high) / (rest + 1) of every entry; false where some exp(z - high) would
// leave float's normal range. float64 rows take double.
inline bool mask_float_row(std::size_t k, const float* z, const std::uint8_t* m, float high,
                           float low, double& rest, double& negatives, float* g) {
  if (!(low - high >= -80.0f)) return false;
  const mask_sums sums = mask_sums_kernel(k, z, m, high);
  rest = sums.rest;
  negatives = sums.negatives;
  float_gradient_kernel(float32(), k, z, high, float(std::sqrt(1 / (rest + 1))), 1.0f, g);
  return true;
}
inline bool mask_float_row(std::size_t, const double*, const std::uint8_t*, double, double, double&,
                           double&, double*) {
  return false;
}

// The sum of exp(z[j] - high) over the negatives (m[j] == 0) of a row, on the scale of their
// own maximum and rounded once into subnormals (as single_entropy_row's competitors),
// compensated: beyond double's normal spread each term would round there first.
template <typename Data>
double scaled_negatives(std::size_t k, const Data* z, const std::uint8_t* m, double high) {
  double top = -std::numeric_limits<double>::infinity(), sum = 0, carry = 0;
  for (std::size_t j = 0; j < k; ++j)
    if (!m[j] && double(z[j]) > top) top = double(z[j]);
  if (top == -std::numeric_limits<double>::infinity()) return 0;
  for (std::size_t j = 0; j < k; ++j) {
    if (m[j]) continue;
    const double y = std::exp(double(z[j]) - top) - carry, t = sum + y;
    carry = (t - sum) - y;
    sum = t;
  }
  const double half = std::exp((top - high) / 2);
  return half * (half * sum);
}

// The exponential sums of a mask row in double, terms[j] = exp(z[j] - high), compensated
// (exp_sum's lanes, and Kahan lanes for the negatives), and the unscaled gradient
// terms / (rest + 1) of every entry.
template <typename Data>
void mask_double_row(std::size_t k, const Data* z, const std::uint8_t* m, Data high, Data low,
                     double* terms, double& rest, double& negatives, Data* g) {
  std::size_t top = 0;
  while (z[top] != high) ++top;
  const bool normal = double(low) - double(high) >= -708;
  rest = detail::exp_sum(k, z, top, double(high), terms, normal);
  terms[top] = 1;
  const double den = rest + 1;
  for (std::size_t j = 0; j < k; ++j) g[j] = Data(terms[j] / den);
  if (!normal) {
    negatives = scaled_negatives(k, z, m, double(high));
    return;
  }
  double sum[4] = {}, carry[4] = {};
  for (std::size_t j = 0; j < k; ++j) {
    const std::size_t l = j % 4;
    const double y = (m[j] ? 0.0 : terms[j]) - carry[l], t = sum[l] + y;
    carry[l] = (t - sum[l]) - y;
    sum[l] = t;
  }
  const double low_half = (sum[0] - carry[0]) + (sum[1] - carry[1]);
  negatives = low_half + ((sum[2] - carry[2]) + (sum[3] - carry[3]));
}

// Cross-entropy of the uniform target on the p positives (positions index) of the mask row
// m, in place: the value, and the gradient into g; one positive takes entropy_label_loss.
// Float work (common_type<Data, Result> float) takes float32 exponentials with double sums
// where the row's spread keeps them normal, else double; the positives take double, their
// centered differences e0 expm1(z - anchor) (as sdca::entropy_loss_gradient). terms holds k
// doubles.
template <typename Data, typename Result>
double mask_entropy_row(std::size_t k, const std::uint8_t* m, const Data* z, std::size_t p,
                        const std::size_t* index, double* terms, Data* g) {
  const std::size_t first = index[0];
  if (p == 1) return entropy_label_loss<Data, Result>(k, first, z, g);
  Data low;
  const Data high = detail::max_except(k, z, k, low);
  const bool fp32 = std::is_same<typename std::common_type<Data, Result>::type, float>::value;
  double rest, negatives;
  // Tied positive maxima may have tiny gradients: float-rounded score gaps
  // would lose relative digits even while their exponentials remain normal.
  bool positive_maximum = false;
  for (std::size_t i = 0; i < p; ++i) positive_maximum = positive_maximum || z[index[i]] == high;
  const bool sensitive = positive_maximum && double(low) - double(high) < -16.0;
  const bool floats = fp32 && !sensitive && mask_float_row(k, z, m, high, low, rest, negatives, g);
  if (!floats) mask_double_row(k, z, m, high, low, terms, rest, negatives, g);
  const double top = double(high), anchor = double(z[first]), count = double(p);
  const double den = rest + 1, e0 = std::exp(anchor - top);
  double gap = 0, differences = 0;
  for (std::size_t i = 0; i < p; ++i) {
    const std::size_t at = index[i];
    const double value = double(z[at]), difference = top - value, dz = value - anchor;
    gap += std::isfinite(difference) ? difference / count : top / count - value / count;
    terms[at] = floats || dz < 0 ? e0 * std::expm1(dz) : -terms[at] * std::expm1(-dz);
    differences += terms[at];
  }
  const double share = (differences / den + negatives / den) / count;
  for (std::size_t i = 0; i < p; ++i) g[index[i]] = Data(terms[index[i]] / den - share);
  return gap + std::log1p(rest);
}

// The rows [begin, end) of sdca_loss_gradient from any storage: one label per
// row or a 0/1 mask (positive), each row checked before its loss; values and
// gradients (Data: float for 16-bit storage) out.
template <typename Format, typename Data, typename Result>
struct loss_gradient_rows {
  loss_kind kind;
  std::size_t k, topk;
  double gamma, alpha;
  projection_precision mode;
  const typename Format::type* z;
  const std::int64_t* labels;
  const std::uint8_t* positive;
  double* values;
  Data* gradients;

  // The positive labels of row i first, then the others, each in order;
  // returns their count.
  std::size_t order_row(std::size_t i, std::size_t* order) const {
    if (labels) {
      const std::size_t label = checked_label(labels[i], k);
      order[0] = label;
      for (std::size_t j = 0; j < k; ++j)
        if (j != label) order[j < label ? j + 1 : j] = j;
      return 1;
    }
    const std::uint8_t* row = positive + i * k;
    std::size_t p = 0, q = k;
    for (std::size_t j = 0; j < k; ++j)
      if (row[j]) order[p++] = j;
    for (std::size_t j = k; j-- > 0;)
      if (!row[j]) order[--q] = j;
    return p;
  }

  void check_positives(std::size_t p) const {
    if (p == 0) throw std::invalid_argument("Every row needs a positive label");
    const bool single = kind == loss_kind::topk_cross_entropy ||
                        kind == loss_kind::multiclass_hinge ||
                        kind == loss_kind::topk_hinge_alpha || kind == loss_kind::topk_hinge_beta;
    if (single && p > 1) throw std::invalid_argument("This loss takes one positive label per row");
    const bool negative =
        kind == loss_kind::multilabel_hinge || kind == loss_kind::proximal_cross_entropy;
    if (negative && p == k) throw std::invalid_argument("This loss needs a negative label per row");
  }

  bool entropy_kind() const {
    return kind == loss_kind::cross_entropy || kind == loss_kind::multilabel_cross_entropy;
  }

  void run_mask_entropy(std::size_t begin, std::size_t end, Data* converted) const {
    detail::entropy_buffer<double> terms(k);
    detail::entropy_buffer<std::size_t> index(k);
    detail::entropy_buffer<Data> grad(gradients ? 0 : k);
    for (std::size_t i = begin; i < end; ++i) {
      const Data* row = scores_of(i, converted);
      const std::size_t p = mask_positions(k, positive + i * k, index.data());
      check_positives(p);
      Data* g = gradients ? gradients + i * k : grad.data();
      values[i] = mask_entropy_row<Data, Result>(k, positive + i * k, row, p, index.data(),
                                                 terms.data(), g);
    }
  }

  void operator()(std::size_t begin, std::size_t end) const {
    using Work = typename std::common_type<Data, Result>::type;
    detail::entropy_buffer<Data> converted(sizeof(typename Format::type) == 2 ? k : 0);
    if (!labels && entropy_kind()) {
      run_mask_entropy(begin, end, converted.data());
      return;
    }
    detail::entropy_buffer<Data> scores(k), grad(k);
    detail::entropy_buffer<std::size_t> order_buffer(k);
    detail::entropy_buffer<Work> terms(topk_entropy() ? k : 0);
    std::size_t* order = order_buffer.data();
    const bool single = entropy_kind() || topk_entropy() || top1_hinge();
    for (std::size_t i = begin; i < end; ++i) {
      if (labels && top1_hinge()) {
        values[i] = hinge_row(z + i * k, i, gradient_row(i), converted.data());
        continue;
      }
      const Data* row = scores_of(i, converted.data());
      Data* g = gradients ? gradients + i * k : grad.data();
      // Rows of one label take the row in place (labels need no permutation).
      const std::size_t p = labels && single ? 1 : order_row(i, order);
      check_positives(p);
      if (single && p == 1) {
        const std::size_t label = labels ? checked_label(labels[i], k) : order[0];
        values[i] = single_row(label, row, gradients ? g : nullptr, g, terms.data());
        continue;
      }
      for (std::size_t j = 0; j < k; ++j) scores.data()[j] = row[order[j]];
      values[i] = loss_and_gradient<Data, Result>(kind, k, p, scores.data(), grad.data(), topk,
                                                  gamma, alpha, mode);
      if (gradients)
        for (std::size_t j = 0; j < k; ++j) g[order[j]] = grad.data()[j];
    }
  }

  // The loss of a row of one label; gradient may be null (top-k entropy writes into work).
  double single_row(std::size_t label, const Data* row, Data* gradient, Data* work,
                    typename std::common_type<Data, Result>::type* terms) const {
    if (topk_entropy())
      return topk_entropy_label_loss<Data, Result>(k, label, row, topk, work, terms);
    if (top1_hinge()) return hinge_label_loss<Data, Result>(k, label, row, gradient);
    return entropy_label_loss<Data, Result>(k, label, row, gradient);
  }

  // Whether labeled rows take topk_entropy_label_loss (1 < topk < k - 1).
  bool topk_entropy() const {
    return kind == loss_kind::topk_cross_entropy && topk > 1 && topk + 1 < k;
  }

  // Whether labeled rows take hinge_label_loss (the nonsmooth hinges at top-1).
  bool top1_hinge() const {
    const bool topk_hinge =
        kind == loss_kind::topk_hinge_alpha || kind == loss_kind::topk_hinge_beta;
    return gamma == 0 && (kind == loss_kind::multiclass_hinge || (topk_hinge && topk == 1));
  }

  // The gradient of row i, or null for the losses alone.
  Data* gradient_row(std::size_t i) const { return gradients ? gradients + i * k : nullptr; }

  // hinge_label_loss of the labeled row i, 16-bit rows widened into buffer.
  double hinge_row(const std::uint16_t*, std::size_t i, Data* g, Data* buffer) const {
    const Data* row = scores_of(i, buffer);
    return hinge_label_loss<Data, Result>(k, checked_label(labels[i], k), row, g);
  }
  // Stored rows check their finiteness in the extrema pass.
  double hinge_row(const Data* row, std::size_t i, Data* g, Data*) const {
    const bool valid = labels[i] >= 0 && static_cast<std::uint64_t>(labels[i]) < k;
    const std::size_t skip = valid ? static_cast<std::size_t>(labels[i]) : 0;
    Data lowest;
    bool finite = false;
    const Data competitor = detail::finite_extrema_kernel(k, row, skip, lowest, finite);
    if (!finite) throw std::invalid_argument(finite_message);
    const std::size_t label = checked_label(labels[i], k);
    // The arithmetic of hinge_label_loss: T where it holds the spread of the scores.
    using T = typename std::common_type<Data, Result>::type;
    const T high = T(std::max(competitor, row[label])), low = T(std::min(lowest, row[label]));
    if (!std::isfinite(high - low))
      return detail::hinge_label<Data, double>(k, label, row, competitor, g);
    return detail::hinge_label<Data, T>(k, label, row, competitor, g);
  }

  // The finite scores of row i: the stored row, or 16-bit ones in buffer.
  const Data* scores_of(std::size_t i, Data* buffer) const {
    const typename Format::type* row = z + i * k;
    const Data* scores = stored_scores(row, buffer);
    if (!scores) throw std::invalid_argument(finite_message);
    return scores;
  }
  const Data* stored_scores(const Data* row, Data*) const {
    // In lanes, branch-free: v - v is NaN exactly at the nonfinite scores.
    const std::size_t lanes = 16;
    Data bad[lanes] = {};
    std::size_t j = 0;
    for (; j + lanes <= k; j += lanes)
      for (std::size_t l = 0; l < lanes; ++l) bad[l] += row[j + l] - row[j + l];
    for (std::size_t l = 0; j + l < k; ++l) bad[l] += row[j + l] - row[j + l];
    return std::accumulate(bad, bad + lanes, Data(0)) == 0 ? row : nullptr;
  }
  const Data* stored_scores(const std::uint16_t* row, Data* buffer) const {
    return widen_row(Format(), k, row, buffer) ? buffer : nullptr;
  }
};

// Distillation rows (sdca_distillation_forward, sdca_distillation_backward and
// sdca_proximal_distillation_rows) of student and teacher logits zs, zt at the
// temperature T: q = softmax(zs / T), p = softmax(zt / T), the loss H(p, q)
// and the gradient (q - p) / T, with u = q - p taken as p expm1(log(q / p))
// where student and teacher agree: equal logits give a zero gradient.

SDCA_ROW_INLINE float select_float(bool c, float a, float b) {
  return bits_float(select(c, float_bits(a), float_bits(b)));
}

// exp(x) - 1 where small (|x| <= 1 there), else exp(x) (0 below -86.6, x
// clamped at 88): one reduction, 2^n expm1(r) + (2^n - 1) or + 2^n.
SDCA_ROW_INLINE float exp_or_expm1(float x, bool small) {
  const bool tiny = x < -86.6f;
  x = select_float(tiny, -86.6f, select_float(x > 88.0f, 88.0f, x));
  const float shift = 12582912.0f;  // 1.5 * 2^23: adding it rounds to an integer
  const float t = x * 1.44269504088896341f + shift, n = t - shift;
  const float r = (x - n * 0.693359375f) - n * -2.12194440e-4f;
  float p = 1.0f / 40320;
  p = p * r + 1.0f / 5040;
  p = p * r + 1.0f / 720;
  p = p * r + 1.0f / 120;
  p = p * r + 1.0f / 24;
  p = p * r + 1.0f / 6;
  p = p * r + 0.5f;
  const float m = (p * r) * r + r;
  const float scale = bits_float((float_bits(t) - float_bits(shift) + 127u) << 23);
  const float e = scale * m + scale, em1 = scale * m + (scale - 1.0f);
  return bits_float(select(small, float_bits(em1), select(tiny, 0u, float_bits(e))));
}

// Columns of the float row statistics: 1 (-1 for a row in double), the
// student's and the teacher's maximum, 1 / sum exp(t) and 1 / sum exp(s) of the
// centered scaled logits, the p-weighted mean of delta and log1p of the
// correction of the log-ratio.
const std::size_t distillation_stats = 7;

// Outcomes of a float row.
enum distillation_status { row_nonfinite, row_double, row_float };

// The constants of a float row: the maxima, 1 / T and s_max - t_max.
struct distillation_scale {
  float s_max, t_max, inv_temp;
  double offset, inv_temp_double;
};

// The centered scaled logits of an entry, (z - max) / T, and delta = (zs - zt)
// - (s_max - t_max), retaining the rounding errors of both double differences.
struct distillation_entry {
  float s, t, delta;
};
template <typename Format>
SDCA_ROW_INLINE distillation_entry scaled_entry(const distillation_scale& k,
                                                typename Format::type a, typename Format::type b) {
  const float x = widen(Format(), a), y = widen(Format(), b);
  const distillation_entry e = {(x - k.s_max) * k.inv_temp, (y - k.t_max) * k.inv_temp,
                                float(sdca::detail::offset_difference(
                                    double(x), double(y), double(k.s_max), double(k.t_max)))};
  return e;
}

// The log-ratio (delta - mean) / T - correction in double, rounded once: it
// cancels to a small value where student and teacher agree.
SDCA_ROW_INLINE float log_ratio(const distillation_scale& k, float delta, double mean,
                                double correction) {
  return float((double(delta) - mean) * k.inv_temp_double - correction);
}

// The extrema of z[0, c); false at a nonfinite score.
template <typename Format>
SDCA_ROW_INLINE bool row_extrema(std::size_t c, const typename Format::type* z, float& high,
                                 float& low) {
  const float infinity = std::numeric_limits<float>::infinity();
  float hi[float_lanes], lo[float_lanes], bad[float_lanes];
  for (std::size_t l = 0; l < float_lanes; ++l) {
    hi[l] = -infinity;
    lo[l] = infinity;
    bad[l] = 0;
  }
  extrema_lanes<Format>(c, z, hi, lo, bad);
  high = -infinity;
  low = infinity;
  bool finite = true;
  for (std::size_t l = 0; l < float_lanes; ++l) {
    high = hi[l] > high ? hi[l] : high;
    low = lo[l] < low ? lo[l] : low;
    finite = finite && bad[l] == 0;
  }
  return finite;
}

// The forward terms of an entry: exp(s) and exp(t) where s and t are not 0
// (else 1 in the zero counts: exp(0) is exactly 1, so the tails without the
// leading entries take one less), exp(t) (-s) and exp(t) delta.
struct forward_term {
  float s_tail, s_zero, t_tail, t_zero, cross, mean;
};
template <typename Format>
SDCA_ROW_INLINE forward_term forward_terms(const distillation_scale& k, typename Format::type a,
                                           typename Format::type b, float* et, float* s,
                                           float* delta) {
  const distillation_entry v = scaled_entry<Format>(k, a, b);
  const float es = exp_float(v.s), e = exp_float(v.t);
  const bool s_zero = v.s == 0.0f, t_zero = v.t == 0.0f;
  // exp(t) (-s), without 0 * inf where both underflow.
  const forward_term f = {select_float(s_zero, 0.0f, es),         select_float(s_zero, 1.0f, 0.0f),
                          select_float(t_zero, 0.0f, e),          select_float(t_zero, 1.0f, 0.0f),
                          select_float(e > 0.0f, e * -v.s, 0.0f), e * v.delta};
  *et = e;
  *s = v.s;
  *delta = v.delta;
  return f;
}

// The correction term of entry i: p expm1(dn) of the log-ratio dn where small,
// else exp(s + shift) - p from the student (also where p underflows).
SDCA_ROW_INLINE float correction_term(const distillation_scale& k, double mean, float inv_p,
                                      float shift, float et, float s, float delta) {
  const float p = et * inv_p, dn = log_ratio(k, delta, mean, 0.0);
  const bool small = std::fabs(dn) < 1.0f;
  const float e = exp_or_expm1(select_float(small, dn, s + shift), small);
  return select_float(small, p * e, e - p);
}

// The sums of the forward terms of a float row (the tails of exp(s) and
// exp(t) without their leading 1, exp(t) (-s) and exp(t) delta); exp(t), s
// and delta into et, s, delta.
struct forward_sums {
  double s_tail, t_tail, cross, mean;
};
template <typename Format>
SDCA_ROW_INLINE forward_sums forward_pass(const distillation_scale& k, std::size_t c,
                                          const typename Format::type* zs,
                                          const typename Format::type* zt, float* et, float* s,
                                          float* delta) {
  double sum[6][float_lanes] = {};
  for (std::size_t begin = 0; begin < c; begin += float_chunk) {
    const std::size_t end = c - begin < float_chunk ? c : begin + float_chunk;
    float part[6][float_lanes] = {};
    std::size_t i = begin;
    for (; i + float_lanes <= end; i += float_lanes)
      for (std::size_t l = 0; l < float_lanes; ++l) {
        const forward_term f =
            forward_terms<Format>(k, zs[i + l], zt[i + l], et + i + l, s + i + l, delta + i + l);
        part[0][l] += f.s_tail, part[1][l] += f.s_zero, part[2][l] += f.t_tail;
        part[3][l] += f.t_zero, part[4][l] += f.cross, part[5][l] += f.mean;
      }
    for (std::size_t l = 0; i + l < end; ++l) {
      const forward_term f =
          forward_terms<Format>(k, zs[i + l], zt[i + l], et + i + l, s + i + l, delta + i + l);
      part[0][l] += f.s_tail, part[1][l] += f.s_zero, part[2][l] += f.t_tail;
      part[3][l] += f.t_zero, part[4][l] += f.cross, part[5][l] += f.mean;
    }
    for (std::size_t m = 0; m < 6; ++m)
      for (std::size_t l = 0; l < float_lanes; ++l) sum[m][l] += double(part[m][l]);
  }
  double total[6] = {};
  for (std::size_t m = 0; m < 6; ++m)
    for (std::size_t l = 0; l < float_lanes; ++l) total[m] += sum[m][l];
  // The tails without one leading exp(0) = 1 (zero counts are exact).
  const forward_sums sums = {total[0] + (total[1] - 1), total[2] + (total[3] - 1), total[4],
                             total[5]};
  return sums;
}

// The correction sum p exp(dn) - 1 of the log-ratio dn + const over a forward
// pass's et, s and delta.
SDCA_ROW_INLINE double correction_pass(const distillation_scale& k, std::size_t c, double mean,
                                       float inv_p, float shift, const float* et, const float* s,
                                       const float* delta) {
  double corrections[float_lanes] = {};
  for (std::size_t begin = 0; begin < c; begin += float_chunk) {
    const std::size_t end = c - begin < float_chunk ? c : begin + float_chunk;
    float part[float_lanes] = {};
    std::size_t i = begin;
    for (; i + float_lanes <= end; i += float_lanes)
      for (std::size_t l = 0; l < float_lanes; ++l)
        part[l] += correction_term(k, mean, inv_p, shift, et[i + l], s[i + l], delta[i + l]);
    for (std::size_t l = 0; i + l < end; ++l)
      part[l] += correction_term(k, mean, inv_p, shift, et[i + l], s[i + l], delta[i + l]);
    for (std::size_t l = 0; l < float_lanes; ++l) corrections[l] += double(part[l]);
  }
  double correction = 0;
  for (std::size_t l = 0; l < float_lanes; ++l) correction += corrections[l];
  return correction;
}

// The loss of a float row and, unless st is NULL, its statistics; work holds 3 c
// floats. row_double where the row needs double: spreads beyond half
// the float range, tails below 1e-30 or an overflowing shift.
template <typename Format>
SDCA_ROW_INLINE int forward_lanes(std::size_t c, const typename Format::type* zs,
                                  const typename Format::type* zt, double temperature, float* work,
                                  double* loss, double* st) {
  float s_max, s_min, t_max, t_min;
  const bool finite_s = row_extrema<Format>(c, zs, s_max, s_min);
  if (!(row_extrema<Format>(c, zt, t_max, t_min) && finite_s)) return row_nonfinite;
  const double half_range = double(std::numeric_limits<float>::max()) / 2;
  if (double(s_max) - double(s_min) > half_range || double(t_max) - double(t_min) > half_range)
    return row_double;
  const distillation_scale k = {s_max, t_max, float(1 / temperature), double(s_max) - double(t_max),
                                1 / temperature};
  float *et = work, *s = work + c, *delta = work + 2 * c;
  const forward_sums sums = forward_pass<Format>(k, c, zs, zt, et, s, delta);
  if (!(sums.s_tail >= 1e-30 && sums.t_tail >= 1e-30)) return row_double;
  const double p_norm = 1 + sums.t_tail, q_norm = 1 + sums.s_tail, mean = sums.mean / p_norm;
  const double shift = -mean / temperature - std::log1p(sums.t_tail);
  if (!std::isfinite(shift)) return row_double;
  *loss = std::log1p(sums.s_tail) + sums.cross / p_norm;
  if (!st) return row_float;
  // exp_or_expm1 clamps s + shift at 88 rather than overflow: a larger shift needs float64.
  if (!(std::fabs(shift) <= 32)) return row_double;
  const double correction =
      correction_pass(k, c, mean, float(1 / p_norm), float(shift), et, s, delta);
  st[0] = 1;
  st[1] = double(s_max);
  st[2] = double(t_max);
  st[3] = 1 / p_norm;
  st[4] = 1 / q_norm;
  st[5] = mean;
  st[6] = std::log1p(correction);
  return row_float;
}

// The backward terms of an entry: u = q - p as p expm1(log(q / p)) where the
// log-ratio is small (q = p + u there), else from q = exp(s) / sum, one
// exponential either way; u and q into work.
template <typename Format>
SDCA_ROW_INLINE float backward_terms(const distillation_scale& k, double mean, double correction,
                                     float inv_p, float inv_q, typename Format::type a,
                                     typename Format::type b, float* u, float* q) {
  const distillation_entry v = scaled_entry<Format>(k, a, b);
  const float p = exp_float(v.t) * inv_p, ratio = log_ratio(k, v.delta, mean, correction);
  const bool small = std::fabs(ratio) < 1.0f;
  const float e = exp_or_expm1(select_float(small, ratio, v.s), small);
  const float pu = p * e, qi = select_float(small, p + pu, e * inv_q);
  const float ui = select_float(small, pu, qi - p);
  *u = ui;
  *q = qi;
  return ui;
}

// The float gradient of a forward row times upstream / T, rounded once into
// the format: g = u - q sum(u), every entry formed on its own (a zero-sum
// entry formed from the others would carry their roundings). work holds 2 c
// floats.
template <typename Format>
SDCA_ROW_INLINE void backward_lanes(std::size_t c, const typename Format::type* zs,
                                    const typename Format::type* zt, double temperature,
                                    const double* st, double upstream, float* work,
                                    typename Format::type* out) {
  const distillation_scale k = {float(st[1]), float(st[2]), float(1 / temperature),
                                double(float(st[1])) - double(float(st[2])), 1 / temperature};
  const float inv_p = float(st[3]), inv_q = float(st[4]);
  float *u = work, *q = work + c;
  double u_sums[float_lanes] = {};
  for (std::size_t begin = 0; begin < c; begin += float_chunk) {
    const std::size_t end = c - begin < float_chunk ? c : begin + float_chunk;
    float part[float_lanes] = {};
    std::size_t i = begin;
    for (; i + float_lanes <= end; i += float_lanes)
      for (std::size_t l = 0; l < float_lanes; ++l)
        part[l] += backward_terms<Format>(k, st[5], st[6], inv_p, inv_q, zs[i + l], zt[i + l],
                                          u + i + l, q + i + l);
    for (std::size_t l = 0; i + l < end; ++l)
      part[l] += backward_terms<Format>(k, st[5], st[6], inv_p, inv_q, zs[i + l], zt[i + l],
                                        u + i + l, q + i + l);
    for (std::size_t l = 0; l < float_lanes; ++l) u_sums[l] += double(part[l]);
  }
  double total_u = 0;
  for (std::size_t l = 0; l < float_lanes; ++l) total_u += u_sums[l];
  const float sum_u = float(total_u);
  // One float rounding of upstream / T where it fits, else the entries over T first.
  const float factor = float(upstream / temperature);
  const bool fits = std::isfinite(factor);
  const float first = fits ? factor : k.inv_temp, second = fits ? 1.0f : float(upstream);
  for (std::size_t j = 0; j < c; ++j)
    out[j] = narrow(Format(), ((u[j] - q[j] * sum_u) * first) * second);
}

#define SDCA_TENSOR_DISTILLATION_KERNELS(F)                                                \
  SDCA_ROW_KERNEL int distillation_forward_kernel(F, std::size_t c, const F::type* zs,     \
                                                  const F::type* zt, double temperature,   \
                                                  float* work, double* loss, double* st) { \
    return forward_lanes<F>(c, zs, zt, temperature, work, loss, st);                       \
  }                                                                                        \
  SDCA_ROW_KERNEL void distillation_backward_kernel(                                       \
      F, std::size_t c, const F::type* zs, const F::type* zt, double temperature,          \
      const double* st, double upstream, float* work, F::type* out) {                      \
    backward_lanes<F>(c, zs, zt, temperature, st, upstream, work, out);                    \
  }
SDCA_TENSOR_DISTILLATION_KERNELS(float16)
SDCA_TENSOR_DISTILLATION_KERNELS(bfloat16)
SDCA_TENSOR_DISTILLATION_KERNELS(float32)
#undef SDCA_TENSOR_DISTILLATION_KERNELS

// float64 storage has no float path (its rows take double).
inline int distillation_forward_kernel(float64, std::size_t, const double*, const double*, double,
                                       float*, double*, double*) {
  return row_double;
}
inline void distillation_backward_kernel(float64, std::size_t, const double*, const double*, double,
                                         const double*, double, float*, double*) {
  throw std::logic_error("float64 storage takes double arithmetic");
}

using detail::distillation_double;
using detail::distillation_double_gradient;
using detail::distillation_row;
using detail::proximal_distillation_row;

// out[j] = double(z[j]).
inline void to_double(float16 f, std::size_t n, const std::uint16_t* z, double* out) {
  widen_row(f, n, z, out);
}
inline void to_double(bfloat16 f, std::size_t n, const std::uint16_t* z, double* out) {
  widen_row(f, n, z, out);
}
inline void to_double(float32, std::size_t n, const float* z, double* out) {
  std::copy(z, z + n, out);
}
inline void to_double(float64, std::size_t n, const double* z, double* out) {
  std::copy(z, z + n, out);
}

// out[j] = g[j] rounded once into the format.
inline void from_double(float16 f, std::size_t n, const double* g, std::uint16_t* out) {
  narrow_kernel(f, n, g, out);
}
inline void from_double(bfloat16 f, std::size_t n, const double* g, std::uint16_t* out) {
  narrow_kernel(f, n, g, out);
}
inline void from_double(float32, std::size_t n, const double* g, float* out) {
  for (std::size_t j = 0; j < n; ++j) out[j] = float(g[j]);
}
inline void from_double(float64, std::size_t n, const double* g, double* out) {
  std::copy(g, g + n, out);
}

// The logits of a distillation call: student and teacher rows of k classes.
template <typename Format>
struct distillation_logits {
  std::size_t k;
  const typename Format::type *zs, *zt;
  double temperature;

  // distillation_double of row i in w (6 k doubles): log-ratio, u, s, t, p, q.
  distillation_row double_row(std::size_t i, double* w) const {
    to_double(Format(), k, zs + i * k, w);
    to_double(Format(), k, zt + i * k, w + k);
    const distillation_row row =
        distillation_double(k, temperature, w, w + k, w + 2 * k, w + 3 * k, w + 4 * k, w + 5 * k);
    if (!row.finite) throw std::invalid_argument(finite_message);
    return row;
  }
  // Whether rows take float work: float rows that index in 31 bits.
  bool float_rows(bool fp32) const { return fp32 && k < 0x7fffffffu; }
};

// distillation_loss of the rows [begin, end): values and, unless NULL, stats.
template <typename Format>
struct distillation_forward_rows {
  distillation_logits<Format> x;
  bool fp32;
  double* values;
  double* stats;

  void operator()(std::size_t begin, std::size_t end) const {
    row_buffer<float> narrow;
    row_buffer<double> wide;
    const std::size_t k = x.k;
    for (std::size_t i = begin; i < end; ++i) {
      double* st = stats ? stats + i * distillation_stats : nullptr;
      if (x.float_rows(fp32)) {
        const int status =
            distillation_forward_kernel(Format(), k, x.zs + i * k, x.zt + i * k, x.temperature,
                                        narrow.get(3 * k), values + i, st);
        if (status == row_nonfinite) throw std::invalid_argument(finite_message);
        if (status == row_float) continue;
      }
      values[i] = x.double_row(i, wide.get(6 * k)).loss;
      if (st) std::fill(st, st + distillation_stats, -1.0);
    }
  }
};

// Gradients of the rows [begin, end) times scale[i * stride], from their stats.
template <typename Format>
struct distillation_backward_rows {
  distillation_logits<Format> x;
  bool fp32;
  const double* stats;
  const double* scale;
  std::size_t stride;
  typename Format::type* gradients;

  void operator()(std::size_t begin, std::size_t end) const {
    row_buffer<float> narrow;
    row_buffer<double> wide;
    const std::size_t k = x.k;
    for (std::size_t i = begin; i < end; ++i) {
      const double* st = stats + i * distillation_stats;
      typename Format::type* g = gradients + i * k;
      if (x.float_rows(fp32) && st[0] > 0) {
        distillation_backward_kernel(Format(), k, x.zs + i * k, x.zt + i * k, x.temperature, st,
                                     scale[i * stride], narrow.get(2 * k), g);
        continue;
      }
      double* w = wide.get(7 * k);
      const distillation_row row = x.double_row(i, w);
      distillation_double_gradient(k, row, w + k, w + 5 * k, x.temperature, scale[i * stride],
                                   w + 6 * k);
      from_double(Format(), k, w + 6 * k, g);
    }
  }
};

// proximal_distillation_loss of the rows [begin, end) in double: values and
// gradients (Data: float for float and 16-bit storage).
template <typename Format, typename Data>
struct proximal_distillation_rows {
  distillation_logits<Format> x;
  double alpha;
  double* values;
  Data* gradients;

  void operator()(std::size_t begin, std::size_t end) const {
    row_buffer<double> wide;
    row_buffer<std::size_t> index;
    const std::size_t k = x.k;
    for (std::size_t i = begin; i < end; ++i) {
      double* w = wide.get(13 * k + 5 * detail::row_lanes);
      const distillation_row row = x.double_row(i, w);
      double* r = w + 6 * k;
      values[i] = proximal_distillation_row(k, alpha, row, w, w + k, w + 2 * k, w + 3 * k,
                                            w + 4 * k, w + 5 * k, r, w + 7 * k, index.get(2 * k));
      Data* g = gradients + i * k;
      for (std::size_t j = 0; j < k; ++j) g[j] = Data(-r[j] / x.temperature);
    }
  }
};

}  // namespace tensor

}  // namespace sdca

#endif
