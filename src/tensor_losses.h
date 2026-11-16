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
#include "sdca/math/distillation_rows.h"
#include "sdca/math/entropy.h"
#include "sdca/math/exp_rows.h"
#include "sdca/math/lambert.h"
#include "sdca/math/log_exp.h"
#include "sdca/math/prox_rows.h"
#include "sdca/math/storage.h"
#include "sdca/utility/parallel.h"

namespace sdca {

namespace tensor {

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
