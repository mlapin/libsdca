#ifndef SDCA_MATH_DISTILLATION_ROWS_H
#define SDCA_MATH_DISTILLATION_ROWS_H

// Shared float row kernels for the C++ loss helpers and native tensor APIs.
#include <cmath>
#include <limits>
#include <stdexcept>
#include "sdca/math/distillation.h"
#include "sdca/math/tensor_float.h"

namespace sdca {
namespace tensor {

// Distillation rows (the distillation losses of sdca_loss_evaluate and the
// sdca_internal_*distillation* entry points) of student and teacher logits zs, zt at the
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

}  // namespace tensor
}  // namespace sdca
#endif
