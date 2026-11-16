#ifndef SDCA_CPU_DISTILL_H
#define SDCA_CPU_DISTILL_H
#include "_cpu_lanes.h"
#include <vector>

namespace {
namespace sdca_cpu {
struct DistillationStats {
  double loss, temp, s_lognorm, t_lognorm, u_sum;
  int64_t pivot;
};
// log1p of a row scalar (finite, > -1) by dlanes' inlined polynomial (within an ulp): small
// rows take three per row, where the library call costs as much as the row's lanes.
inline double log1p_fast(double x) { return dlanes::log1p(dlanes::splat(x))[0]; }

// distillation_loss rows in float32: float exponentials, float64 sums and row scalars.
namespace distill32 {
using lanes::vd;
using lanes::vf;
using lanes::vi;

// Centered scaled logits of a row lane, s and t (times the reciprocal temperature), and
// delta = (cs - s_max) - (ct - t_max), including subtraction errors in double.
// Widening raw cs - ct alone still loses small teacher gaps beside a huge student offset.
struct Scaled {
  vf s, t, delta;
};
template <bool Exact = false>
inline Scaled scaled(vf cs, vf ct, float s_max, float t_max, float inv_temp) {
  const vd x = __builtin_convertvector(cs, vd), y = __builtin_convertvector(ct, vd);
  const vd a = x - double(s_max), b = y - double(t_max);
  if constexpr (Exact) {
    return {(cs - s_max) * inv_temp, (ct - t_max) * inv_temp, __builtin_convertvector(a - b, vf)};
  }
  const vd ea = (x - (a - (a - x))) + (-double(s_max) - (a - x));
  const vd eb = (y - (b - (b - y))) + (-double(t_max) - (b - y));
  const vd delta = (a - b) + (ea - eb);
  return {(cs - s_max) * inv_temp, (ct - t_max) * inv_temp, __builtin_convertvector(delta, vf)};
}

// The log-ratio (delta - mean) / T - shift in double, rounded once: it cancels to a small
// value where student and teacher agree.
inline vf log_ratio(vf delta, double mean, double inv_temp, double shift) {
  return __builtin_convertvector((__builtin_convertvector(delta, vd) - mean) * inv_temp - shift,
                                 vf);
}

// Lanes i < count of v, zeros elsewhere.
inline vf head(vf v, int count) {
  if (count == lanes::L) return v;
  return lanes::select(lanes::iota() < count, v, lanes::splat(0.0f));
}

// Lanes i < count of v into p (float or 16-bit, rounded once).
template <typename T>
inline void put(T* p, vf v, int count) {
  if (count == lanes::L) {
    lanes::store(p, v);
  } else {
    for (int i = 0; i < count; ++i) p[i] = static_cast<T>(v[i]);
  }
}

// The lane of float or 16-bit x at j (count entries, the rest pad), widened to float.
template <typename T>
inline vf part(const T* x, int64_t j, int count, float pad) {
  if (count == lanes::L) return lanes::load(x + j);
  float tail[lanes::L];
  for (int i = 0; i < lanes::L; ++i) tail[i] = i < count ? static_cast<float>(x[j + i]) : pad;
  return lanes::load(tail);
}

// Statistics per row: 1 (-1 for rows in float64), student and teacher maxima, the reciprocal
// sums of exp(t) and exp(s), the p-weighted mean of delta and log1p of the correction.
constexpr int64_t kStats = 7;
enum Status { kNonfinite, kDouble, kFloat };

// A forward row: float views of the logits and padded work buffers of exp(t), s and delta.
template <typename T>
struct RowOf {
  const T *cs, *ct;
  float *et, *s, *delta;
  int64_t c;
};
using Row = RowOf<float>;

// The row sums of a forward pass: extrema, the tails, the p-weighted mean of delta, the shift
// of the student's log-ratio, the cross term sum p (-s) and the loss.
struct Totals {
  float s_max, t_max;
  double s_sum, t_sum, mean_delta, shift, cross, loss;
  bool exact;
};

// What the scaled pass of a forward row computes: the loss; also the p-weighted mean of delta
// and the shift (the statistics); or, without exp(s), its tail and the loss, the work rows
// exp(t), s and delta (forward_gradient, whose correction gives the student's tail).
enum Pass { kLoss, kMean, kTeacher };

// The scaled pass of a forward row in float32 (with es, it also keeps exp(s) there): kDouble
// where it needs float64 (spreads beyond half the float32 range, tails below 1e-30, a shift
// that overflows where it forms one).
template <Pass P, typename T>
SDCA_ROW Status scaled_pass(const RowOf<T>& r, float temp, Totals& o, float* es_row) {
  const float inv_temp = 1 / temp;
  bool finite = true;
  float s_min, s_max, t_min, t_max;
  lanes::extrema(r.cs, r.c, s_min, s_max, finite);
  lanes::extrema(r.ct, r.c, t_min, t_max, finite);
  if (!finite) return kNonfinite;
  const double half_range = std::numeric_limits<float>::max() / 2.0;
  if (double(s_max) - s_min > half_range || double(t_max) - t_min > half_range) return kDouble;
  const bool exact = std::max(std::abs(s_min), std::abs(s_max)) <= 1e4f &&
                     std::max(std::abs(t_min), std::abs(t_max)) <= 1e4f;
  const int64_t s_idx = find_at_least(r.cs, 0, r.c, s_max);
  const int64_t t_idx = find_at_least(r.ct, 0, r.c, t_max);
  // The tails without their leading terms (exp(0) = 1): 1 + tail would round them.
  lanes::Accumulator s_tail, t_tail, cross, mean;
  // The lanes of a vector at j, count entries, but the one at index (if there).
  auto without = [](vf v, int64_t index, int64_t j, int count) {
    if (static_cast<uint64_t>(index - j) >= static_cast<uint64_t>(count)) return v;
    return lanes::select(lanes::iota() == static_cast<int32_t>(index - j), lanes::splat(0.0f), v);
  };
  lanes::each(r.cs, r.c, s_max, [&](vf cs, int64_t j, int count) {
    const vf ct = part(r.ct, j, count, t_max);
    const vf s = (cs - s_max) * inv_temp, t = (ct - t_max) * inv_temp;
    const vf et = head(lanes::exp(t), count);
    if constexpr (P != kTeacher) {
      const vf es = lanes::exp(s);
      s_tail.add(without(head(es, count), s_idx, j, count));
      if (es_row) put(es_row + j, es, count);
    }
    t_tail.add(without(et, t_idx, j, count));
    // exp(t) (-s), without 0 * inf where both underflow.
    cross.add(lanes::select(et > 0.0f, et * -s, lanes::splat(0.0f)));
    if constexpr (P != kLoss) {
      const vf delta = exact ? scaled<true>(cs, ct, s_max, t_max, inv_temp).delta
                             : scaled<false>(cs, ct, s_max, t_max, inv_temp).delta;
      mean.add(et * delta);
      if constexpr (P == kTeacher)
        put(r.et + j, et, count), put(r.s + j, s, count), put(r.delta + j, delta, count);
    }
  });
  o.s_max = s_max, o.t_max = t_max, o.exact = exact;
  o.t_sum = t_tail.value();
  if (!(o.t_sum >= 1e-30)) return kDouble;
  const double p_norm = 1 + o.t_sum;
  o.cross = cross.value() / p_norm;
  if constexpr (P != kLoss) {
    o.mean_delta = mean.value() / p_norm;
    o.shift = -o.mean_delta / temp - log1p_fast(o.t_sum);
    if (!std::isfinite(o.shift)) return kDouble;
  }
  if constexpr (P != kTeacher) {
    o.s_sum = s_tail.value();
    if (!(o.s_sum >= 1e-30)) return kDouble;
    o.loss = log1p_fast(o.s_sum) + o.cross;
  }
  return kFloat;
}

// The correction sum p exp(dn) - 1 of the log-ratio dn + const of a scaled row: p expm1(dn)
// where dn is small, else exp(s + shift) - p from the student (also where p underflows), as
// exp(s) exp(shift) where exp_shift is given. With em and dn, it keeps each lane's exponential
// (expm1(dn) where |dn| < 1, else exp(s) or exp(s + shift)) and dn there.
SDCA_ROW double correction(const Row& r, float temp, const Totals& o, float* em_row, float* dn_row,
                           float exp_shift = 0) {
  const float inv_temp = 1 / temp;
  const float inv_p = static_cast<float>(1 / (1 + o.t_sum));
  const float shift_f = exp_shift ? 0.0f : static_cast<float>(o.shift);
  const float factor = exp_shift ? exp_shift : 1.0f;
  lanes::Accumulator total;
  lanes::each(r.delta, r.c, 0.0f, [&](vf delta, int64_t j, int count) {
    const vf p = lanes::load(r.et + j) * inv_p;
    const vf dn = log_ratio(delta, o.mean_delta, inv_temp, 0.0);
    const vi small = lanes::abs(dn) < 1.0f;
    const vf e =
        lanes::exp_or_expm1(lanes::select(small, dn, lanes::load(r.s + j) + shift_f), small);
    total.add(head(lanes::select(small, p * e, e * factor - p), count));
    if (em_row) put(em_row + j, e, count), put(dn_row + j, dn, count);
  });
  return total.value();
}

// The exponentials of a row's gradient pass: exp(t) (p times 1 + the teacher's tail), the
// log-ratio dn (rounded once from delta), the correction's expm1(dn) where |dn| < 1 and
// exp(s) (q times 1 + the student's tail) elsewhere.
struct Exponentials {
  const float *et, *delta, *dn, *em, *es;
  float inv_p, inv_q;
};

// The gradient times T of a row from its exponentials and correction C (total): u = p
// (expm1(dn) - C) / (1 + C) where |dn| < 1 (= p expm1(dn - log1p(C))), u = q - p elsewhere,
// then u - q sum(u); scale(each entry) rounded into out (which may be u). Every entry is
// formed on its own: a zero-sum entry formed from the others would carry their roundings
// (hundreds of eps of the row's largest entry at 128k classes). u and q hold C floats
// (padded).
template <typename T, typename Scale>
SDCA_ROW void gradient_pass(const Exponentials& x, int64_t c, float temp, double mean_delta,
                            double total, float* g, float* q_row, T* out, Scale scale) {
  const double log1p_corr = log1p_fast(total);
  const float inv_temp = 1 / temp, lc = static_cast<float>(log1p_corr);
  const float c_f = static_cast<float>(total), inv_c = static_cast<float>(1 / (1 + total));
  lanes::Accumulator u_sum;
  lanes::each(x.dn, c, 0.0f, [&](vf dn, int64_t j, int count) {
    const vf p = lanes::load(x.et + j) * x.inv_p;
    vi small = lanes::abs(dn) < 1.0f;
    vf pu = p * ((lanes::load(x.em + j) - c_f) * inv_c);
    // |dn| >= 1 but |dn - log1p(C)| < 1 (rare): p expm1 of the shifted log-ratio itself. The
    // float difference finds every such lane (it errs by far less than 1e-3 there).
    const vi maybe = ~small & (lanes::abs(dn - lc) < 1.001f);
    if (any_lane(maybe)) {
      const vf ratio = log_ratio(lanes::load(x.delta + j), mean_delta, inv_temp, log1p_corr);
      const vi shifted = maybe & (lanes::abs(ratio) < 1.0f);
      pu = lanes::select(shifted, p * lanes::exp_or_expm1(ratio, shifted), pu);
      small = small | shifted;
    }
    const vf q = lanes::select(small, p + pu, lanes::load(x.es + j) * x.inv_q);
    const vf u = head(lanes::select(small, pu, q - p), count);
    u_sum.add(u);
    put(g + j, u, count), put(q_row + j, q, count);
  });
  const float total_u = static_cast<float>(u_sum.value());
  lanes::each(g, c, 0.0f, [&](vf u, int64_t j, int count) {
    put(out + j, scale(u - part(q_row, j, count, 0.0f) * total_u), count);
  });
}

// The largest |shift| of rows this kernel takes with a gradient: backward and forward_gradient
// form exp(s + shift) as exp(s) exp(shift), whose float factor then neither overflows nor loses
// an exp(s) that underflows.
constexpr double kShift = 32;

// The forward row in float32: kDouble where it needs float64 (spreads beyond half the
// float32 range, tails below 1e-30, an overflowing shift; with need_stats, |shift| > kShift).
// Its statistics st: 1, the maxima, the reciprocal sums of exp(t) and exp(s), the p-weighted
// mean of delta and the shift; backward computes the correction.
template <typename T>
inline Status forward(const RowOf<T>& r, float temp, bool need_stats, double& loss, double* st) {
  Totals o;
  const Status status = need_stats ? scaled_pass<kMean>(r, temp, o, nullptr)
                                   : scaled_pass<kLoss>(r, temp, o, nullptr);
  if (status != kFloat) return status;
  loss = o.loss;
  if (!need_stats) return kFloat;
  if (!(std::abs(o.shift) <= kShift)) return kDouble;
  st[0] = o.exact ? 1.5 : 1.0, st[1] = o.s_max, st[2] = o.t_max;
  st[3] = static_cast<float>(1 / (1 + o.t_sum)), st[4] = static_cast<float>(1 / (1 + o.s_sum));
  st[5] = o.mean_delta, st[6] = o.shift;
  return kFloat;
}

// The forward row and its gradient times T into g (gradient_pass): two exponentials per entry,
// exp(t) and the correction's, which is exp(s) where |dn| >= 1. The student's sum follows from
// the correction C: sum exp(s) = exp(-shift) sum p exp(dn) = exp(-shift) (1 + C), so
// log(1 + its tail) = log1p(C) - shift. kDouble also for |shift| > kShift (as forward's
// statistics) and tails below 1e-30. work holds 3 C floats (padded).
inline Status forward_gradient(const Row& r, float temp, double& loss, float* g, float* work,
                               int64_t cp) {
  float *em_row = work, *dn_row = work + cp;
  Totals o;
  const Status status = scaled_pass<kTeacher>(r, temp, o, nullptr);
  if (status != kFloat) return status;
  if (!(std::abs(o.shift) <= kShift)) return kDouble;
  const double exp_shift = std::exp(o.shift);
  const double total = correction(r, temp, o, em_row, dn_row, static_cast<float>(exp_shift));
  // A correction beyond float32 (a log-ratio mean beyond about 88) needs float64.
  if (!std::isfinite(static_cast<float>(total))) return kDouble;
  const double s_lognorm = log1p_fast(total) - o.shift;
  if (!(s_lognorm >= 1e-30)) return kDouble;
  loss = s_lognorm + o.cross;
  const Exponentials x{r.et,
                       r.delta,
                       dn_row,
                       em_row,
                       em_row,
                       static_cast<float>(1 / (1 + o.t_sum)),
                       static_cast<float>(exp_shift / (1 + total))};
  gradient_pass(x, r.c, temp, o.mean_delta, total, g, work + 2 * cp, g, [](auto v) { return v; });
  return kFloat;
}

// The gradient of a forward row of float or 16-bit logits times factor (upstream / temperature)
// into out, rounded once from float32 work, as forward_gradient: one pass for exp(t), delta,
// dn and the correction's exponentials (expm1(dn) where |dn| < 1, else exp(s), its
// exp(s + shift) as exp(s) exp(shift)), then gradient_pass. work holds 6 C floats (padded).
template <typename T>
SDCA_ROW void backward(const T* zs, const T* zt, int64_t c, int64_t cp, float temp,
                       const double* st, double upstream, T* out, float* work) {
  const float s_max = static_cast<float>(st[1]), t_max = static_cast<float>(st[2]);
  const float inv_p = static_cast<float>(st[3]);
  const double mean_delta = st[5];
  const float inv_temp = 1 / temp, exp_shift = static_cast<float>(std::exp(st[6]));
  const bool exact = st[0] == 1.5;
  float *et_row = work, *delta_row = work + cp, *dn_row = work + 2 * cp, *e_row = work + 3 * cp;
  float* g = work + 4 * cp;
  lanes::Accumulator corrections;
  for (int64_t j = 0; j < c; j += lanes::L) {
    const int count = static_cast<int>(std::min<int64_t>(lanes::L, c - j));
    const vf s_lane = part(zs, j, count, s_max), t_lane = part(zt, j, count, t_max);
    const Scaled v = exact ? scaled<true>(s_lane, t_lane, s_max, t_max, inv_temp)
                           : scaled<false>(s_lane, t_lane, s_max, t_max, inv_temp);
    const vf et = lanes::exp(v.t), p = et * inv_p;
    const vf dn = log_ratio(v.delta, mean_delta, inv_temp, 0.0);
    const vi near = lanes::abs(dn) < 1.0f;
    const vf e = lanes::exp_or_expm1(lanes::select(near, dn, v.s), near);
    corrections.add(head(lanes::select(near, p * e, e * exp_shift - p), count));
    put(et_row + j, et, count), put(delta_row + j, v.delta, count);
    put(dn_row + j, dn, count), put(e_row + j, e, count);
  }
  const Exponentials x{et_row, delta_row, dn_row, e_row, e_row, inv_p, static_cast<float>(st[4])};
  // One float rounding of upstream / T where it fits, else the entries over T first; then
  // the rounding into the logits' dtype.
  const float factor = static_cast<float>(upstream / temp), up_f = static_cast<float>(upstream);
  const bool fits = std::isfinite(factor);
  gradient_pass(x, c, temp, mean_delta, corrections.value(), g, work + 5 * cp, out,
                [&](auto v) { return fits ? v * factor : (v * inv_temp) * up_f; });
}
}  // namespace distill32

// distillation_loss rows in float64 on double lanes: rows whose scaled logits stay within
// 708 of their maxima (every exponential normal); others keep distillation_stats.
namespace distill64 {
using dlanes::vd;
using dlanes::vl;
// Scaled logits at or above this have normal exponentials.
constexpr double kLow = -708.0;
// st[0] of a row whose statistics are this kernel's.
constexpr double kTag = 2;

// Centered scaled logits s, t of a lane and delta = (cs - s_max) - (ct - t_max) with the
// rounding errors of both differences (offset_difference).
struct Scaled {
  vd s, t, delta;
};
inline Scaled scaled(vd cs, vd ct, double s_max, double t_max, double inv_temp) {
  const vd a = cs - s_max, b = ct - t_max;
  const vd ea = (cs - (a - (a - cs))) + (-s_max - (a - cs));
  const vd eb = (ct - (b - (b - ct))) + (-t_max - (b - ct));
  return {a * inv_temp, b * inv_temp, (a - b) + (ea - eb)};
}

enum Status { kNonfinite, kGeneral, kFast };

// A forward row: double views of the logits and work rows of exp(t), s and delta (need_stats).
struct Row {
  const double *cs, *ct;
  double *et, *s, *delta;
  int64_t c;
};

// The row sums of a forward pass: extrema, the tails, the p-weighted mean of delta and the loss.
struct Totals {
  double s_max, t_max, s_sum, t_sum, mean_delta, loss;
};

// The scaled pass of a forward row (with es, it also keeps exp(s) there); kGeneral where
// distillation_stats takes the row.
template <bool Compensated>
SDCA_FLAT Status scaled_pass(const Row& r, double temp, bool keep, Totals& o, double* es_row) {
  using namespace dlanes;
  bool finite = true;
  double s_min, s_max, t_min, t_max;
  dlanes::extrema(r.cs, r.c, s_min, s_max, finite);
  dlanes::extrema(r.ct, r.c, t_min, t_max, finite);
  if (!finite) return kNonfinite;
  const double inv_temp = 1 / temp;
  if (!(inv_temp >= std::numeric_limits<double>::min() && inv_temp < HUGE_VAL &&
        (s_min - s_max) * inv_temp >= kLow && (t_min - t_max) * inv_temp >= kLow))
    return kGeneral;
  const int64_t c = r.c;
  const int64_t s_idx = find_at_least(r.cs, 0, c, s_max);
  const int64_t t_idx = find_at_least(r.ct, 0, c, t_max);
  // The tails without their leading terms (exp(0) = 1): 1 + tail would round them.
  Accumulator<Compensated> s_tail, t_tail, cross, mean;
  for (int64_t j = 0; j < c; j += L) {
    const int count = static_cast<int>(std::min<int64_t>(L, c - j));
    const Scaled v =
        scaled(part(r.cs, j, count, s_max), part(r.ct, j, count, t_max), s_max, t_max, inv_temp);
    const vd es = head(exp(v.s), count), et = head(exp(v.t), count);
    const vl index = iota() + j;
    s_tail.add(select(index == s_idx, splat(0.0), es));
    t_tail.add(select(index == t_idx, splat(0.0), et));
    cross.add(et * -v.s);
    mean.add(et * v.delta);
    if (keep) put(r.et + j, et, count), put(r.s + j, v.s, count), put(r.delta + j, v.delta, count);
    if (es_row) put(es_row + j, es, count);
  }
  o.s_max = s_max, o.t_max = t_max, o.s_sum = s_tail.value(), o.t_sum = t_tail.value();
  const double p_norm = 1 + o.t_sum;
  o.mean_delta = mean.value() / p_norm;
  if (!std::isfinite(o.mean_delta)) return kGeneral;
  o.loss = log1p_fast(o.s_sum) + cross.value() / p_norm;
  return kFast;
}

// The correction sum p exp(dn) - 1 of the log-ratio dn + const: p expm1(dn) where dn is
// small, else exp(s + shift) - p from the student. With em, it keeps each lane's exponential
// (expm1(dn) where |dn| < 1) there.
template <bool Compensated>
SDCA_FLAT double correction(const Row& r, double temp, const Totals& o, double* em_row) {
  using namespace dlanes;
  const double inv_temp = 1 / temp, inv_p = 1 / (1 + o.t_sum);
  const double shift = -o.mean_delta * inv_temp - log1p_fast(o.t_sum);
  Accumulator<Compensated> total;
  for (int64_t j = 0; j < r.c; j += L) {
    const int count = static_cast<int>(std::min<int64_t>(L, r.c - j));
    const vd p = part(r.et, j, count, 0.0) * inv_p;
    const vd dn = (part(r.delta, j, count, 0.0) - o.mean_delta) * inv_temp;
    const vl small = abs(dn) < 1.0;
    const vd e = exp_or_expm1(select(small, dn, part(r.s, j, count, 0.0) + shift), small);
    total.add(head(select(small, p * e, e - p), count));
    if (em_row) put(em_row + j, e, count);
  }
  return total.value();
}

// The loss of a row (and with need_stats its statistics st: kTag, the maxima, the
// reciprocal sums of exp(t) and exp(s), the p-weighted mean of delta and log1p of the
// correction); kGeneral where distillation_stats takes the row.
template <bool Compensated>
Status forward(const Row& r, double temp, bool need_stats, double& loss, double* st) {
  Totals o;
  const Status status = scaled_pass<Compensated>(r, temp, need_stats, o, nullptr);
  if (status != kFast) return status;
  loss = o.loss;
  if (!need_stats) return kFast;
  st[0] = kTag, st[1] = o.s_max, st[2] = o.t_max, st[3] = 1 / (1 + o.t_sum);
  st[4] = 1 / (1 + o.s_sum), st[5] = o.mean_delta;
  st[6] = log1p_fast(correction<Compensated>(r, temp, o, nullptr));
  return kFast;
}

// The forward row and its gradient times T (u - q sum(u), every entry formed on its own) into
// g, from the forward pass's exponentials, as distill32::forward_gradient. work holds 2 C
// doubles.
template <bool Compensated>
SDCA_FLAT Status forward_gradient(const Row& r, double temp, double& loss, double* g,
                                  double* work) {
  using namespace dlanes;
  const int64_t c = r.c;
  double *es_row = work, *em_row = work + c;
  Totals o;
  const Status status = scaled_pass<Compensated>(r, temp, true, o, es_row);
  if (status != kFast) return status;
  loss = o.loss;
  const double total = correction<Compensated>(r, temp, o, em_row);
  // A correction beyond double (a log-ratio mean beyond about 709) takes distillation_stats.
  if (!std::isfinite(total)) return kGeneral;
  const double log1p_corr = log1p_fast(total), inv_c = 1 / (1 + total);
  const double inv_temp = 1 / temp, inv_p = 1 / (1 + o.t_sum), inv_q = 1 / (1 + o.s_sum);
  // q overwrites es, u goes to g.
  double* q_row = es_row;
  Accumulator<Compensated> u_sum;
  for (int64_t j = 0; j < c; j += L) {
    const int count = static_cast<int>(std::min<int64_t>(L, c - j));
    const vd p = part(r.et, j, count, 0.0) * inv_p;
    const vd dn = (part(r.delta, j, count, 0.0) - o.mean_delta) * inv_temp;
    const vl near = abs(dn) < 1.0;
    vd pu = p * ((part(em_row, j, count, 0.0) - total) * inv_c);
    // |dn| >= 1 but |dn - log1p(C)| < 1 (rare): p expm1 of the shifted log-ratio itself.
    const vd ratio = dn - log1p_corr;
    const vl shifted = ~near & (abs(ratio) < 1.0);
    if (any_lane(shifted)) pu = select(shifted, p * exp_or_expm1(ratio, shifted), pu);
    const vl small = near | shifted;
    const vd q = head(select(small, p + pu, part(es_row, j, count, 0.0) * inv_q), count);
    const vd u = head(select(small, pu, q - p), count);
    u_sum.add(u);
    put(g + j, u, count), put(q_row + j, q, count);
  }
  const double total_u = u_sum.value();
  for (int64_t j = 0; j < c; j += L) {
    const int count = static_cast<int>(std::min<int64_t>(L, c - j));
    put(g + j, part(g, j, count, 0.0) - part(q_row, j, count, 0.0) * total_u, count);
  }
  return kFast;
}

// The gradient of a forward row times scale / T in double into g: u = q - p, g = u - q
// sum(u), every entry formed on its own. q holds C doubles.
template <bool Compensated>
SDCA_FLAT void backward(const double* cs, const double* ct, int64_t c, double temp,
                        const double* st, double scale, double* g, double* q_row) {
  using namespace dlanes;
  const double s_max = st[1], t_max = st[2], inv_p = st[3], inv_q = st[4];
  const double mean_delta = st[5], log1p_corr = st[6], inv_temp = 1 / temp;
  Accumulator<Compensated> u_sum;
  for (int64_t j = 0; j < c; j += L) {
    const int count = static_cast<int>(std::min<int64_t>(L, c - j));
    const Scaled v =
        scaled(part(cs, j, count, s_max), part(ct, j, count, t_max), s_max, t_max, inv_temp);
    // u = q - p as p expm1(log(q / p)) where the log-ratio is small (q = p + u there), else
    // from q = exp(s) / sum: one exponential either way, besides p's.
    const vd p = exp(v.t) * inv_p;
    const vd ratio = (v.delta - mean_delta) * inv_temp - log1p_corr;
    const vl small = abs(ratio) < 1.0;
    const vd e = exp_or_expm1(select(small, ratio, v.s), small);
    const vd pu = p * e;
    const vd q = head(select(small, p + pu, e * inv_q), count);
    const vd u = head(select(small, pu, q - p), count);
    u_sum.add(u);
    put(g + j, u, count), put(q_row + j, q, count);
  }
  const double total_u = u_sum.value();
  for (int64_t j = 0; j < c; j += L) {
    const int count = static_cast<int>(std::min<int64_t>(L, c - j));
    put(g + j, part(g, j, count, 0.0) - part(q_row, j, count, 0.0) * total_u, count);
  }
  // One rounding of the scaled entry, unless scale / T overflows (tiny temperatures).
  const double factor = scale / temp;
  const bool fits = std::isfinite(factor);
  for (int64_t j = 0; j < c; j += L) {
    const int count = static_cast<int>(std::min<int64_t>(L, c - j));
    const vd v = part(g, j, count, 0.0);
    put(g + j, fits ? v * factor : (v * inv_temp) * scale, count);
  }
}

// The row arrays of proximal_distillation_loss from distill64's passes, as distillation_stats:
// p (et) and q (es) normalized in place, delta into log p, the log-ratio log(q / p) into lr and
// u = q - p into u (from em, the correction pass's exponentials); r.cs and r.ct may alias lr
// and u (each entry is read before it is written). kGeneral where distillation_stats takes
// the row.
template <bool Compensated>
Status prox_stats(const Row& r, double temp, double* es, double* em, double* lr, double* u,
                  DistillationStats& st) {
  using namespace dlanes;
  const int64_t c = r.c;
  Totals o;
  const Status status = scaled_pass<Compensated>(r, temp, true, o, es);
  if (status != kFast) return status;
  const double total = correction<Compensated>(r, temp, o, em);
  // A correction beyond double (a log-ratio mean beyond about 709) takes distillation_stats.
  if (!std::isfinite(total)) return kGeneral;
  const double log1p_corr = log1p_fast(total), inv_c = 1 / (1 + total);
  const double inv_temp = 1 / temp, inv_p = 1 / (1 + o.t_sum), inv_q = 1 / (1 + o.s_sum);
  const double t_lognorm = log1p_fast(o.t_sum);
  double* log_p = r.delta;
  vd top = splat(-1.0);
  vl top_index{};
  for (int64_t j = 0; j < c; j += L) {
    const int count = static_cast<int>(std::min<int64_t>(L, c - j));
    const vd p = part(r.et, j, count, 0.0) * inv_p, q = part(es, j, count, 0.0) * inv_q;
    const vd dn = (part(r.delta, j, count, 0.0) - o.mean_delta) * inv_temp;
    const vd ratio = dn - log1p_corr;
    const vl near = abs(dn) < 1.0, shifted = ~near & (abs(ratio) < 1.0);
    vd pu = p * ((part(em, j, count, 0.0) - total) * inv_c);
    if (any_lane(shifted)) pu = select(shifted, p * exp_or_expm1(ratio, shifted), pu);
    const vd uj = select(near | shifted, pu, q - p);
    const vd t = (part(r.ct, j, count, o.t_max) - o.t_max) * inv_temp;
    put(r.et + j, p, count), put(es + j, q, count), put(log_p + j, t - t_lognorm, count);
    put(lr + j, ratio, count), put(u + j, uj, count);
    const vd pq = count == L ? p + q : select(iota() < count, p + q, splat(-1.0));
    const vl better = pq > top;
    top = select(better, pq, top);
    top_index = (better & (iota() + j)) | (~better & top_index);
  }
  const double best = hmax(top);
  int64_t pivot = c;
  for (int i = 0; i < L; ++i)
    if (top[i] == best) pivot = std::min<int64_t>(pivot, top_index[i]);
  st = {o.loss, temp, log1p_fast(o.s_sum), t_lognorm, 0.0, pivot == c ? 0 : pivot};
  return kFast;
}
}  // namespace distill64

// distillation_loss of rows with at most kMaxClasses classes, a block of rows at a time on
// lanes (a lane per row, a vector per class): the per-row kernels' arithmetic (distill32's in
// float with float64 sums for non-double logits, distill64's for float64 logits) without their
// per-row horizontal sums and library calls, which cost more than a small row's logits. Rows
// the per-row kernels take elsewhere (spreads, tails or shifts out of their range) are left
// to them (ok[r] = false).
namespace distill_small {
constexpr int64_t kMaxClasses = 32;

template <typename V>
struct Shape;
template <>
struct Shape<lanes::vf> {
  using T = float;
  using D = lanes::vd;
  static constexpr int L = lanes::L;
};
template <>
struct Shape<dlanes::vd> {
  using T = double;
  using D = dlanes::vd;
  static constexpr int L = dlanes::L;
};
// The integer mask vector of a vector type.
template <typename X>
using Mask = decltype(X{} < X{});

template <typename X>
inline X sel(Mask<X> m, X a, X b) {
  return (X)((m & (Mask<X>)a) | (~m & (Mask<X>)b));
}
template <typename To, typename From>
inline Mask<To> widen_mask(From m) {
  return __builtin_convertvector(m, Mask<To>);
}
// log1p of double lanes, dlanes::L at a time.
template <typename D>
inline D log1p_all(D x) {
  constexpr int L = sizeof(D) / sizeof(double);
  double lane[L];
  std::memcpy(lane, &x, sizeof x);
  for (int i = 0; i < L; i += dlanes::L) {
    dlanes::vd part;
    std::memcpy(&part, lane + i, sizeof part);
    part = dlanes::log1p(part);
    std::memcpy(lane + i, &part, sizeof part);
  }
  std::memcpy(&x, lane, sizeof x);
  return x;
}
inline lanes::vf exp_v(lanes::vf x) { return lanes::exp(x); }
inline dlanes::vd exp_v(dlanes::vd x) { return dlanes::exp(x); }
inline lanes::vf exp_or_expm1_v(lanes::vf x, lanes::vi m) { return lanes::exp_or_expm1(x, m); }
inline dlanes::vd exp_or_expm1_v(dlanes::vd x, dlanes::vl m) { return dlanes::exp_or_expm1(x, m); }
template <typename X>
inline X abs_v(X x) {
  return sel(x < 0, -x, x);
}

// The per-row sums of a block's forward pass (a lane per row).
template <typename V>
struct Sums {
  typename Shape<V>::D p_norm, s_tail, mean_delta, shift;
};

// The forward pass of rows [0, count) (count <= L) of student and teacher logits with c <=
// kMaxClasses classes: values[r] and ok[r], and exp(s), exp(t), delta and s by column in work
// (from 2 c L on). work holds 6 c L values.
template <typename V, typename scalar_t>
Sums<V> forward_pass(const scalar_t* zs, const scalar_t* zt, int64_t count, int64_t c, double temp,
                     double* values, bool* ok, typename Shape<V>::T* work) {
  using T = typename Shape<V>::T;
  using D = typename Shape<V>::D;
  using M = Mask<V>;
  using DM = Mask<D>;
  constexpr int L = Shape<V>::L;
  constexpr bool kFloat = std::is_same_v<T, float>;
  using I = std::conditional_t<kFloat, int32_t, int64_t>;
  T* col = work;  // col[k * L + r]: k < c the student, c <= k < 2c the teacher
  for (int r = 0; r < L; ++r) {
    const int64_t row = r < count ? r : count - 1;
    for (int64_t j = 0; j < c; ++j) {
      col[j * L + r] = static_cast<T>(zs[row * c + j]);
      col[(c + j) * L + r] = static_cast<T>(zt[row * c + j]);
    }
  }
  auto at = [&](int64_t k) {
    V v;
    std::memcpy(&v, col + k * L, sizeof v);
    return v;
  };
  const T inf = std::numeric_limits<T>::infinity();
  V s_max = V{} - inf, t_max = V{} - inf, s_min = V{} + inf, t_min = V{} + inf, bad{};
  for (int64_t j = 0; j < c; ++j) {
    const V a = at(j), b = at(c + j);
    s_max = sel(a > s_max, a, s_max), s_min = sel(a < s_min, a, s_min);
    t_max = sel(b > t_max, b, t_max), t_min = sel(b < t_min, b, t_min);
    bad = bad + (a * T(0)) + (b * T(0));
  }
  M usable = bad == T(0);
  const T inv_temp = T(1) / static_cast<T>(temp);
  if constexpr (kFloat) {
    const double half_range = std::numeric_limits<float>::max() / 2.0;
    const D sd = __builtin_convertvector(s_max, D) - __builtin_convertvector(s_min, D);
    const D td = __builtin_convertvector(t_max, D) - __builtin_convertvector(t_min, D);
    usable &= __builtin_convertvector((sd <= half_range) & (td <= half_range), M);
  } else {
    usable &= ((s_min - s_max) * inv_temp >= -708.0) & ((t_min - t_max) * inv_temp >= -708.0);
  }
  // The first maxima.
  M s_idx = M{} + static_cast<I>(c), t_idx = M{} + static_cast<I>(c);
  for (int64_t j = 0; j < c; ++j) {
    const I ji = static_cast<I>(j), ci = static_cast<I>(c);
    s_idx = sel<M>((at(j) == s_max) & (s_idx == ci), M{} + ji, s_idx);
    t_idx = sel<M>((at(c + j) == t_max) & (t_idx == ci), M{} + ji, t_idx);
  }
  T *es = work + 2 * c * L, *et = work + 3 * c * L, *dl = work + 4 * c * L, *sc = work + 5 * c * L;
  D s_tail{}, t_tail{}, cross{}, mean{};
  bool exact = false;
  if constexpr (kFloat) {
    exact = !any_lane((abs_v(s_max) > 1e4f) | (abs_v(s_min) > 1e4f) | (abs_v(t_max) > 1e4f) |
                      (abs_v(t_min) > 1e4f));
  }
  for (int64_t j = 0; j < c; ++j) {
    const V a = at(j), b = at(c + j);
    V delta;
    if constexpr (kFloat) {
      const D da = __builtin_convertvector(a, D), db = __builtin_convertvector(b, D);
      const D sm = __builtin_convertvector(s_max, D), tm = __builtin_convertvector(t_max, D);
      const D x = da - sm, y = db - tm;
      if (exact) {
        delta = __builtin_convertvector(x - y, V);
      } else {
        const D ex = (da - (x - (x - da))) + (-sm - (x - da));
        const D ey = (db - (y - (y - db))) + (-tm - (y - db));
        delta = __builtin_convertvector((x - y) + (ex - ey), V);
      }
    } else {
      const V x = a - s_max, y = b - t_max;
      const V ex = (a - (x - (x - a))) + (-s_max - (x - a));
      const V ey = (b - (y - (y - b))) + (-t_max - (y - b));
      delta = (x - y) + (ex - ey);
    }
    const V s = (a - s_max) * inv_temp, t = (b - t_max) * inv_temp;
    const V e_s = exp_v(s), e_t = exp_v(t);
    s_tail += __builtin_convertvector(sel(s_idx == static_cast<I>(j), V{}, e_s), D);
    t_tail += __builtin_convertvector(sel(t_idx == static_cast<I>(j), V{}, e_t), D);
    cross += __builtin_convertvector(sel(e_t > T(0), e_t * -s, V{}), D);
    mean += __builtin_convertvector(e_t * delta, D);
    std::memcpy(es + j * L, &e_s, sizeof e_s), std::memcpy(et + j * L, &e_t, sizeof e_t);
    std::memcpy(dl + j * L, &delta, sizeof delta), std::memcpy(sc + j * L, &s, sizeof s);
  }
  const D p_norm = 1.0 + t_tail, mean_delta = mean / p_norm;
  const D shift = -mean_delta / temp - log1p_all(t_tail);
  const D loss = log1p_all(s_tail) + cross / p_norm;
  DM fine = widen_mask<D>(usable) & (shift - shift == 0.0) & (mean_delta - mean_delta == 0.0);
  if constexpr (kFloat) fine &= (s_tail >= 1e-30) & (t_tail >= 1e-30);
  {
    int64_t flags[L];
    double lane[L];
    std::memcpy(flags, &fine, sizeof fine), std::memcpy(lane, &loss, sizeof loss);
    for (int r = 0; r < count; ++r) ok[r] = flags[r] != 0, values[r] = lane[r];
  }
  return {p_norm, s_tail, mean_delta, shift};
}

// The gradient times T, u - q sum(u), of a block's rows [0, count) into g (row-major, count x
// c) from its forward pass, as forward_gradient; ok[r] turns false where the row's correction
// overflows the work type (forward_gradient's kDouble or kGeneral).
template <typename V, typename G>
void gradient_pass(int64_t count, int64_t c, double temp, const Sums<V>& o, G* g, bool* ok,
                   typename Shape<V>::T* work) {
  using T = typename Shape<V>::T;
  using D = typename Shape<V>::D;
  using M = Mask<V>;
  constexpr int L = Shape<V>::L;
  const T inv_temp = T(1) / static_cast<T>(temp);
  const T *es = work + 2 * c * L, *et = work + 3 * c * L, *dl = work + 4 * c * L;
  const T* sc = work + 5 * c * L;
  const D mean_delta = o.mean_delta;
  const D inv_pd = 1.0 / o.p_norm, inv_qd = 1.0 / (1.0 + o.s_tail);
  const V inv_p = __builtin_convertvector(inv_pd, V), inv_q = __builtin_convertvector(inv_qd, V);
  const V shift_v = __builtin_convertvector(o.shift, V);
  T* em = work + c * L;  // over the student's columns, no longer read
  D total{};
  for (int64_t j = 0; j < c; ++j) {
    V p, e_t, delta, s;
    std::memcpy(&e_t, et + j * L, sizeof e_t), std::memcpy(&delta, dl + j * L, sizeof delta);
    std::memcpy(&s, sc + j * L, sizeof s);
    p = e_t * inv_p;
    const V dn = __builtin_convertvector(
        (__builtin_convertvector(delta, D) - mean_delta) * static_cast<double>(inv_temp), V);
    const M small = abs_v(dn) < T(1);
    const V e = exp_or_expm1_v(sel(small, dn, s + shift_v), small);
    total += __builtin_convertvector(sel(small, p * e, e - p), D);
    std::memcpy(em + j * L, &e, sizeof e);
  }
  const D log1p_corr = log1p_all(total), inv_cd = 1.0 / (1.0 + total);
  const V c_v = __builtin_convertvector(total, V), inv_c = __builtin_convertvector(inv_cd, V);
  {
    // exp_or_expm1_v clamps at 88 rather than overflow: in float32, |shift| > kShift needs
    // float64 (as distill32::forward). The loss alone does not read the shift.
    constexpr double bound =
        std::is_same_v<T, float> ? distill32::kShift : std::numeric_limits<double>::infinity();
    T corrections[L];
    double shifts[L];
    std::memcpy(corrections, &c_v, sizeof c_v), std::memcpy(shifts, &o.shift, sizeof o.shift);
    for (int r = 0; r < count; ++r)
      ok[r] = ok[r] && std::isfinite(corrections[r]) && std::abs(shifts[r]) <= bound;
  }
  D u_sum{};
  T* u_col = work;              // over the student's columns
  T* q_col = work + 2 * c * L;  // over exp(s), read before written
  for (int64_t j = 0; j < c; ++j) {
    V e_t, delta, e_s, e;
    std::memcpy(&e_t, et + j * L, sizeof e_t), std::memcpy(&delta, dl + j * L, sizeof delta);
    std::memcpy(&e_s, es + j * L, sizeof e_s), std::memcpy(&e, em + j * L, sizeof e);
    const V p = e_t * inv_p;
    const D dnd = (__builtin_convertvector(delta, D) - mean_delta) * static_cast<double>(inv_temp);
    const V dn = __builtin_convertvector(dnd, V);
    const M near = abs_v(dn) < T(1);
    V pu = p * ((e - c_v) * inv_c);
    const V ratio = __builtin_convertvector(dnd - log1p_corr, V);
    const M shifted = ~near & (abs_v(ratio) < T(1));
    if (any_lane(shifted)) pu = sel(shifted, p * exp_or_expm1_v(ratio, shifted), pu);
    const M small = near | shifted;
    const V q = sel(small, p + pu, e_s * inv_q);
    const V u = sel(small, pu, q - p);
    u_sum += __builtin_convertvector(u, D);
    std::memcpy(u_col + j * L, &u, sizeof u), std::memcpy(q_col + j * L, &q, sizeof q);
  }
  const V total_u = __builtin_convertvector(u_sum, V);
  for (int64_t j = 0; j < c; ++j) {
    V u, q;
    std::memcpy(&u, u_col + j * L, sizeof u), std::memcpy(&q, q_col + j * L, sizeof q);
    const V gradient = u - q * total_u;
    std::memcpy(u_col + j * L, &gradient, sizeof gradient);
  }
  for (int r = 0; r < count; ++r) {
    G* out = g + r * c;
    for (int64_t j = 0; j < c; ++j) out[j] = static_cast<G>(u_col[j * L + r]);
  }
}

// Rows [0, count) (count <= L) of student and teacher logits with c <= kMaxClasses classes:
// values[r] and ok[r], and with g (row-major, count x c) the gradient times T, u - q sum(u),
// as forward_gradient. work holds 6 c L values.
template <typename V, typename scalar_t, typename G>
void block(const scalar_t* zs, const scalar_t* zt, int64_t count, int64_t c, double temp,
           double* values, bool* ok, G* g, typename Shape<V>::T* work) {
  const Sums<V> sums = forward_pass<V>(zs, zt, count, c, temp, values, ok, work);
  if (g) gradient_pass<V>(count, c, temp, sums, g, ok, work);
}
}  // namespace distill_small
// A task's row buffers of the distillation kernels: float32 work (padded rows of cp), float64
// lanes and distillation_stats' rows.
struct DistillationBuffers {
  std::vector<float> narrow;
  std::vector<double> lanes64, wide;
};

// distillation_forward of a row by the first kernel that takes it (distill32 in float32 work,
// distill64, distillation_stats), with its statistics into row_stats unless null; false on
// nonfinite logits.
template <typename scalar_t, typename DoubleRow, typename Fallback>
bool distillation_forward_row(const scalar_t* s_row, const scalar_t* t_row, int64_t c,
                              double temperature, bool work32, double* row_stats, double& value,
                              DistillationBuffers& b, DoubleRow double_row, Fallback fallback) {
  constexpr bool kSum = std::is_same_v<scalar_t, double>;
  const bool need_stats = row_stats != nullptr;
  double scratch[distill32::kStats];
  double* stats = need_stats ? row_stats : scratch;
  if constexpr (!kSum) {
    if (work32) {
      // The forward reads float or 16-bit rows as they are and keeps no work rows.
      const distill32::RowOf<scalar_t> r{s_row, t_row, nullptr, nullptr, nullptr, c};
      const auto status =
          distill32::forward(r, static_cast<float>(temperature), need_stats, value, stats);
      if (status != distill32::kDouble) return status != distill32::kNonfinite;
    }
  }
  b.lanes64.resize((kSum ? 3 : 5) * c);
  double* w = b.lanes64.data();
  const distill64::Row r{double_row(s_row, c, w + (kSum ? 0 : 3 * c)),
                         double_row(t_row, c, w + (kSum ? 0 : 4 * c)),
                         w,
                         w + c,
                         w + 2 * c,
                         c};
  const auto status = distill64::forward<kSum>(r, temperature, need_stats, value, stats);
  if (status != distill64::kGeneral) return status != distill64::kNonfinite;
  return fallback(s_row, t_row, c, temperature, row_stats, value, b);
}

// Backward from seven saved doubles; scale before rounding to the storage dtype.
template <typename scalar_t, typename DoubleRow, typename Fallback>
void distillation_backward_row(const scalar_t* s_row, const scalar_t* t_row, int64_t c, int64_t cp,
                               double temperature, bool work32, const double* row_stats,
                               double upstream, scalar_t* g, std::vector<float>& narrow,
                               std::vector<double>& wide, DoubleRow double_row, Fallback fallback) {
  if constexpr (!std::is_same_v<scalar_t, double>) {
    if (work32 && (row_stats[0] == 1.0 || row_stats[0] == 1.5)) {
      distill32::backward(s_row, t_row, c, cp, static_cast<float>(temperature), row_stats, upstream,
                          g, narrow.data());
      return;
    }
  }
  if (row_stats[0] == distill64::kTag) {
    constexpr bool kSum = std::is_same_v<scalar_t, double>;
    wide.resize((kSum ? 2 : 5) * c);
    double* b = wide.data();
    const double* cs = double_row(s_row, c, b + 2 * c);
    const double* ct = double_row(t_row, c, b + 3 * c);
    double* out = kSum ? reinterpret_cast<double*>(g) : b + 4 * c;
    distill64::backward<kSum>(cs, ct, c, temperature, row_stats, upstream, out, b);
    if constexpr (!kSum) {
      for (int64_t j = 0; j < c; ++j) g[j] = static_cast<scalar_t>(out[j]);
    }
    return;
  }
  fallback(s_row, t_row, c, temperature, upstream, g, wide);
}

// The rows [begin, end) that distill_small takes a block at a time on V (lanes::vf: float32
// work, dlanes::vd: float64), done[row - begin], with their gradients times T into g unless null.
template <typename V, typename scalar_t, typename G>
SDCA_FLAT std::vector<char> distillation_blocks(const scalar_t* s_ptr, const scalar_t* t_ptr,
                                                int64_t begin, int64_t end, int64_t c,
                                                double temperature, double* values, G* g) {
  constexpr int B = distill_small::Shape<V>::L;
  std::vector<char> done(end - begin, 0);
  std::vector<typename distill_small::Shape<V>::T> work(6 * c * B);
  bool ok[B];
  for (int64_t row = begin; row < end; row += B) {
    const int64_t count = std::min<int64_t>(B, end - row);
    distill_small::block<V>(s_ptr + row * c, t_ptr + row * c, count, c, temperature, values + row,
                            ok, g ? g + row * c : nullptr, work.data());
    for (int64_t r = 0; r < count; ++r) done[row - begin + r] = ok[r];
  }
  return done;
}

// distillation_forward_grad of a row and its gradient times T into g, by the first kernel that
// takes it (as distillation_forward_row); false on nonfinite logits.
template <typename scalar_t, typename G, typename FloatRow, typename DoubleRow, typename Fallback>
bool distillation_grad_row(const scalar_t* s_row, const scalar_t* t_row, int64_t c, int64_t cp,
                           double temperature, double& value, G* g, DistillationBuffers& b,
                           FloatRow float_row, DoubleRow double_row, Fallback fallback) {
  constexpr bool kSum = std::is_same_v<scalar_t, double>;
  if constexpr (!kSum) {
    float* w = b.narrow.data();
    const distill32::Row r{
        float_row(s_row, c, w), float_row(t_row, c, w + cp), w + 2 * cp, w + 3 * cp, w + 4 * cp, c};
    const auto status =
        distill32::forward_gradient(r, static_cast<float>(temperature), value, g, w + 5 * cp, cp);
    if (status != distill32::kDouble) return status != distill32::kNonfinite;
  }
  b.lanes64.resize((kSum ? 5 : 8) * c);
  double* w = b.lanes64.data();
  const distill64::Row r{double_row(s_row, c, w + 5 * c),
                         double_row(t_row, c, w + (kSum ? 5 : 6) * c),
                         w,
                         w + c,
                         w + 2 * c,
                         c};
  double* out;
  if constexpr (kSum)
    out = g;
  else
    out = w + 7 * c;
  const auto status = distill64::forward_gradient<kSum>(r, temperature, value, out, w + 3 * c);
  if (status == distill64::kFast) {
    if constexpr (!kSum) {
      for (int64_t j = 0; j < c; ++j) g[j] = static_cast<G>(out[j]);
    }
  }
  if (status != distill64::kGeneral) return status != distill64::kNonfinite;
  return fallback(s_row, t_row, c, temperature, value, g, b);
}

}  // namespace sdca_cpu
}  // namespace
#endif
