#ifndef SDCA_TORCH_DISTILL_H
#define SDCA_TORCH_DISTILL_H

namespace {

void check_distillation_inputs(const at::Tensor& student, const at::Tensor& teacher) {
  TORCH_CHECK_VALUE(student.is_cpu() && student.dim() == 2 && student.is_contiguous() &&
                        teacher.sizes() == student.sizes() && teacher.is_contiguous() &&
                        teacher.scalar_type() == student.scalar_type() && teacher.is_cpu(),
                    "expected contiguous CPU (N, C) student and teacher logits of one dtype");
}

struct DistillationStats {
  double loss, temp, s_lognorm, t_lognorm, u_sum;
  int64_t pivot;
};

// (x - y) - (u - v) with the rounding errors of both differences: a common student/teacher
// offset cancels in exact differences, and nearly equal logits keep their small difference.
inline double offset_difference(double x, double y, double u, double v) {
  double a = x, ea = 0, b = u, eb = 0;
  two_sum(a, ea, -y);
  two_sum(b, eb, -v);
  return (a - b) + (ea - eb);
}

// distillation_loss of one row in float64. Fills the centered scaled logits s, t, the
// probabilities q, p, log(q / p) in cs and q - p in ct; a NaN loss flags nonfinite logits.
template <typename scalar_t>
DistillationStats distillation_stats(const scalar_t* s_in, const scalar_t* t_in, int64_t c,
                                     double temperature, double* cs, double* ct, double* s,
                                     double* t, double* p_arr, double* q_arr) {
  constexpr bool kSum = std::is_same_v<scalar_t, double>;
  convert_to_double(s_in, cs, c);
  convert_to_double(t_in, ct, c);
  bool finite = true;
  double s_max = -std::numeric_limits<double>::infinity();
  double t_max = -std::numeric_limits<double>::infinity();
  for (int64_t j = 0; j < c; ++j) {
    finite &= std::isfinite(cs[j]) && std::isfinite(ct[j]);
    if (cs[j] > s_max) s_max = cs[j];
    if (ct[j] > t_max) t_max = ct[j];
  }
  if (!finite) return {std::numeric_limits<double>::quiet_NaN(), 0, 0, 0, 0, 0};
  const double temp = temperature;
  const double eps = std::numeric_limits<double>::epsilon();
  int64_t s_idx = 0, t_idx = 0;
  double s_best = -std::numeric_limits<double>::infinity();
  double t_best = -std::numeric_limits<double>::infinity();
  for (int64_t j = 0; j < c; ++j) {
    const double ds = cs[j] - s_max, dt = ct[j] - t_max;
    s[j] = temperature > 1.0 && !std::isfinite(ds) ? (cs[j] / temp) - (s_max / temp) : ds / temp;
    t[j] = temperature > 1.0 && !std::isfinite(dt) ? (ct[j] / temp) - (t_max / temp) : dt / temp;
    if (s[j] > s_best) s_best = s[j], s_idx = j;
    if (t[j] > t_best) t_best = t[j], t_idx = j;
  }
  double s_top = -std::numeric_limits<double>::infinity();
  double t_top = -std::numeric_limits<double>::infinity();
  for (int64_t j = 0; j < c; ++j) {
    if (j != s_idx && s[j] > s_top) s_top = s[j];
    if (j != t_idx && t[j] > t_top) t_top = t[j];
  }
  const double s_orig = std::isfinite(s_top) ? s_top : 0.0;
  const double t_orig = std::isfinite(t_top) ? t_top : 0.0;
  double s_scale = 1.0, t_scale = 1.0;
  Sum<kSum> s_tail_sum, t_tail_sum, es_sum, et_sum;
#ifdef SDCA_SIMD
  double min_s = std::numeric_limits<double>::infinity();
  double min_t = std::numeric_limits<double>::infinity();
  for (int64_t j = 0; j < c; ++j) {
    if (s[j] < min_s) min_s = s[j];
    if (t[j] < min_t) min_t = t[j];
  }
  if (min_s >= -708.0 && min_t >= -708.0 && c >= simd::W) {
    int64_t j = 0;
    VecSum<kSum> ves_acc, vet_acc;
    for (; j + simd::W <= c; j += simd::W) {
      const simd::vd vs = exp_pd_normal(simd::load(s + j));
      const simd::vd vt = exp_pd_normal(simd::load(t + j));
      simd::store(q_arr + j, vs);
      simd::store(p_arr + j, vt);
      ves_acc.add(Vec(vs));
      vet_acc.add(Vec(vt));
    }
    es_sum.add(ves_acc.value());
    et_sum.add(vet_acc.value());
    for (; j < c; ++j) {
      q_arr[j] = std::exp(s[j]);
      p_arr[j] = std::exp(t[j]);
      es_sum.add(q_arr[j]);
      et_sum.add(p_arr[j]);
    }
    // Sum the tails without the leading term: es_sum - 1 cancels.
    for (int64_t k = 0; k < c; ++k) {
      if (k != s_idx) s_tail_sum.add(q_arr[k]);
      if (k != t_idx) t_tail_sum.add(p_arr[k]);
    }
    s_scale = std::exp(-s_orig);
    t_scale = std::exp(-t_orig);
  } else
#else
  // The same on dlanes' portable double lanes (NEON on arm64).
  double min_s = std::numeric_limits<double>::infinity();
  double min_t = std::numeric_limits<double>::infinity();
  for (int64_t j = 0; j < c; ++j) {
    if (s[j] < min_s) min_s = s[j];
    if (t[j] < min_t) min_t = t[j];
  }
  if (min_s >= -708.0 && min_t >= -708.0 && c >= dlanes::L) {
    int64_t j = 0;
    dlanes::Accumulator<kSum> ves_acc, vet_acc;
    for (; j + dlanes::L <= c; j += dlanes::L) {
      const dlanes::vd vs = dlanes::exp(dlanes::load(s + j));
      const dlanes::vd vt = dlanes::exp(dlanes::load(t + j));
      dlanes::store(q_arr + j, vs);
      dlanes::store(p_arr + j, vt);
      ves_acc.add(vs);
      vet_acc.add(vt);
    }
    es_sum.add(ves_acc.value());
    et_sum.add(vet_acc.value());
    for (; j < c; ++j) {
      q_arr[j] = std::exp(s[j]);
      p_arr[j] = std::exp(t[j]);
      es_sum.add(q_arr[j]);
      et_sum.add(p_arr[j]);
    }
    for (int64_t k = 0; k < c; ++k) {
      if (k != s_idx) s_tail_sum.add(q_arr[k]);
      if (k != t_idx) t_tail_sum.add(p_arr[k]);
    }
    s_scale = std::exp(-s_orig);
    t_scale = std::exp(-t_orig);
  } else
#endif
  {
    for (int64_t j = 0; j < c; ++j) {
      if (j != s_idx) s_tail_sum.add(std::exp(s[j] - s_orig));
      if (j != t_idx) t_tail_sum.add(std::exp(t[j] - t_orig));
      p_arr[j] = std::exp(t[j]);
      q_arr[j] = std::exp(s[j]);
      et_sum.add(p_arr[j]);
      es_sum.add(q_arr[j]);
    }
  }
  const double s_half = std::exp(s_top * 0.5);
  const double t_half = std::exp(t_top * 0.5);
  const double s_tail = s_half * (s_half * (s_tail_sum.value() * s_scale));
  const double t_tail = t_half * (t_half * (t_tail_sum.value() * t_scale));
  const double s_lognorm = s_tail < eps * 0.5 ? s_tail : std::log1p(s_tail);
  const double t_lognorm = t_tail < eps * 0.5 ? t_tail : std::log1p(t_tail);

  const double p_norm = et_sum.value(), q_norm = es_sum.value();
  Sum<kSum> val;
  // Weighted means: their rounding cancels in the ratio, so they stay plain sums.
  double p_sum = 0.0, mean_delta_num = 0.0, rel_mean_num = 0.0;
  for (int64_t j = 0; j < c; ++j) {
    p_arr[j] /= p_norm;
    q_arr[j] /= q_norm;
    if (p_arr[j] > 0.0) {
      val.add(p_arr[j] * ((-s[j]) + s_lognorm));
      p_sum += p_arr[j];
      mean_delta_num += p_arr[j] * offset_difference(cs[j], ct[j], s_max, t_max);
      rel_mean_num += p_arr[j] * (s[j] - t[j]);
    }
  }
  const double mean_delta = mean_delta_num / p_sum;
  const bool over = temperature > 1.0 && !std::isfinite(mean_delta);
  const double rel_mean = over ? (rel_mean_num / p_sum) : 0.0;
  const double shift = over ? (-rel_mean - t_lognorm) : (-mean_delta / temp) - t_lognorm;

  // The exponentials of the two loops below, vectorized (Sleef) in passes of their own.
  thread_local std::vector<double> scratch;
  scratch.resize(2 * static_cast<std::size_t>(c));
  double* em1_of = scratch.data();
  double* exp_of = em1_of + c;
  for (int64_t j = 0; j < c; ++j) {
    const double delta_j = offset_difference(cs[j], ct[j], s_max, t_max);
    cs[j] = over ? ((s[j] - t[j]) - rel_mean) : ((delta_j - mean_delta) / temp);
  }
  at::vec::map([](Vec x) { return at::vec::clamp(x, Vec(-1.0), Vec(1.0)).expm1(); }, em1_of, cs, c);
  at::vec::map([shift](Vec x) { return (x + Vec(shift)).exp(); }, exp_of, s, c);
  Sum<kSum> correction;
  for (int64_t j = 0; j < c; ++j) {
    const double dn = cs[j];
    const double abs_dn = std::abs(dn);
    if (abs_dn < 1.0) {
      const double x = std::min(std::max(dn, -1.0), 1.0);
      const double em1 =
          abs_dn < 1e-4 ? x * (1.0 + x * (0.5 + x * (1.0 / 6.0 + x * (1.0 / 24.0)))) : em1_of[j];
      correction.add(p_arr[j] * em1);
    } else {
      correction.add(exp_of[j] - p_arr[j]);
    }
  }
  const double log1p_corr = std::log1p(correction.value());
  at::vec::map(
      [log1p_corr](Vec x) {
        return at::vec::clamp(x - Vec(log1p_corr), Vec(-1.0), Vec(1.0)).expm1();
      },
      em1_of, cs, c);
  Sum<kSum> u_sum;
  int64_t pivot = 0;
  double max_pq = -1.0;
  for (int64_t j = 0; j < c; ++j) {
    const double lr = cs[j] - log1p_corr;
    const double abs_lr = std::abs(lr);
    cs[j] = lr;
    double u;
    if (abs_lr < 1.0) {
      const double x = std::min(std::max(lr, -1.0), 1.0);
      const double em1 =
          abs_lr < 1e-4 ? x * (1.0 + x * (0.5 + x * (1.0 / 6.0 + x * (1.0 / 24.0)))) : em1_of[j];
      u = p_arr[j] * em1;
    } else {
      u = q_arr[j] - p_arr[j];
    }
    ct[j] = u;
    u_sum.add(u);
    if (p_arr[j] + q_arr[j] > max_pq) {
      max_pq = p_arr[j] + q_arr[j];
      pivot = j;
    }
  }
  return {val.value(), temp, s_lognorm, t_lognorm, u_sum.value(), pivot};
}

// The float64 distillation_loss gradient of a row times scale, rounded once into out:
// (q - p) / T, every entry formed on its own. buffer holds 6 C doubles.
template <typename scalar_t, typename O>
bool distillation_gradient_double(const scalar_t* s_in, const scalar_t* t_in, int64_t c,
                                  double temperature, double scale, double* buffer, O* out) {
  double *cs = buffer, *ct = cs + c, *q_arr = cs + 5 * c;
  const DistillationStats st = distillation_stats(s_in, t_in, c, temperature, cs, ct, cs + 2 * c,
                                                  cs + 3 * c, cs + 4 * c, q_arr);
  if (std::isnan(st.loss)) return false;
  for (int64_t j = 0; j < c; ++j) cs[j] = ct[j] - q_arr[j] * st.u_sum;
  // One rounding of the scaled entry, unless scale / T overflows (tiny temperatures).
  const double factor = scale / st.temp;
  for (int64_t j = 0; j < c; ++j)
    out[j] = static_cast<O>(std::isfinite(factor) ? cs[j] * factor : cs[j] / st.temp * scale);
  return true;
}

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
  extrema(r.cs, r.c, s_min, s_max, finite);
  extrema(r.ct, r.c, t_min, t_max, finite);
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

// proximal_distillation_loss of one row in double, as _tensor_prox.proximal_distillation:
// safeguarded Halley steps on sum(r(omega)) = 0 for the residuals r = p - pi, each evaluating
// the residuals on double lanes (proxlanes), near the root the Taylor model of the last one.
// Where alpha (max u - min u) <= 1 the steps start at log sum q exp(alpha (p - pi)) at the
// first-order pi = q exp(alpha (p - q) - omega0), omega0 the same sum at pi = q.
// work holds 7 C doubles and index C int32 slots; NaN flags nonfinite logits. The gradient
// goes to grad_out (float32 for 16-bit logits).
template <typename scalar_t, typename G>
double prox_distillation_row(const scalar_t* s_in, const scalar_t* t_in, int64_t c,
                             double temperature, double alpha, double* cs, double* ct, double* s,
                             double* t, double* p_arr, double* q_arr, double* work, int32_t* index,
                             G* grad_out) {
  using namespace dlanes;
  constexpr bool kSum = std::is_same_v<scalar_t, double>;
  double *r = work, *slope = work + c, *score = work + 2 * c, *pi_arr = work + 3 * c;
  // The weights 1 / (1 + alpha pi) of the last evaluation and the model's residuals and
  // probabilities (proxlanes::Model).
  double *weights = work + 4 * c, *next_r = work + 5 * c, *next_pi = work + 6 * c;
  bool clipped = false;
  // The row statistics on distill64's lanes, else (extreme rows) distillation_stats'; t then
  // holds log p.
  DistillationStats st;
  const distill64::Row rows{double_row(s_in, c, cs), double_row(t_in, c, ct), p_arr, s, t, c};
  const auto status = distill64::prox_stats<kSum>(rows, temperature, q_arr, r, cs, ct, st);
  if (status == distill64::kNonfinite) return std::numeric_limits<double>::quiet_NaN();
  if (status == distill64::kGeneral) {
    st = distillation_stats(s_in, t_in, c, temperature, cs, ct, s, t, p_arr, q_arr);
    if (std::isnan(st.loss)) return st.loss;
    for (int64_t j = 0; j < c; ++j) t[j] -= st.t_lognorm;
  }
  // The entries the model solves again.
  int32_t* fix = index;
  const double log_alpha = std::log(alpha);
  const double eps = std::numeric_limits<double>::epsilon();
  const double tiny = std::numeric_limits<double>::min();
  double u_max = 0.0, u_min = 0.0;
  for (int64_t j = 0; j < c; ++j) {
    u_max = std::max(u_max, ct[j]);
    u_min = std::min(u_min, ct[j]);
  }
  double lo = -alpha * u_max, hi = -alpha * u_min;
  if (!(lo <= hi)) hi = lo;
  // score: log q + alpha p (the Lambert inversion's h at omega = 0).
  const double shift_s = -st.s_lognorm;
  for (int64_t j = 0; j < c; j += L) {
    const int count = static_cast<int>(std::min<int64_t>(L, c - j));
    put(score + j, (part(s, j, count, 0.0) + shift_s) + alpha * part(p_arr, j, count, 0.0), count);
  }
  struct Sums {
    double f, magnitude, d1, d2;
  };
  using proxlanes::WD;
  constexpr int W = proxlanes::Shape<WD>::L;
  // The residuals r(omega), probabilities and slopes pi / den (den = 1 + alpha pi); the sums
  // of r, |r|, pi / den and -pi / den^3.
  auto evaluate = [&](double omega) -> Sums {
    // Every entry from its probability pi, r = p - pi: within 9 rounding errors of pi where
    // |r| >= p / 8, which |arg| >= (2 + alpha p) / 8 implies. Below (the Halley window), one
    // Newton step on alpha r - log1p(-r / p) = arg gives r to its relative accuracy: from p - pi
    // (error eps p, squared after the step: eps^2 p) where |arg| >= 2^-14, else from the
    // second-order estimate in arg (error |arg|^3 p, then |arg|^6 p).
    using proxlanes::part;
    using proxlanes::put;
    using proxlanes::select;
    using MW = proxlanes::MaskOf<WD>;
    clipped = false;
    for (int64_t j = 0; j < c; j += W) {
      const int count = static_cast<int>(std::min<int64_t>(W, c - j));
      // Padding lanes have a negligible probability.
      const WD h = part<WD>(score, j, count, -1e300), lr = part<WD>(cs, j, count, 0.0);
      const WD p = part<WD>(p_arr, j, count, 0.0), arg = omega - lr;
      const WD prob = proxlanes::prox_probability(h - omega, alpha, log_alpha);
      clipped |= any_lane(prob > 1.0);
      WD pi = proxlanes::min(prob, WD{} + 1.0), rj = p - pi;
      const MW window = (p > tiny) & (proxlanes::abs(arg) < (2.0 + alpha * p) * 0.125);
      if (any_lane(window)) {
        const WD q = select(window, p, WD{} + 1.0), a = select(window, arg, WD{});
        const WD linear = a / (alpha + 1.0 / q);
        const WD estimate = linear / (1.0 + (linear / q) / (2.0 * (1.0 + alpha * q)));
        const WD r0 = select(proxlanes::abs(a) < 0x1p-14, estimate, select(window, rj, WD{}));
        const WD g = (alpha * r0 - proxlanes::log1p(-r0 / q)) - a;
        const WD next = r0 - g / (alpha + 1.0 / (q - r0));
        rj = select(window, next, rj), pi = select(window, p - next, pi);
      }
      // A zero argument has a zero residual, also where p is subnormal.
      rj = select(arg == 0.0, WD{}, rj);
      const WD u = 1.0 / (1.0 + alpha * pi);
      put(r + j, rj, count), put(pi_arr + j, pi, count), put(weights + j, u, count);
      put(slope + j, pi * u, count);
    }
    Accumulator<kSum> sum_f;
    WD magnitude{}, d1{}, d2{};
    for (int64_t j = 0; j < c; j += W) {
      const int count = static_cast<int>(std::min<int64_t>(W, c - j));
      const WD rj = part<WD>(r, j, count, 0.0), slope_j = part<WD>(slope, j, count, 0.0);
      const WD u = part<WD>(weights, j, count, 0.0);
      for (int k = 0; k < proxlanes::kDoubleChains; ++k) sum_f.add(rj.v[k]);
      magnitude += proxlanes::abs(rj);
      d1 += slope_j;
      d2 -= (slope_j * u) * u;
    }
    return {sum_f.value(), proxlanes::total<double>(magnitude), proxlanes::total<double>(d1),
            proxlanes::total<double>(d2)};
  };
  double omega = 0.0;
  // Near the root, the Taylor model about the last evaluation at omega (residual sum f) gives
  // the root and its residuals without another evaluation: true with omega, r and pi_arr
  // there; else false with its step in delta (where finite). Entries whose residual it would
  // not keep to its relative accuracy are solved again (exactly) at the root.
  using Model = proxlanes::Model<double>;
  auto model_root = [&](double f, double& delta) -> bool {
    Model model;
    model.add_probabilities(pi_arr, weights, c);
    double d = delta;
    for (int it = 0; it < 8; ++it) {
      model.at(d);
      const double step = (f - model.change()) / model.slope();
      if (!std::isfinite(step)) return false;
      d -= step;
      if (std::abs(step) <= 1e-3 * eps * std::abs(d)) break;
    }
    for (int attempt = 0; attempt < 3; ++attempt) {
      const double next = omega + d;
      if (!(lo <= next && next <= hi)) return false;
      delta = next - omega;
      if (!(std::abs(delta) <= Model::reach)) return false;
      d = delta;
      model.at(d);
      const double bound = std::pow(std::abs(d) / Model::reach, proxlanes::kModelOrder + 1);
      Accumulator<kSum> total;
      double size = 0.0;
      const int64_t fixes = model.expand_probabilities(r, pi_arr, weights, c, bound, next_r,
                                                       next_pi, total, size, fix);
      double sum = total.value();
      if (fixes > 0) {
        for (int64_t k = 0; k < fixes; ++k) {
          const int64_t j = fix[k];
          const double p = p_arr[j], arg = next - cs[j], half = alpha * p / 2;
          double rj;
          if (p > tiny && arg <= half + 0.69314718055994530942 &&
              arg >= -half - 0.40546510810816438198) {
            rj = prox_residual(arg, alpha, p, log_alpha, t[j], next_r[j]);
            next_pi[j] = p - rj;
          } else {
            next_pi[j] = std::min(prox_probability(score[j] - next, alpha, log_alpha), 1.0);
            rj = arg == 0.0 ? 0.0 : p - next_pi[j];
          }
          size += std::abs(rj) - std::abs(next_r[j]);
          next_r[j] = rj;
        }
        Accumulator<kSum> again;
        for (int64_t j = 0; j < c; j += L)
          again.add(part(next_r, j, static_cast<int>(std::min<int64_t>(L, c - j)), 0.0));
        sum = again.value();
      }
      if (std::abs(sum) <= 8.0 * eps * std::max(size, tiny)) {
        std::copy(next_r, next_r + c, r);
        std::copy(next_pi, next_pi + c, pi_arr);
        omega = next;
        return true;
      }
      d -= sum / model.slope();
    }
    delta = d;
    return false;
  };
  if (hi - lo <= 1.0) {
    Accumulator<false> first, second;
    for (int64_t j = 0; j < c; j += L) {
      const int count = static_cast<int>(std::min<int64_t>(L, c - j));
      const vd q = part(q_arr, j, count, 0.0);
      first.add(q * exp(-alpha * part(ct, j, count, 0.0)));
    }
    const double omega0 = std::log(first.value());
    for (int64_t j = 0; j < c; j += L) {
      const int count = static_cast<int>(std::min<int64_t>(L, c - j));
      const vd q = part(q_arr, j, count, 0.0), p = part(p_arr, j, count, 0.0);
      const vd pi = q * exp(alpha * (p - q) - omega0);
      second.add(q * exp(alpha * (p - pi)));
    }
    const double start = std::log(second.value());
    if (lo <= start && start <= hi) omega = start;
  }
  bool solved = lo == 0.0 && hi == 0.0;
  if (solved) evaluate(omega);
  for (int it = 0; it < 100 && !solved; ++it) {
    const Sums e = evaluate(omega);
    solved = true;
    const double f = e.f;
    if (std::abs(f) <= 8.0 * eps * std::max(e.magnitude, tiny)) {
      // Within the tolerance: the residuals at the Newton root omega - f / d1, to first order.
      const double delta = -f / e.d1;
      if (delta != 0.0 && std::isfinite(delta)) {
        for (int64_t j = 0; j < c; j += L) {
          const int count = static_cast<int>(std::min<int64_t>(L, c - j));
          const vd moved_j = delta * part(slope, j, count, 0.0);
          put(r + j, part(r, j, count, 0.0) + moved_j, count);
          put(pi_arr + j, part(pi_arr, j, count, 0.0) - moved_j, count);
        }
      }
      break;
    }
    (f < 0.0 ? lo : hi) = omega;
    const double newton = -f / e.d1, corr = (f * e.d2) / (2.0 * e.d1 * e.d1);
    double delta =
        std::isfinite(corr) ? newton / (1.0 - std::min(std::max(corr, -0.5), 0.5)) : newton;
    if (!(e.d1 > 0.0 && std::isfinite(e.d1) && std::isfinite(newton))) delta = 0.0;
    if (!clipped && delta != 0.0 && std::abs(delta) <= 2 * Model::reach) {
      double root = delta;
      if (model_root(f, root)) break;
      if (std::isfinite(root) && root != 0.0) delta = root;
    }
    double next = omega + delta;
    if (delta != 0.0 && next == omega) break;
    if (!(lo <= next && next <= hi && next != omega && std::isfinite(next)))
      next = lo + (hi - lo) / 2;
    if (next == omega) break;
    omega = next;
    solved = false;
  }
  // The residuals at omega (the last evaluation's, unless the steps ran out).
  if (!solved) evaluate(omega);
  Accumulator<kSum> other_sum;
  auto add_others = [&](int64_t begin, int64_t end) {
    for (int64_t j = begin; j < end; j += L)
      other_sum.add(part(r, j, static_cast<int>(std::min<int64_t>(L, end - j)), 0.0));
  };
  add_others(0, st.pivot);
  add_others(st.pivot + 1, c);
  r[st.pivot] = -other_sum.value();
  Accumulator<kSum> cross, penalty;
  for (int64_t j = 0; j < c; j += W) {
    const int count = static_cast<int>(std::min<int64_t>(W, c - j));
    using proxlanes::select;
    using MW = proxlanes::MaskOf<WD>;
    const WD p = proxlanes::part<WD>(p_arr, j, count, 0.0);
    const WD rj = proxlanes::part<WD>(r, j, count, 0.0), zero{};
    const MW positive = proxlanes::valid<WD>(count) & (p > 0.0);
    // log pi: log p + log1p(-r / p) where |r| <= p / 2 (pi near p, whose logarithm would lose
    // the small -r / p), else the logarithm of the solved probability (accurate where r is
    // not small: a Lambert inversion's, or p - r), below double's normal range h - W.
    const WD pi = proxlanes::part<WD>(pi_arr, j, count, 1.0);
    const MW near = positive & ~(rj > p / 2.0) & ~(rj < -p / 2.0);
    const MW tiny_pi = positive & ~near & ~(pi >= std::numeric_limits<double>::min());
    const MW logarithm = positive & ~near & ~tiny_pi;
    WD log_pi = zero;
    if (any_lane(logarithm)) log_pi = proxlanes::log(select(logarithm, pi, zero + 1.0));
    if (any_lane(near))
      log_pi = select(
          near,
          proxlanes::part<WD>(t, j, count, 0.0) + proxlanes::log1p(select(near, -rj / p, zero)),
          log_pi);
    if (any_lane(tiny_pi)) {
      const WD h = proxlanes::part<WD>(score, j, count, 0.0) - omega;
      const WD w = proxlanes::lambert_w_exp(select(tiny_pi, h + log_alpha, zero));
      log_pi = select(tiny_pi, h - w, log_pi);
    }
    const WD terms = select(positive, -(p * log_pi), zero), squares = (alpha * rj) * rj;
    for (int k = 0; k < proxlanes::kDoubleChains; ++k)
      cross.add(terms.v[k]), penalty.add(squares.v[k]);
  }
  for (int64_t j = 0; j < c; ++j) grad_out[j] = static_cast<G>(-r[j] / st.temp);
  return cross.value() + penalty.value() / 2;
}

// proximal_distillation_loss per row in float64: (values (N,), gradient (N, C)); values
// float32 (float64 for float64 logits), the gradient in gradient_type.
std::tuple<at::Tensor, at::Tensor> prox_distillation_forward(const at::Tensor& student,
                                                             const at::Tensor& teacher,
                                                             double temperature, double alpha) {
  check_distillation_inputs(student, teacher);
  const int64_t n = student.size(0), c = student.size(1);
  auto grad_t = at::empty(student.sizes(), student.options().dtype(gradient_type(student)));
  std::vector<double> values(n);
  std::atomic<bool> nonfinite{false};
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, student.scalar_type(), "sdca_prox_distillation_forward", [&] {
        using G = std::conditional_t<std::is_same_v<scalar_t, double>, double, float>;
        const scalar_t* s_ptr = student.data_ptr<scalar_t>();
        const scalar_t* t_ptr = teacher.data_ptr<scalar_t>();
        G* g_ptr = grad_t.data_ptr<G>();
        const int64_t grain = prox_grain(c);
        at::parallel_for(0, n, grain, [&](int64_t begin, int64_t end) {
          std::vector<double> buf(13 * c);
          std::vector<int32_t> index(c);
          double* b = buf.data();
          for (int64_t row = begin; row < end; ++row) {
            values[row] = prox_distillation_row(
                s_ptr + row * c, t_ptr + row * c, c, temperature, alpha, b, b + c, b + 2 * c,
                b + 3 * c, b + 4 * c, b + 5 * c, b + 6 * c, index.data(), g_ptr + row * c);
            if (std::isnan(values[row])) nonfinite = true;
          }
        });
      });
  TORCH_CHECK_VALUE(!nonfinite, "Logits must be finite floating-point tensors");
  return {at::tensor(values, student.options().dtype(at::kDouble)).to(value_type(student)), grad_t};
}

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

// Rows per task of the distillation kernels: about 4096 logits, and at most 32 rows (small
// rows cost more than their logits). serial: one task below 4096 logits, where an OpenMP
// fork costs more than the rows (the forward without statistics and the gradient forward;
// the statistics kernels cost several times more per logit).
inline int64_t distillation_grain(int64_t n, int64_t c, bool serial = false) {
  return serial && n * c < 4096 ? n + 1 : std::max<int64_t>(1, std::min<int64_t>(32, 4096 / c));
}

// Whether distillation_loss works in float32: non-double logits without precision="float64",
// at temperatures float32 holds.
inline bool distillation_float(const at::Tensor& student, double temperature, bool is_double) {
  const double tiny = std::numeric_limits<float>::min();
  return !is_double && student.scalar_type() != at::kDouble && tiny <= temperature &&
         temperature <= 1 / tiny;
}

// A task's row buffers of the distillation kernels: float32 work (padded rows of cp), float64
// lanes and distillation_stats' rows.
struct DistillationBuffers {
  std::vector<float> narrow;
  std::vector<double> lanes64, wide;
};

// distillation_forward of a row by the first kernel that takes it (distill32 in float32 work,
// distill64, distillation_stats), with its statistics into row_stats unless null; false on
// nonfinite logits.
template <typename scalar_t>
bool distillation_forward_row(const scalar_t* s_row, const scalar_t* t_row, int64_t c,
                              double temperature, bool work32, double* row_stats, double& value,
                              DistillationBuffers& b) {
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
  const distill64::Row r{
      double_row(s_row, c, w + 3 * c), double_row(t_row, c, w + 4 * c), w, w + c, w + 2 * c, c};
  const auto status = distill64::forward<kSum>(r, temperature, need_stats, value, stats);
  if (status != distill64::kGeneral) return status != distill64::kNonfinite;
  b.wide.resize(6 * c);
  double* x = b.wide.data();
  value = distillation_stats(s_row, t_row, c, temperature, x, x + c, x + 2 * c, x + 3 * c,
                             x + 4 * c, x + 5 * c)
              .loss;
  if (row_stats) row_stats[0] = -1;
  return !std::isnan(value);
}

// The rows [begin, end) that distill_small takes a block at a time on V (lanes::vf: float32
// work, dlanes::vd: float64), done[row - begin], with their gradients times T into g unless null.
template <typename V, typename scalar_t, typename G>
std::vector<char> distillation_blocks(const scalar_t* s_ptr, const scalar_t* t_ptr, int64_t begin,
                                      int64_t end, int64_t c, double temperature, double* values,
                                      G* g) {
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
template <typename scalar_t, typename G>
bool distillation_grad_row(const scalar_t* s_row, const scalar_t* t_row, int64_t c, int64_t cp,
                           double temperature, double& value, G* g, DistillationBuffers& b) {
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
  b.wide.resize(6 * c);
  double* x = b.wide.data();
  value = distillation_stats(s_row, t_row, c, temperature, x, x + c, x + 2 * c, x + 3 * c,
                             x + 4 * c, x + 5 * c)
              .loss;
  if (std::isnan(value)) return false;
  distillation_gradient_double(s_row, t_row, c, temperature, temperature, x, g);
  return true;
}

// distillation_loss per row: (values, stats). values (N,) float32 (float64 for float64
// logits); stats float64 (N, distill32::kStats) when need_stats, else undefined. is_double selects
// float64 work for any logits (precision="float64").
std::tuple<at::Tensor, at::Tensor> distillation_forward(const at::Tensor& student,
                                                        const at::Tensor& teacher,
                                                        double temperature, bool is_double,
                                                        bool need_stats) {
  check_distillation_inputs(student, teacher);
  const int64_t n = student.size(0), c = student.size(1);
  const bool work32 = distillation_float(student, temperature, is_double);
  auto stats = need_stats ? at::empty({n, distill32::kStats}, student.options().dtype(at::kDouble))
                          : at::Tensor();
  double* st = need_stats ? stats.data_ptr<double>() : nullptr;
  std::vector<double> values(n);
  std::atomic<bool> nonfinite{false};
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, student.scalar_type(), "sdca_distillation_forward", [&] {
        const scalar_t* s_ptr = student.data_ptr<scalar_t>();
        const scalar_t* t_ptr = teacher.data_ptr<scalar_t>();
        const int64_t task = distillation_grain(n, c, !need_stats);
        at::parallel_for(0, n, task, [&](int64_t begin, int64_t end) {
          DistillationBuffers buffers;
          // Small rows without statistics a block at a time; rows the block leaves take the
          // per-row kernels.
          const bool small = !need_stats && c <= distill_small::kMaxClasses;
          std::vector<char> done;
          if (small)
            done =
                work32
                    ? distillation_blocks<lanes::vf>(s_ptr, t_ptr, begin, end, c, temperature,
                                                     values.data(), static_cast<float*>(nullptr))
                    : distillation_blocks<dlanes::vd>(s_ptr, t_ptr, begin, end, c, temperature,
                                                      values.data(), static_cast<double*>(nullptr));
          for (int64_t row = begin; row < end; ++row) {
            if (small && done[row - begin]) continue;
            double* row_stats = st ? st + row * distill32::kStats : nullptr;
            if (!distillation_forward_row(s_ptr + row * c, t_ptr + row * c, c, temperature, work32,
                                          row_stats, values[row], buffers))
              nonfinite = true;
          }
        });
      });
  TORCH_CHECK_VALUE(!nonfinite, "Logits must be finite floating-point tensors");
  const bool s_is_double = student.scalar_type() == at::kDouble;
  return {at::tensor(values, student.options().dtype(at::kDouble))
              .to(s_is_double ? at::kDouble : at::kFloat),
          stats};
}

// distillation_loss per row and its gradient times T, (u - q sum(u)), unscaled: (values (N,),
// gradient (N, C)) of float32 logits in float32 work or of float64 logits, the gradient in
// their dtype. The backward multiplies the saved gradient by upstream / T (one rounding).
// A float32 row that needs float64 rounds its float64 gradient once into float32.
std::tuple<at::Tensor, at::Tensor> distillation_forward_grad(const at::Tensor& student,
                                                             const at::Tensor& teacher,
                                                             double temperature) {
  check_distillation_inputs(student, teacher);
  const int64_t n = student.size(0), c = student.size(1);
  const bool is_double = student.scalar_type() == at::kDouble;
  TORCH_CHECK_VALUE(is_double || (student.scalar_type() == at::kFloat &&
                                  distillation_float(student, temperature, false)),
                    "distillation_forward_grad: float32 logits in float32 work or float64 logits");
  auto gradient = at::empty_like(student, at::MemoryFormat::Contiguous);
  std::vector<double> values(n);
  std::atomic<bool> nonfinite{false};
  AT_DISPATCH_FLOATING_TYPES(student.scalar_type(), "sdca_distillation_forward_grad", [&] {
    using G = scalar_t;
    const scalar_t* s_ptr = student.data_ptr<scalar_t>();
    const scalar_t* t_ptr = teacher.data_ptr<scalar_t>();
    G* g_ptr = gradient.data_ptr<G>();
    const int64_t cp = (c + lanes::L - 1) / lanes::L * lanes::L;
    constexpr bool kSum = std::is_same_v<scalar_t, double>;
    const int64_t task = distillation_grain(n, c, true);
    at::parallel_for(0, n, task, [&](int64_t begin, int64_t end) {
      DistillationBuffers buffers{std::vector<float>(kSum ? 0 : 10 * cp), {}, {}};
      // Small rows a block at a time; rows the block leaves take the per-row kernels.
      using V = std::conditional_t<kSum, dlanes::vd, lanes::vf>;
      std::vector<char> done;
      if (c <= distill_small::kMaxClasses)
        done =
            distillation_blocks<V>(s_ptr, t_ptr, begin, end, c, temperature, values.data(), g_ptr);
      for (int64_t row = begin; row < end; ++row) {
        if (!done.empty() && done[row - begin]) continue;
        if (!distillation_grad_row(s_ptr + row * c, t_ptr + row * c, c, cp, temperature,
                                   values[row], g_ptr + row * c, buffers))
          nonfinite = true;
      }
    });
  });
  TORCH_CHECK_VALUE(!nonfinite, "Logits must be finite floating-point tensors");
  return {at::tensor(values, student.options().dtype(at::kDouble)), gradient};
}

// The distillation_loss gradient times upstream ((N,), any float dtype), rounded once into the
// student's dtype, from distillation_forward's stats of the same arguments.
at::Tensor distillation_backward(const at::Tensor& student, const at::Tensor& teacher,
                                 const at::Tensor& stats, const at::Tensor& upstream,
                                 double temperature, bool is_double) {
  check_distillation_inputs(student, teacher);
  const int64_t n = student.size(0), c = student.size(1);
  TORCH_CHECK_VALUE(stats.scalar_type() == at::kDouble && stats.is_contiguous() &&
                        stats.numel() == n * distill32::kStats && upstream.numel() == n,
                    "invalid distillation statistics or upstream gradient");
  const bool work32 = distillation_float(student, temperature, is_double);
  const at::Tensor u = upstream.to(at::kDouble).contiguous();
  const double* up = u.data_ptr<double>();
  const double* st = stats.data_ptr<double>();
  auto output = gradient_like(student);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, student.scalar_type(), "sdca_distillation_backward", [&] {
        const scalar_t* s_ptr = student.data_ptr<scalar_t>();
        const scalar_t* t_ptr = teacher.data_ptr<scalar_t>();
        scalar_t* g_ptr = output.data_ptr<scalar_t>();
        at::parallel_for(0, n, distillation_grain(n, c), [&](int64_t begin, int64_t end) {
          // Padded work rows: their lane loads stay inside the buffer.
          const int64_t cp = (c + lanes::L - 1) / lanes::L * lanes::L;
          std::vector<float> narrow(work32 ? 6 * cp : 0);
          std::vector<double> wide;
          for (int64_t row = begin; row < end; ++row) {
            const scalar_t *s_row = s_ptr + row * c, *t_row = t_ptr + row * c;
            scalar_t* g = g_ptr + row * c;
            const double* row_stats = st + row * distill32::kStats;
            if constexpr (!std::is_same_v<scalar_t, double>) {
              if (work32 && (row_stats[0] == 1.0 || row_stats[0] == 1.5)) {
                distill32::backward(s_row, t_row, c, cp, static_cast<float>(temperature), row_stats,
                                    up[row], g, narrow.data());
                continue;
              }
            }
            if (row_stats[0] == distill64::kTag) {
              constexpr bool kSum = std::is_same_v<scalar_t, double>;
              wide.resize((kSum ? 2 : 5) * c);
              double* b = wide.data();
              const double* cs = double_row(s_row, c, b + 2 * c);
              const double* ct = double_row(t_row, c, b + 3 * c);
              double* out = kSum ? reinterpret_cast<double*>(g) : b + 4 * c;
              distill64::backward<kSum>(cs, ct, c, temperature, row_stats, up[row], out, b);
              if constexpr (!kSum) {
                for (int64_t j = 0; j < c; ++j) g[j] = static_cast<scalar_t>(out[j]);
              }
              continue;
            }
            wide.resize(6 * c);
            distillation_gradient_double(s_row, t_row, c, temperature, up[row], wide.data(), g);
          }
        });
      });
  return output;
}

}  // namespace

#endif  // SDCA_TORCH_DISTILL_H
