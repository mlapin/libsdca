#ifndef SDCA_TORCH_DISTILL_H
#define SDCA_TORCH_DISTILL_H

#include "_cpu_prox_distill.h"

namespace {

void check_distillation_inputs(const at::Tensor& student, const at::Tensor& teacher) {
  TORCH_CHECK_VALUE(student.is_cpu() && student.dim() == 2 && student.is_contiguous() &&
                        teacher.sizes() == student.sizes() && teacher.is_contiguous() &&
                        teacher.scalar_type() == student.scalar_type() && teacher.is_cpu(),
                    "expected contiguous CPU (N, C) student and teacher logits of one dtype");
}

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
  constexpr bool kSum = std::is_same_v<scalar_t, double>;
  double* r = work;
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
  return prox_distillation_solve<G, kSum>(c, alpha, st, cs, ct, s, t, p_arr, q_arr, work, index,
                                          grad_out);
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

// API adapters for the shared statistics dispatchers.
template <typename scalar_t>
bool distillation_forward_row(const scalar_t* s, const scalar_t* t, int64_t c, double temperature,
                              bool work32, double* stats, double& value, DistillationBuffers& b) {
  return sdca_cpu::distillation_forward_row(
      s, t, c, temperature, work32, stats, value, b,
      [](const scalar_t* x, int64_t n, double* w) { return double_row(x, n, w); },
      [](const scalar_t* sr, const scalar_t* tr, int64_t n, double temp, double* st, double& v,
         DistillationBuffers& buffers) {
        buffers.wide.resize(6 * n);
        double* x = buffers.wide.data();
        v = distillation_stats(sr, tr, n, temp, x, x + n, x + 2 * n, x + 3 * n, x + 4 * n,
                               x + 5 * n)
                .loss;
        if (st) st[0] = -1;
        return !std::isnan(v);
      });
}

// API adapter for the shared row dispatcher and its rare general-precision fallback.
template <typename scalar_t, typename G>
bool distillation_grad_row(const scalar_t* s, const scalar_t* t, int64_t c, int64_t cp,
                           double temperature, double& value, G* g, DistillationBuffers& b) {
  return sdca_cpu::distillation_grad_row(
      s, t, c, cp, temperature, value, g, b,
      [](const scalar_t* x, int64_t n, float* w) { return float_row(x, n, w); },
      [](const scalar_t* x, int64_t n, double* w) { return double_row(x, n, w); },
      [](const scalar_t* sr, const scalar_t* tr, int64_t n, double temp, double& v, G* out,
         DistillationBuffers& buffers) {
        buffers.wide.resize(6 * n);
        double* x = buffers.wide.data();
        v = distillation_stats(sr, tr, n, temp, x, x + n, x + 2 * n, x + 3 * n, x + 4 * n,
                               x + 5 * n)
                .loss;
        if (std::isnan(v)) return false;
        distillation_gradient_double(sr, tr, n, temp, temp, x, out);
        return true;
      });
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
            sdca_cpu::distillation_backward_row(
                s_row, t_row, c, cp, temperature, work32, row_stats, up[row], g, narrow, wide,
                [](const scalar_t* x, int64_t count, double* w) { return double_row(x, count, w); },
                [](const scalar_t* sr, const scalar_t* tr, int64_t count, double temp, double u,
                   scalar_t* out, std::vector<double>& scratch) {
                  scratch.resize(6 * count);
                  distillation_gradient_double(sr, tr, count, temp, u, scratch.data(), out);
                });
          }
        });
      });
  return output;
}

}  // namespace

#endif  // SDCA_TORCH_DISTILL_H
