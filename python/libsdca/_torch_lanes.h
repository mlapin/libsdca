#ifndef SDCA_TORCH_LANES_H
#define SDCA_TORCH_LANES_H

#include "_cpu_lanes.h"

namespace {
namespace sdca_cpu {
namespace lanes {
template <>
struct Storage<at::BFloat16> : BFloatStorage<at::BFloat16> {};
template <>
struct Storage<at::Half> : HalfStorage<at::Half> {};
}  // namespace lanes
}  // namespace sdca_cpu
}  // namespace

namespace {
using namespace sdca_cpu;
using Vec = at::vec::Vectorized<double>;

template <bool Compensated>
struct VecSum {
  Vec hi = Vec(0.0), lo = Vec(0.0), block = Vec(0.0);
  int count = 0;
  void add(const Vec& x) {
    if constexpr (Compensated) {
      block = block + x;
      if (++count == kBlock) {
        two_sum(hi, lo, block);
        block = Vec(0.0);
        count = 0;
      }
    } else {
      hi = hi + x;
    }
  }
  double value() const {
    if constexpr (!Compensated) {
      return at::vec::vec_reduce_all([](Vec a, Vec b) { return a + b; }, hi);
    } else {
      Vec h = hi, l = lo;
      two_sum(h, l, block);
      double hs[Vec::size()], ls[Vec::size()];
      h.store(hs);
      l.store(ls);
      Sum<true> total;
      for (int i = 0; i < Vec::size(); ++i) two_sum(total.hi, total.lo, hs[i]);
      for (int i = 0; i < Vec::size(); ++i) total.lo += ls[i];
      return total.value();
    }
  }
};

#ifdef SDCA_SIMD
// The double vector of the build and its operations.
namespace simd {
#ifdef SDCA_AVX512
using vd = __m512d;
constexpr int W = 8;
inline vd load(const double* p) { return _mm512_loadu_pd(p); }
inline void store(double* p, vd v) { _mm512_storeu_pd(p, v); }
inline vd set1(double x) { return _mm512_set1_pd(x); }
inline vd zero() { return _mm512_setzero_pd(); }
inline vd add(vd a, vd b) { return _mm512_add_pd(a, b); }
inline vd sub(vd a, vd b) { return _mm512_sub_pd(a, b); }
inline vd mul(vd a, vd b) { return _mm512_mul_pd(a, b); }
inline vd max(vd a, vd b) { return _mm512_max_pd(a, b); }
inline vd min(vd a, vd b) { return _mm512_min_pd(a, b); }
inline vd fmadd(vd a, vd b, vd c) { return _mm512_fmadd_pd(a, b, c); }
inline vd fnmadd(vd a, vd b, vd c) { return _mm512_fnmadd_pd(a, b, c); }
inline vd round(vd x) {
  return _mm512_roundscale_pd(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}
// 2^n of integral n in the normal range.
inline vd pow2(vd n) {
  const __m512i bits = _mm512_add_epi64(_mm512_cvtpd_epi64(n), _mm512_set1_epi64(1023));
  return _mm512_castsi512_pd(_mm512_slli_epi64(bits, 52));
}
inline vd widen(__m256 f) { return _mm512_cvtps_pd(f); }
inline vd load_double(const float* p) { return widen(_mm256_loadu_ps(p)); }
inline vd load_double(const at::Half* p) {
  return widen(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))));
}
inline vd load_double(const at::BFloat16* p) {
  const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  return widen(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16)));
}
inline void store_float(float* p, vd v) { _mm256_storeu_ps(p, _mm512_cvtpd_ps(v)); }
// 2 W floats of two double vectors, for the 16-bit conversions of ATen.
inline __m512 narrow2(vd a, vd b) {
  return _mm512_insertf32x8(_mm512_castps256_ps512(_mm512_cvtpd_ps(a)), _mm512_cvtpd_ps(b), 1);
}
// The two-vector conversions (PyTorch 2.1 has no one-vector overload; the same rounding), the
// first vector's half.
inline void store2(at::Half* p, vd a, vd b) {
  const __m512 f = narrow2(a, b);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(p),
                      _mm512_castsi512_si256(at::vec::cvtfp32_fp16(f, f)));
}
inline void store2(at::BFloat16* p, vd a, vd b) {
  const __m512 f = narrow2(a, b);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(p),
                      _mm512_castsi512_si256(at::vec::cvtfp32_bf16(f, f)));
}
#else
using vd = __m256d;
constexpr int W = 4;
inline vd load(const double* p) { return _mm256_loadu_pd(p); }
inline void store(double* p, vd v) { _mm256_storeu_pd(p, v); }
inline vd set1(double x) { return _mm256_set1_pd(x); }
inline vd zero() { return _mm256_setzero_pd(); }
inline vd add(vd a, vd b) { return _mm256_add_pd(a, b); }
inline vd sub(vd a, vd b) { return _mm256_sub_pd(a, b); }
inline vd mul(vd a, vd b) { return _mm256_mul_pd(a, b); }
inline vd max(vd a, vd b) { return _mm256_max_pd(a, b); }
inline vd min(vd a, vd b) { return _mm256_min_pd(a, b); }
inline vd fmadd(vd a, vd b, vd c) { return _mm256_fmadd_pd(a, b, c); }
inline vd fnmadd(vd a, vd b, vd c) { return _mm256_fnmadd_pd(a, b, c); }
inline vd round(vd x) { return _mm256_round_pd(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
inline vd pow2(vd n) {
  const __m256i ni64 = _mm256_cvtepi32_epi64(_mm256_cvtpd_epi32(n));
  return _mm256_castsi256_pd(
      _mm256_slli_epi64(_mm256_add_epi64(ni64, _mm256_set1_epi64x(1023)), 52));
}
inline vd widen(__m128 f) { return _mm256_cvtps_pd(f); }
inline vd load_double(const float* p) { return widen(_mm_loadu_ps(p)); }
inline vd load_double(const at::Half* p) {
  return widen(_mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p))));
}
inline vd load_double(const at::BFloat16* p) {
  const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
  return widen(_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(raw), 16)));
}
inline void store_float(float* p, vd v) { _mm_storeu_ps(p, _mm256_cvtpd_ps(v)); }
inline __m256 narrow2(vd a, vd b) {
  return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(a)), _mm256_cvtpd_ps(b), 1);
}
// The two-vector conversions (PyTorch 2.1 has no one-vector overload; the same rounding), the
// first vector's half.
inline void store2(at::Half* p, vd a, vd b) {
  const __m256 f = narrow2(a, b);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(p),
                   _mm256_castsi256_si128(at::vec::cvtfp32_fp16(f, f)));
}
inline void store2(at::BFloat16* p, vd a, vd b) {
  const __m256 f = narrow2(a, b);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(p),
                   _mm256_castsi256_si128(at::vec::cvtfp32_bf16(f, f)));
}
#endif
inline vd load_double(const double* p) { return load(p); }
}  // namespace simd

// Inlined double exp for x in [-708, 0] (and -inf -> 0); extreme rows use Sleef.
static inline simd::vd exp_pd_normal(simd::vd x) {
  using namespace simd;
  x = max(x, set1(-709.08956571282405));
  const vd n = round(mul(x, set1(1.44269504088896340736)));
  vd r = fnmadd(n, set1(0.69314718055966295651), x);
  r = fnmadd(n, set1(2.82352905630315771226e-13), r);
  const vd r2 = mul(r, r), r4 = mul(r2, r2);
  const vd p12 = fmadd(set1(1.0 / 6227020800.0), r, set1(1.0 / 479001600.0));
  const vd p10 = fmadd(set1(1.0 / 39916800.0), r, set1(1.0 / 3628800.0));
  const vd p8 = fmadd(set1(1.0 / 362880.0), r, set1(1.0 / 40320.0));
  const vd p6 = fmadd(set1(1.0 / 5040.0), r, set1(1.0 / 720.0));
  const vd p4 = fmadd(set1(1.0 / 120.0), r, set1(1.0 / 24.0));
  const vd p2 = fmadd(set1(1.0 / 6.0), r, set1(0.5));
  const vd q10 = fmadd(p12, r2, p10), q6 = fmadd(p8, r2, p6), q2 = fmadd(p4, r2, p2);
  const vd s2 = fmadd(fmadd(q10, r4, q6), r4, q2);
  return mul(add(fmadd(s2, r2, r), set1(1.0)), pow2(n));
}
#endif

template <typename scalar_t>
inline void convert_to_double(const scalar_t* src, double* dst, int64_t c) {
#ifdef SDCA_SIMD
  if constexpr (!std::is_same_v<scalar_t, double>) {
    int64_t j = 0;
    for (; j + simd::W <= c; j += simd::W) simd::store(dst + j, simd::load_double(src + j));
    for (; j < c; ++j) dst[j] = static_cast<double>(src[j]);
    return;
  }
#endif
  at::vec::convert(src, dst, c);
}

template <typename scalar_t>
inline void scale_and_store(const double* src, scalar_t* dst, int64_t c, double scale) {
  int64_t j = 0;
#ifdef SDCA_SIMD
  const simd::vd vscale = simd::set1(scale);
  if constexpr (std::is_same_v<scalar_t, double>) {
    for (; j + simd::W <= c; j += simd::W)
      simd::store(dst + j, simd::mul(simd::load(src + j), vscale));
  } else if constexpr (std::is_same_v<scalar_t, float>) {
    for (; j + simd::W <= c; j += simd::W)
      simd::store_float(dst + j, simd::mul(simd::load(src + j), vscale));
  } else {
    for (; j + 2 * simd::W <= c; j += 2 * simd::W)
      simd::store2(dst + j, simd::mul(simd::load(src + j), vscale),
                   simd::mul(simd::load(src + j + simd::W), vscale));
  }
#endif
  for (; j < c; ++j) dst[j] = static_cast<scalar_t>(src[j] * scale);
}

#ifdef SDCA_SIMD
// Sum of exp(row - maximum) over normal-range rows.
template <bool Compensated>
double exp_sum_simd(const double* row, int64_t classes, double maximum) {
  constexpr int W = simd::W;
  const simd::vd vmax = simd::set1(maximum);
  VecSum<Compensated> acc;
  int64_t d = 0;
  for (; d + W <= classes; d += W)
    acc.add(Vec(exp_pd_normal(simd::sub(simd::load(row + d), vmax))));
  if (classes - d > 0) {
    double tail_buf[W];
    for (int i = 0; i < W; ++i) tail_buf[i] = maximum;
    std::memcpy(tail_buf, row + d, static_cast<std::size_t>(classes - d) * sizeof(double));
    const Vec v(exp_pd_normal(simd::sub(simd::load(tail_buf), vmax)));
    acc.add(Vec::set(Vec(0.0), v, classes - d));
  }
  return acc.value();
}
#endif

// Sum of exp(row - maximum) for any row.
template <bool Compensated>
double exp_sum_generic(const double* row, int64_t classes, double maximum) {
  if constexpr (Compensated) {
    VecSum<true> acc;
    for (int64_t d = 0; d < classes; d += Vec::size()) {
      const int64_t count = std::min<int64_t>(Vec::size(), classes - d);
      const Vec v = (Vec::loadu(row + d, count) - Vec(maximum)).exp();
      acc.add(Vec::set(Vec(0.0), v, count));
    }
    return acc.value();
  } else {
    return at::vec::map_reduce_all<double>([maximum](Vec x) { return (x - Vec(maximum)).exp(); },
                                           [](Vec a, Vec b) { return a + b; }, row, classes);
  }
}

// The extrema of v[0, n) (NaN if any v[j] is NaN).
inline std::pair<double, double> extrema(const double* v, int64_t n) {
  return at::vec::reduce2_all<double>([](Vec a, Vec b) { return at::vec::maximum(a, b); },
                                      [](Vec a, Vec b) { return at::vec::minimum(a, b); }, v, n);
}

// Loss values are at least float32: promote(logits dtype, float32).
inline at::ScalarType value_type(const at::Tensor& z) {
  return z.scalar_type() == at::kDouble ? at::kDouble : at::kFloat;
}

// The kernels write their losses and read their upstream derivatives in place: a zeroed
// tensor, a conversion or a copy through the dispatcher costs more than a 256 x 10 call's rows.

// Losses in the value dtype of z, each rounded once from double: (count,) for rows, else the
// scalar values[0].
inline at::Tensor loss_values(const double* values, int64_t count, bool rows, const at::Tensor& z) {
  const at::ScalarType dtype = value_type(z);
  at::Tensor out =
      rows ? at::empty({count}, z.options().dtype(dtype)) : at::empty({}, z.options().dtype(dtype));
  const int64_t size = rows ? count : 1;
  if (dtype == at::kDouble) {
    std::copy(values, values + size, out.data_ptr<double>());
  } else {
    float* o = out.data_ptr<float>();
    for (int64_t i = 0; i < size; ++i) o[i] = static_cast<float>(values[i]);
  }
  return out;
}
inline at::Tensor loss_value(double value, const at::Tensor& z) {
  return loss_values(&value, 1, false, z);
}

// The bytes of a boolean or uint8 mask (0 or 1), read in place.
inline const uint8_t* mask_bytes(const at::Tensor& mask) {
  TORCH_CHECK_VALUE(mask.scalar_type() == at::kBool || mask.scalar_type() == at::kByte,
                    "expected a boolean or uint8 mask");
  return static_cast<const uint8_t*>(mask.data_ptr());
}

// Row losses as losses for reduction 0 (none), else their long double sum (2) or mean (1).
inline at::Tensor reduced_losses(const std::vector<double>& values, int64_t reduction,
                                 const at::Tensor& z) {
  const int64_t n = static_cast<int64_t>(values.size());
  if (reduction == 0) return loss_values(values.data(), n, true, z);
  long double total = 0;
  for (double v : values) total += v;
  if (reduction == 1) total /= n;
  return loss_value(static_cast<double>(total), z);
}

// A scalar or vector of any floating dtype and stride (an upstream derivative, class weights)
// widened to double.
inline std::vector<double> double_values(const at::Tensor& t) {
  TORCH_CHECK_VALUE(t.dim() <= 1, "invalid upstream gradient");
  std::vector<double> out(t.numel());
  const int64_t stride = t.dim() == 0 ? 0 : t.stride(0);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, t.scalar_type(), "sdca_double_values", [&] {
        const scalar_t* p = t.data_ptr<scalar_t>();
        for (std::size_t i = 0; i < out.size(); ++i)
          out[i] = static_cast<double>(p[static_cast<int64_t>(i) * stride]);
      });
  return out;
}

// Differentiable fallback for second and higher derivatives (create_graph=True).
at::Tensor differentiable_backward(const at::Tensor& logits, const at::Tensor& target,
                                   const at::Tensor& upstream_in, int64_t reduction,
                                   const at::Tensor& invalid = {}, bool bounded_scale = false) {
  const auto z = logits.to(at::kDouble);
  // CUDA reuses forward's validation status; CPU forward already rejected bad
  // inputs. Clamping keeps the higher-derivative gather/scatter safe on CUDA.
  const auto idx = target.clamp(0, logits.size(1) - 1).unsqueeze(1);
  const auto label = z.gather(1, idx).squeeze(1);
  const auto neg_inf = -std::numeric_limits<double>::infinity();
  const auto masked = z.scatter(1, idx, neg_inf);
  const auto maximum = std::get<0>(masked.max(1));
  const auto log_sum = (masked - maximum.unsqueeze(1)).exp().sum(1).log();
  const auto relative = (maximum - label) + log_sum;
  const auto log_mass = -at::logaddexp(at::zeros_like(relative), -relative);
  const auto log_grad = ((z - maximum.unsqueeze(1)) + (log_mass - log_sum).unsqueeze(1))
                            .scatter(1, idx, log_mass.unsqueeze(1));
  const auto upstream = upstream_in.to(at::kDouble).expand({logits.size(0)});
  if (bounded_scale ||
      (logits.scalar_type() != at::kDouble && upstream_in.scalar_type() != at::kDouble)) {
    // The log-domain path is accurate enough for float32 storage. Keep factored
    // roots so later derivatives can rescue initially vanished probability tails.
    const double divisor = reduction == 1 ? std::sqrt(static_cast<double>(logits.size(0))) : 1.0;
    auto root = (log_grad * 0.5).exp();
    if (reduction == 1) root = root / divisor;
    auto gradient = root * (root * upstream.unsqueeze(1));
    gradient = gradient.scatter(1, idx, -gradient.gather(1, idx));
    if (invalid.defined()) {
      gradient =
          gradient.masked_fill(invalid.unsqueeze(1), std::numeric_limits<double>::quiet_NaN());
    }
    return gradient.to(logits.scalar_type());
  }
  const double divisor = reduction == 1 ? std::sqrt(static_cast<double>(logits.size(0))) : 1.0;
  auto root = (log_grad * 0.5).exp();
  if (reduction == 1) root = root / divisor;
  auto gradient = root * (root * upstream.unsqueeze(1));
  gradient = gradient.scatter(1, idx, -gradient.gather(1, idx));
  const auto far = relative < -37.0;
  if (!logits.is_cuda() && !far.any().item<bool>()) {
    return gradient.to(logits.scalar_type());
  }
  const auto far_col = far.unsqueeze(1);
  auto half = at::where(far_col, (z - label.unsqueeze(1)) * 0.5, neg_inf).exp();
  if (reduction == 1) half = half / divisor;
  auto far_grad = half * (half * upstream.unsqueeze(1));
  auto half_mass = at::where(far, (maximum - label) * 0.5, neg_inf).exp();
  if (reduction == 1) half_mass = half_mass / divisor;
  const auto target_mass = half_mass * ((half_mass * upstream) * log_sum.exp());
  far_grad = far_grad.scatter(1, idx, -target_mass.unsqueeze(1));
  gradient = at::where(far_col, far_grad, gradient);
  if (invalid.defined()) {
    gradient = gradient.masked_fill(invalid.unsqueeze(1), std::numeric_limits<double>::quiet_NaN());
  }
  return gradient.to(logits.scalar_type());
}

// Rows of 16-bit logits to and from float buffers (exact widening, round to nearest).
inline void to_float(const at::BFloat16* src, float* dst, int64_t n) {
  int64_t j = 0;
  for (; j + lanes::L <= n; j += lanes::L) {
    lanes::vh h;
    std::memcpy(&h, src + j, sizeof h);
    const lanes::vu u = __builtin_convertvector(h, lanes::vu) << 16;
    std::memcpy(dst + j, &u, sizeof u);
  }
  for (; j < n; ++j) dst[j] = static_cast<float>(src[j]);
}

inline void from_float(const float* src, at::BFloat16* dst, int64_t n) {
  int64_t j = 0;
  for (; j + lanes::L <= n; j += lanes::L) {
    const lanes::vf f = lanes::load(src + j);
    const lanes::vu u = (lanes::vu)f;
    const lanes::vu rounded = (u + (0x7fffu + ((u >> 16) & 1u))) >> 16;
    const lanes::vu bits =
        (lanes::vu)lanes::select(f != f, (lanes::vf)(lanes::vu{} + 0x7fc0u), (lanes::vf)rounded);
    const lanes::vh h = __builtin_convertvector(bits, lanes::vh);
    std::memcpy(dst + j, &h, sizeof h);
  }
  for (; j < n; ++j) dst[j] = static_cast<at::BFloat16>(src[j]);
}

inline void to_float(const at::Half* src, float* dst, int64_t n) {
#if defined(__aarch64__)
  const __fp16* h = reinterpret_cast<const __fp16*>(src);
  for (int64_t j = 0; j < n; ++j) dst[j] = static_cast<float>(h[j]);
#elif defined(SDCA_SIMD)
  int64_t j = 0;
  for (; j + 8 <= n; j += 8)
    _mm256_storeu_ps(dst + j,
                     _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + j))));
  for (; j < n; ++j) dst[j] = static_cast<float>(src[j]);
#else
  at::vec::convert(src, dst, n);
#endif
}

inline void from_float(const float* src, at::Half* dst, int64_t n) {
#if defined(__aarch64__)
  __fp16* h = reinterpret_cast<__fp16*>(dst);
  for (int64_t j = 0; j < n; ++j) h[j] = static_cast<__fp16>(src[j]);
#elif defined(SDCA_SIMD)
  int64_t j = 0;
  for (; j + 8 <= n; j += 8)
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + j),
                     _mm256_cvtps_ph(_mm256_loadu_ps(src + j), _MM_FROUND_TO_NEAREST_INT));
  for (; j < n; ++j) dst[j] = static_cast<at::Half>(src[j]);
#else
  at::vec::convert(src, dst, n);
#endif
}

// Instantiated for float64 logits, which never take the float32 paths.
inline void to_float(const double* src, float* dst, int64_t n) { std::copy(src, src + n, dst); }
inline void from_float(const float* src, double* dst, int64_t n) { std::copy(src, src + n, dst); }

// Where a float row result goes: float rows in place, others through buffer (from_float).
inline float* float_output(float* out, float*) { return out; }
template <typename scalar_t>
inline float* float_output(scalar_t*, float* buffer) {
  return buffer;
}
inline void from_float(const float* src, float* dst, int64_t n) {
  if (src != dst) std::copy(src, src + n, dst);
}

// A float view of a row: the row itself for float logits, else its widened copy in buffer.
template <typename scalar_t>
inline const float* float_row(const scalar_t* src, int64_t n, float* buffer) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return src;
  } else {
    to_float(src, buffer, n);
    return buffer;
  }
}

// A double view of a row: the row itself for float64 logits, else its widened copy in buffer.
template <typename scalar_t>
inline const double* double_row(const scalar_t* src, int64_t n, double* buffer) {
  if constexpr (std::is_same_v<scalar_t, double>) {
    return src;
  } else {
    convert_to_double(src, buffer, n);
    return buffer;
  }
}

// Input errors of parallel rows, raised after the loop in the order of the torch layer's checks.
struct RowErrors {
  std::atomic<bool> finite{false}, target{false}, positive{false}, negative{false};
  void raise() const {
    TORCH_CHECK_VALUE(!finite, "logits must be finite floating-point values");
    TORCH_CHECK_VALUE(!target, "Target index out of range");
    TORCH_CHECK_VALUE(!positive, "Every example needs a positive label");
    TORCH_CHECK_VALUE(!negative, "This loss needs a negative label");
  }
};

template <typename scalar_t>
inline bool all_finite(const scalar_t* row, int64_t c) {
  for (int64_t j = 0; j < c; ++j)
    if (!std::isfinite(static_cast<double>(row[j]))) return false;
  return true;
}

// Validates the target of a row and, through margins (hinge_margins or a row loss returning
// NaN for nonfinite logits), its logits; false records the error and skips the row.
template <typename scalar_t, typename F>
bool checked_row(const scalar_t* z, int64_t c, int64_t target, RowErrors& errors, F&& margins) {
  if (target < 0 || target >= c) {
    errors.target = true;
    if (!all_finite(z, c)) errors.finite = true;
    return false;
  }
  if (!margins()) {
    errors.finite = true;
    return false;
  }
  return true;
}

// An uninitialized gradient like z. Large ones ask Linux for transparent huge pages: their
// first touch (in the parallel row loops) then faults 2 MiB at a time, not 4 KiB.
inline at::Tensor gradient_like(const at::Tensor& z) {
  at::Tensor output = at::empty_like(z);
#ifdef __linux__
  constexpr std::uintptr_t kHuge = std::uintptr_t{1} << 21;
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(output.data_ptr());
  const std::uintptr_t end = begin + output.nbytes();
  const std::uintptr_t first = (begin + kHuge - 1) & ~(kHuge - 1), last = end & ~(kHuge - 1);
  if (output.nbytes() >= (std::size_t{32} << 20) && last > first)
    madvise(reinterpret_cast<void*>(first), last - first, MADV_HUGEPAGE);
#endif
  return output;
}

// Rows split by the parallel loops: about 4096 logits per task, one task for small inputs.
inline int64_t grain(int64_t n, int64_t c) {
  return n * c >= 16384 ? std::max<int64_t>(1, 4096 / c) : n + 1;
}

}  // namespace

#endif  // SDCA_TORCH_LANES_H
