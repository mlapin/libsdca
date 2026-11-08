#ifndef SDCA_TORCH_LANES_H
#define SDCA_TORCH_LANES_H

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
using Vec = at::vec::Vectorized<double>;

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
inline vf load(const at::BFloat16* p) {
  vh h;
  std::memcpy(&h, p, sizeof h);
  return (vf)(__builtin_convertvector(h, vu) << 16);
}
inline vf load(const at::Half* p) {
  vf v;
#if defined(SDCA_SIMD)
  // Eight lanes at a time (L is 8 or 16).
  for (int k = 0; k < L; k += 8) {
    const __m256 x = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + k)));
    std::memcpy(reinterpret_cast<float*>(&v) + k, &x, sizeof x);
  }
#else
  for (int i = 0; i < L; ++i) v[i] = static_cast<float>(p[i]);
#endif
  return v;
}
inline void store(float* p, vf v) { std::memcpy(p, &v, sizeof v); }
inline vf splat(float x) { return vf{} + x; }
inline vf select(vi mask, vf a, vf b) { return (vf)((mask & (vi)a) | (~mask & (vi)b)); }
// Lanes rounded to nearest (even) into 16-bit logits.
inline void store(at::BFloat16* p, vf v) {
  const vu u = (vu)v;
  const vu rounded = (u + (0x7fffu + ((u >> 16) & 1u))) >> 16;
  const vu bits = (vu)select(v != v, (vf)(vu{} + 0x7fc0u), (vf)rounded);
  const vh h = __builtin_convertvector(bits, vh);
  std::memcpy(p, &h, sizeof h);
}
inline void store(at::Half* p, vf v) {
#if defined(SDCA_SIMD)
  for (int k = 0; k < L; k += 8) {
    __m256 x;
    std::memcpy(&x, reinterpret_cast<const float*>(&v) + k, sizeof x);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p + k),
                     _mm256_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT));
  }
#else
  for (int i = 0; i < L; ++i) p[i] = static_cast<at::Half>(v[i]);
#endif
}
// select(a > b, a, b) and select(a < b, a, b) (b where either is NaN or both are zeros): the
// x86 max and min, one instruction where the select takes a compare and a blend. GCC's lanes
// are the ISA's registers (Clang compiles the conditionals to max and min itself).
inline vf max(vf a, vf b) {
#if defined(SDCA_AVX512) && !defined(__clang__)
  return (vf)_mm512_max_ps((__m512)a, (__m512)b);
#elif defined(SDCA_SIMD) && !defined(__clang__)
  return (vf)_mm256_max_ps((__m256)a, (__m256)b);
#else
  return a > b ? a : b;
#endif
}
inline vf min(vf a, vf b) {
#if defined(SDCA_AVX512) && !defined(__clang__)
  return (vf)_mm512_min_ps((__m512)a, (__m512)b);
#elif defined(SDCA_SIMD) && !defined(__clang__)
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
#elif defined(SDCA_AVX512)
  return (vf)_mm512_fmadd_ps((__m512)a, (__m512)b, (__m512)c);
#elif defined(SDCA_SIMD)
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
  for (; j + L <= n; j += L) body(load(x + j), j, L);
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
  for (; j + L <= n; j += L) body(load(x + j), j);
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
#if defined(SDCA_AVX512) && !defined(__clang__)
  return (vd)_mm512_max_pd((__m512d)a, (__m512d)b);
#elif defined(SDCA_SIMD) && !defined(__clang__)
  return (vd)_mm256_max_pd((__m256d)a, (__m256d)b);
#else
  return a > b ? a : b;
#endif
}
inline vd min(vd a, vd b) {
#if defined(SDCA_AVX512) && !defined(__clang__)
  return (vd)_mm512_min_pd((__m512d)a, (__m512d)b);
#elif defined(SDCA_SIMD) && !defined(__clang__)
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
#elif defined(SDCA_AVX512)
  return (vd)_mm512_fmadd_pd((__m512d)a, (__m512d)b, (__m512d)c);
#elif defined(SDCA_SIMD)
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
    vf v = load(x + j);
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
