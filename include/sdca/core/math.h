#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_CORE_MATH_H)
#if SDCA_ISA_ID == 1
#define SDCA_CORE_MATH_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_CORE_MATH_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_CORE_MATH_H3 1
#else
#define SDCA_CORE_MATH_H4 1
#endif

// Exponentials and logarithms on lanes, inlined (no library calls): float exp within about
// 1.5 ulp, double within about 2 ulp, subnormal results rounded once (+0.0 below half the least
// subnormal, -inf included), as the C library's, and results beyond max saturated at max (a
// product with a zero sum stays zero). One lane (isa::scalar) and scalars take the C library's.
//
// The exponentials take x clamped to [low, high] (ExpRange) as 2^k exp(r), k = round(x / ln 2),
// and form 2^k exp(r) as (2^h exp(r)) 2^(k - h), h = floor(k / 2), or for x <= 0 as
// (2^(k + S) exp(r)) 2^-S, S = digits + 2 (ExpRange::lift): both factors are normal over the
// whole range, so the product is exact for normal results and rounds once into the subnormals
// and at overflow.
#include <cmath>
#include <limits>

#include "lanes.h"
#include "sdca/core/limits.h"

SDCA_ISA_BEGIN
namespace core {

namespace detail {
// 2^k for k in [min_exponent - 1, max_exponent - 1].
template <typename T, typename Isa, int K>
SDCA_LANE Lanes<T, Isa, K> pow2(const Ints<T, Isa, K>& k) {
  typedef std::numeric_limits<T> N;
  return from_bits<T, Isa, K>((k + (N::max_exponent - 1)) << (N::digits - 1));
}

// low = log(2^(min_exponent - digits - 2)): 2^k exp(r) <= 2^(1/2 + min_exponent - digits - 2) is
// below half the least subnormal and rounds to +0.0; k + lift >= min_exponent. high =
// log(2^(max_exponent + 1)): 2^k exp(r) >= 2^(max_exponent + 1/2) overflows; floor(k / 2) and
// k - floor(k / 2) are at most max_exponent / 2 + 1.
template <typename T>
struct ExpRange {
  typedef std::numeric_limits<T> N;
  static const int lift = N::digits + 2;
  static constexpr T low() { return limits::log_pow2<T>(N::min_exponent - N::digits - 2); }
  static constexpr T high() { return limits::log_pow2<T>(N::max_exponent + 1); }
};

// x clamped to ExpRange (to low only for x <= 0).
template <bool Nonpositive, typename T, typename Isa, int K>
SDCA_LANE Lanes<T, Isa, K> exp_range(const Lanes<T, Isa, K>& x) {
  constexpr T low = ExpRange<T>::low(), high = ExpRange<T>::high();
  const Lanes<T, Isa, K> c = max(x, low);
  return Nonpositive ? c : min(c, high);
}

// floor(k / 2) of |k| < 2 max_exponent, by a logical shift of k + 4 max_exponent.
template <typename T, typename Isa, int K>
SDCA_LANE Ints<T, Isa, K> exp_half(const Ints<T, Isa, K>& k) {
  const int offset = 2 * std::numeric_limits<T>::max_exponent;
  return shift_right_logical(k + 2 * offset, 1) - offset;
}

// exp(r) (Cephes expf) of the reduction of x, and k.
template <typename Isa, int K>
SDCA_LANE Lanes<float, Isa, K> exp_reduced(const Lanes<float, Isa, K>& x, Ints<float, Isa, K>& k) {
  typedef Lanes<float, Isa, K> F;
  const F n = round(x * 1.44269504088896341f, k);
  const F r = fma(n, 2.12194440e-4f, fma(n, -0.693359375f, x));
  F p = fma(F::splat(1.9875691500e-4f), r, 1.3981999507e-3f);
  p = fma(p, r, 8.3334519073e-3f);
  p = fma(p, r, 4.1665795894e-2f);
  p = fma(p, r, 1.6666665459e-1f);
  p = fma(p, r, 5.0000001201e-1f);
  return fma(p * r, r, r) + 1.0f;
}
}  // namespace detail

// exp(x) (Cephes expf): h added to the exponent bits of exp(r), times 2^(k - h).
template <typename Isa, int K>
SDCA_LANE Lanes<float, Isa, K> exp(const Lanes<float, Isa, K>& x) {
  typedef Lanes<float, Isa, K> F;
  Ints<float, Isa, K> k;
  const F y = detail::exp_reduced(detail::exp_range<false>(x), k);
  const Ints<float, Isa, K> h = detail::exp_half(k);
  const F e = from_bits<float, Isa, K>(bits(y) + (h << 23)) * detail::pow2<float>(k - h);
  return min(e, std::numeric_limits<float>::max());
}

// The floor of exp(x) 2^E = 2^(k + E) exp(r) in T, log(2^(min_exponent - 2 - E)): from
// (min_exponent - 2.5 - E) ln 2 down, k + E is min_exponent - 2 and 2^(k + E) has the bits of
// +0.0.
template <typename T, int E>
constexpr T exp_times_floor() {
  return limits::log_pow2<T>(std::numeric_limits<T>::min_exponent - 2 - E);
}

// exp(x) 2^E for x <= 0 without exp's flag: x is clamped at exp_times_floor, so the polynomial
// times 2^(k + E) is +0.0 below (min_exponent - 2.5 - E) ln 2 (-inf included); above, that
// product is exact.
template <int E, typename Isa, int K>
SDCA_LANE Lanes<float, Isa, K> exp_nonpositive_times(Lanes<float, Isa, K> x) {
  typedef Lanes<float, Isa, K> F;
  x = max(x, exp_times_floor<float, E>());
  const float magic = 12582912.0f;
  const F t = x * 1.44269504088896341f + magic;
  const F n = t - magic;
  const Ints<float, Isa, K> biased = bits(t) - (bits(F::splat(magic)) - 127 - E);
  const F r = fma(n, 2.12194440e-4f, fma(n, -0.693359375f, x));
  // Minimize the relative error of 1 + r + r^2 q(r) over the reduced interval
  // (|r| <= ln(2)/2 plus float range-reduction rounding); q has degree four.
  const F r2 = r * r;
  const F p = fma(F::splat(0.008368712f), r, 0.04166839f);
  const F q = fma(F::splat(0.16666521f), r, 0.49999994f);
  const F poly = fma(fma(F::splat(0.0013814609f), r2, p), r2, q);
  const F y = fma(poly, r2, r) + 1.0f;
  return y * from_bits<float, Isa, K>(biased << 23);
}
// exp(x) for x <= 0: exactly +0.0 below -87.68.
template <typename Isa, int K>
SDCA_LANE Lanes<float, Isa, K> exp_nonpositive(const Lanes<float, Isa, K>& x) {
  return exp_nonpositive_times<0>(x);
}
// exp(x) 2^64 for x <= 0 (Wide<float>): exact and normal down to -132, exactly +0.0 below
// -132.04.
template <typename Isa, int K>
SDCA_LANE Lanes<float, Isa, K> exp_nonpositive_wide(const Lanes<float, Isa, K>& x) {
  return exp_nonpositive_times<64>(x);
}

// Whether float holds x within its range (false for NaN and infinities).
template <typename T>
inline bool fits_float(T x) {
  return std::fabs(static_cast<double>(x)) <=
         static_cast<double>(std::numeric_limits<float>::max());
}
// Whether x is a positive normal float.
inline bool normal_float(double x) {
  return x >= static_cast<double>(std::numeric_limits<float>::min()) && fits_float(x);
}

// f times unwide as factor times power, with a normal factor whenever possible.
template <typename W>
struct Wide {
  static double unwide() {
    constexpr double u = limits::detail::pow2<double>(-64);
    return u;
  }
  static bool split(double f, W& factor, W& power) {
    const double scaled = f * unwide(), a = std::fabs(scaled);
    const double most = static_cast<double>(std::numeric_limits<W>::max());
    power = 1;
    if (a >= 4 * static_cast<double>(std::numeric_limits<W>::min()) && a <= most) {
      factor = static_cast<W>(scaled);
      return true;
    }
    if (!(a <= most)) return false;
    factor = static_cast<W>(f == 0 ? scaled : f);
    if (f != 0) power = static_cast<W>(unwide());
    return true;
  }
};
namespace detail {
// expm1(r) (Taylor, degree 8) of the reduction of x, and k.
template <typename Isa, int K>
SDCA_LANE Lanes<float, Isa, K> expm1_parts(const Lanes<float, Isa, K>& x, Ints<float, Isa, K>& k) {
  typedef Lanes<float, Isa, K> F;
  const F n = round(x * 1.44269504088896341f, k);
  const F r = fma(n, 2.12194440e-4f, fma(n, -0.693359375f, x));
  F p = fma(F::splat(1.0f / 40320), r, 1.0f / 5040);
  p = fma(p, r, 1.0f / 720);
  p = fma(p, r, 1.0f / 120);
  p = fma(p, r, 1.0f / 24);
  p = fma(p, r, 1.0f / 6);
  p = fma(p, r, 0.5f);
  return fma(p * r, r, r);
}
}  // namespace detail

// expm1(r) for |r| <= log(2) / 2: degree 13, within 1e-17 of r before rounding.
template <typename Isa, int K>
SDCA_LANE Lanes<double, Isa, K> expm1_reduced(const Lanes<double, Isa, K>& r) {
  typedef Lanes<double, Isa, K> D;
  const D r2 = r * r, r4 = r2 * r2;
  const D p12 = fma(r, 1.0 / 6227020800.0, 1.0 / 479001600.0);
  const D p10 = fma(r, 1.0 / 39916800.0, 1.0 / 3628800.0);
  const D p8 = fma(r, 1.0 / 362880.0, 1.0 / 40320.0);
  const D p6 = fma(r, 1.0 / 5040.0, 1.0 / 720.0);
  const D p4 = fma(r, 1.0 / 120.0, 1.0 / 24.0);
  const D p2 = fma(r, 1.0 / 6.0, 0.5);
  const D q10 = fma(p12, r2, p10), q6 = fma(p8, r2, p6), q2 = fma(p4, r2, p2);
  return fma(fma(fma(q10, r4, q6), r4, q2), r2, r);
}

namespace detail {
// expm1(r) of the reduction of x, and k.
template <typename Isa, int K>
SDCA_LANE Lanes<double, Isa, K> expm1_parts(const Lanes<double, Isa, K>& x,
                                            Ints<double, Isa, K>& k) {
  typedef Lanes<double, Isa, K> D;
  const D n = round(x * 1.44269504088896338700, k);
  // log(2) in two parts (fdlibm): n log2_hi is exact for |n| < 2^20.
  const double hi = static_cast<double>(limits::kLn2Hi), lo = static_cast<double>(limits::kLn2Lo);
  const D r = (x - n * hi) - n * lo;
  return expm1_reduced(r);
}

// 2^k (1 + m), m = expm1(r).
template <bool Nonpositive, typename T, typename Isa, int K>
SDCA_LANE Lanes<T, Isa, K> exp_compose(const Lanes<T, Isa, K>& m, const Ints<T, Isa, K>& k) {
  typedef Lanes<T, Isa, K> L;
  if (Nonpositive) {
    constexpr T unlift = limits::detail::pow2<T>(-ExpRange<T>::lift);
    const L lifted = pow2<T>(k + ExpRange<T>::lift);
    return fma(lifted, m, lifted) * unlift;
  }
  const Ints<T, Isa, K> h = exp_half(k);
  const L first = pow2<T>(h);
  return min(fma(first, m, first) * pow2<T>(k - h), std::numeric_limits<T>::max());
}

template <bool Nonpositive, typename T, typename Isa, int K>
SDCA_LANE Lanes<T, Isa, K> exp_of(const Lanes<T, Isa, K>& x) {
  Ints<T, Isa, K> k;
  const Lanes<T, Isa, K> m = expm1_parts(exp_range<Nonpositive>(x), k);
  return exp_compose<Nonpositive>(m, k);
}
}  // namespace detail

// exp(x) - 1 in the lanes of small (|x| <= 1 there, so 2^k unscaled; float within about 2 ulp),
// exp(x) elsewhere: one range reduction, 2^k expm1(r) + (2^k - 1) or + 2^k.
template <typename T, typename Isa, int K>
SDCA_LANE Lanes<T, Isa, K> exp_or_expm1(const Lanes<T, Isa, K>& x, const Ints<T, Isa, K>& small) {
  typedef Lanes<T, Isa, K> L;
  Ints<T, Isa, K> k;
  const L m = detail::expm1_parts(detail::exp_range<false>(x), k), scale = detail::pow2<T>(k);
  return select(small, fma(scale, m, scale - T(1)), detail::exp_compose<false>(m, k));
}

template <typename Isa, int K>
SDCA_LANE Lanes<double, Isa, K> exp(const Lanes<double, Isa, K>& x) {
  return detail::exp_of<false>(x);
}
template <typename Isa, int K>
SDCA_LANE Lanes<double, Isa, K> exp_nonpositive(const Lanes<double, Isa, K>& x) {
  return detail::exp_of<true>(x);
}
// exp(x) 2^64 for x <= 0. Lift the scale by another power of two so a
// subnormal scaled term can contribute to a representable sum over a wide row.
template <typename Isa, int K>
SDCA_LANE Lanes<double, Isa, K> exp_nonpositive_wide(const Lanes<double, Isa, K>& x) {
  typedef Lanes<double, Isa, K> D;
  Ints<double, Isa, K> k;
  const D m = detail::expm1_parts(
      max(x, exp_times_floor<double, 64 + detail::ExpRange<double>::lift>()), k);
  return detail::exp_compose<true>(m, k + 64);
}

// expm1(x) for the near lanes (|x| < 1), exp(x) times 2^64 otherwise (x <= 0).
// Apply the same lift to the unit subtracted in a near lane.
template <typename T, typename Isa, int K>
SDCA_LANE Lanes<T, Isa, K> expm1_or_wide(const Lanes<T, Isa, K>& x, const Ints<T, Isa, K>& near) {
  typedef Lanes<T, Isa, K> A;
  constexpr int lift = sizeof(T) == sizeof(double) ? detail::ExpRange<T>::lift : 0;
  constexpr T unit = limits::detail::pow2<T>(lift);
  Ints<T, Isa, K> k;
  const A m = detail::expm1_parts(max(x, exp_times_floor<T, 64 + lift>()), k);
  const A scale = detail::pow2<T>(k + (64 + lift) - (near & 64));
  return fma(scale, m, scale - select(near, A::splat(unit), A::zero())) * (T(1) / unit);
}

// expm1(x): exp_or_expm1 with the lanes |x| > 1 less one.
template <typename T, typename Isa, int K>
SDCA_LANE Lanes<T, Isa, K> expm1(const Lanes<T, Isa, K>& x) {
  const Ints<T, Isa, K> small = abs(x) <= T(1);
  const Lanes<T, Isa, K> e = exp_or_expm1(x, small);
  return select(small, e, e - T(1));
}

namespace detail {
// log(u) + c / u for u = 1 + x > 0 rounded with error c (musl's log1p): u = 2^k (1 + f) with
// 1 + f in [sqrt(2) / 2, sqrt(2)), and log(1 + f) by the polynomial in s = f / (2 + f). Where
// direct, f = x exactly.
template <typename Isa, int K>
SDCA_LANE Lanes<double, Isa, K> log_reduced(const Lanes<double, Isa, K>& u,
                                            const Lanes<double, Isa, K>& x, bool from_one) {
  typedef Lanes<double, Isa, K> D;
  typedef Ints<double, Isa, K> I;
  const I b = bits(u);
  const I high = (b >> 32) + (0x3ff00000 - 0x3fe6a09e);
  const I k = (high >> 20) - 0x3ff;
  const I reduced = (((high & 0x000fffff) + 0x3fe6a09e) << 32) | (b & 0xffffffff);
  const D dk = convert<double, Isa, K>(k);
  D c = D::zero(), f;
  if (from_one) {
    const I direct = k == 0;
    c = select(k >= 2, 1.0 - (u - x), x - (u - 1.0)) / u;
    c = select(direct | (k >= 54), D::zero(), c);
    f = select(direct, x, from_bits<double, Isa, K>(reduced) - 1.0);
  } else {
    f = from_bits<double, Isa, K>(reduced) - 1.0;
  }
  const D hfsq = 0.5 * f * f, s = f / (2.0 + f), z = s * s, w = z * z;
  const D t1 = w * (3.999999999940941908e-01 +
                    w * (2.222219843214978396e-01 + w * 1.531383769920937332e-01));
  const D t2 = z * (6.666666666666735130e-01 +
                    w * (2.857142874366239149e-01 +
                         w * (1.818357216161805012e-01 + w * 1.479819860511658591e-01)));
  const D r = t2 + t1;
  return s * (hfsq + r) + (dk * 1.90821492927058770002e-10 + c) - hfsq + f +
         dk * 6.93147180369123816490e-01;
}
}  // namespace detail

// log1p(x) within an ulp for finite x > -1; -inf at -1, NaN below, +inf at +inf.
template <typename Isa, int K>
SDCA_LANE Lanes<double, Isa, K> log1p(const Lanes<double, Isa, K>& x) {
  typedef Lanes<double, Isa, K> D;
  const D u = 1.0 + x;
  const D finite = detail::log_reduced(u, x, true);
  const double inf = std::numeric_limits<double>::infinity();
  D r = select(u == inf, D::splat(inf), finite);
  r = select(u == 0.0, D::splat(-inf), r);
  return select((u < 0.0) | (x != x), D::splat(std::numeric_limits<double>::quiet_NaN()), r);
}

// log(x) within an ulp for normal x > 0; -inf at 0, NaN below, +inf at +inf. Subnormal x
// scale by 2^54 first.
template <typename Isa, int K>
SDCA_LANE Lanes<double, Isa, K> log(const Lanes<double, Isa, K>& x) {
  typedef Lanes<double, Isa, K> D;
  const Ints<double, Isa, K> sub = (x < std::numeric_limits<double>::min()) & (x > 0.0);
  const D y = select(sub, x * 18014398509481984.0, x);
  D r = detail::log_reduced(y, y, false);
  r = select(sub, r - 37.42994775023704676, r);  // 54 log(2)
  const double inf = std::numeric_limits<double>::infinity();
  r = select(x == inf, D::splat(inf), r);
  r = select(x == 0.0, D::splat(-inf), r);
  return select((x < 0.0) | (x != x), D::splat(std::numeric_limits<double>::quiet_NaN()), r);
}

// Float logarithms through double lanes, rounded once.
template <typename Isa, int K>
SDCA_LANE Lanes<float, Isa, K> log1p(const Lanes<float, Isa, K>& x) {
  return narrow(log1p(widen(x)));
}
template <typename Isa, int K>
SDCA_LANE Lanes<float, Isa, K> log(const Lanes<float, Isa, K>& x) {
  return narrow(log(widen(x)));
}

// One lane (isa::scalar) takes the C library's functions, which keep subnormal results; exp
// and expm1 saturate at max as the lanes do.
namespace detail {
template <typename T, int K, typename F>
SDCA_LANE Lanes<T, isa::scalar, K> each_lane(Lanes<T, isa::scalar, K> x, F f) {
  for (int k = 0; k < K; ++k) x.v[k][0] = f(x.v[k][0]);
  return x;
}
template <typename T>
SDCA_LANE T saturated(T x) {
  return x > std::numeric_limits<T>::max() ? std::numeric_limits<T>::max() : x;
}
struct Exp {
  template <typename T>
  T operator()(T x) const {
    return saturated(std::exp(x));
  }
};
struct Expm1 {
  template <typename T>
  T operator()(T x) const {
    return saturated(std::expm1(x));
  }
};
struct Log {
  template <typename T>
  T operator()(T x) const {
    return std::log(x);
  }
};
struct Log1p {
  template <typename T>
  T operator()(T x) const {
    return std::log1p(x);
  }
};
}  // namespace detail
#define SDCA_SCALAR_MATH(T)                                                               \
  template <int K>                                                                        \
  SDCA_LANE Lanes<T, isa::scalar, K> exp(const Lanes<T, isa::scalar, K>& x) {             \
    return detail::each_lane(x, detail::Exp());                                           \
  }                                                                                       \
  template <int K>                                                                        \
  SDCA_LANE Lanes<T, isa::scalar, K> exp_nonpositive(const Lanes<T, isa::scalar, K>& x) { \
    return detail::each_lane(x, detail::Exp());                                           \
  }                                                                                       \
  template <int K>                                                                        \
  SDCA_LANE Lanes<T, isa::scalar, K> expm1(const Lanes<T, isa::scalar, K>& x) {           \
    return detail::each_lane(x, detail::Expm1());                                         \
  }                                                                                       \
  template <int K>                                                                        \
  SDCA_LANE Lanes<T, isa::scalar, K> exp_or_expm1(const Lanes<T, isa::scalar, K>& x,      \
                                                  const Ints<T, isa::scalar, K>& small) { \
    return select(small, expm1(x), exp(x));                                               \
  }                                                                                       \
  template <int K>                                                                        \
  SDCA_LANE Lanes<T, isa::scalar, K> log(const Lanes<T, isa::scalar, K>& x) {             \
    return detail::each_lane(x, detail::Log());                                           \
  }                                                                                       \
  template <int K>                                                                        \
  SDCA_LANE Lanes<T, isa::scalar, K> log1p(const Lanes<T, isa::scalar, K>& x) {           \
    return detail::each_lane(x, detail::Log1p());                                         \
  }
SDCA_SCALAR_MATH(float)
SDCA_SCALAR_MATH(double)
#undef SDCA_SCALAR_MATH

// exp(x) for x <= 0, retaining the full subnormal range of exp. Double and scalar
// exp_nonpositive already retain it. For SIMD float, lift the polynomial by an exact
// power of two so its final scaling is the only rounding in the subnormal range.
template <typename Isa, int K>
SDCA_LANE Lanes<double, Isa, K> exp_nonpositive_full(const Lanes<double, Isa, K>& x) {
  return exp_nonpositive(x);
}
template <typename Isa, int K>
SDCA_LANE Lanes<float, Isa, K> exp_nonpositive_full(const Lanes<float, Isa, K>& x) {
  if (std::is_same<Isa, isa::scalar>::value) return exp(x);
  constexpr int lift = detail::ExpRange<float>::lift;
  Ints<float, Isa, K> k;
  const Lanes<float, Isa, K> y = detail::exp_reduced(detail::exp_range<true>(x), k);
  return (detail::pow2<float>(k + lift) * y) * limits::detail::pow2<float>(-lift);
}

typedef Lanes<double, isa::scalar> Scalar64;
typedef Lanes<float, isa::scalar> Scalar32;
SDCA_LANE double exp(double x) { return std::exp(x); }
SDCA_LANE double expm1(double x) { return std::expm1(x); }
SDCA_LANE double log(double x) { return std::log(x); }
SDCA_LANE double log1p(double x) { return std::log1p(x); }
SDCA_LANE float exp(float x) { return std::exp(x); }

// log(1 + exp(x)) and log(1 + exp(-|x|)), without overflow.
SDCA_LANE double softplus(double x) {
  const double e = log1p(exp(-std::fabs(x)));
  return x > 0 ? x + e : e;
}

}  // namespace core
SDCA_ISA_END

#endif  // SDCA_CORE_MATH_H
