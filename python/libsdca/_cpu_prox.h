#ifndef SDCA_CPU_PROX_H
#define SDCA_CPU_PROX_H
#include "_cpu_lanes.h"

namespace {
namespace sdca_cpu {
// Defined by the adapter: the shared solve does not depend on a tensor framework.
inline dlanes::vd proximal_log(dlanes::vd x);

// Lambert W(exp(x)) in float32 or float64, matching include/sdca/math/lambert.h.
template <typename T>
inline T lambert_w_iter5(T w, T y) {
  const T f0 = w - y, f1 = T(1) + y;
  const T f11 = f1 * f1, f0y = f0 * y;
  const T f00y = f0 * f0y;
  return w - T(4) * f0 * (T(6) * f1 * (f11 + f0y) + f00y) /
                 (f11 * (T(24) * f11 + T(36) * f0y) + f00y * (T(14) * y + f0 + T(8)));
}

template <typename T>
inline T exp_approx_1024(T x) {
  T y = T(1) + x / T(1024);
  for (int i = 0; i < 10; ++i) y *= y;
  return y;
}

inline double lambert_w_exp_large_d(double x) {
  if (std::isinf(x)) return x;
  const double l1 = std::log(x);
  double w = x - l1;
  if (x < 1.0 / std::numeric_limits<double>::epsilon()) {
    const double inv_x = 1.0 / x;
    w +=
        l1 * inv_x * (1.0 + inv_x * (0.5 * l1 - 1.0 + inv_x * (l1 * (2.0 * l1 - 9.0) + 6.0) / 6.0));
  }
  const double eps = std::numeric_limits<double>::epsilon();
  const double tol = 2.0 * eps * x;
  const double step_tol = std::cbrt(eps);
  for (int i = 0; i < 6; ++i) {
    const double f = (w - x) + std::log(w);
    if (std::abs(f) <= tol) break;
    const double w1 = w + 1.0;
    const double step = f / ((w1 / w) + (f / w1) / (2.0 * w));
    const double next = w - step;
    if (next == w) break;
    w = next;
    if (std::abs(step) <= step_tol * w) break;
  }
  return w;
}

inline double lambert_w_exp_t(double x) {
  if (std::isnan(x)) return x;
  double w;
  if (x > 0.0) {
    if (x <= 4.0)
      w = lambert_w_iter5(x, 1.0);
    else
      return lambert_w_exp_large_d(x);
  } else {
    if (x > -36.0) {
      w = exp_approx_1024(x);
      if (x > -20.0) w = lambert_w_iter5(w, exp_approx_1024(x - w));
    } else {
      return (x > -746.0) ? std::exp(x) : 0.0;
    }
  }
  return lambert_w_iter5(w, std::exp(x - w));
}

inline float lambert_w_exp_t(float x) {
  if (std::isnan(x)) return x;
  float w;
  if (x > -1.0f) {
    if (x <= 8.0f)
      w = lambert_w_iter5(x, 1.0f);
    else
      return static_cast<float>(lambert_w_exp_large_d(static_cast<double>(x)));
  } else {
    if (x > -18.0f)
      w = exp_approx_1024(x);
    else
      return (x > -104.0f) ? std::exp(x) : 0.0f;
  }
  return lambert_w_iter5(w, std::exp(x - w));
}

template <typename T>
inline T prox_probability(T h, T alpha, T log_alpha) {
  const T arg = h + log_alpha;
  const T cut = -T(std::numeric_limits<T>::digits + 2) * T(0.69314718055994530942);
  if (arg < cut && std::abs(h) >= T(0.5)) return std::exp(h);
  const T w = lambert_w_exp_t(arg);
  const T p_cut = std::is_same<T, float>::value ? T(-9) : T(-18);
  return (arg < p_cut) ? std::exp(h - w) : (w / alpha);
}

// The residual r = q - pi of one entry at the shifted score t: alpha r - log1p(-r / q) = t,
// by safeguarded Halley steps from guess where it is finite (else a second-order estimate).
template <typename T>
inline T prox_residual(T t, T alpha, T q, T log_alpha, T log_q,
                       T guess = std::numeric_limits<T>::quiet_NaN()) {
  if (t == T(0)) return T(0);
  T lo = q - T(1), hi = q;
  if (t <= alpha * lo + log_q) return lo;
  const T half = alpha * q / T(2);
  if (t > half + T(0.69314718055994530942) || t < -half - T(0.40546510810816438198)) {
    const T prob = prox_probability(log_q + (alpha * q - t), alpha, log_alpha);
    return std::max(lo, std::min(q, q - prob));
  }
  const T inv_q = T(1) / q;
  T initial = guess;
  if (!std::isfinite(initial)) {
    const T linear = t / (alpha + inv_q);
    initial = linear / (T(1) + (linear * inv_q) / (T(2) * (T(1) + alpha * q)));
  }
  T r = std::max(lo, std::min(initial, std::nextafter(q, T(0))));
  const T tol = T(8) * std::numeric_limits<T>::epsilon() * std::abs(t);
  for (int it = 0; it < 100; ++it) {
    const T f = (alpha * r - std::log1p(-r * inv_q)) - t;
    if (std::abs(f) <= tol) break;
    if (f < T(0))
      lo = r;
    else
      hi = r;
    const T inv_qr = T(1) / (q - r);
    const T d1 = alpha + inv_qr;
    const T denom = d1 - (f * inv_qr * inv_qr) / (T(2) * d1);
    T next = (denom > T(0)) ? (r - f / denom) : (r - f / d1);
    if (next == r) break;
    if (!(lo <= next && next <= hi)) next = lo + (hi - lo) / T(2);
    if (next == q) return q;
    if (next == r) break;
    r = next;
  }
  return r;
}

// Lane kernels of the proximal losses: the scalar functions above on the lanes of
// dlanes (double) or lanes (float), with the same branches, steps and stopping rules per lane
// (a lane stops on its own), and Sleef's exp, log and log1p (within an ulp) for libm's.
namespace proxlanes {
// K vectors of lanes taken as one: K independent dependency chains, which the core overlaps
// (the iterations below are chains of divisions and polynomials, one vector alone leaves the
// pipelines idle). B is a dlanes or lanes vector of values or of their comparison masks.
template <typename B, int K>
struct Wide {
  B v[K];
};
template <typename B>
using MaskOf = decltype(B{} < B{});

#define SDCA_WIDE_BINARY(op)                                                                     \
  template <typename B, int K>                                                                   \
  inline Wide<B, K> operator op(Wide<B, K> a, const Wide<B, K>& b) {                             \
    for (int k = 0; k < K; ++k) a.v[k] = a.v[k] op b.v[k];                                       \
    return a;                                                                                    \
  }                                                                                              \
  template <typename B, int K, typename S, typename = std::enable_if_t<std::is_arithmetic_v<S>>> \
  inline Wide<B, K> operator op(Wide<B, K> a, S b) {                                             \
    for (int k = 0; k < K; ++k) a.v[k] = a.v[k] op b;                                            \
    return a;                                                                                    \
  }                                                                                              \
  template <typename B, int K, typename S, typename = std::enable_if_t<std::is_arithmetic_v<S>>> \
  inline Wide<B, K> operator op(S a, Wide<B, K> b) {                                             \
    for (int k = 0; k < K; ++k) b.v[k] = a op b.v[k];                                            \
    return b;                                                                                    \
  }                                                                                              \
  template <typename B, int K>                                                                   \
  inline Wide<B, K>& operator op##=(Wide<B, K>& a, const Wide<B, K>& b) {                        \
    for (int k = 0; k < K; ++k) a.v[k] = a.v[k] op b.v[k];                                       \
    return a;                                                                                    \
  }
SDCA_WIDE_BINARY(+)
SDCA_WIDE_BINARY(-)
SDCA_WIDE_BINARY(*)
SDCA_WIDE_BINARY(/)
SDCA_WIDE_BINARY(&)
SDCA_WIDE_BINARY(|)
#undef SDCA_WIDE_BINARY

#define SDCA_WIDE_COMPARE(op)                                                                    \
  template <typename B, int K>                                                                   \
  inline Wide<MaskOf<B>, K> operator op(const Wide<B, K>& a, const Wide<B, K>& b) {              \
    Wide<MaskOf<B>, K> r;                                                                        \
    for (int k = 0; k < K; ++k) r.v[k] = a.v[k] op b.v[k];                                       \
    return r;                                                                                    \
  }                                                                                              \
  template <typename B, int K, typename S, typename = std::enable_if_t<std::is_arithmetic_v<S>>> \
  inline Wide<MaskOf<B>, K> operator op(const Wide<B, K>& a, S b) {                              \
    Wide<MaskOf<B>, K> r;                                                                        \
    for (int k = 0; k < K; ++k) r.v[k] = a.v[k] op b;                                            \
    return r;                                                                                    \
  }
SDCA_WIDE_COMPARE(<)
SDCA_WIDE_COMPARE(<=)
SDCA_WIDE_COMPARE(>)
SDCA_WIDE_COMPARE(>=)
SDCA_WIDE_COMPARE(==)
SDCA_WIDE_COMPARE(!=)
#undef SDCA_WIDE_COMPARE

template <typename B, int K>
inline Wide<B, K> operator-(Wide<B, K> a) {
  for (int k = 0; k < K; ++k) a.v[k] = -a.v[k];
  return a;
}
template <typename B, int K>
inline Wide<B, K> operator~(Wide<B, K> a) {
  for (int k = 0; k < K; ++k) a.v[k] = ~a.v[k];
  return a;
}

using dlanes::abs;
using dlanes::max;
using dlanes::min;
using dlanes::select;
using dlanes::splat;
using lanes::abs;
using lanes::max;
using lanes::min;
using lanes::select;
using lanes::splat;
template <typename B, typename M, int K>
inline Wide<B, K> select(const Wide<M, K>& m, Wide<B, K> a, const Wide<B, K>& b) {
  for (int k = 0; k < K; ++k) a.v[k] = select(m.v[k], a.v[k], b.v[k]);
  return a;
}
template <typename B, int K>
inline Wide<B, K> abs(Wide<B, K> a) {
  for (int k = 0; k < K; ++k) a.v[k] = abs(a.v[k]);
  return a;
}
template <typename B, int K>
inline Wide<B, K> min(Wide<B, K> a, const Wide<B, K>& b) {
  for (int k = 0; k < K; ++k) a.v[k] = min(a.v[k], b.v[k]);
  return a;
}
template <typename B, int K>
inline Wide<B, K> max(Wide<B, K> a, const Wide<B, K>& b) {
  for (int k = 0; k < K; ++k) a.v[k] = max(a.v[k], b.v[k]);
  return a;
}

// The scalar type and lane count of a lane vector.
template <typename V>
struct Shape;
template <>
struct Shape<dlanes::vd> {
  using T = double;
  using I = int64_t;
  static constexpr int L = dlanes::L;
};
template <>
struct Shape<lanes::vf> {
  using T = float;
  using I = int32_t;
  static constexpr int L = lanes::L;
};
template <typename B, int K>
struct Shape<Wide<B, K>> {
  using T = typename Shape<B>::T;
  using I = typename Shape<B>::I;
  static constexpr int L = K * Shape<B>::L;
};

// The lanes of x at j (count entries, the rest pad), and back; the sum of the lanes.
template <typename V, typename T>
inline V part(const T* x, int64_t j, int count, T pad) {
  constexpr int L = Shape<V>::L;
  T lane[L];
  for (int i = 0; i < L; ++i) lane[i] = i < count ? x[j + i] : pad;
  V v;
  static_assert(sizeof v == sizeof lane, "lanes");
  std::memcpy(&v, lane, sizeof v);
  return v;
}
template <typename V, typename T>
inline void put(T* x, const V& v, int count) {
  constexpr int L = Shape<V>::L;
  T lane[L];
  std::memcpy(lane, &v, sizeof v);
  for (int i = 0; i < count; ++i) x[i] = lane[i];
}
template <typename T, typename V>
inline T total(const V& v) {
  constexpr int L = sizeof(V) / sizeof(T);
  T lane[L];
  std::memcpy(lane, &v, sizeof v);
  T s = 0;
  for (int i = 0; i < L; ++i) s += lane[i];
  return s;
}
// Lanes i >= count zeroed.
template <typename V>
inline V head(const V& v, int count) {
  using I = typename Shape<V>::I;
  constexpr int L = Shape<V>::L;
  if (count >= L) return v;
  I index[L];
  for (int i = 0; i < L; ++i) index[i] = i < count ? -1 : 0;
  MaskOf<V> keep;
  std::memcpy(&keep, index, sizeof keep);
  return select(keep, v, V{});
}
// v[i] = f(x[i]) in the lanes of m, one at a time.
template <typename V, typename M, typename F>
inline V per_lane(V v, const V& x, const M& m, F f) {
  using T = typename Shape<V>::T;
  using I = typename Shape<V>::I;
  constexpr int L = Shape<V>::L;
  T values[L], args[L];
  I flags[L];
  std::memcpy(values, &v, sizeof v);
  std::memcpy(args, &x, sizeof x);
  std::memcpy(flags, &m, sizeof m);
  for (int i = 0; i < L; ++i)
    if (flags[i]) values[i] = f(args[i]);
  std::memcpy(&v, values, sizeof v);
  return v;
}

// exp on lanes: the inlined polynomial of dlanes or lanes where the result is normal, libm's
// below (subnormal results).
inline dlanes::vd exp(dlanes::vd x) {
  const dlanes::vd e = dlanes::exp(x);
  const dlanes::vl low = x < -708.0;
  return any_lane(low) ? per_lane(e, x, low, [](double a) { return std::exp(a); }) : e;
}
inline lanes::vf exp(lanes::vf x) {
  const lanes::vf e = lanes::exp(x);
  const lanes::vi low = x < -86.6f;
  return any_lane(low) ? per_lane(e, x, low, [](float a) { return std::exp(a); }) : e;
}
// The API adapter supplies its vector logarithm.
inline dlanes::vd log(dlanes::vd x) { return proximal_log(x); }
inline dlanes::vd log1p(dlanes::vd x) { return dlanes::log1p(x); }
// nextafter(q, 0) of positive normal q.
inline dlanes::vd below(dlanes::vd q) { return (dlanes::vd)((dlanes::vl)q - 1); }
template <typename B, int K>
inline Wide<B, K> exp(Wide<B, K> a) {
  for (int k = 0; k < K; ++k) a.v[k] = exp(a.v[k]);
  return a;
}
template <typename B, int K>
inline Wide<B, K> log(Wide<B, K> a) {
  for (int k = 0; k < K; ++k) a.v[k] = log(a.v[k]);
  return a;
}
template <typename B, int K>
inline Wide<B, K> log1p(Wide<B, K> a) {
  for (int k = 0; k < K; ++k) a.v[k] = log1p(a.v[k]);
  return a;
}
template <typename B, int K>
inline Wide<B, K> below(Wide<B, K> a) {
  for (int k = 0; k < K; ++k) a.v[k] = below(a.v[k]);
  return a;
}

// Vectors per wide lane group: two dependency chains (more spill the AVX2 registers; measured
// on Zen 3, two take 0.78 of one's time and 0.88 of four's).
#if defined(SDCA_PROX_CHAINS)
constexpr int kDoubleChains = SDCA_PROX_CHAINS;
constexpr int kFloatChains = SDCA_PROX_CHAINS;
#else
constexpr int kDoubleChains = 2;
constexpr int kFloatChains = 2;
#endif
using WD = Wide<dlanes::vd, kDoubleChains>;
using WF = Wide<lanes::vf, kFloatChains>;
template <typename T>
using WideOf = std::conditional_t<std::is_same_v<T, double>, WD, WF>;

template <typename V, typename T>
inline V iter5(const V& w, const V& y) {
  const V f0 = w - y, f1 = T(1) + y;
  const V f11 = f1 * f1, f0y = f0 * y;
  const V f00y = f0 * f0y;
  return w - T(4) * f0 * (T(6) * f1 * (f11 + f0y) + f00y) /
                 (f11 * (T(24) * f11 + T(36) * f0y) + f00y * (T(14) * y + f0 + T(8)));
}

template <typename V, typename T>
inline V exp_approx_1024(const V& x) {
  V y = T(1) + x / T(1024);
  for (int i = 0; i < 10; ++i) y = y * y;
  return y;
}

// lambert_w_exp_large_d on the lanes of m (x > 4 there); others keep w.
template <typename V>
inline V lambert_w_exp_large(const V& x, const MaskOf<V>& m, const V& w_in) {
  const double eps = std::numeric_limits<double>::epsilon();
  const V safe = select(m, x, V{} + 8.0);
  const V l1 = log(safe);
  V w = safe - l1;
  const V inv_x = 1.0 / safe;
  const V series =
      l1 * inv_x * (1.0 + inv_x * (0.5 * l1 - 1.0 + inv_x * (l1 * (2.0 * l1 - 9.0) + 6.0) / 6.0));
  w = select(safe < 1.0 / eps, w + series, w);
  const V tol = 2.0 * eps * safe;
  const double step_tol = std::cbrt(eps);
  MaskOf<V> active = m & (safe < std::numeric_limits<double>::infinity());
  for (int i = 0; i < 6 && any_lane(active); ++i) {
    const V f = (w - safe) + log(w);
    active = active & ~(abs(f) <= tol);
    if (!any_lane(active)) break;
    const V w1 = w + 1.0;
    const V step = f / ((w1 / w) + (f / w1) / (2.0 * w));
    const V next = w - step;
    active = active & (next != w);
    w = select(active, next, w);
    active = active & ~(abs(step) <= step_tol * w);
  }
  w = select(safe == std::numeric_limits<double>::infinity(), safe, w);
  return select(m, w, w_in);
}

// lambert_w_exp_t on lanes (double or float, as the scalar functions' branches).
template <typename V>
inline V lambert_w_exp(const V& x) {
  using T = typename Shape<V>::T;
  using M = MaskOf<V>;
  constexpr bool kDouble = std::is_same_v<T, double>;
  const T pos_cut = kDouble ? T(0) : T(-1), low_cut = kDouble ? T(-36) : T(-18);
  const T zero_cut = kDouble ? T(-746) : T(-104), big_cut = kDouble ? T(4) : T(8);
  const M pos = x > pos_cut, low = ~pos & ~(x > low_cut), big = x > big_cut;
  const V zero{};
  V w;
  if (!any_lane(~low)) {
    w = select(x > zero_cut, exp(x), zero);
  } else {
    w = select(pos, iter5<V, T>(x, zero + T(1)), exp_approx_1024<V, T>(x));
    if constexpr (kDouble) {
      const M mid = ~pos & (x > T(-20));
      if (any_lane(mid)) w = select(mid, iter5<V, T>(w, exp_approx_1024<V, T>(x - w)), w);
    }
    w = iter5<V, T>(w, exp(select(low | big, zero, x - w)));
    if (any_lane(low)) w = select(low, select(x > zero_cut, exp(x), zero), w);
  }
  if (any_lane(big)) {
    if constexpr (kDouble) {
      w = lambert_w_exp_large(x, big, w);
    } else {
      w = per_lane(w, x, big, [](float a) { return lambert_w_exp_t(a); });
    }
  }
  return select(x != x, x, w);
}

// W(y) / y = exp(-W(y)) for 0 <= y <= exp(-3.5) < 2^-5, by its series
// sum_n (-(n + 1))^n / (n + 1)! y^n, whose terms decay as (e y)^n: 15 of them are within 2^-56
// in double, 7 within 2^-27 in float.
template <typename V, typename T>
inline V lambert_ratio_small(const V& y) {
  constexpr double a[] = {1.0,
                          -1.0,
                          1.5,
                          -2.6666666666666665,
                          5.208333333333333,
                          -10.8,
                          23.343055555555555,
                          -52.01269841269841,
                          118.62522321428571,
                          -275.5731922398589,
                          649.7871723434745,
                          -1551.1605194805195,
                          3741.4497029592385,
                          -9104.500241158019,
                          22324.3085127066};
  constexpr int n = std::is_same_v<T, double> ? 15 : 7;
  V g = V{} + T(a[n - 1]);
  for (int k = n - 2; k >= 0; --k) g = g * y + T(a[k]);
  return g;
}

// prox_probability on lanes: exp(h) W(y) / y at y = alpha exp(h) where every lane has a small
// W (arg <= -3.5), else the Lambert function.
template <typename V, typename T>
inline V prox_probability(const V& h, T alpha, T log_alpha) {
  const V arg = h + log_alpha, zero{};
  const T top = std::is_same_v<T, double> ? T(700) : T(85);
  if (!any_lane(~((arg <= T(-3.5)) & (h <= top)))) {
    const V e = exp(h);
    return e * lambert_ratio_small<V, T>(alpha * e);
  }
  const T cut = -T(std::numeric_limits<T>::digits + 2) * T(0.69314718055994530942);
  const T p_cut = std::is_same_v<T, float> ? T(-9) : T(-18);
  // exp(h) where W is negligible: a zero W there takes the exp(h - W) branch.
  const auto negligible = (arg < cut) & (abs(h) >= T(0.5));
  const V w = select(negligible, zero, lambert_w_exp(select(negligible, zero, arg)));
  const auto tail = arg < p_cut;
  const V scaled = w / alpha;
  return any_lane(tail) ? select(tail, exp(select(tail, h - w, zero)), scaled) : scaled;
}

// prox_residual on the lanes of m (whose Lambert window the caller excludes): the residual r
// of alpha r - log1p(-r / q) = t, by the same safeguarded Halley steps from guess where it is
// finite (else the second-order estimate). Lanes outside m return 0.
template <typename V>
inline V prox_residual(const V& t, double alpha, const V& q, const V& log_q, const V& guess,
                       const MaskOf<V>& m) {
  using M = MaskOf<V>;
  const V lo0 = q - 1.0, zero{};
  V lo = lo0, hi = q;
  const M is_zero = t == 0.0, floor = t <= alpha * lo0 + log_q;
  M run = m & ~is_zero & ~floor;
  const V q_run = select(run, q, zero + 1.0);
  const V inv_q = 1.0 / q_run;
  const V linear = t / (alpha + inv_q);
  const V estimate = linear / (1.0 + (linear * inv_q) / (2.0 * (1.0 + alpha * q)));
  const V initial = select(guess - guess == 0.0, guess, estimate);
  V r = max(lo, min(initial, below(q_run)));
  const V tol = 8.0 * std::numeric_limits<double>::epsilon() * abs(t);
  M at_q{};
  for (int it = 0; it < 100 && any_lane(run); ++it) {
    const V f = (alpha * r - log1p(-r * inv_q)) - t;
    run = run & ~(abs(f) <= tol);
    if (!any_lane(run)) break;
    const M negative = f < 0.0;
    lo = select(run & negative, r, lo);
    hi = select(run & ~negative, r, hi);
    const V inv_qr = 1.0 / (q - r);
    const V d1 = alpha + inv_qr;
    const V denom = d1 - (f * inv_qr * inv_qr) / (2.0 * d1);
    V next = select(denom > 0.0, r - f / denom, r - f / d1);
    run = run & (next != r);
    next = select((lo <= next) & (next <= hi), next, lo + (hi - lo) / 2.0);
    const M reached = run & (next == q);
    at_q = at_q | reached;
    run = run & ~reached & (next != r);
    r = select(run, next, r);
  }
  r = select(at_q, q, r);
  r = select(floor, lo0, r);
  return select(m & ~is_zero, r, zero);
}

// The lanes i < count of a mask (true) and the double sums of float lanes.
template <typename V>
inline MaskOf<V> valid(int count) {
  using I = typename Shape<V>::I;
  constexpr int L = Shape<V>::L;
  I index[L];
  for (int i = 0; i < L; ++i) index[i] = i < count ? -1 : 0;
  MaskOf<V> m;
  std::memcpy(&m, index, sizeof m);
  return m;
}
inline WD widen(const WD& v) { return v; }
inline Wide<lanes::vd, kFloatChains> widen(const WF& v) {
  Wide<lanes::vd, kFloatChains> r;
  for (int k = 0; k < kFloatChains; ++k) r.v[k] = __builtin_convertvector(v.v[k], lanes::vd);
  return r;
}

// A lane sum in double. Float64 lanes are compensated (dlanes::Accumulator): their plain sums
// of C - 1 equal terms drift by about C / L epsilons, which moves the proximal threshold.
template <typename T>
struct LaneSum {
  decltype(widen(WideOf<T>{})) acc{};
  void add(const WideOf<T>& x) { acc += widen(x); }
  double value() const { return total<double>(acc); }
};
template <>
struct LaneSum<double> {
  dlanes::Accumulator<true> acc[kDoubleChains];
  void add(const WD& x) {
    for (int k = 0; k < kDoubleChains; ++k) acc[k].add(x.v[k]);
  }
  double value() const {
    double total = 0;
    for (const auto& chain : acc) total += chain.value();
    return total;
  }
};

// The start of prox_entropy_row's threshold dt: the softmax limit log q + log sum exp(v - v_ref),
// and where alpha (max - min) of q_j - pi_j over its probabilities pi is at most 2, one
// fixed-point step, log q + log sum exp((v - v_ref) + alpha (q_j - pi_j)) (q_j = q on the labels,
// 0 elsewhere). In double lanes; scratch holds C entries.
template <typename T>
inline T prox_start(const T* v, int64_t c, int64_t p, T v_ref, T q, T log_q, T alpha, T* scratch) {
  using dlanes::vd;
  dlanes::Accumulator<false> total;
  vd top = splat(0.0);
  for (int64_t j = 0; j < c; j += dlanes::L) {
    const int count = static_cast<int>(std::min<int64_t>(dlanes::L, c - j));
    const vd x = dlanes::part(v, j, count, -HUGE_VAL) - static_cast<double>(v_ref);
    const vd e = select(x >= -708.0, dlanes::exp(x), splat(0.0));
    total.add(e);
    top = max(top, j >= p ? e : select(dlanes::iota() + j >= p, e, splat(0.0)));
    if constexpr (std::is_same_v<T, double>) {
      dlanes::put(scratch + j, e, count);
    } else {
      for (int i = 0; i < count; ++i) scratch[j + i] = static_cast<T>(e[i]);
    }
  }
  const double sum = total.value(), qd = q, a = alpha, inv_sum = 1.0 / sum;
  const double softmax = static_cast<double>(log_q) + std::log1p(sum - 1.0);
  // The gaps q_j - pi_j: -pi_j of the others, q - pi_j of the labels.
  double high = 0.0, low = -dlanes::hmax(top) * inv_sum;
  for (int64_t j = 0; j < p; ++j) {
    const double gap = qd - static_cast<double>(scratch[j]) * inv_sum;
    high = std::max(high, gap), low = std::min(low, gap);
  }
  if (!(a * (high - low) <= 2.0)) return static_cast<T>(softmax);
  dlanes::Accumulator<false> shifted;
  for (int64_t j = 0; j < c; j += dlanes::L) {
    const int count = static_cast<int>(std::min<int64_t>(dlanes::L, c - j));
    const vd x = dlanes::part(v, j, count, -HUGE_VAL) - static_cast<double>(v_ref);
    const vd pi = dlanes::part(scratch, j, count, 0.0) * inv_sum;
    const vd target = j >= p ? splat(0.0) : select(dlanes::iota() + j < p, splat(qd), splat(0.0));
    const vd y = x + a * (target - pi);
    shifted.add(select(y >= -708.0, dlanes::exp(y), splat(0.0)));
  }
  const double fixed = static_cast<double>(log_q) + std::log(shifted.value());
  return static_cast<T>(std::isfinite(fixed) ? fixed : softmax);
}

// The sum of x over [begin, end) in double, on lanes. lane_sum and lane_penalty run once per
// row: inlined, they slowed the float64 root solve of prox_entropy_row by 5% (Apple M-series).
template <typename T>
__attribute__((noinline)) double lane_sum(const T* x, int64_t begin, int64_t end) {
  using V = WideOf<T>;
  constexpr int L = Shape<V>::L;
  LaneSum<T> acc;
  for (int64_t j = begin; j < end; j += L)
    acc.add(part<V>(x, j, static_cast<int>(std::min<int64_t>(L, end - j)), T(0)));
  return acc.value();
}

// sum_j (alpha x_j) x_j over [0, n), on lanes, summed as LaneSum.
template <typename T>
__attribute__((noinline)) T lane_penalty(const T* x, int64_t n, T alpha) {
  using V = WideOf<T>;
  constexpr int L = Shape<V>::L;
  LaneSum<T> acc;
  for (int64_t j = 0; j < n; j += L) {
    const V xj = part<V>(x, j, static_cast<int>(std::min<int64_t>(L, n - j)), T(0));
    acc.add((alpha * xj) * xj);
  }
  return static_cast<T>(acc.value());
}

// The other classes' terms of prox_entropy_row's residual sum at the threshold dt, on lanes:
// r[j] = -pi_j (clipped at one) and u[j] = 1 / (1 + alpha pi_j) (0 where clipped) over
// [begin, end); subtracts sum(pi) from sum (in double) and adds it to magnitude and other_mass,
// and the slopes of the unclipped to derivative and curvature. Whether any is clipped.
template <typename T>
inline bool prox_others(const T* v, int64_t begin, int64_t end, T v_ref, T dt, T log_q, T alpha,
                        T log_alpha, T* r, T* u, double& sum, T& derivative, T& curvature,
                        T& magnitude, T& other_mass) {
  using V = WideOf<T>;
  using M = MaskOf<V>;
  constexpr int L = Shape<V>::L;
  LaneSum<T> minus;
  V mass{}, slope{}, bend{};
  M any_clipped{};
  const V one = V{} + T(1), zero{};
  for (int64_t j = begin; j < end; j += L) {
    const int count = static_cast<int>(std::min<int64_t>(L, end - j));
    // Padding lanes have a zero probability.
    const V vj = part<V>(v, j, count, -std::numeric_limits<T>::infinity());
    V prob = prox_probability(((vj - v_ref) - dt) + log_q, alpha, log_alpha);
    const M clipped = prob > T(1);
    any_clipped = any_clipped | clipped;
    prob = select(clipped, one, prob);
    put(r + j, -prob, count);
    minus.add(prob);
    mass += prob;
    const V uj = select(clipped, zero, one / (T(1) + alpha * prob)), d1 = prob * uj;
    put(u + j, uj, count);
    slope += d1;
    bend += -((d1 * uj) * uj);
  }
  sum -= minus.value();
  const T m = total<T>(mass);
  magnitude += m;
  other_mass += m;
  derivative += total<T>(slope);
  curvature += total<T>(bend);
  return any_lane(any_clipped);
}

// The Taylor model of the probabilities about an evaluated threshold dt: at dt + d,
// pi_j (1 + s(d, u_j)) with u = 1 / (1 + alpha pi) and s = sum_k P_k(u) (-d)^k / k!, k <= 7
// (d log W(e^x) / dx = u and du / dx = u^2 (u - 1): P_1 = u, P_{k+1} = u P_k + u^2 (u - 1) P_k').
// Its remainder is below eps / 32 of pi_j for |d| <= reach (|P_8| <= 1.61 on [0, 1]), and
// the residual sum f(dt + d) = f(dt) - sum_m b_m(d) sum_j pi_j u_j^m is a polynomial in d.
constexpr int kModelOrder = 7, kModelPowers = 2 * kModelOrder;
struct ModelTable {
  double p[kModelOrder + 1][kModelPowers];
};
constexpr ModelTable model_table() {
  ModelTable t{};
  t.p[1][1] = 1;
  for (int k = 1; k < kModelOrder; ++k) {
    for (int m = 1; m + 2 < kModelPowers; ++m) {
      t.p[k + 1][m + 1] += (1 - m) * t.p[k][m];
      t.p[k + 1][m + 2] += m * t.p[k][m];
    }
  }
  return t;
}
constexpr ModelTable kModel = model_table();

template <typename T>
struct Model {
  static constexpr double reach = std::is_same_v<T, float> ? 0.25 : 0.015625;
  double moment[kModelPowers] = {};                    // sum_j pi_j u_j^m
  double b[kModelPowers] = {}, db[kModelPowers] = {};  // s and ds/dd: coefficients of u^m

  void at(double d) {
    double t[kModelOrder + 1];
    t[0] = 1;
    for (int k = 1; k <= kModelOrder; ++k) t[k] = t[k - 1] * -d / k;
    for (int m = 0; m < kModelPowers; ++m) {
      b[m] = db[m] = 0;
      for (int k = 1; k <= kModelOrder; ++k) {
        b[m] += kModel.p[k][m] * t[k];
        db[m] -= kModel.p[k][m] * t[k - 1];
      }
    }
  }
  // sum_j pi_j s_j and its derivative in d at the last at(d).
  double change() const {
    double total = 0;
    for (int m = kModelPowers - 1; m > 0; --m) total += b[m] * moment[m];
    return total;
  }
  double slope() const {
    double total = 0;
    for (int m = kModelPowers - 1; m > 0; --m) total -= db[m] * moment[m];
    return total;
  }
  // The moments of the probabilities pi = target - r with weights u over [begin, end).
  void add(const T* r, const T* u, int64_t begin, int64_t end, T target) {
    using V = WideOf<T>;
    constexpr int L = Shape<V>::L;
    V acc[kModelPowers] = {};
    for (int64_t j = begin; j < end; j += L) {
      const int count = static_cast<int>(std::min<int64_t>(L, end - j));
      const V uj = part<V>(u, j, count, T(0));
      V x = part<V>(r, j, count, target);
      x = target - x;
      for (int m = 1; m < kModelPowers; ++m) {
        x = x * uj;
        acc[m] += x;
      }
    }
    for (int m = 1; m < kModelPowers; ++m) moment[m] += total<double>(widen(acc[m]));
  }
  // The moments of the probabilities pi with weights u over [0, n).
  void add_probabilities(const T* pi, const T* u, int64_t n) {
    using V = WideOf<T>;
    constexpr int L = Shape<V>::L;
    V acc[kModelPowers] = {};
    for (int64_t j = 0; j < n; j += L) {
      const int count = static_cast<int>(std::min<int64_t>(L, n - j));
      const V uj = part<V>(u, j, count, T(0));
      V x = part<V>(pi, j, count, T(0));
      for (int m = 1; m < kModelPowers; ++m) {
        x = x * uj;
        acc[m] += x;
      }
    }
    for (int m = 1; m < kModelPowers; ++m) moment[m] += total<double>(widen(acc[m]));
  }
  // The residuals r - pi s and probabilities pi (1 + s) over [0, n) into out_r and out_pi,
  // their sum and magnitude into sum (Accumulator) and magnitude; the entries whose residual
  // would lose its relative accuracy (|pi s| > |r| / 2, or pi bound > |r| for the remainder
  // pi bound eps / 32) go to fix. Their count.
  template <typename Sum>
  int64_t expand_probabilities(const T* r, const T* pi, const T* u, int64_t n, T bound, T* out_r,
                               T* out_pi, Sum& sum, T& magnitude, int32_t* fix) const {
    using V = WideOf<T>;
    using M = MaskOf<V>;
    constexpr int L = Shape<V>::L;
    V size{};
    int64_t fixes = 0;
    for (int64_t j = 0; j < n; j += L) {
      const int count = static_cast<int>(std::min<int64_t>(L, n - j));
      const V uj = part<V>(u, j, count, T(0)), rj = part<V>(r, j, count, T(0));
      const V pj = part<V>(pi, j, count, T(0));
      V s = V{} + T(b[kModelPowers - 1]);
      for (int m = kModelPowers - 2; m > 0; --m) s = s * uj + T(b[m]);
      const V change = pj * (s * uj), next = rj - change;
      put(out_r + j, next, count), put(out_pi + j, pj + change, count);
      for (const auto& lanes_k : next.v) sum.add(lanes_k);
      size += abs(next);
      const M bad = ~(abs(change) <= abs(rj) * T(0.5)) | ~(pj * bound <= abs(rj));
      if (any_lane(bad)) {
        T changes[L];
        std::memcpy(changes, &change, sizeof change);
        for (int i = 0; i < count; ++i) {
          const T ri = std::abs(r[j + i]);
          if (!(std::abs(changes[i]) <= ri / 2 && pi[j + i] * bound <= ri))
            fix[fixes++] = static_cast<int32_t>(j + i);
        }
      }
    }
    magnitude += total<T>(size);
    return fixes;
  }
  // out = -pi (1 + s) of the other classes' pi = -r over [begin, end); adds sum(out) to sum
  // and sum |out| to magnitude.
  void expand(const T* r, const T* u, int64_t begin, int64_t end, T* out, double& sum,
              T& magnitude) const {
    using V = WideOf<T>;
    constexpr int L = Shape<V>::L;
    LaneSum<T> acc;
    V size{};
    for (int64_t j = begin; j < end; j += L) {
      const int count = static_cast<int>(std::min<int64_t>(L, end - j));
      const V uj = part<V>(u, j, count, T(0)), rj = part<V>(r, j, count, T(0));
      V s = V{} + T(b[kModelPowers - 1]);
      for (int m = kModelPowers - 2; m > 0; --m) s = s * uj + T(b[m]);
      const V next = rj + rj * (s * uj);
      put(out + j, next, count);
      acc.add(next);
      size += abs(next);
    }
    sum += acc.value();
    magnitude += total<T>(size);
  }
};
}  // namespace proxlanes

}  // namespace sdca_cpu
}  // namespace
#endif
