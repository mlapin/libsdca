#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_CORE_PROJECT_SOFTMAX_H)
#if SDCA_ISA_ID == 1
#define SDCA_CORE_PROJECT_SOFTMAX_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_CORE_PROJECT_SOFTMAX_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_CORE_PROJECT_SOFTMAX_H3 1
#else
#define SDCA_CORE_PROJECT_SOFTMAX_H4 1
#endif

#include <vector>

#include "sdca/core/project_base.h"
#include "sdca/core/lanes.h"
#include "sdca/core/math.h"
#include "sdca/core/reduce.h"

SDCA_ISA_BEGIN
namespace core {
namespace softmax {

// One lane of a type without vector lanes (long double), with the operations of Lanes that
// softmax_prox takes.
struct one_mask {
  bool v;
  static one_mask zero() {
    const one_mask m = {false};
    return m;
  }
  bool operator[](int) const { return v; }
};
inline one_mask operator&(const one_mask a, const one_mask b) {
  const one_mask m = {a.v && b.v};
  return m;
}
inline one_mask operator|(const one_mask a, const one_mask b) {
  const one_mask m = {a.v || b.v};
  return m;
}
inline one_mask operator~(const one_mask a) {
  const one_mask m = {!a.v};
  return m;
}
inline bool any(const one_mask m) { return m.v; }
inline bool all(const one_mask m) { return m.v; }

template <typename T>
struct one_lane {
  typedef T value_type;
  typedef one_mask mask_type;
  static const int L = 1;
  T v;
  static one_lane splat(const T x) {
    const one_lane a = {x};
    return a;
  }
  static one_lane zero() { return splat(T(0)); }
  static one_lane load(const T* p) { return splat(*p); }
  static one_lane load(const T* p, int, T) { return splat(*p); }
  void store(T* p) const { *p = v; }
  T operator[](int) const { return v; }
  void set(int, const T x) { v = x; }
  one_lane& operator+=(const one_lane& b) {
    v += b.v;
    return *this;
  }
  one_lane& operator*=(const one_lane& b) {
    v *= b.v;
    return *this;
  }
};

#define SDCA_ONE_LANE_BINARY(op)                                             \
  template <typename T>                                                      \
  inline one_lane<T> operator op(const one_lane<T> a, const one_lane<T> b) { \
    return one_lane<T>::splat(a.v op b.v);                                   \
  }                                                                          \
  template <typename T>                                                      \
  inline one_lane<T> operator op(const one_lane<T> a, const T b) {           \
    return one_lane<T>::splat(a.v op b);                                     \
  }                                                                          \
  template <typename T>                                                      \
  inline one_lane<T> operator op(const T a, const one_lane<T> b) {           \
    return one_lane<T>::splat(a op b.v);                                     \
  }
SDCA_ONE_LANE_BINARY(+)
SDCA_ONE_LANE_BINARY(-)
SDCA_ONE_LANE_BINARY(*)
SDCA_ONE_LANE_BINARY(/)
#undef SDCA_ONE_LANE_BINARY
#define SDCA_ONE_LANE_COMPARE(op)                                         \
  template <typename T>                                                   \
  inline one_mask operator op(const one_lane<T> a, const one_lane<T> b) { \
    const one_mask m = {a.v op b.v};                                      \
    return m;                                                             \
  }                                                                       \
  template <typename T>                                                   \
  inline one_mask operator op(const one_lane<T> a, const T b) {           \
    const one_mask m = {a.v op b};                                        \
    return m;                                                             \
  }
SDCA_ONE_LANE_COMPARE(<)
SDCA_ONE_LANE_COMPARE(<=)
SDCA_ONE_LANE_COMPARE(>)
SDCA_ONE_LANE_COMPARE(>=)
SDCA_ONE_LANE_COMPARE(==)
SDCA_ONE_LANE_COMPARE(!=)
#undef SDCA_ONE_LANE_COMPARE

template <typename T>
inline one_lane<T> operator-(const one_lane<T> a) {
  return one_lane<T>::splat(-a.v);
}
template <typename T>
inline one_lane<T> select(const one_mask m, const one_lane<T> a, const one_lane<T> b) {
  return m.v ? a : b;
}
template <typename T>
inline one_lane<T> abs(const one_lane<T> a) {
  return one_lane<T>::splat(std::abs(a.v));
}
template <typename T>
inline one_lane<T> min(const one_lane<T> a, const one_lane<T> b) {
  return a.v < b.v ? a : b;
}
template <typename T>
inline one_lane<T> max(const one_lane<T> a, const one_lane<T> b) {
  return a.v > b.v ? a : b;
}
template <typename T>
inline T sum(const one_lane<T> a) {
  return a.v;
}
template <typename T>
inline T max_lane(const one_lane<T> a) {
  return a.v;
}
template <typename T>
inline T min_lane(const one_lane<T> a) {
  return a.v;
}
template <typename T>
inline one_lane<T> exp(const one_lane<T> a) {
  return one_lane<T>::splat(std::exp(a.v));
}
template <typename T>
inline one_lane<T> expm1(const one_lane<T> a) {
  return one_lane<T>::splat(std::expm1(a.v));
}
template <typename T>
inline one_lane<T> log(const one_lane<T> a) {
  return one_lane<T>::splat(std::log(a.v));
}
template <typename T>
inline one_lane<T> log1p(const one_lane<T> a) {
  return one_lane<T>::splat(std::log1p(a.v));
}
template <typename T>
inline one_lane<T> lambert_w_exp(const one_lane<T> a) {
  return one_lane<T>::splat(core::lambert_w_exp(a.v));
}
template <typename T>
inline one_lane<T> lambert_probability(const one_lane<T> h, const T alpha, const T log_alpha) {
  return one_lane<T>::splat(core::lambert_probability(h.v, alpha, log_alpha));
}

// The lanes of T: two vectors of the instruction set (two dependency chains, which the core
// overlaps), or one_lane.
template <typename T>
struct lanes_of {
  typedef one_lane<T> type;
};
template <>
struct lanes_of<float> {
  typedef Lanes<float, isa::native, 2> type;
};
template <>
struct lanes_of<double> {
  typedef Lanes<double, isa::native, 2> type;
};

// One vector of the lanes A: the loops of the Taylor model, whose 13 moments or coefficients
// stay in registers only at that width.
template <typename A>
struct one_chain;
template <typename T, typename Isa, int K>
struct one_chain<Lanes<T, Isa, K>> {
  typedef Lanes<T, Isa, 1> type;
};
template <typename T>
struct one_chain<one_lane<T>> {
  typedef one_lane<T> type;
};

// Sums of lanes in double at least: Accumulator, compensated in T for one_lane.
template <typename T>
struct one_lane_sum {
  compensated_sum<T> s;
  void add(const one_lane<T> x) { s.add(x.v); }
  T value() const { return s.value(); }
};
template <typename A>
struct lane_sum {
  typedef Accumulator<A> type;
};
template <typename T>
struct lane_sum<one_lane<T>> {
  typedef one_lane_sum<T> type;
};

// The sum of the lanes, float lanes in double.
template <typename Isa, int K>
SDCA_LANE double total(const Lanes<float, Isa, K>& a) {
  return sum(to_double(a));
}
template <typename Isa, int K>
SDCA_LANE double total(const Lanes<double, Isa, K>& a) {
  return sum(a);
}
template <typename T>
inline T total(const one_lane<T> a) {
  return a.v;
}

// The coefficients of model's P_k(u) in the powers u^m (nonzero for k <= m <= 2k - 1).
const double model_table[8][14] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, -2, 3, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 6, -20, 15, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, -24, 130, -210, 105, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 120, -924, 2380, -2520, 945, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, -720, 7308, -26432, 44100, -34650, 10395}};

// The Taylor model of the proximal probabilities about an evaluated threshold omega: at
// omega + d, pi_j (1 + s(d, u_j)) with u = 1 / (1 + alpha pi) and
// s = sum_k P_k(u) (-d)^k / k!, k <= 7 (d log W(e^x) / dx = u and du / dx = u^2 (u - 1):
// P_1 = u, P_{k+1} = u P_k + u^2 (u - 1) P_k'). Its remainder |P_8| d^8 / 8! of pi_j is below
// eps / 32 for |d| <= reach, the largest power of two that keeps it there (|P_8| <= 1.62 on
// [0, 1], 1.6123 at its peak; tests/python/test_limits.py): 2^-2 for float, 2^-6 for double.
// The residual sum is a polynomial in d.
const int kModelOrder = 7;
constexpr double kModelRemainder = 1.62;
constexpr double factorial(const int n) { return n <= 1 ? 1 : n * factorial(n - 1); }
template <typename T>
constexpr T model_reach(const int r = 0) {
  return kModelRemainder * limits::detail::pow2<double>(-(kModelOrder + 1) * r) /
                     factorial(kModelOrder + 1) <=
                 double(limits::eps<T>()) / 32
             ? limits::detail::pow2<T>(-r)
             : model_reach<T>(r + 1);
}
template <typename T>
struct model {
  static const int order = kModelOrder, powers = 2 * order;
  // A constant member: a call of the recursive model_reach would be evaluated at run time.
  static constexpr T kReach = model_reach<T>();
  static constexpr T reach() { return kReach; }
  double moment[powers];  // sum_j pi_j u_j^m
  double c[order + 1];    // sum_j pi_j P_k(u_j)
  double b[powers];       // s: the coefficients of u^m

  model() : moment(), c(), b() {}

  // The moments of pi with weights u over [0, n), n whole lanes A, summed in lanes of T.
  template <typename A>
  void add(const T* pi, const T* u, const std::size_t n) {
    A acc[powers];
    for (int m = 0; m < powers; ++m) acc[m] = A::zero();
    for (std::size_t j = 0; j < n; j += A::L) {
      const A uj = A::load(u + j);
      A x = A::load(pi + j);
      for (int m = 1; m < powers; ++m) {
        x = x * uj;
        acc[m] += x;
      }
    }
    for (int m = 1; m < powers; ++m) moment[m] += double(total(acc[m]));
    for (int k = 1; k <= order; ++k) {
      c[k] = 0;
      for (int m = k; m < 2 * k; ++m) c[k] += model_table[k][m] * moment[m];
    }
  }
  // The coefficients b of s at d (for relative).
  void at(const double d) {
    double t[order + 1];
    t[0] = 1;
    for (int k = 1; k <= order; ++k) t[k] = t[k - 1] * -d / k;
    for (int m = 0; m < powers; ++m) {
      b[m] = 0;
      for (int k = (m + 2) / 2; k <= m && k <= order; ++k) b[m] += model_table[k][m] * t[k];
    }
  }
  // sum_j pi_j s_j at d and its derivative in d.
  void series(const double d, double& change, double& slope) const {
    double t = 1;
    change = slope = 0;
    for (int k = 1; k <= order; ++k) {
      slope += c[k] * t;
      t = t * -d / k;
      change += c[k] * t;
    }
  }
  // s(u) u at the last at(d).
  template <typename A>
  A relative(const A& u) const {
    A s = A::splat(T(b[powers - 1]));
    for (int m = powers - 2; m > 0; --m) s = s * u + T(b[m]);
    return s * u;
  }
  // Newton steps on f - change(d) from d, a start within a factor two of the root (one bit):
  // the quadratic steps of steps_of_order and one to confirm, until a step is below the
  // rounding of d in T. False where a step is not finite.
  bool newton(const double f, double& d) const {
    const double u = double(limits::roundoff<T>());
    const int steps = std::integral_constant<int, limits::steps_of_order<T>(2, 1) + 1>::value;
    for (int it = 0; it < steps; ++it) {
      double change, slope;
      series(d, change, slope);
      const double step = (f - change) / slope;
      if (!std::isfinite(step)) return false;
      d -= step;
      if (std::abs(step) <= u * std::abs(d)) break;
    }
    return true;
  }
};

// The convergence test of the residual sum, |sum r| <= tolerance sum |r| (also of the Triton
// and composed solves): the error of the compensated sum (2 u sum |r|) and of the entries in
// the window (one Newton step from the second-order estimate or from p - pi: a few roundings
// u |r_j| each), 8 u, with a factor two of margin. Where the entries outside the window carry
// more (W's error), a step below the resolution of omega ends the solve instead.
template <typename T>
constexpr T tolerance() {
  return 16 * limits::roundoff<T>();
}

// The residual r = q - pi of one entry at the shifted score t: alpha r - log1p(-r / q) = t
// for r in [q - 1, q), by safeguarded Halley steps from guess where it is finite (else a
// second-order estimate). The steps are accurate inside the window |r| < q / 2.
template <typename T>
inline T residual(const T t, const T alpha, const T q, const T log_q, const T guess) {
  if (t == 0) return 0;
  T lo = q - 1, hi = q;
  if (t <= alpha * lo + log_q) return lo;
  const T inv_q = T(1) / q, tol = T(8) * std::numeric_limits<T>::epsilon() * std::abs(t);
  T initial = guess;
  if (!(guess - guess == 0)) {
    const T linear = t / (alpha + inv_q);
    initial = linear / (T(1) + (linear * inv_q) / (T(2) * (T(1) + alpha * q)));
  }
  T r = std::max(lo, std::min(initial, std::nextafter(q, T(0))));
  for (int it = 0; it < root_steps<T>::total(); ++it) {
    const T f = (alpha * r - std::log1p(-r * inv_q)) - t;
    if (std::abs(f) <= tol) break;
    (f < 0 ? lo : hi) = r;
    const T inv_qr = T(1) / (q - r), d1 = alpha + inv_qr;
    const T denom = d1 - (f * inv_qr * inv_qr) / (T(2) * d1);
    T next = it >= root_steps<T>::free_steps() ? bisect(lo, hi)
             : denom > 0                       ? r - f / denom
                                               : r - f / d1;
    // A step below the resolution of r ends it: a subnormal r misses tol.
    if (next == r) break;
    if (!(lo <= next && next <= hi)) next = bisect(lo, hi);
    if (next == q) return q;
    if (next == r) break;
    r = next;
  }
  return r;
}

}  // namespace softmax

/**
 * The entropic proximal step of a softmax toward a target distribution p:
 *    pi = argmin_x  <x, log x> - <v, x> + (alpha / 2) ||x - p||^2,  <1, x> = 1,  x >= 0,
 * that is alpha pi_j + log pi_j = v_j + alpha p_j - t (pi = softmax(v + alpha (p - pi))).
 * It is solved for the threshold omega = t - log sum exp(v) in the residuals r = p - pi, with
 * q = softmax(v): every entry from its Lambert probability pi_j = W(alpha exp(h)) / alpha at
 * h = log q_j + alpha p_j - omega, and, near its target (p_j > 0 and |omega - log(q_j / p_j)|
 * < (2 + alpha p_j) / 8), r_j by one Newton step on alpha r - log1p(-r / p) =
 * omega - log(q_j / p_j). Safeguarded Halley steps on sum(r(omega)) = 0; near the root the
 * Taylor model of the probabilities gives the root without another evaluation. The steps
 * start at fixed-point steps from pi = q where alpha (max(q - p) - min(q - p)) is small (see
 * start).
 *
 * The loops run on lanes (softmax::lanes_of: two vectors of the instruction set for float and
 * double, one entry for other types) over the arrays padded to whole lanes; probabilities
 * below the normal range of float or double lanes are +0.0. The SDCA steps of the entropy
 * objectives, the proximal cross-entropy and distillation losses, prox_two_entropy and the
 * top-1 entropy projection all take this solver.
 **/
template <typename T>
struct softmax_prox {
  typedef typename softmax::lanes_of<T>::type A;
  typedef typename A::mask_type M;
  typedef typename softmax::lane_sum<A>::type LaneSum;
  typedef typename softmax::one_chain<A>::type B;
  // Sums in double at least: a float sum of many classes misplaces the root.
  typedef typename std::common_type<T, double>::type Sum;

  std::size_t n;
  std::size_t width;  // n rounded up to whole lanes: the arrays' entries past n are padding
  T alpha, log_alpha;
  T v_max, log_norm;  // lse(v) = v_max + log_norm
  T omega;            // the threshold after solve()
  // Inputs (set_scores): p, log p, log(q / p) (+inf where p = 0), q - p, q and
  // log q + alpha p; work: r, the model's weights 1 / (1 + alpha pi) (the slopes are pi times
  // them), pi and the model's residuals.
  T *p, *log_p, *ratio, *u, *q, *score, *r, *weights, *pi, *next_r;
  std::size_t pivot;  // the entry of largest p + q

  explicit softmax_prox(const std::size_t n_, const T alpha_)
      : storage_(10 * whole(n_)), index_(n_) {
    reset(n_, alpha_);
  }

  // A step of n entries and scale alpha in this solver's buffers (grown, never shrunk, and
  // outside the per-thread pool): one solver per thread, thread_local, serves every row.
  void reset(const std::size_t n_, const T alpha_) {
    if (n_ < 1 || !(alpha_ > 0) || !std::isfinite(alpha_))
      throw std::invalid_argument("Invalid entropy block dimensions or scale");
    n = n_;
    width = whole(n_);
    alpha = alpha_;
    log_alpha = std::log(alpha_);
    v_max = log_norm = omega = 0;
    pivot = 0;
    storage_.resize(10 * width);
    index_.resize(n_);
    T* w = storage_.data();
    T** parts[] = {&p, &log_p, &ratio, &u, &q, &score, &r, &weights, &pi, &next_r};
    for (std::size_t i = 0; i < 10; ++i) *parts[i] = w + i * width;
  }

  // The scores v (finite) and the target (nonnegative, summing to one); target(j) gives p_j.
  template <typename Target>
  void set_scores(const T* v, Target target) {
    const T inf = std::numeric_limits<T>::infinity();
    A top = A::splat(-inf);
    M bad = M::zero();
    for (std::size_t j = 0; j < width; j += A::L) {
      const int count = lanes(j);
      const A x = A::load(v + j, count, v[0]);
      top = max(top, x);
      bad = bad | ~(x - x == T(0));
    }
    if (any(bad)) throw std::invalid_argument("Entropy scores must be finite");
    v_max = max_lane(top);
    // sum q = n + e: n the entries near the largest (x = v - v_max >= -1/2), e the sum of
    // their expm1(x) and of the others' exp(x). log(q_j / p_j) = x_j - log(n p_j) -
    // log1p(e / n) keeps both the scores' small differences near the top and the mass of a
    // far tail, which log(sum q) would round away; e rounds relative to its own terms, which
    // are small where those matter, so the sum needs no compensation.
    LaneSum excess;
    A near_count = A::zero();
    for (std::size_t j = 0; j < width; j += A::L) {
      const int count = lanes(j);
      const A x = A::load(v + j, count, -inf) - v_max, e = exp(x);
      e.store(q + j);
      const M near = x >= T(-0.5);
      if (any(near)) {
        excess.add(select(near, expm1(x), e));
        near_count += select(near, A::splat(T(1)), A::zero());
      } else {
        excess.add(e);
      }
    }
    const T count_near = sum(near_count);
    const Sum e = Sum(excess.value());
    const T tail = T(std::log1p(e / Sum(count_near))), log_count = std::log(count_near);
    const T inv_q = T(1 / (Sum(count_near) + e));
    log_norm = log_count + tail;
    for (std::size_t j = 0; j < n; ++j) p[j] = T(target(j));
    pad();
    A best = A::splat(-inf);
    for (std::size_t j = 0; j < width; j += A::L) {
      const int count = lanes(j);
      const A pj = A::load(p + j);
      const A x = A::load(v + j, count, v_max) - v_max;
      const M positive = pj > T(0);
      A lp = A::splat(-inf), lr = A::splat(inf);
      if (all(positive)) {
        const A lnp = log(count_near * pj);
        lr = (x - lnp) - tail;
        lp = lnp - log_count;
      } else if (any(positive)) {
        for (int i = 0; i < A::L; ++i)
          if (positive[i]) {
            const T lnp = std::log(count_near * pj[i]);
            lr.set(i, (x[i] - lnp) - tail);
            lp.set(i, lnp - log_count);
          }
      }
      input(j, pj, lp, x - log_norm, lr, A::load(q + j) * inv_q, best);
    }
    pad();
    find_pivot(best);
  }

  // The inputs of set_scores from a caller's own statistics of q = softmax(v) and the target
  // p, with lse(v) = v_max + log_norm: entry(j, p_j, log p_j, log q_j, log(q_j / p_j), q_j)
  // fills them for j = 0, 1, ..., n - 1 in order (log p_j = -inf and the ratio +inf where
  // p_j = 0).
  template <typename Entry>
  void set_inputs(const T v_max_, const T log_norm_, Entry entry) {
    v_max = v_max_, log_norm = log_norm_;
    for (std::size_t j = 0; j < n; ++j) entry(j, p[j], log_p[j], score[j], ratio[j], q[j]);
    pad();
    A best = A::splat(-std::numeric_limits<T>::infinity());
    for (std::size_t j = 0; j < width; j += A::L)
      input(j, A::load(p + j), A::load(log_p + j), A::load(score + j), A::load(ratio + j),
            A::load(q + j), best);
    pad();
    find_pivot(best);
  }

  // The target uniform on the labels (mask(j) true), zero elsewhere.
  template <typename Mask>
  void set_labels(const T* v, const std::size_t labels, Mask mask) {
    const T share = T(1) / T(labels);
    set_scores(v, [&](std::size_t j) { return mask(j) ? share : T(0); });
  }

  // The threshold omega (the root of sum r = 0); leaves its residuals in r. prior, if given,
  // holds residuals of a nearby solution (an SDCA block's current duals over C) to start from.
  T solve(const T* prior = nullptr) {
    const T tiny = std::numeric_limits<T>::min();
    T lo, hi;
    bracket(lo, hi);
    omega = prior ? warm_start(prior, lo, hi) : start(lo, hi);
    bool solved = lo == 0 && hi == 0;
    if (solved) evaluate(omega);
    for (int it = 0; it < root_steps<T>::total() && !solved; ++it) {
      const sums e = evaluate(omega);
      solved = true;
      if (std::abs(e.f) <= softmax::tolerance<T>() * std::max(e.magnitude, tiny)) {
        finish(e);
        break;
      }
      (e.f < 0 ? lo : hi) = omega;
      const bool free = it < root_steps<T>::free_steps();
      T next = free ? step(omega, e, lo, hi) : bisect(lo, hi);
      if (free && !e.clipped && std::abs(next - omega) <= 2 * softmax::model<T>::reach()) {
        T delta = next - omega;
        if (model_root(delta, e.f, lo, hi)) break;
        if (std::isfinite(delta) && delta != 0 && lo <= omega + delta && omega + delta <= hi)
          next = omega + delta;
      }
      if (next == omega) break;
      omega = next;
      solved = false;
    }
    // The residuals at omega (the last evaluation's, unless the steps ran out).
    if (!solved) evaluate(omega);
    if (!std::isfinite(omega)) throw std::runtime_error("Entropy normalization did not converge");
    return omega;
  }

  // The log-normalizer t of the solution: alpha pi_j + log pi_j = v_j + alpha p_j - t.
  T normalizer() const { return v_max + relative_normalizer(); }
  T relative_normalizer() const { return log_norm + omega; }

  // log pi_j: log p + log1p(-r / p) where pi is near p, log(p - r) far above it (where -r / p
  // can overflow), else h - W of the Lambert inversion (p - r would cancel).
  T log_probability(const std::size_t j) const {
    const T pj = p[j] - r[j];
    if (p[j] > 0 && r[j] < -p[j] / 2) return std::log(pj);
    if (p[j] > 0 && !(r[j] > p[j] / 2)) return log_p[j] + std::log1p(-r[j] / p[j]);
    if (p[j] == 0 && pj >= std::numeric_limits<T>::min()) return std::log(pj);
    const T h = score[j] - omega;
    return h - lambert_w_exp(h + log_alpha);
  }

  // The proximal loss min_x <p, -log x> ... at the solution: -<p, log pi> + (alpha / 2)
  // ||r||^2, both sums compensated and nonnegative (log_probability on lanes).
  T value() const {
    LaneSum cross, penalty;
    for (std::size_t j = 0; j < width; j += A::L) {
      const A pj = A::load(p + j), rj = A::load(r + j);
      penalty.add((alpha * rj) * rj);
      const M positive = pj > T(0);
      if (!any(positive)) continue;
      if (!all(positive)) {
        A terms = A::zero();
        for (int i = 0; i < A::L; ++i)
          if (positive[i]) terms.set(i, -(pj[i] * log_probability(j + std::size_t(i))));
        cross.add(terms);
        continue;
      }
      const M above = positive & (rj < -pj / T(2)), near = positive & ~above & ~(rj > pj / T(2));
      const M far = positive & ~above & ~near;
      const A one = A::splat(T(1));
      A lp = A::zero();
      if (any(above)) lp = select(above, log(select(above, pj - rj, one)), lp);
      if (any(near))
        lp = select(near, A::load(log_p + j) + log1p(select(near, -rj / pj, A::zero())), lp);
      if (any(far)) {
        const A h = A::load(score + j) - omega;
        lp = select(far, h - lambert_w_exp(select(far, h + log_alpha, A::zero())), lp);
      }
      cross.add(select(positive, -(pj * lp), A::zero()));
    }
    const T inf = std::numeric_limits<T>::infinity();
    const T c = T(cross.value()), s = T(penalty.value());
    if (!(c < inf) || !(s < inf)) return inf;
    return c + s / 2;
  }

  // r with the pivot restoring the zero sum of the others.
  void balance() {
    LaneSum others;
    for (std::size_t j = 0; j < width; j += A::L) {
      A x = A::load(r + j);
      if (pivot >= j && pivot - j < std::size_t(A::L)) x.set(int(pivot - j), T(0));
      others.add(x);
    }
    r[pivot] = -T(others.value());
  }

  // The dual block c r into out (Data), zero-sum up to one rounding in Data: the pivot is
  // formed from the stored others. An entry without target whose probability is below the
  // normal range (zero in lanes) takes -exp(log c + log pi), rounded once: normal where c
  // lifts it, subnormal rather than zero where it does not. The solve did not see the mass so
  // recovered; the targets take it in the proportions in which a shift d of the threshold
  // moves them to first order: alpha r_j - log1p(-r_j / p_j) = d gives
  // r_j = d p_j / (1 + alpha p_j).
  template <typename Data>
  void duals(const T c, Data* out) const {
    const T tiny = std::numeric_limits<T>::min();
    Sum recovered = 0;
    for (std::size_t j = 0; j < n; ++j) {
      out[j] = Data(c * r[j]);
      if (r[j] > -tiny && p[j] == 0) {
        out[j] = Data(-std::exp(std::log(c) + log_probability(j)));
        recovered -= Sum(out[j]) - Sum(c * r[j]);
      }
    }
    if (recovered != 0) {
      Sum shares = 0;
      for (std::size_t j = 0; j < n; ++j)
        if (p[j] > 0) shares += Sum(p[j]) / (1 + Sum(alpha) * Sum(p[j]));
      for (std::size_t j = 0; j < n; ++j)
        if (p[j] > 0)
          out[j] =
              Data(Sum(c * r[j]) + recovered * (Sum(p[j]) / (1 + Sum(alpha) * Sum(p[j]))) / shares);
    }
    row_sum<Data, Sum> others;
    for (std::size_t j = 0; j < n; ++j)
      if (j != pivot) others.add(Sum(out[j]));
    out[pivot] = Data(-others.value());
  }

  // The normalized probabilities: p - r where r is small, the Lambert inversion where pi is
  // (never exp of the threshold equation, whose O(alpha) terms cancel).
  template <typename Out>
  void probabilities(Out* out) {
    A mass = A::zero();
    for (std::size_t j = 0; j < width; j += A::L) {
      const A pj = A::load(p + j), rj = A::load(r + j);
      A prob = pj - rj;
      const M far = (pj > T(0)) & (rj > pj / T(2));
      if (any(far))
        prob = select(far, lambert_probability(A::load(score + j) - omega, alpha, log_alpha), prob);
      prob = min(A::splat(T(1)), prob);
      prob.store(pi + j);
      mass += prob;
    }
    const T total_mass = sum(mass);
    if (!(total_mass > 0) || !std::isfinite(total_mass))
      throw std::runtime_error("Invalid entropy probability normalization");
    for (std::size_t j = 0; j < n; ++j) out[j] = Out(pi[j] / total_mass);
  }

 private:
  struct sums {
    T f, magnitude, d1, d2;
    bool clipped;
  };
  row_buffer<T> storage_;
  row_buffer<std::size_t> index_;

  // n rounded up to whole lanes.
  static std::size_t whole(const std::size_t n_) { return (n_ + A::L - 1) / A::L * A::L; }
  int lanes(const std::size_t j) const { return int(std::min<std::size_t>(A::L, n - j)); }

  // The entries [j, j + L) of the inputs; best, the largest p + q so far, picks the pivot
  // (padding has zero p and q).
  void input(const std::size_t j, const A& pj, const A& lp, const A& log_q, const A& lr,
             const A& qj, A& best) {
    A gap = qj - pj;
    // q - p from the ratio where q is near p: their difference would cancel.
    const M near = abs(lr) < T(1);
    if (any(near)) gap = select(near, pj * expm1(select(near, lr, A::zero())), gap);
    pj.store(p + j);
    qj.store(q + j);
    lp.store(log_p + j);
    lr.store(ratio + j);
    gap.store(u + j);
    (log_q + alpha * pj).store(score + j);
    best = max(best, pj + qj);
  }
  // The inputs' padding: no target, no probability, a zero residual in every pass.
  void pad() {
    const T inf = std::numeric_limits<T>::infinity();
    for (std::size_t j = n; j < width; ++j) {
      p[j] = log_p[j] = u[j] = q[j] = r[j] = pi[j] = weights[j] = 0;
      ratio[j] = inf;
      score[j] = -inf;
    }
  }
  // The first entry of the largest p + q.
  void find_pivot(const A& best) {
    const T most = max_lane(best);
    pivot = 0;
    while (pivot + 1 < n && !(p[pivot] + q[pivot] == most)) ++pivot;
  }

  // The residuals, probabilities and weights at omega = at, and the sums of the residuals,
  // their magnitudes, the slopes pi / (1 + alpha pi) and their derivatives. Outside the
  // window r = p - pi keeps r within 9 rounding errors of pi (|r| >= p / 8 there); inside, one
  // Newton step gives r to its relative accuracy: from p - pi (error eps p, after the step
  // eps^2 p) where |arg| >= cbrt(eps), else from the second-order estimate in arg (error
  // |arg|^3 p, then |arg|^6 p); relative to |r| ~ |arg| p these are eps^2 / |arg| and
  // |arg|^5, which meet at |arg|^6 = eps^2.
  // Rows of at most 16 entries on one vector (less padding), the others on A.
  sums evaluate(const T at) {
    if (n <= 16) return evaluate_on<B>(at, (n + B::L - 1) / B::L * B::L);
    return evaluate_on<A>(at, width);
  }
  template <typename V>
  sums evaluate_on(const T at, const std::size_t end) {
    typedef typename V::mask_type N;
    const T tiny = std::numeric_limits<T>::min();
    constexpr T cut = limits::cbrt_eps<T>();
    const V one = V::splat(T(1));
    N clipped = N::zero();
    typename softmax::lane_sum<V>::type f;
    V magnitude = V::zero(), d1 = V::zero(), d2 = V::zero();
    for (std::size_t j = 0; j < end; j += V::L) {
      // Padding lanes have a zero probability and residual.
      const V h = V::load(score + j) - at;
      const V pj = V::load(p + j), arg = at - V::load(ratio + j);
      const V prob = lambert_probability(h, alpha, log_alpha);
      clipped = clipped | (prob > T(1));
      V pij = min(prob, one), rj = pj - pij;
      const N window = (pj > tiny) & (abs(arg) < (T(2) + alpha * pj) * T(0.125));
      if (any(window)) {
        const V qj = select(window, pj, one), a = select(window, arg, V::zero());
        const V linear = a / (alpha + one / qj);
        const V estimate = linear / (T(1) + (linear / qj) / (T(2) * (T(1) + alpha * qj)));
        const V r0 = select(abs(a) < cut, estimate, select(window, rj, V::zero()));
        const V g = (alpha * r0 - log1p(-r0 / qj)) - a;
        const V next = r0 - g / (alpha + one / (qj - r0));
        rj = select(window, next, rj);
        pij = select(window, pj - next, pij);
      }
      // A zero argument has a zero residual, also where p is subnormal.
      rj = select(arg == T(0), V::zero(), rj);
      const V w = one / (T(1) + alpha * pij), s = pij * w;
      rj.store(r + j);
      pij.store(pi + j);
      w.store(weights + j);
      f.add(rj);
      magnitude += abs(rj);
      d1 += s;
      d2 += -(s * (w * w));
    }
    const sums result = {T(f.value()), sum(magnitude), sum(d1), sum(d2), any(clipped)};
    return result;
  }

  // The bracket of omega: -alpha max(q - p, 0) and -alpha min(q - p, 0).
  void bracket(T& lo, T& hi) const {
    A u_max = A::zero(), u_min = A::zero();
    for (std::size_t j = 0; j < width; j += A::L) {
      const A x = A::load(u + j);
      u_max = max(u_max, x);
      u_min = min(u_min, x);
    }
    lo = -alpha * max_lane(u_max), hi = -alpha * min_lane(u_min);
    if (!(lo <= hi)) hi = lo;
  }

  // 0, or fixed-point steps from pi = q: log sum q exp(alpha (p - q)) where
  // alpha (max u - min u) <= 2 (double), and from its pi (exp(-alpha u) kept in next_r)
  // log sum q exp(alpha (p - pi)) where alpha (max u - min u) <= 1 (float, whose first
  // evaluation then often meets the tolerance without the model).
  T start(const T lo, const T hi) {
    const bool single = sizeof(T) < sizeof(double);
    if (!(hi - lo <= (single ? T(1) : T(2)))) return 0;
    LaneSum first;
    for (std::size_t j = 0; j < width; j += A::L) {
      const A e = exp(-alpha * A::load(u + j));
      if (single) e.store(next_r + j);
      first.add(A::load(q + j) * e);
    }
    const T total = T(first.value());
    T at = T(std::log(total));
    if (single) {
      LaneSum second;
      const T inv_first = T(1) / total;
      for (std::size_t j = 0; j < width; j += A::L) {
        const A qj = A::load(q + j), pj = A::load(p + j);
        const A pij = (qj * A::load(next_r + j)) * inv_first;
        second.add(qj * exp(alpha * (pj - pij)));
      }
      at = T(std::log(second.value()));
    }
    return lo <= at && at <= hi ? at : T(0);
  }

  // The threshold at which the prior residuals solve the equation of their most probable
  // entry, alpha pi + log pi = score - omega; the ordinary start where that is not finite or
  // outside the bracket.
  T warm_start(const T* prior, const T lo, const T hi) {
    std::size_t best = 0;
    T best_pi = -std::numeric_limits<T>::infinity();
    bool zero = true;
    for (std::size_t j = 0; j < n; ++j) {
      zero = zero && prior[j] == 0;
      const T pj = p[j] - prior[j];
      if (pj > best_pi) best_pi = pj, best = j;
    }
    if (zero || !(best_pi > 0) || !(best_pi < 1)) return start(lo, hi);
    const T at = (score[best] - std::log(best_pi)) - alpha * best_pi;
    return lo <= at && at <= hi ? at : start(lo, hi);
  }

  // The next threshold of a safeguarded Halley step from at (the bracket's midpoint where
  // the step leaves it), or at where the steps stall.
  static T step(const T at, const sums& e, const T lo, const T hi) {
    const T newton = -e.f / e.d1, halley = (e.f * e.d2) / (2 * e.d1 * e.d1);
    T delta =
        std::isfinite(halley) ? newton / (1 - std::min(std::max(halley, T(-0.5)), T(0.5))) : newton;
    if (!(e.d1 > 0 && std::isfinite(e.d1) && std::isfinite(newton))) delta = 0;
    const T next = at + delta;
    if (delta != 0 && next == at) return at;
    if (!(lo <= next && next <= hi && next != at && std::isfinite(next))) return bisect(lo, hi);
    return next;
  }

  // Within the tolerance: the residuals at the Newton root omega - f / f', to first order.
  void finish(const sums& e) {
    const T delta = -e.f / e.d1;
    if (delta != 0 && std::isfinite(delta))
      for (std::size_t j = 0; j < n; ++j) r[j] += delta * (pi[j] * weights[j]);
  }

  // The root of the Taylor model about the last evaluation at omega (sum f) from its step
  // delta: true with omega and r there where the model's residuals meet the tolerance (those
  // it would not keep to their relative accuracy solved again); else false with its step in
  // delta (where finite).
  bool model_root(T& delta, const T f, const T lo, const T hi) {
    const T tiny = std::numeric_limits<T>::min();
    const T reach = softmax::model<T>::reach();
    softmax::model<T> m;
    m.template add<B>(pi, weights, width);
    double d = double(delta);
    if (!m.newton(double(f), d)) return false;
    // The model's root, and once more after the Newton step of its re-solved residuals; a
    // root that still misses the tolerance is left to the safeguarded steps.
    for (int attempt = 0; attempt < 2; ++attempt) {
      const T next = omega + T(d);
      if (!(lo <= next && next <= hi)) return false;
      delta = next - omega;
      if (!(std::abs(delta) <= reach)) return false;
      d = double(delta);
      m.at(d);
      // The remainder relative to eps / 32 of the probability.
      const T bound = T(std::pow(std::abs(d) / double(reach), softmax::model<T>::order + 1));
      std::size_t* fix = index_.data();
      std::size_t fixes = 0;
      typename softmax::lane_sum<B>::type total;
      B size = B::zero();
      for (std::size_t j = 0; j < width; j += B::L) {
        const B rj = B::load(r + j), pij = B::load(pi + j);
        const B change = pij * m.relative(B::load(weights + j));
        const B next_rj = rj - change;
        next_rj.store(next_r + j);
        total.add(next_rj);
        size += abs(next_rj);
        const typename B::mask_type kept =
            (abs(change) <= abs(rj) * T(0.5)) & (pij * bound <= abs(rj));
        if (!all(kept))
          for (int i = 0; i < B::L; ++i)
            if (!kept[i]) fix[fixes++] = j + std::size_t(i);
      }
      T sum_r = T(total.value()), sum_size = sum(size);
      if (fixes > 0) {
        for (std::size_t k = 0; k < fixes; ++k) {
          const std::size_t j = fix[k];
          const T arg = next - ratio[j], half = alpha * p[j] / 2, old = next_r[j];
          if (p[j] > tiny && arg <= half + T(0.69314718055994530942) &&
              arg >= -half - T(0.40546510810816438198)) {
            next_r[j] = softmax::residual(arg, alpha, p[j], log_p[j], old);
          } else {
            const T pj = std::min(lambert_probability(score[j] - next, alpha, log_alpha), T(1));
            next_r[j] = arg == 0 ? T(0) : p[j] - pj;
          }
          sum_size += std::abs(next_r[j]) - std::abs(old);
        }
        LaneSum again;
        for (std::size_t j = 0; j < width; j += A::L) again.add(A::load(next_r + j));
        sum_r = T(again.value());
      }
      if (std::abs(sum_r) <= softmax::tolerance<T>() * std::max(sum_size, tiny)) {
        std::copy(next_r, next_r + n, r);
        omega = next;
        return true;
      }
      double change, slope;
      m.series(d, change, slope);
      d -= double(sum_r) / slope;
    }
    delta = T(d);
    return false;
  }
};

}  // namespace core
SDCA_ISA_END

#endif  // SDCA_CORE_PROJECT_SOFTMAX_H
