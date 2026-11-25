#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_LOSS_DISTILL_H)
#if SDCA_ISA_ID == 1
#define SDCA_LOSS_DISTILL_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_LOSS_DISTILL_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_LOSS_DISTILL_H3 1
#else
#define SDCA_LOSS_DISTILL_H4 1
#endif

// Distillation rows (sdca/loss/row.h) of student scores z and teacher scores t at temperature
// T, with p = softmax(t / T) and q = softmax(z / T): distillation is H(p, q) = -sum_j p_j log
// q_j with the gradient (q - p) / T; proximal_distillation is the Moreau envelope with step
// gamma of H(p, softmax(.)) at z / T, with the gradient (pi - p) / T at the envelope's optimum
// pi (core::softmax_prox).
//
// One pass from the maxima m_z, m_t and the first classes that take them, in float and double
// work alike: s = (z - m_z) / T, t' = (t - m_t) / T and the log-ratio d = ((z - m_z) - (t -
// m_t)) / T with the rounding errors of both differences, exact to its own rounding at any
// offset between the rows. Where |d| < 1, v = exp(t') expm1(d) and exp(s) = exp(t') + v;
// elsewhere v = exp(s) - exp(t'), terms a factor e apart. The exponentials are times 2^64
// (core::Wide<W>), exact down to the subnormal probabilities of the work type. The loss is
// log(1 + tail(s)) - sum_j p_j s_j (the tails without their first largest term); C = sum_j p_j
// expm1(d_j) = sum(v) / sum(exp(t')), or (1 + tail(s)) / (1 + tail(t')) - 1 below -1/2; q - p
// = p (expm1(d) - C) / (1 + C) = (v - exp(t') C) / sum(exp(s)) entry by entry, within the
// rounding of p (|expm1(d)| + |C|) (Barron et al.'s log-space difference), and the gradient
// takes q - p less q sum(q - p), rows summing to zero to rounding. Equal student and teacher
// scores give an exact zero gradient.
//
// Masked classes (-inf): a class the teacher masks has p = 0, one both mask is removed (its
// gradient +0.0). One only the student masks where p is not zero in the work type gives the
// loss +inf and the limit of the gradient: the passes leave its p out of the cross term and the
// mean of delta (its "lost" mass), and the rest follows, exp(s) = 0 making q - p = -p there;
// proximal_distillation spreads the lost mass evenly over the classes the student keeps. A
// teacher or student row of -inf only, NaN and +inf are invalid.
//
// Float work: float lanes and double row sums. A temperature beyond float's normal range takes
// double work for the call (float_parameters); rows rerun in double where the value is not
// finite (a spread beyond float, or an invalid row) or the gradient factor is beyond float times
// 2^64. proximal_distillation computes in double.
#include <cmath>
#include <cstddef>
#include <limits>

#include "sdca/core/lanes.h"
#include "sdca/core/limits.h"
#include "sdca/core/math.h"
#include "sdca/core/project_softmax.h"
#include "sdca/core/reduce.h"
#include "sdca/core/storage.h"
#include "sdca/loss/row.h"

SDCA_ISA_BEGIN
namespace loss {
namespace distill {

const double kInf = std::numeric_limits<double>::infinity();

enum Status { done, invalid, retry, large };

// The outcome of a row: its value, NaN, or the rerun in double.
inline double finish(Status status, double value, Scratch& s) {
  if (status == retry) s.needs_double = true;
  return status == done ? value : (status == invalid ? Row::invalid() : 0);
}

// The double lanes of lanes A (A itself for double work), and back.
template <typename A, typename W = typename A::value_type>
struct Doubles {
  typedef A type;
  static SDCA_LANE A narrow(const A& d) { return d; }
};
template <typename A>
struct Doubles<A, float> {
  typedef typename core::Accumulator<A>::lanes_type type;
  static SDCA_LANE A narrow(const type& d) { return core::narrow(d); }
};

// The factor k of the cross term exp(t') s k of a double row of the spread (max - min) of its
// scores: 1, and 2^-128 where |s| may exceed 2^896, so that it stays finite (float work reruns
// a row whose cross term overflows).
inline double cross_factor(double spread, double inv_temp) {
  return spread * inv_temp > limits::detail::pow2<double>(896) ? limits::detail::pow2<double>(-128)
                                                               : 1;
}

// The maxima of a row (M: scalars, or lanes of a block's rows), their difference as gap + lo
// exactly, the reciprocal temperature and the cross term's factor.
template <typename M, typename W>
struct Maxima {
  M s_max, t_max, gap, lo, k;
  W inv_temp;
};
template <typename W>
struct Head : Maxima<W, W> {
  std::size_t s_first, t_first;
  bool ordinary, moderate, broad;
};

// Ordinary rows: no masked class and both spreads within ordinary_spread<W>() in units of T, so
// that |s| and |t'| are at most log(max / 4) (87.3 for float, 709 for double) and the
// exponentials of classic's passes are normal.
template <typename W>
constexpr W ordinary_spread() {
  return limits::log_quarter_max<W>();
}

// The two-part normalizer is for moderate spreads. At this bound the mass outside
// a largest class is at least 2^-16, even when all other classes take the minimum.
// Smaller tails retain the centered calculation.
inline double partition_spread(std::size_t classes) {
  int exponent;
  std::frexp(static_cast<double>(classes), &exponent);
  return limits::log_pow2<double>(exponent + 14);
}

// Find both sets of extrema in one pass on finite gradient rows. In double work the
// diameter of the classwise student-teacher differences selects the gradient algorithm.
// Center at each row's first score so a large offset does not erase that diameter.
// Nonfinite lane sums fall back to extrema, which handles both masks and overflow.
template <typename A, typename S>
inline bool paired_extrema(const S* z, const S* t, std::size_t n, double temperature,
                           typename A::value_type& s_min, typename A::value_type& t_min,
                           Head<typename A::value_type>& h) {
  typedef typename A::value_type W;
  if (!n) return false;
  const W sz = core::to_work<W>(z[0]), st = core::to_work<W>(t[0]);
  A s_lo = A::splat(sz), s_hi = s_lo, t_lo = A::splat(st), t_hi = t_lo;
  A d_lo = A::zero(), d_hi = A::zero(), total = A::zero(), teacher_total = A::zero();
  // The prefix avoids scanning a masked float row twice before selecting the general pass.
  if (sizeof(W) == sizeof(float) && n >= 4 * A::L) {
    A first = A::zero();
    for (int j = 0; j < 4 * A::L; j += A::L) first += core::load<A>(z + j) + core::load<A>(t + j);
    if (!core::all(first > -std::numeric_limits<W>::infinity())) return false;
  }
  core::each<A>(z, n, sz, [&](const A& x, std::size_t j, int count) {
    const A y = core::load<A>(t + j, count, st);
    s_lo = core::min(s_lo, x), s_hi = core::max(s_hi, x);
    t_lo = core::min(t_lo, y), t_hi = core::max(t_hi, y);
    if (sizeof(W) == sizeof(double)) {
      const A d = (x - sz) - (y - st);
      d_lo = core::min(d_lo, d), d_hi = core::max(d_hi, d);
      total += x + y;
    } else {
      total += x;
      teacher_total += y;
    }
  });
  const W inf = std::numeric_limits<W>::infinity();
  if (!core::all((total > -inf) & (total < inf))) return false;
  if (sizeof(W) == sizeof(float) && !core::all((teacher_total > -inf) & (teacher_total < inf)))
    return false;
  h.s_max = core::max_lane(s_hi), h.t_max = core::max_lane(t_hi);
  s_min = core::min_lane(s_lo), t_min = core::min_lane(t_lo);
  // With diameter > 2T, some |(delta - mean) / T| necessarily exceeds
  // classic::correction_entry's unit expm1 interval for every choice of mean.
  h.broad = sizeof(W) == sizeof(double) &&
            static_cast<double>(core::max_lane(d_hi) - core::min_lane(d_lo)) / temperature > 2;
  return true;
}

// NaN, +inf and a row without a finite teacher or student score are invalid. A score beyond
// half the work type's range (where the differences of the pass could overflow) is "large":
// float work reruns the row, double work takes it scaled by 1/4.
template <typename A, typename S>
inline Status head(const S* z, const S* t, std::size_t n, double temperature,
                   Head<typename A::value_type>& h, bool gradient = false) {
  typedef typename A::value_type W;
  const W inf = std::numeric_limits<W>::infinity(), half = std::numeric_limits<W>::max() / 2;
  bool valid = true, unbounded = false;
  W s_min, t_min;
  h.broad = false;
  if (!(gradient && n > 32 && paired_extrema<A>(z, t, n, temperature, s_min, t_min, h))) {
    core::extrema<A>(z, n, s_min, h.s_max, valid, &unbounded);
    core::extrema<A>(t, n, t_min, h.t_max, valid, &unbounded);
  }
  if (!valid || !(h.s_max > -inf) || !(h.t_max > -inf)) return invalid;
  const double inv = 1 / temperature, spread = static_cast<double>(ordinary_spread<W>());
  h.ordinary = !unbounded && static_cast<double>(h.s_max - s_min) * inv <= spread &&
               static_cast<double>(h.t_max - t_min) * inv <= spread;
  h.moderate =
      sizeof(W) == sizeof(double) && std::isnormal(inv) &&
      static_cast<double>(std::fmax(h.s_max - s_min, h.t_max - t_min)) * inv <= partition_spread(n);
  if (!(std::fmax(std::fabs(s_min), std::fabs(h.s_max)) <= half &&
        std::fmax(std::fabs(t_min), std::fabs(h.t_max)) <= half))
    return sizeof(W) < sizeof(double) ? retry : large;
  h.gap = core::sum_error(h.s_max, -h.t_max, h.lo);
  h.k = static_cast<W>(sizeof(W) < sizeof(double)
                           ? 1
                           : cross_factor(static_cast<double>(h.s_max - s_min), 1 / temperature));
  h.inv_temp = static_cast<W>(1 / temperature);
  h.s_first = core::find_at_least<A>(z, 0, n, h.s_max);
  h.t_first = core::find_at_least<A>(t, 0, n, h.t_max);
  return done;
}

// expm1(x) for |x| < 1 in float work. For x < -1/2 add ln(2) in two parts
// and use expm1(x) = (expm1(x + ln(2)) - 1) / 2. On [-1/2, 1] the degree-six
// polynomial approximates (expm1(x) - x) / x^2, including its limit at zero.
template <typename A>
SDCA_LANE A expm1_near(const A& input) {
  typedef typename A::value_type W;
  const auto shifted = input < W(-0.5);
  const A x = core::select(shifted, (input + W(0.693359375)) - W(2.12194440e-4), input);
  const A x2 = x * x, x4 = x2 * x2;
  const A p0 = core::fma(A::splat(W(0.16666661f)), x, A::splat(W(0.5f)));
  const A p2 = core::fma(A::splat(W(0.008334026f)), x, A::splat(W(0.041666906f)));
  const A p4 = core::fma(A::splat(W(0.00019710849f)), x, A::splat(W(0.0013866909f)));
  const A p6 = A::splat(W(3.0475796e-05f));
  const A poly = core::fma(core::fma(p6, x2, p4), x4, core::fma(p2, x2, p0));
  const A m = core::fma(poly, x2, x);
  return core::select(shifted, core::fma(m, W(0.5), A::splat(W(-0.5))), m);
}

// One lane block of the pass at the maxima m: s, d, exp(t') and v, and exp(s) returned, the
// exponentials times 2^64: v is exp(t') expm1(d) where |d| < 1 (near), else exp(s), whose
// entries store -exp(t'). d = ((z - t) - (m_z - m_t)) / T from z - t exactly (TwoSum). The sum
// of exp(t') expm1(d) (exp(s) - exp(t') where far), and the cross term p s k (where exp(t') is
// not zero), over 2^64 too.
template <bool Ordinary, typename A, typename M>
SDCA_LANE A entry(const A& cz, const A& ct, const Maxima<M, typename A::value_type>& m,
                  core::Accumulator<A>& v_sum, core::Accumulator<A>& cross,
                  core::Accumulator<A>& lost, A& s, A& d, A& wt, A& v) {
  typedef typename A::value_type W;
  A w = cz, ew = A::zero();
  core::two_sum(w, ew, A::zero() - ct);
  s = (cz - m.s_max) * m.inv_temp;
  d = ((w - m.gap) + (ew - m.lo)) * m.inv_temp;
  wt = core::exp_nonpositive_wide((ct - m.t_max) * m.inv_temp);
  const typename A::mask_type taken = wt > W(0);
  const auto near = Ordinary ? core::abs(d) < W(1) : (core::abs(d) < W(1)) & taken;
  const A student = Ordinary ? core::exp_nonpositive_wide(s) : A::zero();
  if (Ordinary && core::all(near)) {
    v = wt * expm1_near(d);
    v_sum.add(v);
    cross.add(wt * s);
    return student;
  }
  const A e = Ordinary ? core::select(near, expm1_near(core::select(near, d, A::zero())), student)
                       : core::expm1_or_wide(core::select(near, d, s), near);
  v = core::select(near, wt * e, e);
  v_sum.add(core::select(near, v, e - wt));
  if (Ordinary) {
    // A partial vector has masked padding with zero weight and an infinite score.
    cross.add(core::select(cz > -std::numeric_limits<W>::infinity(), wt * s, A::zero()));
  } else {
    const auto kept = cz > -std::numeric_limits<W>::infinity();
    lost.add(core::select(kept, A::zero(), wt));
    cross.add(
        core::select(taken & kept, wt * (sizeof(W) < sizeof(double) ? s : s * m.k), A::zero()));
  }
  const A es = core::select(near, wt + v, e);
  wt = core::select(near, wt, -wt);
  return Ordinary ? student : es;
}

inline double pick(bool m, double a, double b) { return m ? a : b; }
template <typename M, typename D>
SDCA_LANE D pick(const M& m, const D& a, const D& b) {
  return core::select(m, a, b);
}

// A row's statistics (X: double, or double lanes of a block's rows) from the sums of the pass
// (the cross term's over 2^64 k already): the tails, C and 1 + C, and the loss. C is the sum of
// v over exp(t')'s, or exp(s)'s sum over exp(t')'s less one where that is below -1/2.
template <typename X>
struct Stats {
  X ss, ts, c, c1, value, lost;
};
template <typename W, typename X>
SDCA_LANE Stats<X> stats(const X& s_tail, const X& t_tail, const X& v_sum, const X& cross,
                         const X& lost) {
  const double unwide = core::Wide<W>::unwide();
  Stats<X> r;
  r.ss = s_tail * unwide;
  r.ts = t_tail * unwide;
  const X c = v_sum * unwide / (1.0 + r.ts);
  const auto low = c < -0.5;
  r.c1 = pick(low, (1.0 + r.ss) / (1.0 + r.ts), 1.0 + c);
  r.c = pick(low, r.c1 - 1.0, c);
  r.lost = lost * unwide / (1.0 + r.ts);
  r.value = core::log1p(r.ss) + cross / (1.0 + r.ts);
  return r;
}

// At extreme finite spreads, a negative log probability can overflow before it is
// multiplied by a small teacher probability. Combine the exponents of the gap,
// reciprocal temperature and teacher weight before rounding each positive term.
template <typename S, typename W>
inline double extreme_value(const S* z, const S* t, std::size_t n, const Head<W>& h,
                            const Stats<double>& st) {
  int inv_exp;
  const double inv = std::frexp(static_cast<double>(h.inv_temp), &inv_exp);
  double sum = 0, correction = 0;
  for (std::size_t j = 0; j < n; ++j) {
    const double score = core::to_work<double>(z[j]);
    if (score == -kInf || score == static_cast<double>(h.s_max)) continue;
    const double teacher = (core::to_work<double>(t[j]) - static_cast<double>(h.t_max)) *
                           static_cast<double>(h.inv_temp);
    // A quarter exponent preserves weights whose unscaled exponential underflows.
    // Below -2304 even 2^64 terms of size exp(teacher) * DBL_MAX^2 round to zero.
    if (teacher < -2304) continue;
    int weight_exp, gap_exp;
    const double root = std::frexp(std::exp(teacher * 0.25), &weight_exp);
    const double square = root * root;
    const double gap = std::frexp(static_cast<double>(h.s_max) - score, &gap_exp);
    const double term = std::ldexp((gap * inv) * (square * square) / (1 + st.ts),
                                   gap_exp + inv_exp + 4 * weight_exp);
    const double next = sum + term;
    if (next == kInf) return kInf;
    correction += (sum >= term) ? (sum - next) + term : (term - next) + sum;
    sum = next;
  }
  return (sum + correction) + core::log1p(st.ss);
}

// The work rows of a row (np entries each, whole lanes): exp(t') and v, and s and d for
// proximal_distillation.
template <typename W>
struct Rows {
  W *wt, *v, *s, *d;
};
template <typename A>
inline Rows<typename A::value_type> work_rows(std::size_t n, int count, Scratch& s,
                                              std::size_t& np) {
  typedef typename A::value_type W;
  np = (n + A::L - 1) / A::L * A::L;
  W* w = s.buffer<W>(static_cast<std::size_t>(count) * np, 0);
  const Rows<W> r = {w, w + np, count > 2 ? w + 2 * np : 0, count > 3 ? w + 3 * np : 0};
  return r;
}

// The pass over a row: its statistics, and the work rows (exp(t') and v where Keep > 0, s and
// d too where Keep > 1).
template <int Keep, typename A, bool Ordinary, typename S>
SDCA_FLAT inline Stats<double> pass_at_maxima(const S* z, const S* t, std::size_t n,
                                              const Head<typename A::value_type>& h,
                                              const Rows<typename A::value_type>& r) {
  typedef typename A::value_type W;
  const W inf = std::numeric_limits<W>::infinity();
  core::Accumulator<A> s_tail, t_tail, v_sum, cross, lost;
  // A loop of its own: Clang's flatten leaves core::each's lambda out of line.
  for (std::size_t j = 0; j < n; j += A::L) {
    const int count = n - j < std::size_t(A::L) ? static_cast<int>(n - j) : A::L;
    A s, d, wt, v;
    const A es =
        entry<Ordinary>(core::load<A>(z + j, count, -inf), core::load<A>(t + j, count, -inf), h,
                        v_sum, cross, lost, s, d, wt, v);
    const A student = !Ordinary && Keep == 1 && h.ordinary ? core::exp_nonpositive_wide(s) : es;
    s_tail.add(core::with_lane(student, j, h.s_first, W(0)));
    t_tail.add(core::with_lane(core::abs(wt), j, h.t_first, W(0)));
    if (Keep > 0) wt.store(r.wt + j), v.store(r.v + j);
    if (Keep > 1) s.store(r.s + j), d.store(r.d + j);
  }
  const double undo = -core::Wide<W>::unwide() / static_cast<double>(h.k);
  const double weighted = cross.value();
  Stats<double> st =
      stats<W>(s_tail.value(), t_tail.value(), v_sum.value(), weighted * undo, lost.value());
  // At extreme spreads the sum over tied teacher maxima can exceed double even
  // when its weighted mean is finite. Divide before restoring the cross scale.
  if (h.k != W(1)) {
    if (Keep < 2 && std::isfinite(h.inv_temp))
      st.value = extreme_value(z, t, n, h, st);
    else if (st.value == kInf)
      st.value = core::log1p(st.ss) + (weighted / (1 + st.ts)) * undo;
  }
  return st;
}

template <int Keep, typename A, typename S>
inline Stats<double> pass(const S* z, const S* t, std::size_t n,
                          const Head<typename A::value_type>& h,
                          const Rows<typename A::value_type>& r) {
  // The finite float gradient uses the short expm1 polynomial on fused SIMD lanes.
  if (Keep == 1 && sizeof(typename A::value_type) == sizeof(float) && A::B > 1 && h.ordinary)
    return pass_at_maxima<Keep, A, true>(z, t, n, h, r);
  return pass_at_maxima<Keep, A, false>(z, t, n, h, r);
}

// The distillation kernel for ordinary rows (head): the teacher, correction and gradient
// passes of the log-ratio about its p-weighted mean (Eq. 9 of Barron et al.), and the loss
// alone from exp(s) and exp(t'). Rows whose shift exceeds kShift take the general pass.
namespace classic {

// The largest |shift| of the gradient passes: they form exp(s + shift) as exp(s) exp(shift).
const double kShift = 32;

// The helpers below take a row's statistics X as scalars (a row's lanes) or as double lanes
// (a block's lanes, one row each).

// (x - x_max) - (y - y_max) in double with the rounding errors of both differences: the
// difference of the offsets alone loses a small gap beside a large one.
template <typename A, typename X>
SDCA_LANE A offset_difference(const A& x, const A& y, const X& x_max, const X& y_max) {
  typedef typename Doubles<A>::type D;
  D a = core::to_double(x), b = core::to_double(y), ea = D::zero(), eb = D::zero();
  core::two_sum(a, ea, D::zero() - x_max);
  core::two_sum(b, eb, D::zero() - y_max);
  return Doubles<A>::narrow((a - b) + (ea - eb));
}

// (delta - mean) / T - shift in double, rounded once: it cancels where student and teacher
// agree.
template <typename A, typename X, typename Y>
SDCA_LANE A log_ratio(const A& delta, const X& mean, double inv_temp, const Y& shift) {
  return Doubles<A>::narrow((core::to_double(delta) - mean) * inv_temp - shift);
}

// One lane block of the teacher pass: s, delta and exp(t') at the maxima (M: scalars or lanes;
// X: in double); the cross term and the mean of delta add p's weights.
template <typename A, typename M, typename X>
SDCA_LANE A teacher_entry(const A& cs, const A& ct, const M& s_max, const M& t_max,
                          const X& s_max_d, const X& t_max_d, typename A::value_type inv_temp,
                          core::Accumulator<A>& cross, core::Accumulator<A>& mean, A& s, A& delta) {
  s = (cs - s_max) * inv_temp;
  const A et = core::exp_nonpositive((ct - t_max) * inv_temp);
  delta = offset_difference(cs, ct, s_max_d, t_max_d);
  // A masked teacher entry: exp(t') is zero and delta infinite or NaN.
  const typename A::mask_type taken = et > typename A::value_type(0);
  cross.add(core::select(taken, et * -s, A::zero()));
  mean.add(core::select(taken, et * delta, A::zero()));
  return et;
}

// One lane block of the correction pass: expm1(dn) where |dn| < 1, else exp(s); C adds
// p expm1(dn) there and exp(s + shift) - p elsewhere.
template <typename A, typename X>
SDCA_LANE A correction_entry(const A& p, const A& delta, const A& s, const X& mean, double inv_temp,
                             core::Accumulator<A>& near, core::Accumulator<A>& far, A& dn) {
  typedef typename A::value_type W;
  dn = log_ratio(delta, mean, inv_temp, 0.0);
  const typename A::mask_type small = core::abs(dn) < W(1);
  const A e = core::exp_or_expm1(core::select(small, dn, s), small);
  near.add(core::select(small, p * e, -p));
  far.add(core::select(small, A::zero(), e));
  return e;
}

// The factors of a row's gradient pass from p's normalizer, the correction C and log(1 +
// tail(s)): lanes A of every row's (or a row's) value, and the double statistics X.
template <typename A, typename X>
struct Factors {
  A inv_p, c, inv_c, lc, inv_q;
  X mean, log1p_c;
};
template <typename A, typename X>
SDCA_LANE Factors<A, X> factors(const X& t_tail, const X& correction, const X& log1p_c,
                                const X& lognorm, const X& mean) {
  typedef typename Doubles<A>::type D;
  const D one = D::splat(1.0);
  const Factors<A, X> f = {Doubles<A>::narrow(one / (one + t_tail)),
                           Doubles<A>::narrow(D::zero() + correction),
                           Doubles<A>::narrow(one / (one + correction)),
                           Doubles<A>::narrow(D::zero() + log1p_c),
                           Doubles<A>::narrow(core::exp(D::zero() - lognorm)),
                           mean,
                           log1p_c};
  return f;
}

// One lane block of the gradient pass: u = q - p and q. Where |dn| >= 1 but the log-ratio
// dn - log1p(C) of q / p is small (rare), u is p expm1 of that ratio, from delta.
template <typename A, typename X>
SDCA_LANE void gradient_entry(const Factors<A, X>& f, double inv_temp, const A& et, const A& dn,
                              const A& e, const typename A::value_type* delta, A& u, A& q) {
  typedef typename A::value_type W;
  typedef typename A::mask_type M;
  const A p = et * f.inv_p;
  M small = core::abs(dn) < W(1);
  A pu = p * ((e - f.c) * f.inv_c);
  // The rounded dn finds every such lane: it errs by far less than 1e-3 there.
  const M maybe = ~small & (core::abs(dn - f.lc) < W(1.001));
  if (core::any(maybe)) {
    const A x = log_ratio(A::load(delta), f.mean, inv_temp, f.log1p_c);
    const M shifted = maybe & (core::abs(x) < W(1));
    pu = core::select(shifted, p * core::exp_or_expm1(x, shifted), pu);
    small = small | shifted;
  }
  q = core::select(small, p + pu, e * f.inv_q);
  u = core::select(small, pu, q - p);
}

// One lane block of the loss alone: exp(s) and exp(t') over core::Wide::unwide() into ws and
// wt; the cross term adds -p s, over unwide too (where exp(t') is not zero, s > -133).
template <typename A, typename M>
SDCA_LANE void value_entry(const A& cs, const A& ct, const M& s_max, const M& t_max,
                           typename A::value_type inv_temp, core::Accumulator<A>& cross, A& ws,
                           A& wt) {
  typedef typename A::value_type W;
  const A s = (cs - s_max) * inv_temp;
  wt = core::exp_nonpositive_wide((ct - t_max) * inv_temp), ws = core::exp_nonpositive_wide(s);
  cross.add(core::select(wt > W(0), wt * -s, A::zero()));
}

// The loss from the tails of both rows and the cross term in one pass.
template <typename A, typename S>
SDCA_FLAT inline double value_row(const S* z, const S* t, std::size_t n,
                                  const Head<typename A::value_type>& h, Status& status) {
  typedef typename A::value_type W;
  const W inf = std::numeric_limits<W>::infinity();
  core::Accumulator<A> s_tail, t_tail, cross;
  core::each<A>(z, n, -inf, [&](const A& cs, std::size_t j, int count) {
    A ws, wt;
    value_entry(cs, core::load<A>(t + j, count, -inf), h.s_max, h.t_max, h.inv_temp, cross, ws, wt);
    s_tail.add(core::with_lane(ws, j, h.s_first, W(0)));
    t_tail.add(core::with_lane(wt, j, h.t_first, W(0)));
  });
  // The tails of exact terms over core::Wide::unwide(), also below float's normal range.
  const double unwide = core::Wide<W>::unwide();
  const double ss = s_tail.value() * unwide, ts = t_tail.value() * unwide;
  const double value = core::log1p(ss) + cross.value() * unwide / (1 + ts);
  // Float: a spread beyond float (s of -inf at a positive p), or an invalid row, which double
  // work tells apart.
  if (!(value < kInf)) status = sizeof(W) < sizeof(double) ? retry : invalid;
  return value;
}

// The work rows (np entries each, whole lanes) and sums of a row's gradient passes.
template <typename W>
struct Rows {
  W *et, *s, *delta, *e, *dn;
  double t_tail, cross, mean, shift, correction, log1p_c;

  // log(1 + tail(s)).
  double student_lognorm() const { return log1p_c - shift; }
};

// The teacher pass: exp(t'), s and delta into the work rows; the teacher's tail, the cross
// term and the p-weighted mean of delta.
template <typename A, typename S>
SDCA_FLAT inline Status teacher_pass(const S* z, const S* t, std::size_t n, double temperature,
                                     const Head<typename A::value_type>& h,
                                     Rows<typename A::value_type>& r) {
  typedef typename A::value_type W;
  const W inf = std::numeric_limits<W>::infinity();
  core::Accumulator<A> t_tail, cross, mean;
  core::each<A>(z, n, -inf, [&](const A& cs, std::size_t j, int count) {
    A s, delta;
    const A et = teacher_entry(cs, core::load<A>(t + j, count, -inf), h.s_max, h.t_max,
                               double(h.s_max), double(h.t_max), h.inv_temp, cross, mean, s, delta);
    t_tail.add(core::with_lane(et, j, h.t_first, W(0)));
    et.store(r.et + j), s.store(r.s + j), delta.store(r.delta + j);
  });
  r.t_tail = t_tail.value();
  r.cross = cross.value() / (1 + r.t_tail);
  r.mean = mean.value() / (1 + r.t_tail);
  r.shift = -r.mean / temperature - core::log1p(r.t_tail);
  return std::fabs(r.shift) < kInf ? done : invalid;
}

// The correction pass: C, and per entry expm1(dn) where |dn| < 1, else exp(s), into e, and dn.
// C sums exp(s + shift) - p as exp(shift) sum exp(s) - sum p, in double.
template <typename A>
SDCA_FLAT inline Status correction_pass(std::size_t np, double temperature,
                                        Rows<typename A::value_type>& r) {
  typedef typename A::value_type W;
  const W inv_p = static_cast<W>(1 / (1 + r.t_tail));
  core::Accumulator<A> near, far;
  for (std::size_t j = 0; j < np; j += A::L) {
    A dn;
    const A e = correction_entry(A::load(r.et + j) * inv_p, A::load(r.delta + j), A::load(r.s + j),
                                 r.mean, 1 / temperature, near, far, dn);
    e.store(r.e + j), dn.store(r.dn + j);
  }
  r.correction = near.value() + core::exp(r.shift) * far.value();
  r.log1p_c = core::log1p(r.correction);
  const double value = r.student_lognorm() + r.cross;
  return value < kInf ? done : invalid;
}

// The gradient pass: u = q - p into s and q into e, and sum(u).
template <typename A>
SDCA_FLAT inline double gradient_pass(std::size_t np, double temperature,
                                      Rows<typename A::value_type>& r) {
  const Factors<A, double> f =
      factors<A>(r.t_tail, r.correction, r.log1p_c, r.student_lognorm(), r.mean);
  core::Accumulator<A> total;
  for (std::size_t j = 0; j < np; j += A::L) {
    A u, q;
    gradient_entry(f, 1 / temperature, A::load(r.et + j), A::load(r.dn + j), A::load(r.e + j),
                   r.delta + j, u, q);
    total.add(u);
    u.store(r.s + j), q.store(r.e + j);
  }
  return total.value();
}

// The work rows of n classes: five rows of whole lanes from the scratch.
template <typename A>
inline Rows<typename A::value_type> work_rows(std::size_t n, Scratch& s, std::size_t& np) {
  typedef typename A::value_type W;
  np = (n + A::L - 1) / A::L * A::L;
  W* w = s.buffer<W>(5 * np, 0);
  Rows<W> r;
  r.et = w, r.s = w + np, r.delta = w + 2 * np, r.e = w + 3 * np, r.dn = w + 4 * np;
  return r;
}

template <typename A, typename S, typename G, typename Targets>
SDCA_LANE void pack_block(const BlockArgs<S, G>& b, const Targets& targets, std::size_t first,
                          typename A::value_type* zc, typename A::value_type* tc) {
  typedef typename A::value_type W;
  const int L = A::L;
  const std::size_t c = b.n, rows = b.rows;
  for (int r = 0; r < L; ++r) {
    const std::size_t row =
        static_cast<std::size_t>(r) < rows ? static_cast<std::size_t>(r) : rows - 1;
    const S *z = b.scores + row * c, *t = targets.row(first + row, c).t;
    for (std::size_t j = 0; j < c; ++j)
      zc[j * L + r] = core::to_work<W>(z[j]), tc[j * L + r] = core::to_work<W>(t[j]);
  }
}

template <typename A>
SDCA_LANE void ordinary_block_rows(const A& finite, const A& s_min, const A& s_max, const A& t_min,
                                   const A& t_max, double temperature, bool* ordinary) {
  typedef typename A::value_type W;
  // Ordinary rows (head): the others take row().
  const double spread = static_cast<double>(ordinary_spread<W>()) * temperature;
  const W half = std::numeric_limits<W>::max() / 2;
  for (int i = 0; i < A::L; ++i)
    ordinary[i] = finite[i] == 0 && static_cast<double>(s_max[i] - s_min[i]) <= spread &&
                  static_cast<double>(t_max[i] - t_min[i]) <= spread &&
                  std::fmax(std::fmax(std::fabs(s_min[i]), std::fabs(s_max[i])),
                            std::fmax(std::fabs(t_min[i]), std::fabs(t_max[i]))) <= half;
}

// Normalizers and cross term for a block whose gradients are not requested.
template <typename A, typename S, typename G>
SDCA_LANE void block_values(const BlockArgs<S, G>& b, const typename A::value_type* zc,
                            const typename A::value_type* tc, const A& s_max, const A& t_max,
                            const typename A::mask_type& s_first,
                            const typename A::mask_type& t_first, typename A::value_type inv_temp,
                            const bool* ordinary, double* values, bool* done) {
  typedef typename A::value_type W;
  typedef typename A::mask_type M;
  typedef typename M::value_type I;
  typedef typename Doubles<A>::type D;
  const int L = A::L;
  const std::size_t c = b.n, rows = b.rows;
  core::Accumulator<A> s_tail, t_tail, cross;
  for (std::size_t j = 0; j < c; ++j) {
    A ws, wt;
    const M at = s_first == static_cast<I>(j), t_at = t_first == static_cast<I>(j);
    value_entry(A::load(zc + j * L), A::load(tc + j * L), s_max, t_max, inv_temp, cross, ws, wt);
    s_tail.add(core::select(at, A::zero(), ws));
    t_tail.add(core::select(t_at, A::zero(), wt));
  }
  const double unwide = core::Wide<W>::unwide();
  const D ss = s_tail.lanes() * unwide, ts = t_tail.lanes() * unwide;
  const D value = core::log1p(ss) + cross.lanes() * unwide / (1.0 + ts);
  for (std::size_t r = 0; r < rows; ++r) {
    const int i = static_cast<int>(r);
    done[r] = ordinary[i] && value[i] < kInf;
    values[r] = value[i];
  }
}

// The centered gradient pass requires a representable scale and a bounded logarithmic shift.
template <typename W>
SDCA_LANE bool valid_gradient_lane(double value, double shift, double scale) {
  return value < kInf && std::fabs(shift) <= kShift &&
         std::fabs(scale) <= static_cast<double>(std::numeric_limits<W>::max());
}

// rows by column into the scratch, then the passes of row() on one vector of the rows' scores
// per class, without per-row horizontal sums. done[r] for the rows of finite scores whose
// results the work type holds (as row() tests them); the others take row().
template <typename A, typename S, typename G, typename Targets>
SDCA_FLAT inline void block(const BlockArgs<S, G>& b, const Targets& targets, std::size_t first,
                            double temperature, double* values, bool* done, Scratch& scratch) {
  typedef typename A::value_type W;
  typedef typename A::mask_type M;
  typedef typename M::value_type I;
  typedef typename Doubles<A>::type D;
  const W inf = std::numeric_limits<W>::infinity();
  const int L = A::L;
  const std::size_t c = b.n, rows = b.rows, cl = c * static_cast<std::size_t>(L);
  W* col = scratch.buffer<W>(7 * cl, 1);
  W *zc = col, *tc = col + cl, *et_c = col + 2 * cl, *s_c = col + 3 * cl, *delta_c = col + 4 * cl;
  W *e_c = col + 5 * cl, *dn_c = col + 6 * cl;
  pack_block<A>(b, targets, first, zc, tc);
  A s_max = A::splat(-inf), t_max = A::splat(-inf), s_min = A::splat(inf), t_min = s_min;
  A finite = A::zero();
  for (std::size_t j = 0; j < c; ++j) {
    const A a = A::load(zc + j * L), y = A::load(tc + j * L);
    s_max = core::max(s_max, a), t_max = core::max(t_max, y);
    s_min = core::min(s_min, a), t_min = core::min(t_min, y);
    finite += a * W(0) + y * W(0);
  }
  bool ordinary[A::L];
  ordinary_block_rows(finite, s_min, s_max, t_min, t_max, temperature, ordinary);
  // The first largest class of each row (c: none yet).
  M s_first = M::splat(static_cast<I>(c)), t_first = s_first;
  for (std::size_t j = 0; j < c; ++j) {
    const M at = M::splat(static_cast<I>(j)), none = M::splat(static_cast<I>(c));
    s_first = core::select((A::load(zc + j * L) == s_max) & (s_first == none), at, s_first);
    t_first = core::select((A::load(tc + j * L) == t_max) & (t_first == none), at, t_first);
  }
  const W inv_temp = static_cast<W>(1 / temperature);
  D value;
  bool fine[A::L];
  if (!b.grad) {
    block_values(b, zc, tc, s_max, t_max, s_first, t_first, inv_temp, ordinary, values, done);
    return;
  }
  const D s_max_d = core::to_double(s_max), t_max_d = core::to_double(t_max);
  core::Accumulator<A> t_tail, cross, mean;
  for (std::size_t j = 0; j < c; ++j) {
    A s, delta;
    const A et = teacher_entry(A::load(zc + j * L), A::load(tc + j * L), s_max, t_max, s_max_d,
                               t_max_d, inv_temp, cross, mean, s, delta);
    t_tail.add(core::select(t_first == static_cast<I>(j), A::zero(), et));
    et.store(et_c + j * L), s.store(s_c + j * L), delta.store(delta_c + j * L);
  }
  const D tt = t_tail.lanes(), md = mean.lanes() / (1.0 + tt);
  const D shift = D::zero() - md / temperature - core::log1p(tt);
  const A inv_p = Doubles<A>::narrow(1.0 / (1.0 + tt));
  core::Accumulator<A> near, far;
  for (std::size_t j = 0; j < c; ++j) {
    A dn;
    const A e = correction_entry(A::load(et_c + j * L) * inv_p, A::load(delta_c + j * L),
                                 A::load(s_c + j * L), md, 1 / temperature, near, far, dn);
    e.store(e_c + j * L), dn.store(dn_c + j * L);
  }
  const D correction = near.lanes() + core::exp(shift) * far.lanes();
  const D lc = core::log1p(correction), lognorm = lc - shift;
  value = lognorm + cross.lanes() / (1.0 + tt);
  A factor = A::zero();
  for (std::size_t r = 0; r < rows; ++r) {
    const int i = static_cast<int>(r);
    const double f = b.scale[r] / temperature;
    fine[i] = ordinary[i] && valid_gradient_lane<W>(value[i], shift[i], f);
    if (fine[i]) factor.set(i, static_cast<W>(f));
  }
  const Factors<A, D> fs = factors<A>(tt, correction, lc, lognorm, md);
  core::Accumulator<A> total;
  M sensitive = M::zero(), separate = M::zero();
  for (std::size_t j = 0; j < c; ++j) {
    A u, q;
    gradient_entry(fs, 1 / temperature, A::load(et_c + j * L), A::load(dn_c + j * L),
                   A::load(e_c + j * L), delta_c + j * L, u, q);
    if (sizeof(W) == sizeof(double)) {
      sensitive = sensitive | (core::abs(u) < (q + q - u) * W(1.0 / 16));
      separate = separate | (core::abs(A::load(delta_c + j * L)) > W(temperature / 32));
    }
    total.add(u);
    u.store(s_c + j * L), q.store(e_c + j * L);
  }
  const A sum = Doubles<A>::narrow(total.lanes());
  for (std::size_t j = 0; j < c; ++j)
    ((A::load(s_c + j * L) - A::load(e_c + j * L) * sum) * factor + W(0)).store(zc + j * L);
  for (std::size_t r = 0; r < rows; ++r) {
    const int i = static_cast<int>(r);
    const bool refine =
        sizeof(W) == sizeof(double) && sensitive[i] && separate[i] &&
        static_cast<double>(std::fmax(s_max[i] - s_min[i], t_max[i] - t_min[i])) / temperature <=
            partition_spread(c);
    done[r] = fine[i] && !refine;
    if (!done[r]) continue;
    values[r] = value[i];
    G* g = b.grad + r * c;
    for (std::size_t j = 0; j < c; ++j) g[j] = core::to_storage<G>(zc[j * L + i]);
  }
}

}  // namespace classic

// A double row of large scores (head) at a quarter of its scores and of the temperature,
// which leaves s, t' and d as they are: f(the scaled row, its teacher, T / 4).
template <typename S, typename G, typename F>
inline double quarter(const RowArgs<S, G>& row, const S* t, double temperature, Scratch& s, F f) {
  const std::size_t n = row.n;
  double* q = s.doubles(2 * n, 2);
  for (std::size_t j = 0; j < n; ++j)
    q[j] = core::to_work<double>(row.scores[j]) / 4, q[n + j] = core::to_work<double>(t[j]) / 4;
  if (!(temperature / 4 > 0)) return Row::invalid();
  const RowArgs<double, G> r = {q, row.grad, n, row.scale};
  return f(r, q + n, temperature / 4);
}

namespace precise {

// Two parts of a double-lane number. The low part preserves normalizer differences
// that can be much smaller than one ulp of either partition function.
template <typename A>
struct Pair {
  A hi, lo;
};
// TwoProduct requires a single rounding even when the compiler can reuse a rounded product.
template <typename A>
SDCA_LANE A strict_fma(const A& a, const A& b, const A& c) {
  A r;
  for (int i = 0; i < A::L; ++i) r.set(i, std::fma(a[i], b[i], c[i]));
  return r;
}
template <typename A>
SDCA_LANE Pair<A> shifted(const A& x, double top, double temperature, double inv) {
  A h = x, lo = A::zero();
  core::two_sum(h, lo, A::splat(-top));
  const A q = h * inv;
  const Pair<A> value = {q, (strict_fma(q, A::splat(-temperature), h) + lo) * inv};
  return value;
}
template <typename A>
SDCA_LANE Pair<A> difference(const Pair<A>& x, const Pair<A>& y) {
  A hi = x.hi, lo = x.lo - y.lo;
  core::two_sum(hi, lo, A::zero() - y.hi);
  const Pair<A> r = {hi, lo};
  return r;
}

// 2^(k/256), k = 0,...,255, from the high and low parts of 2^(1/256).
struct ExpTable {
  double entries[256][2];
  ExpTable() {
    entries[0][0] = 1;
    entries[0][1] = 0;
    const double step = 1.0027112750502025, step_low = -3.636615928692264e-17;
    for (int i = 1; i < 256; ++i) {
      const double a = entries[i - 1][0], al = entries[i - 1][1];
      double err;
      const double product = core::two_product(a, step, err);
      const double low = err + (a * step_low + al * step);
      entries[i][0] = core::sum_error(product, low, entries[i][1]);
    }
  }
};
// exp(hi + lo) = 2^k a exp(r), |r| <= log(2)/512. The split log(2) makes the first
// subtraction exact. Taking the product a*r and the smaller a*(expm1(r)-r) separately
// leaves the low part of exp available to the partition sum.
template <typename A>
SDCA_LANE Pair<A> precise_exp(const Pair<A>& x) {
  typename A::mask_type k;
  const A n = core::round(x.hi * (256 * 1.4426950408889634074), k);
  A r = x.hi - n * (static_cast<double>(limits::kLn2Hi) / 256), rl = x.lo;
  core::two_sum(r, rl, n * (-static_cast<double>(limits::kLn2Lo) / 256));
  static const ExpTable table;
  const auto index = k & static_cast<int64_t>(255);
  A a = A::zero(), al = A::zero();
  for (int i = 0; i < A::L; ++i) {
    a.set(i, table.entries[index[i]][0]);
    al.set(i, table.entries[index[i]][1]);
  }
  const A r2 = r * r;
  const A p = core::fma(r, 1.0 / 720, 1.0 / 120);
  const A phi = r2 * core::fma(r, core::fma(r, core::fma(r, p, 1.0 / 24), 1.0 / 6), 0.5);
  const A ar = a * r, h = a + ar;
  const A l = ((a - h) + ar) + (strict_fma(a, r, A::zero() - ar) + al * (1.0 + r) + a * (phi + rl));
  const A main = h + l, low = (h - main) + l, scale = core::detail::pow2<double>(k >> 8);
  const Pair<A> e = {scale * main, scale * low};
  return e;
}

// Begin each positive sum above every term, so FastTwoSum collects its low parts.
// Remove this bias when summing the lanes.
template <typename A>
struct PairSum {
  const double bias;
  A hi, lo;
  explicit PairSum(double largest) : bias(largest), hi(A::splat(largest)), lo(A::zero()) {}
  SDCA_LANE void add(const Pair<A>& x) {
    const A total = hi + x.hi;
    lo += ((hi - total) + x.hi) + x.lo;
    hi = total;
  }
  Pair<double> value() const {
    core::Neumaier sum;
    for (int i = 0; i < A::L; ++i) {
      sum.add(hi[i]);
      sum.add(-bias);
      sum.add(lo[i]);
    }
    double l;
    const double h = core::sum_error(sum.hi, sum.lo, l);
    const Pair<double> result = {h, l};
    return result;
  }
};

// Refine log(S_hi / T_hi) with (S - T exp(l)) / S using both normalizer parts.
template <typename A>
SDCA_LANE Pair<double> log_normalizer_ratio(Pair<double> ss, Pair<double> ts) {
  const double l = std::log(ss.hi / ts.hi);
  const Pair<A> in = {A::splat(l), A::zero()}, e = precise_exp(in);
  double pe, se;
  const double product = core::two_product(ts.hi, e.hi[0], pe);
  const double residual = core::sum_error(ss.hi, -product, se);
  const double ll = (residual + (ss.lo + se - pe - ts.hi * e.lo[0] - ts.lo * e.hi[0])) / ss.hi;
  return {l, ll};
}

// Independently rounded probability differences can exceed n/2 ulps of the gradient
// 1-norm on a short row. Distribute the residual by |g| to retain the relative
// accuracy of a nearly cancelling class.
template <typename S, typename G>
inline void conserve_short_gradient(const RowArgs<S, G>& row, std::size_t m) {
  if (sizeof(G) == sizeof(double) && m <= 32) {
    core::Neumaier total, norm;
    for (std::size_t j = 0; j < m; ++j) {
      const double g = core::to_work<double>(row.grad[j]);
      total.add(g);
      norm.add(std::fabs(g));
    }
    if (total.value() != 0 && norm.value() > 0) {
      const double correction = total.value() / norm.value();
      for (std::size_t j = 0; j < m; ++j) {
        const double g = core::to_work<double>(row.grad[j]);
        row.grad[j] = core::to_storage<G>(std::fma(-std::fabs(g), correction, g) + 0.0);
      }
    }
  }
}

// Resolve close probabilities with p*expm1((s-t) - log(S/T)).
// Raw scores are scaled exactly when the temperature is a power of two and the
// exponentials fit without shifting; there the backward pass rereads the two inputs.
template <bool Unshifted, typename A, typename S, typename G, typename W>
SDCA_FLAT inline double pair_row(const RowArgs<S, G>& row, const S* teacher, double temperature,
                                 const Head<W>& h, Scratch& scratch, double divisor) {
  const std::size_t m = row.n;
  const std::size_t n = ((m + A::L - 1) / A::L) * A::L;
  double* et = scratch.doubles((Unshifted ? 2 : 4) * n, 0);
  double* es = et + n;
  double *dh = Unshifted ? 0 : es + n, *dl = Unshifted ? 0 : dh + n;
  const double inv = 1 / temperature;
  const double s_max = static_cast<double>(h.s_max), t_max = static_cast<double>(h.t_max);
  double max_hi = 1, max_lo = 0;
  PairSum<A> student(Unshifted ? 2 * std::exp(s_max * inv) : 2),
      target(Unshifted ? 2 * std::exp(t_max * inv) : 2);
  core::Accumulator<A> cross;
  for (std::size_t j = 0; j < m; j += A::L) {
    const int count = static_cast<int>(std::min<std::size_t>(A::L, m - j));
    const A z = core::load<A>(row.scores + j, count, s_max),
            t = core::load<A>(teacher + j, count, t_max);
    const Pair<A> x = Unshifted ? Pair<A>{z * inv, A::zero()} : shifted(z, s_max, temperature, inv);
    const Pair<A> y = Unshifted ? Pair<A>{t * inv, A::zero()} : shifted(t, t_max, temperature, inv);
    if (!Unshifted) {
      const Pair<A> d = difference(x, y);
      d.hi.store(dh + j);
      d.lo.store(dl + j);
    }
    Pair<A> pt = precise_exp(y), qs = precise_exp(x);
    if (count != A::L) {
      const auto live =
          core::iota<double, typename A::isa_type, A::chains>() < static_cast<int64_t>(count);
      pt.hi = core::select(live, pt.hi, A::zero());
      pt.lo = core::select(live, pt.lo, A::zero());
      qs.hi = core::select(live, qs.hi, A::zero());
      qs.lo = core::select(live, qs.lo, A::zero());
    }
    if (h.s_first >= j && h.s_first - j < A::L) {
      const int lane = static_cast<int>(h.s_first - j);
      max_hi = qs.hi[lane];
      max_lo = qs.lo[lane];
    }
    Pair<A> others = {core::with_lane(qs.hi, j, h.s_first, 0.0),
                      core::with_lane(qs.lo, j, h.s_first, 0.0)};
    student.add(others);
    target.add(pt);
    cross.add(pt.hi * ((Unshifted ? A::splat(s_max * inv) : A::zero()) - x.hi));
    pt.hi.store(et + j);
    qs.hi.store(es + j);
  }
  Pair<double> ss = student.value();
  const double tail = ss.hi / max_hi;
  core::two_sum(ss.hi, ss.lo, max_hi);
  ss.lo += max_lo;
  const Pair<double> ts = target.value();
  const Pair<double> lr = log_normalizer_ratio<A>(ss, ts);
  const double factor = row.scale / divisor, inv_t = 1 / ts.hi, inv_s = 1 / ss.hi;
  for (std::size_t j = 0; j < m; j += A::L) {
    const int count = static_cast<int>(std::min<std::size_t>(A::L, m - j));
    A x, y, gap;
    if (Unshifted) {
      x = core::load<A>(row.scores + j, count, s_max) * inv;
      y = core::load<A>(teacher + j, count, t_max) * inv;
      gap = x - y;
    } else {
      gap = A::load(dh + j);
    }
    const A a = gap - lr.hi, p = A::load(et + j) * inv_t, q = A::load(es + j) * inv_s;
    const auto near = core::abs(a) < 0.125;
    A u = q - p;
    if (core::any(near)) {
      A low;
      if (Unshifted) {
        const A b = gap - x;
        low = (x - (gap - b)) - (y + b);
      } else {
        low = A::load(dl + j);
      }
      const A centered = a + (low - lr.lo);
      u = core::select(near, p * core::expm1_reduced(core::select(near, centered, A::zero())), u);
    }
    core::store(row.grad + j, u * factor + 0.0, count);
  }
  conserve_short_gradient(row, m);
  return core::log1p(tail) + cross.value() / ts.hi;
}

// A nearly constant difference between teacher and student is better resolved by the centered
// log-ratio algorithm. As soon as a centered logit differs by more than 1/32 in units of T,
// the two normalizers resolve differences to additional bits without subtracting large offsets.
template <typename A, typename S, typename W>
inline bool separated(const S* z, const S* t, std::size_t n, const Head<W>& h, double temperature) {
  for (std::size_t j = 0; j < n; j += A::L) {
    const int count = static_cast<int>(std::min<std::size_t>(A::L, n - j));
    A delta = core::load<A>(z + j, count, static_cast<double>(h.s_max)), low = A::zero();
    core::two_sum(delta, low,
                  A::zero() - core::load<A>(t + j, count, static_cast<double>(h.t_max)));
    if (core::any(core::abs((delta - static_cast<double>(h.gap)) +
                            (low - static_cast<double>(h.lo))) > temperature / 32))
      return true;
  }
  return false;
}
}  // namespace precise

// A row at temperature T whose gradient divides by divisor (T, but for quarter()'s rows).
// Return the ordinary double gradient and its loss when the centered passes apply.
template <typename A, typename S, typename G>
SDCA_LANE bool ordinary_gradient(const RowArgs<S, G>& row, const S* t, double temperature,
                                 const Head<typename A::value_type>& h, Scratch& s, double divisor,
                                 double& value) {
  typedef typename A::value_type W;
  typedef core::Lanes<double, typename A::isa_type> D;
  const double factor = row.scale / divisor;
  if (h.moderate && std::fabs(factor) <= limits::detail::pow2<double>(256) &&
      (h.broad || (row.n <= 32 && precise::separated<D>(row.scores, t, row.n, h, temperature)))) {
    int exponent;
    if (h.s_max >= 0 && h.t_max >= 0 &&
        static_cast<double>(std::fmax(h.s_max, h.t_max)) / temperature < 32 &&
        std::frexp(temperature, &exponent) == 0.5)
      value = precise::pair_row<true, D>(row, t, temperature, h, s, divisor);
    else
      value = precise::pair_row<false, D>(row, t, temperature, h, s, divisor);
    return true;
  }
  std::size_t cp;
  classic::Rows<W> r = classic::work_rows<A>(row.n, s, cp);
  Status st = classic::teacher_pass<A>(row.scores, t, row.n, temperature, h, r);
  if (st == done && std::fabs(r.shift) <= classic::kShift)
    st = classic::correction_pass<A>(cp, temperature, r);
  else
    st = invalid;
  const double f = row.scale / divisor;
  if (st == done && std::fabs(f) <= static_cast<double>(std::numeric_limits<W>::max())) {
    const W total = static_cast<W>(classic::gradient_pass<A>(cp, temperature, r));
    const W fw = static_cast<W>(f);
    for (std::size_t j = 0; j < row.n; j += A::L) {
      const int count = row.n - j < std::size_t(A::L) ? static_cast<int>(row.n - j) : A::L;
      core::store(row.grad + j, (A::load(r.s + j) - A::load(r.e + j) * total) * fw + W(0), count);
    }
    value = r.student_lognorm() + r.cross;
    return true;
  }
  return false;
}

// Float retries a nonfinite cross-entropy in double. In double a valid loss
// can overflow while its student gradient remains representable.
template <typename W>
inline bool valid_value(double value) {
  return value <= (sizeof(W) == sizeof(double) ? kInf : std::numeric_limits<double>::max());
}

template <typename A, typename S, typename G>
inline double row(const RowArgs<S, G>& row, const S* t, double temperature, Scratch& s,
                  double divisor) {
  typedef typename A::value_type W;
  typedef core::Wide<W> Wide;
  const Status beyond = sizeof(W) < sizeof(double) ? retry : invalid;
  Head<W> h;
  const Status status = head<A>(row.scores, t, row.n, temperature, h, row.grad != 0);
  if (status == large)
    return quarter(
        row, t, temperature, s, [&](const RowArgs<double, G>& r, const double* u, double t4) {
          return distill::row<core::Lanes<double, typename A::isa_type>>(r, u, t4, s, divisor);
        });
  if (status != done) return finish(status, 0, s);
  const Rows<W> none = {0, 0, 0, 0};
  std::size_t np;
  // The ordinary forward needs only the normalizers and the cross term. The two-row pass
  // below suffices for float gradients; double gradients retain the centered log ratio.
  if (h.ordinary) {
    Status st = done;
    if (!row.grad) {
      const double value = classic::value_row<A>(row.scores, t, row.n, h, st);
      if (st == done) return value;
    } else if (sizeof(W) == sizeof(double)) {
      double value;
      if (ordinary_gradient<A>(row, t, temperature, h, s, divisor, value)) return value;
    }
  }
  if (!row.grad) {
    const auto st = pass<0, A>(row.scores, t, row.n, h, none);
    if (st.lost > 0) return kInf;
    return valid_value<W>(st.value) ? st.value : finish(beyond, 0, s);
  }
  const Rows<W> r = work_rows<A>(row.n, 2, s, np);
  const Stats<double> st = pass<1, A>(row.scores, t, row.n, h, r);
  if (!valid_value<W>(st.value)) return finish(beyond, 0, s);
  // (q - p) T = u / norm over 2^64: u = exp(t') (expm1(d) - C) where near, exp(s) - exp(t') (1 +
  // C) where far, into v; q = exp(t') (1 + C) + u.
  const double norm = (1 + st.ts) * st.c1, factor = row.scale / divisor / norm;
  W c, power;
  if (!Wide::split(factor, c, power)) return finish(beyond, 0, s);
  const W cw = static_cast<W>(st.c), c1 = static_cast<W>(st.c1);
  core::Accumulator<A> total;
  for (std::size_t j = 0; j < np; j += A::L) {
    const A wt = A::load(r.wt + j);
    const A u =
        core::fma(wt, core::select(wt < W(0), A::splat(c1), A::splat(-cw)), A::load(r.v + j));
    total.add(u);
    u.store(r.v + j);
  }
  const W sum = static_cast<W>(total.value() * Wide::unwide() / norm);
  if (power == W(1)) {
    for (std::size_t j = 0; j < row.n; j += A::L) {
      const int count = row.n - j < std::size_t(A::L) ? static_cast<int>(row.n - j) : A::L;
      const A u = A::load(r.v + j), q = core::fma(core::abs(A::load(r.wt + j)), c1, u);
      core::store(row.grad + j, core::fma(q, -sum, u) * c + W(0), count);
    }
  } else {
    const double unwide = Wide::unwide();
    const double c1d = static_cast<double>(c1), sumd = static_cast<double>(sum);
    for (std::size_t j = 0; j < row.n; ++j) {
      const double u = static_cast<double>(r.v[j]);
      const double q = std::fabs(static_cast<double>(r.wt[j])) * c1d + u;
      row.grad[j] = core::to_storage<G>((u - q * sumd) * factor * unwide + 0.0);
    }
  }
  return st.lost > 0 ? kInf : st.value;
}

// proximal_distillation in double: the pass gives the inputs of the proximal step on the
// classes the student keeps (log p from the teacher's scores, log q / p = d - log1p(C)).
template <typename A, typename S, typename G>
inline double proximal_row(const RowArgs<S, G>& row, const S* t, double temperature, double gamma,
                           Scratch& s, double divisor) {
  const double inf = kInf;
  const S* z = row.scores;
  const std::size_t n = row.n;
  Head<double> h;
  const Status status = head<A>(z, t, n, temperature, h);
  if (status == large)
    return quarter(row, t, temperature, s,
                   [&](const RowArgs<double, G>& r, const double* u, double t4) {
                     return distill::proximal_row<A>(r, u, t4, gamma, s, divisor);
                   });
  if (status != done) return finish(status, 0, s);
  std::size_t np;
  const Rows<double> r = work_rows<A>(n, 4, s, np);
  const Stats<double> st = pass<2, A>(z, t, n, h, r);
  if (!(st.value < inf)) return Row::invalid();
  std::size_t kept = 0;
  for (std::size_t j = 0; j < n; ++j) kept += r.s[j] > -inf;
  const double inv_p = core::Wide<double>::unwide() / (1 + st.ts);
  const double log_tail = core::log1p(st.ts), lognorm = core::log1p(st.ss);
  const double inv_q = 1 / (1 + st.ss);
  const double extra = st.lost / static_cast<double>(kept);
  const double log1p_c = st.c >= -0.5 ? core::log1p(st.c) : core::log(st.c1);
  try {
    core::softmax_prox<double> prox(kept, gamma);
    std::size_t j = 0;
    prox.set_inputs(
        h.s_max * h.inv_temp, lognorm,
        [&](std::size_t, double& p, double& log_p, double& log_q, double& ratio, double& q) {
          while (!(r.s[j] > -inf)) ++j;
          p = std::fabs(r.wt[j]) * inv_p;
          log_q = r.s[j] - lognorm;
          q = core::exp(r.s[j]) * inv_q;
          log_p = p > 0 ? (core::to_work<double>(t[j]) - h.t_max) * h.inv_temp - log_tail : -inf;
          ratio = p > 0 ? r.d[j] - log1p_c : inf;
          if (extra > 0) {
            p += extra;
            log_p = core::log(p);
            ratio = log_q - log_p;
          }
          ++j;
        });
    prox.solve();
    prox.balance();
    const double value = st.lost > 0 ? inf : prox.value();
    if (!(st.lost > 0) && !(value < inf)) return Row::invalid();
    if (row.grad) {
      for (std::size_t k = 0; k < kept; ++k) prox.r[k] -= extra;
      // The solver keeps classes by their scaled score, which can reach -inf
      // even at finite logits when the spread exceeds double.
      core::scatter_kept(
          r.s, n, prox.r, -row.scale / divisor,
          [&](std::size_t i) { return std::fabs(r.wt[i]) * inv_p; }, row.grad);
    }
    return value;
  } catch (...) {
    return Row::invalid();
  }
}

}  // namespace distill

// distillation H(softmax(t / T), softmax(z / T)) of teacher scores t.
struct Distillation {
  static const std::size_t block_classes = 32;
  double temperature;
  Distillation() : temperature(1) {}
  bool float_parameters() const {
    return core::normal_float(temperature) && core::normal_float(1 / temperature);
  }

  template <typename W, typename Isa, typename S, typename G, typename Targets>
  bool block(const BlockArgs<S, G>& b, const Targets& t, std::size_t first, double* values,
             bool* done, Scratch& s) const {
    distill::classic::block<core::Lanes<W, Isa>>(b, t, first, temperature, values, done, s);
    return true;
  }

  template <typename W, typename Isa, typename S, typename G>
  double operator()(const RowArgs<S, G>& row, const TeacherRow<S>& t, Scratch& s) const {
    return distill::row<core::Lanes<W, Isa>>(row, t.t, temperature, s, temperature);
  }
};

// proximal_distillation: the Moreau envelope with step gamma of H(softmax(t / T), softmax(.))
// at z / T, in double work.
struct ProximalDistillation {
  double temperature, gamma;
  ProximalDistillation() : temperature(1), gamma(1) {}

  template <typename W, typename Isa, typename S, typename G>
  double operator()(const RowArgs<S, G>& row, const TeacherRow<S>& t, Scratch& s) const {
    return distill::proximal_row<core::Lanes<double, Isa>>(row, t.t, temperature, gamma, s,
                                                           temperature);
  }
};

}  // namespace loss
SDCA_ISA_END

#endif  // SDCA_LOSS_DISTILL_H
