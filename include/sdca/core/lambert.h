#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_CORE_LAMBERT_H)
#if SDCA_ISA_ID == 1
#define SDCA_CORE_LAMBERT_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_CORE_LAMBERT_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_CORE_LAMBERT_H3 1
#else
#define SDCA_CORE_LAMBERT_H4 1
#endif

#include <cmath>
#include <limits>
#include <type_traits>
#include "sdca/core/lanes.h"
#include "sdca/core/limits.h"
#include "sdca/core/math.h"

SDCA_ISA_BEGIN
namespace core {

// The scalar type of T: T itself, or that of lanes.
template <typename T>
struct lane_value {
  typedef T type;
};
template <typename T, typename Isa, int K>
struct lane_value<Lanes<T, Isa, K>> {
  typedef T type;
};

// One Householder step of order 5 for w - z exp(-w) = 0 from w = w_n, y = z exp(-w_n)
// (Fukushima, J. Comput. Appl. Math. 244, 2013).
template <typename T>
SDCA_LANE T lambert_w_iter_5(const T w, const T y) {
  typedef typename lane_value<T>::type S;
  const T f0 = w - y, f1 = S(1) + y;
  const T f11 = f1 * f1, f0y = f0 * y;
  const T f00y = f0 * f0y;
  return w - S(4) * f0 * (S(6) * f1 * (f11 + f0y) + f00y) /
                 (f11 * (S(24) * f11 + S(36) * f0y) + f00y * (S(14) * y + f0 + S(8)));
}

// (1 + x / n)^n, n = 2^10 by 10 squarings: within 0.001 max(1, exp(x)) of exp(x) on [-n, 1].
template <typename T>
SDCA_LANE T exp_approx(const T x) {
  typedef typename lane_value<T>::type S;
  const int squarings = 10;
  T y = S(1) + x / S(1 << squarings);
  for (int i = 0; i < squarings; ++i) y *= y;
  return y;
}

// exp(x - w) with the rounding of x - w restored: d = x - w rounds by e = (x - d) - w, exactly
// where |x| >= w (Fast2Sum: x <= -0.42 and x >= 1, where w = W(exp(x))), and exp(d) (1 + e)
// is within exp's error (|e| <= u |d|, e^2 below a rounding). A rounded x - w would cost
// u |x - w| more, 16 eps at x = -33; between -0.42 and 1, |x - w| < 1 and the correction is at
// most that rounding.
template <typename T>
inline T exp_difference(const T x, const T w) {
  const T d = x - w, e = (x - d) - w, y = std::exp(d);
  return y + y * e;
}
template <typename T, typename Isa, int K>
SDCA_LANE Lanes<T, Isa, K> exp_difference(const Lanes<T, Isa, K>& x, const Lanes<T, Isa, K>& w) {
  typedef Lanes<T, Isa, K> A;
  const A d = x - w, e = (x - d) - w, y = exp(d);
  return fma(y, e, y);
}

// The start of lambert_w_exp_large, the asymptotic series to (log(x) / x)^3, has 10 correct
// bits from x = 4 (lambert_branches::large) on (2^-10.46 at x = 4; tests/cpp/
// rewrite_smoke_limits.cpp), and its Halley steps triple them: those of steps_of_order, and an
// evaluation that confirms the last (3 for double).
const int kLambertLargeStartBits = 10;
template <typename T>
constexpr int lambert_large_steps() {
  return std::integral_constant<int,
                                limits::steps_of_order<T>(3, kLambertLargeStartBits) + 1>::value;
}

// W(exp(x)) for large positive x in log space: no polynomial intermediate overflows and
// x - w keeps its leading bits. A Halley step on f = w + log(w) - x moves w by f w / (w + 1):
// |f| <= u (w + 1) leaves w within u w; a step below cbrt(eps) w leaves the next within eps.
template <typename T>
inline T lambert_w_exp_large(const T x) {
  if (std::isinf(x)) return x;
  const T l1 = std::log(x);
  T w = x - l1;
  if (x < T(1) / std::numeric_limits<T>::epsilon()) {
    const T inv_x = T(1) / x;
    w += l1 * inv_x *
         (T(1) + inv_x * (T(0.5) * l1 - T(1) + inv_x * (l1 * (T(2) * l1 - T(9)) + T(6)) / T(6)));
  }
  static const T step_tol = std::cbrt(limits::eps<T>());
  for (int i = 0; i < lambert_large_steps<T>(); ++i) {
    const T f = (w - x) + std::log(w);
    if (std::abs(f) <= limits::roundoff<T>() * (w + T(1))) break;
    const T w1 = w + T(1);
    const T step = f / ((w1 / w) + (f / w1) / (T(2) * w));
    const T next = w - step;
    if (next == w) break;
    w = next;
    if (std::abs(step) <= step_tol * w) break;
  }
  return w;
}

// The branch points of lambert_w_exp: above `large` the log-space solve, above `start` the
// first Householder step from w = x, above `tiny` from exp_approx (a second one above
// `refine`), and below `tiny` w = exp(x). float solves its large branch in double.
// tiny = log(u): below it W(exp(x)) = exp(x) (1 - exp(x) + ...) rounds to exp(x). large and
// refine choose the cheaper of two accurate methods: two Householder steps from w = x are
// within the error of the log-space solve up to x = 4 (double) and beyond x = 12 (float), one
// step from exp_approx within half an ulp of two below x = -4 (double) and -2 (float)
// (tests/cpp/rewrite_smoke_limits.cpp, against long double).
template <typename T>
struct lambert_branches {
  typedef T large_type;
  static T large() { return T(4); }
  static T start() { return T(0); }
  static T refine() { return T(-20); }
  static T tiny() { return limits::log_roundoff<T>(); }
};
template <>
struct lambert_branches<float> {
  typedef double large_type;
  static float large() { return 8.0f; }
  static float start() { return -1.0f; }
  static float refine() { return 0.0f; }
  static float tiny() { return limits::log_roundoff<float>(); }
};

// W(exp(x)), the solution w of w + log(w) = x: |w + log(w) - x| < 4 eps max(1, x) for x > 0
// and |w exp(w) - exp(x)| < 4 eps for x <= 0.
template <typename T>
inline T lambert_w_exp(const T x) {
  typedef lambert_branches<T> B;
  if (std::isnan(x)) return x;
  T w;
  if (x > B::start()) {
    if (x > B::large()) return T(lambert_w_exp_large(static_cast<typename B::large_type>(x)));
    w = lambert_w_iter_5(x, T(1));
  } else if (x > B::tiny()) {
    w = exp_approx(x);
    if (x > B::refine()) w = lambert_w_iter_5(w, exp_approx(x - w));
  } else {
    return std::exp(x);
  }
  return lambert_w_iter_5(w, exp_difference(x, w));
}

// W(exp(a)) by one Householder step of order 5 from guess = W(exp(a0)), |a - a0| <= reach:
// the guess errs by at most e^|a - a0| - 1 relative (d log W / da <= 1), and the step leaves
// its fifth power (times at most one; tests/cpp/rewrite_smoke_limits.cpp sweeps it), below
// u / 4 for reach = 2^-ceil((digits + 2) / 5) (2^-11 for double): lambert_w_exp's accuracy.
template <typename T>
constexpr T lambert_warm_reach() {
  return limits::detail::pow2<T>(-((limits::digits<T>() + 2 + 4) / 5));
}
template <typename T>
inline T lambert_w_exp_from(const T a, const T guess) {
  return lambert_w_iter_5(guess, exp_difference(a, guess));
}

// lambert_w_exp errs by at most 4 u (3 u over float and double, tests/cpp/
// rewrite_smoke_limits.cpp, which checks this bound).
const int kLambertWRoundings = 4;
// lambert_probability errs by at most 6 u (5 u over float and double, same test).
const int kLambertProbabilityRoundings = 6;

// The coefficients c_n = (-(n + 1))^n / (n + 1)! of the series W(y) / y = exp(-W(y)) =
// sum_n c_n y^n, alternating with terms decreasing as (e y)^n.
constexpr long double kLambertRatio[] = {1.0L,
                                         -1.0L,
                                         1.5L,
                                         -2.666666666666666666666666666666666666667L,
                                         5.208333333333333333333333333333333333333L,
                                         -10.8L,
                                         23.34305555555555555555555555555555555556L,
                                         -52.01269841269841269841269841269841269841L,
                                         118.6252232142857142857142857142857142857L,
                                         -275.5731922398589065255731922398589065256L,
                                         649.7871723434744268077601410934744268078L,
                                         -1551.160519480519480519480519480519480519L,
                                         3741.44970295923854951632729410507188285L,
                                         -9104.50024115801893579671357449135226913L,
                                         22324.30851270660143428000570857713714857L,
                                         -55103.62197290383533769777156020542263929L,
                                         136808.8609039429356334221578930573639568L,
                                         -341422.0506658383633173549139935694557543L,
                                         855992.9659966075514633630250633224662342L,
                                         -2154990.206091088289321708745358647164724L,
                                         5445552.922314462431642316420035073285847L,
                                         -13807330.00216662949061923813184508062416L,
                                         35117044.98513923292853869855945334884375L,
                                         -89568002.56102797693072819557780090776314L,
                                         229041684.6187949813964782641734774133119L,
                                         -587103504.1171798492020292225245235010313L,
                                         1508256053.85779291964131713881295719607L,
                                         -3882630161.293188940385873468413841750925L,
                                         1.00139431366548296801391360156572342654e+10L,
                                         -2.587356736265760638992878359024929653546e+10L,
                                         6.696209709358073856946120522333454843979e+10L,
                                         -1.735711659599198077777078238043644666493e+11L,
                                         4.505680465642353886756098108484670411872e+11L,
                                         -1.171223178256487391904047636564823190916e+12L};

// The series' domain y <= 2^-5 (arg = log(y) <= -5 ln 2), where its terms fall by e y < 1/11
// each; a wider one takes more terms.
const int kLambertRatioLog2 = -5;

namespace detail {
constexpr long double lambert_ratio_term(const int n) {
  return (kLambertRatio[n] < 0 ? -kLambertRatio[n] : kLambertRatio[n]) *
         limits::detail::pow2<long double>(kLambertRatioLog2 * n);
}
constexpr int lambert_ratio_terms(const long double bound, const int n = 0) {
  return n + 1 >= int(sizeof kLambertRatio / sizeof kLambertRatio[0]) ||
                 lambert_ratio_term(n) <= bound
             ? n
             : lambert_ratio_terms(bound, n + 1);
}
}  // namespace detail

// The terms of the series kept on its domain: up to the first below eps / 32, which bounds the
// rest (alternating, decreasing): 7 for float, 15 for double, 18 for x87 long double.
template <typename T>
constexpr int lambert_ratio_terms() {
  return detail::lambert_ratio_terms(static_cast<long double>(limits::eps<T>()) / 32);
}

// log(2^-5), the largest argument of the series.
template <typename T>
constexpr T lambert_ratio_arg() {
  return limits::log_pow2<T>(kLambertRatioLog2);
}

// W(y) / y for 0 <= y <= 2^-5 by its series (scalars or lanes).
template <typename T>
SDCA_LANE T lambert_ratio_small(const T& y) {
  typedef typename lane_value<T>::type S;
  const int terms = lambert_ratio_terms<S>();
  static_assert(
      detail::lambert_ratio_term(terms) <= static_cast<long double>(limits::eps<S>()) / 32,
      "the series needs more coefficients");
  T g = y * S(kLambertRatio[terms - 1]) + S(kLambertRatio[terms - 2]);
  for (int n = terms - 3; n >= 0; --n) g = g * y + S(kLambertRatio[n]);
  return g;
}

// The probability W(alpha exp(h)) / alpha = exp(h - W(alpha exp(h))) of a proximal step at
// arg = h + log(alpha) (alpha > 0). On the series' domain exp(h) W(y) / y at y = alpha exp(h)
// (exp(h) <= max / 4): neither form of W is as accurate there, W / alpha erring by W's error,
// up to u |arg - W| from the rounding of arg - W in its last step, and exp(h - W) by u |h - W|.
// Below y = u / 4 the series rounds to one. Elsewhere W / alpha, and exp(h - W) on the
// series' domain beyond exp(h) <= max / 4 (above one, the callers' cap).
template <typename T>
inline T lambert_probability_at(const T arg, const T h, const T alpha) {
  const bool small = arg <= lambert_ratio_arg<T>();
  if (small && h <= limits::log_quarter_max<T>()) {
    const T e = std::exp(h);
    return arg < limits::log_pow2<T>(-(limits::digits<T>() + 2))
               ? e
               : e * lambert_ratio_small(alpha * e);
  }
  const T w = lambert_w_exp(arg);
  return small ? std::exp(h - w) : w / alpha;
}

template <typename T>
inline T lambert_probability(const T h, const T alpha, const T log_alpha) {
  return lambert_probability_at(h + log_alpha, h, alpha);
}

// lambert_w_exp_large on the double lanes of m; the others keep w.
template <typename Isa, int K>
SDCA_LANE Lanes<double, Isa, K> lambert_w_exp_large(const Lanes<double, Isa, K>& x,
                                                    const Ints<double, Isa, K>& m,
                                                    const Lanes<double, Isa, K>& w_in) {
  typedef Lanes<double, Isa, K> D;
  typedef Ints<double, Isa, K> M;
  const double eps = std::numeric_limits<double>::epsilon();
  const double inf = std::numeric_limits<double>::infinity();
  const D safe = select(m, x, D::splat(8.0)), l1 = log(safe), inv_x = 1.0 / safe;
  D w = safe - l1;
  const D series =
      l1 * inv_x * (1.0 + inv_x * (0.5 * l1 - 1.0 + inv_x * (l1 * (2.0 * l1 - 9.0) + 6.0) / 6.0));
  w = select(safe < 1.0 / eps, w + series, w);
  static const double step_tol = std::cbrt(eps);
  M active = m & (safe < inf);
  for (int i = 0; i < lambert_large_steps<double>() && any(active); ++i) {
    const D f = (w - safe) + log(w);
    active = active & ~(abs(f) <= limits::roundoff<double>() * (w + 1.0));
    if (!any(active)) break;
    const D w1 = w + 1.0;
    const D step = f / ((w1 / w) + (f / w1) / (2.0 * w));
    const D next = w - step;
    active = active & (next != w);
    w = select(active, next, w);
    active = active & ~(abs(step) <= step_tol * w);
  }
  w = select(safe == inf, safe, w);
  return select(m, w, w_in);
}
// Float lanes solve their large branch in double, lane by lane.
template <typename Isa, int K>
inline Lanes<float, Isa, K> lambert_w_exp_large(const Lanes<float, Isa, K>& x,
                                                const Ints<float, Isa, K>& m,
                                                const Lanes<float, Isa, K>& w) {
  struct large {
    float operator()(const float a) const {
      return static_cast<float>(lambert_w_exp_large(static_cast<double>(a)));
    }
  };
  return per_lane(w, x, m, large());
}

// lambert_w_exp on lanes: the branches of the scalar function per lane, exponentials of the
// lanes (results below the normal range are +0.0).
template <typename T, typename Isa, int K>
SDCA_LANE Lanes<T, Isa, K> lambert_w_exp(const Lanes<T, Isa, K>& x) {
  typedef Lanes<T, Isa, K> A;
  typedef Ints<T, Isa, K> M;
  typedef lambert_branches<T> B;
  const M start = x > B::start(), low = ~start & ~(x > B::tiny()), large = x > B::large();
  A w;
  if (!any(~low)) {
    w = exp(x);
  } else {
    w = select(start, lambert_w_iter_5(x, A::splat(T(1))), exp_approx(x));
    if (B::refine() < B::start()) {
      const M refine = ~start & (x > B::refine());
      if (any(refine)) w = select(refine, lambert_w_iter_5(w, exp_approx(x - w)), w);
    }
    const M other = low | large;
    w = lambert_w_iter_5(w,
                         exp_difference(select(other, A::zero(), x), select(other, A::zero(), w)));
    if (any(low)) w = select(low, exp(x), w);
  }
  if (any(large)) w = lambert_w_exp_large(x, large, w);
  return select(x != x, x, w);
}

// lambert_probability on lanes, as the scalar function: the series where every lane has a
// small W, else W / alpha with the series and exp(h - W) in the lanes of theirs.
template <typename T, typename Isa, int K>
SDCA_LANE Lanes<T, Isa, K> lambert_probability(const Lanes<T, Isa, K>& h, const T alpha,
                                               const T log_alpha) {
  typedef Lanes<T, Isa, K> A;
  typedef Ints<T, Isa, K> M;
  const A arg = h + log_alpha;
  const M small = arg <= lambert_ratio_arg<T>(),
          series = small & (h <= limits::log_quarter_max<T>());
  if (all(series)) {
    const A e = exp(h);
    return e * lambert_ratio_small(alpha * e);
  }
  const A w = lambert_w_exp(select(series, A::zero(), arg));
  A p = w / alpha;
  if (any(series)) {
    const A e = exp(select(series, h, A::zero()));
    p = select(series, e * lambert_ratio_small(alpha * e), p);
  }
  const M tail = small & ~series;
  if (any(tail)) p = select(tail, exp(select(tail, h - w, A::zero())), p);
  return p;
}

}  // namespace core
SDCA_ISA_END

#endif  // SDCA_CORE_LAMBERT_H
