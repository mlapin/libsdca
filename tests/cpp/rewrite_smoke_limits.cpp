// Smoke test of sdca/core/limits.h and the Lambert branch points of sdca/core/lambert.h: the
// limits against std::log and std::exp, the iteration counts at compile time, and the
// accuracy behind each branch point (float against double; double against long double where
// that is wider). With "print", the limits as "type name value" lines for
// tests/python/test_limits.py.
//
//   c++ -std=c++11 -O2 -Wall -Wextra -Wshadow -Werror -Wdouble-promotion -Iinclude THIS -o smoke
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

#include "sdca/core/lambert.h"
#include "sdca/core/limits.h"
#include "sdca/core/project_base.h"
#include "sdca/core/project_softmax.h"

namespace {

namespace core = sdca::SDCA_ISA_NAMESPACE::core;
namespace limits = sdca::limits;

static_assert(limits::key_bisection_steps<float>() == 32, "float keys");
static_assert(limits::key_bisection_steps<double>() == 64, "double keys");
static_assert(limits::value_bisection_steps<float>() == 278, "float values");
static_assert(limits::value_bisection_steps<double>() == 2099, "double values");
static_assert(core::root_steps<double>::total() == 129, "double root steps");
static_assert(core::lambert_ratio_terms<float>() == 7, "float series");
static_assert(core::lambert_ratio_terms<double>() == 15, "double series");
static_assert(limits::steps_of_order<double>(3, 1) == 4, "Halley from one bit");
static_assert(limits::steps_of_order<float>(2, 1) == 5, "Newton from one bit");
static_assert(limits::cbrt_eps<double>() == 1.0 / 131072, "2^-17");
static_assert(limits::cbrt_eps<float>() == 1.0f / 128, "2^-7");
static_assert(limits::min_over_eps<double>() ==
                  std::numeric_limits<double>::min() / std::numeric_limits<double>::epsilon(),
              "min / eps");

int failures = 0;

void check(const bool ok, const char* what, const char* type, const double value) {
  if (ok) return;
  ++failures;
  std::printf("FAIL %s %s %a (%.17g)\n", type, what, value, value);
}

// The limits against the C library: each logarithm within an ulp of std::log of its power of
// two, and the property its comment states.
template <typename T>
void limits_against_libm(const char* type) {
  typedef std::numeric_limits<T> L;
  const T ulp_tol = 1;
  auto near_log = [&](const char* name, const T value, const int n) {
    const T ref = std::log(std::ldexp(T(1), n));
    const T ulp = std::nextafter(std::abs(ref), std::numeric_limits<T>::infinity()) - std::abs(ref);
    check(std::abs(value - ref) <= ulp_tol * ulp, name, type, double(value));
  };
  near_log("log_eps", limits::log_eps<T>(), 1 - L::digits);
  near_log("log_roundoff", limits::log_roundoff<T>(), -L::digits);
  near_log("log_min", limits::log_min<T>(), L::min_exponent - 1);
  near_log("log_denorm_min", limits::log_denorm_min<T>(), L::min_exponent - L::digits);
  near_log("log_quarter_max", limits::log_quarter_max<T>(), L::max_exponent - 2);
  check(limits::log_max<T>() == std::log(L::max()), "log_max", type, double(limits::log_max<T>()));
  check(limits::eps<T>() == L::epsilon(), "eps", type, double(limits::eps<T>()));
  check(T(1) + limits::roundoff<T>() == T(1), "roundoff", type, double(limits::roundoff<T>()));
  check(limits::one_below<T>() == std::nextafter(T(1), T(0)), "one_below", type,
        double(limits::one_below<T>()));
  check(limits::inv_eps<T>() + T(0.5) == limits::inv_eps<T>(), "inv_eps", type,
        double(limits::inv_eps<T>()));
  check(limits::eps<T>() * limits::min_over_eps<T>() == L::min() &&
            std::fpclassify(limits::eps<T>() * (limits::min_over_eps<T>() / 2)) == FP_SUBNORMAL,
        "min_over_eps", type, double(limits::min_over_eps<T>()));
  // A logarithm x rounded to T moves exp(x) by up to |x| u.
  const T x_max = limits::log_quarter_max<T>(), x_u = limits::log_roundoff<T>();
  check(std::exp(x_max) <= L::max() / 4 * (1 + x_max * L::epsilon()), "log_quarter_max bound", type,
        double(x_max));
  check(std::fpclassify(std::exp(limits::log_min<T>() - limits::log_pow2<T>(1))) == FP_SUBNORMAL,
        "log_min bound", type, double(limits::log_min<T>()));
  check(std::exp(x_u) <= limits::roundoff<T>() * (1 - x_u * L::epsilon()), "log_roundoff bound",
        type, double(x_u));
  // 2^k exp(r) at the ends of the inline exp's range: normal, and finite.
  const T lo = limits::exp_reduction_min<T>(), hi = limits::exp_reduction_max<T>();
  const int k_lo = int(std::lround(lo / limits::log_pow2<T>(1))),
            k_hi = int(std::lround(hi / limits::log_pow2<T>(1)));
  check(k_lo >= L::min_exponent - 1 &&
            std::ldexp(std::exp(lo - k_lo * limits::log_pow2<T>(1)), k_lo) >= L::min(),
        "exp_reduction_min", type, double(lo));
  check(k_hi <= L::max_exponent && std::exp(hi) < std::numeric_limits<T>::infinity(),
        "exp_reduction_max", type, double(hi));
  const T c = limits::cbrt_eps<T>();
  check(c * c * c >= L::epsilon() && c * c * c <= 4 * L::epsilon(), "cbrt_eps", type, double(c));
}

// The float keys bisect any bracket to adjacent values in key_bisection_steps halvings.
template <typename T>
void bisection(const char* type) {
  const T brackets[][2] = {{-std::numeric_limits<T>::max(), std::numeric_limits<T>::max()},
                           {0, std::numeric_limits<T>::denorm_min()},
                           {-1, std::numeric_limits<T>::max()},
                           {T(1), T(2)}};
  for (const auto& b : brackets) {
    T lo = b[0], hi = b[1];
    int steps = 0;
    for (;;) {
      const T mid = core::bisect(lo, hi);
      if (mid == lo || mid == hi) break;
      (mid < T(0.5) ? hi : lo) = mid;  // toward the root 0.5 (or an end)
      ++steps;
    }
    check(steps <= limits::key_bisection_steps<T>() && std::nextafter(lo, hi) == hi,
          "key bisection", type, double(steps));
  }
}

// Relative error of x against ref in units of eps of T.
template <typename T, typename R>
double eps_error(const T x, const R ref) {
  if (ref == 0) return x == 0 ? 0 : std::numeric_limits<double>::infinity();
  return double(std::abs((R(x) - ref) / ref) / R(std::numeric_limits<T>::epsilon()));
}

// The Lambert branch points of T against the wider type R: tiny (exp(x) below log(u)), refine
// (one step below it as accurate as two), large (two steps from w = x below it as accurate as
// the log-space solve), and the probability.
template <typename T, typename R>
void lambert(const char* type) {
  typedef core::lambert_branches<T> B;
  double worst_w = 0, worst_refine = 0, worst_start = 0, worst_p = 0, worst_lanes = 0;
  for (int i = 0; i <= 40000; ++i) {
    const T x = T(-45 + 60.0 * i / 40000);
    const T w = core::lambert_w_exp(x);
    const R ref = core::lambert_w_exp(R(x));
    worst_w = std::max(worst_w, eps_error(w, ref));
    if (x > B::tiny() && x <= B::refine() && B::refine() < B::start()) {
      const T e = core::exp_approx(x), one = core::lambert_w_iter_5(e, core::exp_difference(x, e));
      worst_refine = std::max(worst_refine, eps_error(one, ref));
    }
    if (x > B::start() && x <= B::large()) {
      const T s1 = core::lambert_w_iter_5(x, T(1)),
              s2 = core::lambert_w_iter_5(s1, core::exp_difference(x, s1));
      worst_start = std::max(worst_start, eps_error(s2, ref));
    }
  }
  // One step from W(exp(x)) to W(exp(x + d)), |d| <= lambert_warm_reach.
  double worst_warm = 0;
  const T reach = core::lambert_warm_reach<T>();
  for (int i = 0; i <= 20000; ++i) {
    const T x = T(-45 + 55.0 * i / 20000), w0 = core::lambert_w_exp(x);
    if (!(w0 > 0)) continue;
    for (const T d : {reach, -reach, reach / 2, -reach / 2}) {
      const R ref = core::lambert_w_exp(R(x + d));
      worst_warm = std::max(worst_warm, eps_error(core::lambert_w_exp_from(T(x + d), w0), ref));
    }
  }
  check(worst_warm <= core::kLambertWRoundings / 2.0, "lambert_warm_reach", type, worst_warm);
  // The large branch to 1e12, within its lambert_large_steps.
  for (int i = 0; i <= 4000; ++i) {
    const T x = T(double(B::large()) * std::pow(10.0, 12.0 * i / 4000));
    worst_w = std::max(worst_w, eps_error(core::lambert_w_exp(x), core::lambert_w_exp(R(x))));
  }
  typedef core::softmax::lanes_of<T> lanes;
  typedef typename lanes::type A;
  for (int a = -12; a <= 12; ++a) {
    const T log_alpha = T(2.5 * a), alpha = std::exp(log_alpha);
    for (int i = 0; i <= 4000; i += A::L) {
      A h;
      for (int l = 0; l < A::L; ++l) h.set(l, T(-60 + 70.0 * (i + l) / 4000) - log_alpha);
      const A p = core::lambert_probability(h, alpha, log_alpha);
      for (int l = 0; l < A::L; ++l) {
        const R ra = R(alpha), rh = R(h[l]);
        const R ref = core::lambert_probability_at(rh + std::log(ra), rh, ra);
        // Above the flush of the lanes' exp to zero (core::exp, sdca/core/math.h).
        if (!(ref < R(1) && ref >= 4 * R(std::numeric_limits<T>::min()))) continue;
        worst_p =
            std::max(worst_p, eps_error(core::lambert_probability(h[l], alpha, log_alpha), ref));
        worst_lanes = std::max(worst_lanes, eps_error(p[l], ref));
      }
    }
  }
  std::printf(
      "lambert %-6s W %.2f, one step below refine %.2f, two from w = x below large %.2f, "
      "probability %.2f (lanes %.2f), warm %.2f eps\n",
      type, worst_w, worst_refine, worst_start, worst_p, worst_lanes, worst_warm);
  // The bounds the tolerances take (in u; 2 u per eps).
  check(worst_w <= core::kLambertWRoundings / 2.0, "lambert_w_exp", type, worst_w);
  check(worst_refine <= core::kLambertWRoundings / 2.0, "refine", type, worst_refine);
  check(worst_start <= core::kLambertWRoundings / 2.0, "large", type, worst_start);
  check(worst_p <= core::kLambertProbabilityRoundings / 2.0 &&
            worst_lanes <= core::kLambertProbabilityRoundings / 2.0,
        "lambert_probability", type, worst_p);
}

template <typename T>
void print(const char* type) {
  auto row = [&](const char* name, const double v) { std::printf("%s %s %a\n", type, name, v); };
  row("eps", double(limits::eps<T>()));
  row("roundoff", double(limits::roundoff<T>()));
  row("one_below", double(limits::one_below<T>()));
  row("inv_eps", double(limits::inv_eps<T>()));
  row("min_over_eps", double(limits::min_over_eps<T>()));
  row("log_eps", double(limits::log_eps<T>()));
  row("log_roundoff", double(limits::log_roundoff<T>()));
  row("log_min", double(limits::log_min<T>()));
  row("log_denorm_min", double(limits::log_denorm_min<T>()));
  row("log_max", double(limits::log_max<T>()));
  row("log_quarter_max", double(limits::log_quarter_max<T>()));
  row("exp_reduction_min", double(limits::exp_reduction_min<T>()));
  row("exp_reduction_max", double(limits::exp_reduction_max<T>()));
  row("cbrt_eps", double(limits::cbrt_eps<T>()));
  row("key_bisection_steps", limits::key_bisection_steps<T>());
  row("value_bisection_steps", limits::value_bisection_steps<T>());
  row("root_free_steps", core::root_steps<T>::free_steps());
  row("root_steps", core::root_steps<T>::total());
  row("lambert_ratio_terms", core::lambert_ratio_terms<T>());
  row("lambert_ratio_arg", double(core::lambert_ratio_arg<T>()));
  row("prox_tolerance", double(core::softmax::tolerance<T>()));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "print") {
    print<float>("float32");
    print<double>("float64");
    return 0;
  }
  limits_against_libm<float>("float");
  limits_against_libm<double>("double");
  bisection<float>("float");
  bisection<double>("double");
  lambert<float, double>("float");
  if (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits)
    lambert<double, long double>("double");
  std::printf(failures ? "%d failures\n" : "ok\n", failures);
  return failures ? 1 : 0;
}
