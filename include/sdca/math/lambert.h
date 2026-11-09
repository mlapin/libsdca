#ifndef SDCA_MATH_LAMBERT_H
#define SDCA_MATH_LAMBERT_H

#include <cmath>
#include <limits>

namespace sdca {

/**
 * Omega constant, see
 *    https://oeis.org/A030178.
 * Omega = lambert_w(1) = lambert_w_exp(0);
 * it is the solution to w * exp(w) = 1, w + log(w) = 0.
 */
const long double kOmega =
    0.5671432904097838729999686622103555497538157871865125081351310792230457930866L;

/**
 * Householder's iteration for the equation
 *    w - z * exp(-w) = 0
 * with convergence rate of order 5; see
 * [1] A. Householder, The numerical treatment of a single nonlinear equation.
 *     McGraw-Hill, 1970.
 * [2] T. Fukushima, Precise and fast computation of Lambert W-functions
 *     without transcendental function evaluations.
 *     Journal of Computational and Applied Mathematics 244 (2013): 77-89.
 *
 * Input: w = w_n, y = z * exp(-w_n).
 * Returns: w_{n+1}.
 **/
template <typename Type>
inline Type lambert_w_iter_5(const Type w, const Type y) {
  Type f0 = w - y, f1 = 1 + y;
  Type f11 = f1 * f1, f0y = f0 * y;
  Type f00y = f0 * f0y;
  return w - 4 * f0 * (6 * f1 * (f11 + f0y) + f00y) /
                 (f11 * (24 * f11 + 36 * f0y) + f00y * (14 * y + f0 + 8));
}

/**
 * Fast (crude) approximation of the exponential function:
 *    (1 + x/1024)^1024.
 * Note: not accurate for x < -1024 and x > 1.
 * For x in [-1024, 1], the following holds
 *    | std::exp(x) - sdca::exp_approx(x) | < tol,
 * where
 *    tol = 0.001 * max(1, std::exp(x)).
 **/
template <typename Type>
inline Type exp_approx(const Type x) {
  Type y = 1 + x / static_cast<Type>(1024);
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  return y;
}

// Positive large arguments are solved in log space so polynomial intermediates
// never overflow and (x - w) never loses leading mantissa bits.
template <typename T>
inline T lambert_w_exp_large(T x) {
  if (std::isinf(x)) return x;
  T l1 = std::log(x);
  T w = x - l1;
  if (x < T(1) / std::numeric_limits<T>::epsilon()) {
    T inv_x = T(1) / x;
    w += l1 * inv_x *
         (T(1) + inv_x * (T(0.5) * l1 - T(1) + inv_x * (l1 * (T(2) * l1 - T(9)) + T(6)) / T(6)));
  }
  const T eps = std::numeric_limits<T>::epsilon();
  const T tol = T(2) * eps * x;
  const T step_tol = std::cbrt(eps);
  for (int i = 0; i < 6; ++i) {
    T f = (w - x) + std::log(w);
    if (std::abs(f) <= tol) break;
    T w1 = w + T(1);
    T step = f / ((w1 / w) + (f / w1) / (T(2) * w));
    T next = w - step;
    if (next == w) break;
    w = next;
    if (std::abs(step) <= step_tol * w) break;
  }
  return w;
}

/**
 * Lambert W function of exp(x),
 *    w = W(exp(x)).
 * Computed w satisfies
 *    w + log(w) = x
 * or, equivalently,
 *    w * exp(w) = exp(x).
 * For positive x,
 *    (w + log(w) - x) < 4 * eps * max(1, x),
 * and for negative x,
 *    (w * exp(w) - exp(x)) < 4 * eps,
 * where eps = std::numeric_limits<float>::epsilon().
 **/
inline float lambert_w_exp(const float x) {
  /* Initialize w for the Householder's iteration; consider intervals:
   * (-Inf, -104]         - exp underflows (exp(x)=0), return 0
   * (-104, -18]          - w = exp(x), return exp(x)
   * (-18, -1]            - w_0 = exp(x), return w_1
   * (-1, 8]              - w_0 = x, return w_2
   * (8, +Inf)           - log-space Newton in double precision
   * Note: these intervals were chosen approximately
   * and could be optimized further (as well as the if-else branching).
   */
  if (std::isnan(x)) return x;
  float w;
  if (x > -1) {    // (-1, +Inf)
    if (x <= 8) {  // (-1, 8]
      w = lambert_w_iter_5(x, 1.0f);
    } else {  // (8, +Inf)
      return static_cast<float>(lambert_w_exp_large(static_cast<double>(x)));
    }
  } else {          // (-Inf, -1]
    if (x > -18) {  // (-18, -1]
      w = exp_approx(x);
    } else {  // (-Inf, -18]
      return (x > -104) ? std::exp(x) : 0.0f;
    }
  }
  return lambert_w_iter_5(w, std::exp(x - w));
}

/**
 * Lambert W function of exp(x),
 *    w = W(exp(x)).
 * Computed w satisfies
 *    w + log(w) = x
 * or, equivalently,
 *    w * exp(w) = exp(x)
 * For positive x,
 *    (w + log(w) - x) < 4 * eps * max(1, x),
 * and for negative x,
 *    (w * exp(w) - exp(x)) < 4 * eps,
 * where eps = std::numeric_limits<double>::epsilon().
 **/
inline double lambert_w_exp(const double x) {
  /* Initialize w for the Householder's iteration; consider intervals:
   * (-Inf, -746]                 - exp underflows (exp(x)=0), return 0
   * (-746, -36]                  - w = exp(x), return exp(x)
   * (-36, -20]                   - w_0 = exp(x), return w_1
   * (-20, 0]                     - w_0 = exp(x), return w_2
   * (0, 4]                       - w_0 = x, return w_2
   * (4, +Inf)                   - log-space Newton
   */
  if (std::isnan(x)) return x;
  double w;
  if (x > 0) {     // (0, +Inf)
    if (x <= 4) {  // (0, 4]
      w = lambert_w_iter_5(x, 1.0);
    } else {  // (4, +Inf)
      return lambert_w_exp_large(x);
    }
  } else {          // (-Inf, 0]
    if (x > -36) {  // (-36, 0]
      w = exp_approx(x);
      if (x > -20) {  // (-20, 0]
        w = lambert_w_iter_5(w, exp_approx(x - w));
      }
    } else {  // (-Inf, -36]
      return (x > -746) ? std::exp(x) : 0.0;
    }
  }
  return lambert_w_iter_5(w, std::exp(x - w));
}

/**
 * Lambert W function of exp(x),
 *    w = W_0(exp(x)).
 * Computed w satisfies
 *    w + log(w) = x
 * or, equivalently,
 *    w * exp(w) = exp(x)
 * For positive x,
 *    (w + log(w) - x) < 4 * eps * max(1, x),
 * and for negative x,
 *    (w * exp(w) - exp(x)) < 4 * eps,
 * where eps = std::numeric_limits<double>::epsilon().
 * Note: this implementation was optimized for double, not for long double.
 **/
inline long double lambert_w_exp(const long double x) {
  /* Initialize w for the Householder's iteration;
   * use the same intervals as for double
   */
  if (std::isnan(x)) return x;
  long double w;
  if (x > 0) {     // (0, +Inf)
    if (x <= 4) {  // (0, 4]
      w = lambert_w_iter_5(x, 1.0L);
    } else {  // (4, +Inf)
      return lambert_w_exp_large(x);
    }
  } else {          // (-Inf, 0]
    if (x > -36) {  // (-36, 0]
      w = exp_approx(x);
      if (x > -20) {  // (-20, 0]
        w = lambert_w_iter_5(w, exp_approx(x - w));
      }
    } else {  // (-Inf, -36]
      w = std::exp(x);
      return w == 0 ? w : lambert_w_iter_5(w, std::exp(x - w));
    }
  }
  return lambert_w_iter_5(w, std::exp(x - w));
}

/**
 * Inverse of the Lambert W function of exp(x).
 * Computes
 *    x = w + log(w)
 **/
template <typename Type>
inline Type lambert_w_exp_inverse(const Type w) {
  return w + std::log(w);
}

}  // namespace sdca

#endif
