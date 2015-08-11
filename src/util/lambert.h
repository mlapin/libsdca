#ifndef SDCA_UTIL_LAMBERT_H
#define SDCA_UTIL_LAMBERT_H

#include <iostream>
#include <cmath>
#include <limits>

#include "util/fmath.h"

namespace sdca {

/**
 * Omega constant, see
 *    https://oeis.org/A030178.
 * Omega = lambert_w(1); it is the solution to x * exp(x) = 1.
 */
const long double kOmega =
0.5671432904097838729999686622103555497538157871865125081351310792230457930866L;

/**
 * Schroeder's / Householder's iteration for the equation
 *    w - z * exp(-w) = 0
 * with expected convergence rate of order 5; see
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
inline Type
lambert_w_householder_5(
    const Type w,
    const Type y
  ) {
  Type f0 = w - y, f1 = static_cast<Type>(1) + y;
  Type f11 = f1 * f1, f0y = f0 * y;
  Type f00y = f0 * f0y;
  return w - static_cast<Type>(4) * f0 * (
      static_cast<Type>(6) * f1 * (f11 + f0y) + f00y
    ) / (
      f11 * (static_cast<Type>(24) * f11 + static_cast<Type>(36) * f0y) +
      f00y * (static_cast<Type>(14) * y + f0 + static_cast<Type>(8))
    );
}

/**
 * Fast approximation of the exponential function: (1 + x/1024)^1024.
 * For double, it is about 2 times faster than fmath::expd,
 * which in turn is about 4 times faster than std::exp.
 * Not accurate for x > 1; accuracy increases for x < -5 as x -> -Inf;
 * for x <= -36, the difference to std::exp is below 2^(-52);
 * for x in [-5, 1], it is accurate to about 1e-3 (more around 0).
 **/
template <typename Type>
inline Type
exp_approx(
    const Type x
  ) {
  Type y = static_cast<Type>(1) + x / static_cast<Type>(1024);
  y *= y; y *= y; y *= y; y *= y; y *= y;
  y *= y; y *= y; y *= y; y *= y; y *= y;
  return y;
}

/**
 * Lambert W function of exp(x),
 *    w = W_0(exp(x)).
 * Computed w satisfies the equation
 *    w + ln(w) = x.
 **/
inline double
lambert_w_exp(
    const double x
  ) {
  /* Initialize w for the Householder's iteration; consider intervals:
   * (-Inf, -700], (-700, -36], (-36, -20], (-20, -1],
   * (-1, 0.5], (0.5, 2], (2, 5.7647e+17], (5.7647e+17, +Inf)
   */
  double w;
  if (x > -1) { // (-1, +Inf)
    if (x <= 2.0) { // (-1, 2]
      if (x <= 0.5) { // (-1, 0.5]
        w = static_cast<double>(kOmega);
        w = lambert_w_householder_5(w, exp_approx(x - w));
      } else { // (0.5, 2]
        w = lambert_w_householder_5(x, 1.0);
      }
    } else { // (2, +Inf)
      if (x <= 5.7647e+17) { // (2, 5.7647e+17]
        w = x - static_cast<double>(fmath::log(static_cast<float>(x)));
        w = lambert_w_householder_5(w, x);
      } else { // (5.7647e+17, +Inf)
        return x;
      }
    }
  } else { // (-Inf, -1]
    if (x > -36.0) { // (-36, -1]
      if (x > -20.0) { // (-20, -1]
        w = exp_approx(x);
        w = lambert_w_householder_5(w, exp_approx(x - w));
      } else { // (-36, -20]
        w = exp_approx(x);
      }
    } else { // (-Inf, -36]
      if (x > -700.0) { // (-700, -36]
        return exp_approx(x);
      } else { // (-Inf, -700]
        return 0.0;
      }
    }
  }
  return lambert_w_householder_5(w, fmath::expd(x - w));
}

}

#endif
