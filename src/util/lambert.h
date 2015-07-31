#ifndef SDCA_UTIL_LAMBERT_H
#define SDCA_UTIL_LAMBERT_H
#include <iostream>
#include <cmath>
#include <limits>

namespace sdca {

/**
 * Omega constant, see
 *    https://oeis.org/A030178.
 * Omega = lambert_w(1); it is the solution to x * exp(x) = 1.
 */
const long double kOmega =
0.5671432904097838729999686622103555497538157871865125081351310792230457930866L;

/**
 * Schroeder's / Householder's iteration of order 4 for the equation
 *    w - z * exp(-w) = 0.
 * Expected convergence rate of order 5, see
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
lambert_w_householder_4(
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
 * Lambert W function of exp(x),
 *    w = W_0(exp(x)).
 * Computed w satisfies the equation
 *    w + ln(w) = x.
 **/
template <typename Type>
inline Type
lambert_w_exp(
    const Type x
  ) {
  Type w, y, w1(0);
  if (x > static_cast<Type>(50)) {
    // x > 0, w \approx x, w_0 = x, y_0 = exp(x - w_0) = 1
    w = x - std::log(x);
  } else if (x > static_cast<Type>(0.1)) {
    // x > 0, w \approx x, w_0 = x, y_0 = exp(x - w_0) = 1
    w = x;
  } else if (x < - static_cast<Type>(1)) {
    // x < 0, ln(w) \approx x, w_0 = exp(x), y_0 = exp(x - w_0)
    if (x < static_cast<Type>(-500)) return static_cast<Type>(0);
    w = std::exp(x);
  } else {
    // x = 0, w = Omega
    w = static_cast<Type>(kOmega);
  }
  int count = 0;
  while (w != w1 && count < 40) {
    w1 = w;
    y = std::exp(x - w);
    w = lambert_w_householder_4(w, y);
    if (++count > 40) {
      std::cout << x << ", " << count << std::endl;
    }
  }
  return w;
}

}

#endif
