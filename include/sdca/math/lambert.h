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
inline Type
lambert_w_iter_5(
    const Type w,
    const Type y
  ) {
  Type f0 = w - y, f1 = 1 + y;
  Type f11 = f1 * f1, f0y = f0 * y;
  Type f00y = f0 * f0y;
  return w - 4 * f0 * (6 * f1 * (f11 + f0y) + f00y)
    / (f11 * (24 * f11 + 36 * f0y) + f00y * (14 * y + f0 + 8));
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
inline Type
exp_approx(
    const Type x
  ) {
  Type y = 1 + x / static_cast<Type>(1024);
  y *= y; y *= y; y *= y; y *= y; y *= y;
  y *= y; y *= y; y *= y; y *= y; y *= y;
  return y;
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
inline float
lambert_w_exp(
    const float x
  ) {
  /* Initialize w for the Householder's iteration; consider intervals:
   * (-Inf, -104]         - exp underflows (exp(x)=0), return 0
   * (-104, -18]          - w = exp(x), return exp(x)
   * (-18, -1]            - w_0 = exp(x), return w_1
   * (-1, 8]              - w_0 = x, return w_2
   * (8, 536870912]       - w_0 = x - log(x), return w_1
   * (536870912, +Inf)    - (x + log(x)) = x, return x
   * Note: these intervals were chosen approximately
   * and could be optimized further (as well as the if-else branching).
   */
  float w;
  if (x > -1) { // (-1, +Inf)
    if (x <= 8) { // (-1, 8]
      w = lambert_w_iter_5(x, 1.0f);
    } else { // (8, +Inf)
      return (x <= 536870912) ? lambert_w_iter_5(x - std::log(x), x) : x;
    }
  } else { // (-Inf, -1]
    if (x > -18) { // (-18, -1]
      w = exp_approx(x);
    } else { // (-Inf, -18]
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
inline double
lambert_w_exp(
    const double x
  ) {
  /* Initialize w for the Householder's iteration; consider intervals:
   * (-Inf, -746]                 - exp underflows (exp(x)=0), return 0
   * (-746, -36]                  - w = exp(x), return exp(x)
   * (-36, -20]                   - w_0 = exp(x), return w_1
   * (-20, 0]                     - w_0 = exp(x), return w_2
   * (0, 4]                       - w_0 = x, return w_2
   * (4, 576460752303423488]      - w_0 = x - log(x), return w_2
   * (576460752303423488, +Inf)   - (x + log(x)) = x, return x
   */
  double w;
  if (x > 0) { // (0, +Inf)
    if (x <= 4) { // (0, 4]
      w = lambert_w_iter_5(x, 1.0);
    } else { // (4, +Inf)
      if (x <= 576460752303423488.0) { // (4, 576460752303423488]
        w = x - std::log(x);
        w = lambert_w_iter_5(w, x);
      } else { // (576460752303423488, +Inf)
        return x;
      }
    }
  } else { // (-Inf, 0]
    if (x > -36) { // (-36, 0]
      w = exp_approx(x);
      if (x > -20) { // (-20, 0]
        w = lambert_w_iter_5(w, exp_approx(x - w));
      }
    } else { // (-Inf, -36]
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
inline long double
lambert_w_exp(
    const long double x
  ) {
  /* Initialize w for the Householder's iteration;
   * use the same intervals as for double
   */
  long double w;
  if (x > 0) { // (0, +Inf)
    if (x <= 4) { // (0, 4]
      w = lambert_w_iter_5(x, 1.0L);
    } else { // (4, +Inf)
      if (x <= 576460752303423488.0L) { // (4, 576460752303423488]
        w = x - std::log(x);
        w = lambert_w_iter_5(w, x);
      } else { // (576460752303423488, +Inf)
        return x;
      }
    }
  } else { // (-Inf, 0]
    if (x > -36) { // (-36, 0]
      w = exp_approx(x);
      if (x > -20) { // (-20, 0]
        w = lambert_w_iter_5(w, exp_approx(x - w));
      }
    } else { // (-Inf, -36]
      return (x > -746) ? std::exp(x) : 0.0;
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
inline Type
lambert_w_exp_inverse(const Type w) {
  return w + std::log(w);
}


/**
 * Evaluates the function
 *    f0 = f(t) = sum_i W(exp(a_i + t))
 *
 * Note: the variable f0 must be properly initialized
 * (e.g., set to 0) before calling this function.
 **/
template <typename Type,
          typename Iterator>
inline void
sum_lambert_w_exp(
    const Iterator first,
    const Iterator last,
    const Type t,
    Type& f0
  ) {
  for (auto a = first; a != last; ++a) {
    f0 += lambert_w_exp(static_cast<Type>(*a) + t);
  }
}


/**
 * Evaluates the function
 *    f0 = f(t) = sum_i W(exp(a_i + t))
 * and its derivatives
 *    f1 = df/dt,
 *    f2 = d^2f/dt^2,
 *    ...
 *
 * Note: the variables f0, f1, ... must be properly initialized
 * (e.g., set to 0) before calling this function.
 **/
template <typename Type,
          typename Iterator>
inline void
sum_lambert_w_exp_derivatives(
    const Iterator first,
    const Iterator last,
    const Type t,
    Type& f0,
    Type& f1
  ) {
  for (auto a = first; a != last; ++a) {
    Type v = lambert_w_exp(static_cast<Type>(*a) + t);
    f0 += v;
    f1 += v / (1 + v);
  }
}


/**
 * Evaluates the function
 *    f0 = f(t) = sum_i W(exp(a_i + t))
 * and its derivatives
 *    f1 = df/dt,
 *    f2 = d^2f/dt^2,
 *    ...
 *
 * Note: the variables f0, f1, ... must be properly initialized
 * (e.g., set to 0) before calling this function.
 **/
template <typename Type,
          typename Iterator>
inline void
sum_lambert_w_exp_derivatives(
    const Iterator first,
    const Iterator last,
    const Type t,
    Type& f0,
    Type& f1,
    Type& f2
  ) {
  for (auto a = first; a != last; ++a) {
    Type v = lambert_w_exp(static_cast<Type>(*a) + t);
    Type d = 1 + v;
    f0 += v;
    f1 += v / d;
    f2 += v / (d * d * d);
  }
}


/**
 * Evaluates the function
 *    f0 = f(t) = sum_i W(exp(a_i + t))
 * and its derivatives
 *    f1 = df/dt,
 *    f2 = d^2f/dt^2,
 *    ...
 *
 * Note: the variables f0, f1, ... must be properly initialized
 * (e.g., set to 0) before calling this function.
 **/
template <typename Type,
          typename Iterator>
inline void
sum_lambert_w_exp_derivatives(
    const Iterator first,
    const Iterator last,
    const Type t,
    Type& f0,
    Type& f1,
    Type& f2,
    Type& f3
  ) {
  for (auto a = first; a != last; ++a) {
    Type v = lambert_w_exp(static_cast<Type>(*a) + t);
    Type d = 1 + v;
    Type d3 = d * d * d;
    f0 += v;
    f1 += v / d;
    f2 += v / d3;
    f3 += v * (1 - 2 * v) / (d3 * d * d);
  }
}

}

#endif
