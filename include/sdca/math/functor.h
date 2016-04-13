#ifndef SDCA_MATH_FUNCTOR_H
#define SDCA_MATH_FUNCTOR_H

#include <cmath>

#include "sdca/math/lambert.h"

namespace sdca {

/**
 * Computes
 *    exp(x).
 **/
template <typename Type>
struct exp_map {
  const Type operator() (const Type x) const { return std::exp(x); }
};

/**
 * Computes
 *    lambert_w_exp(x).
 **/
template <typename Type>
struct lambert_w_exp_map {
  const Type operator() (const Type x) const { return lambert_w_exp(x); }
};

/**
 * Computes
 *    a * lambert_w_exp(x),
 * where 'a' is a pre-defined constant.
 **/
template <typename Type>
struct a_lambert_w_exp_map {
  Type a;
  a_lambert_w_exp_map(const Type __a) : a(__a) {}
  const Type operator() (const Type x) const { return a * lambert_w_exp(x); }
};

}

#endif
