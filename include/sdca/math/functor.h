#ifndef SDCA_MATH_FUNCTOR_H
#define SDCA_MATH_FUNCTOR_H

#include <algorithm>
#include <cmath>

#include "sdca/math/lambert.h"

namespace sdca {

/**
 * Computes
 *    exp(x).
 **/
template <typename Type>
struct exp_map {
  const Type operator()(const Type x) const { return std::exp(x); }
};

/**
 * Computes
 *    lambert_w_exp(x).
 **/
template <typename Type>
struct lambert_w_exp_map {
  const Type operator()(const Type x) const { return lambert_w_exp(x); }
};

// Evaluate W(exp(x+shift))/alpha without an underflowing intermediate W
// or an overflowing reciprocal alpha. Shift is kept separate from the
// caller's common score offset.
template <typename Type>
struct scaled_lambert_w_exp_map {
  Type alpha, shift, log_alpha;
  scaled_lambert_w_exp_map(Type a, Type s = 0) : alpha(a), shift(s), log_alpha(std::log(a)) {}
  Type operator()(Type x) const {
    Type argument = x + shift, w = lambert_w_exp(argument);
    return argument < -18 ? std::exp((argument - log_alpha) - w) : w / alpha;
  }
};

// Centered capped entropy probability, optionally with a quadratic term.
template <typename Type>
struct capped_entropy_map {
  Type shift, cap, bound;
  bool quadratic;
  capped_entropy_map(Type s, Type c, bool q)
      : shift(s), cap(c), bound(c > 0 ? std::log(c) + (q ? c : Type(0)) : Type(0)), quadratic(q) {}
  Type operator()(Type x) const {
    if (cap == 0) return 0;
    Type a = std::min(x + shift, bound);
    return quadratic ? lambert_w_exp(a) : std::exp(a);
  }
};

}  // namespace sdca

#endif
