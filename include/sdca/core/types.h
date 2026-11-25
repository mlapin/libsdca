#ifndef SDCA_CORE_TYPES_H
#define SDCA_CORE_TYPES_H

// Types shared by every instruction set: the 16-bit storage types as their raw bits (IEEE half
// and bfloat16; sdca/core/storage.h converts them) and the comparison of the selections.
#include <cstdint>

namespace sdca {

struct half {
  uint16_t bits;
};
struct bfloat16 {
  uint16_t bits;
};

namespace detail {
// a > b, for the standard algorithms: compiled once, without target attributes.
struct Greater {
  template <typename T>
  bool operator()(const T& a, const T& b) const {
    return a > b;
  }
};
}  // namespace detail

}  // namespace sdca

#endif  // SDCA_CORE_TYPES_H
