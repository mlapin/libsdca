#include <algorithm>

#include "projector.hpp"

namespace sdca {

template <typename RealType>
void Projector<RealType>::Project(
    RealType *x,
    const std::size_t n) {
  RealType t, lo, hi;
  std::vector<RealType> aux(x, x + n);
  ComputeThresholds(aux, t, lo, hi);
  Clamp(x, x + n, t, lo, hi);
}

template <typename RealType>
void Projector<RealType>::Project(
    RealType *x,
    const std::size_t n,
    const std::size_t num_col) {
  RealType t, lo, hi, *last = x + n, *end = x + n*num_col;
  std::vector<RealType> aux(n);
  for (; x != end; last += n) {
    std::copy(x, last, &aux[0]);
    ComputeThresholds(aux, t, lo, hi);
    Clamp(x, last, t, lo, hi);
    x = last;
  }
}

template <typename RealType>
void Projector<RealType>::Clamp(
    RealType *first,
    RealType *last,
    const RealType t,
    const RealType lo,
    const RealType hi) {
  if (hi <= lo) {
    std::fill(first, last, lo);
  } else if (hi == std::numeric_limits<RealType>::infinity()) {
    for (; first != last; ++first) {
      *first = std::max(lo, *first - t);
    }
  } else if (lo == -std::numeric_limits<RealType>::infinity()) {
    for (; first != last; ++first) {
      *first = std::min(*first - t, hi);
    }
  } else {
    for (; first != last; ++first) {
      *first = std::min(std::max(lo, *first - t), hi);
    }
  }
}

template class Projector<float>;
template class Projector<double>;

}
