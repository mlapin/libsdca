#include <algorithm>
#include <cassert>

#include "projector.hpp"

namespace sdca {

template <typename RealType>
void Projector<RealType>::Project(
    const std::size_t n,
    RealType *first) const {
  Project(first, first + n);
}

template <typename RealType>
void Projector<RealType>::Project(
    RealType *first,
    RealType *last) const {
  RealType t, lo, hi;
  std::vector<RealType> aux(first, last);
  ComputeThresholds(aux, t, lo, hi);
  Clamp(first, last, t, lo, hi);
}

template <typename RealType>
void Projector<RealType>::Project(
    const std::size_t n,
    RealType *first,
    std::vector<RealType> &aux) const {
  Project(first, first + n, aux);
}

template <typename RealType>
void Projector<RealType>::Project(
    RealType *first,
    RealType *last,
    std::vector<RealType> &aux) const {
  RealType t, lo, hi;
  std::copy(first, last, &aux[0]);
  ComputeThresholds(aux, t, lo, hi);
  Clamp(first, last, t, lo, hi);
}

template <typename RealType>
void Projector<RealType>::Project(
    const std::size_t num_row,
    const std::size_t num_col,
    RealType *x) const {
  RealType t, lo, hi, *last = x + num_row, *end = x + num_row*num_col;
  std::vector<RealType> aux(num_row);
  for (; x != end; last += num_row) {
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
    const RealType hi) const {
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
