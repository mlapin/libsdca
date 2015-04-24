#include "topkprojector.hpp"

namespace sdca {

template <typename RealType>
void TopKProjector<RealType>::Clamp(RealType *x, const std::size_t m,
    const RealType t, const RealType hi, const RealType lo) {
  if (hi <= lo) {
    std::fill_n(x, m, lo);
  } else if (hi == std::numeric_limits<RealType>::infinity()) {
    RealType *end = x + m;
    for (; x != end; ++x) {
      *x = std::max(lo, *x + t);
    }
  } else if (lo == -std::numeric_limits<RealType>::infinity()) {
    RealType *end = x + m;
    for (; x != end; ++x) {
      *x = std::min(*x + t, hi);
    }
  } else {
    RealType *end = x + m;
    for (; x != end; ++x) {
      *x = std::min(std::max(lo, *x + t), hi);
    }
  }
}

template class TopKProjector<float>;
template class TopKProjector<double>;

}
