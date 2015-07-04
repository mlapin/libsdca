#ifndef SDCA_PROX_THRESHOLD_HPP
#define SDCA_PROX_THRESHOLD_HPP

#include <algorithm>

namespace sdca {

template <class ForwardIterator, class Type>
inline void
clamp(
    ForwardIterator first,
    ForwardIterator last,
    const Type& t,
    const Type& lo = -std::numeric_limits<Type>::infinity(),
    const Type& hi = +std::numeric_limits<Type>::infinity()) {
  if (hi <= lo) {
    std::fill(first, last, lo);
  } else {
    std::for_each(first, last, );
    for (; first != last; ++first) {
      *first = std::min(std::max(lo, *first - t), hi);
    }
  }
}

template class Projector<float>;
template class Projector<double>;

}

#endif
