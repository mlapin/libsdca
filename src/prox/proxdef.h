#ifndef SDCA_PROX_PROXDEF_H
#define SDCA_PROX_PROXDEF_H

#include <algorithm>
#include <iterator>
#include <limits>
#include <vector>

namespace sdca {

enum class projection {
  zero = 0,
  constant,
  general
};

template <typename ForwardIterator>
struct thresholds {
  typedef ForwardIterator iterator_type;
  typedef typename std::iterator_traits<ForwardIterator>::value_type value_type;

  value_type t;
  value_type lo;
  value_type hi;
  ForwardIterator first;
  ForwardIterator last;

  thresholds() :
    t(0),
    lo(-std::numeric_limits<value_type>::infinity()),
    hi(+std::numeric_limits<value_type>::infinity())
  {}

  thresholds(const value_type& __t) :
    t(__t),
    lo(-std::numeric_limits<value_type>::infinity()),
    hi(+std::numeric_limits<value_type>::infinity())
  {}

  thresholds(const value_type& __t,
             const value_type& __lo,
             const value_type& __hi) :
    t(__t),
    lo(__lo),
    hi(__hi)
  {}

  thresholds(const value_type& __t,
             const value_type& __lo,
             const value_type& __hi,
             const iterator_type& __first,
             const iterator_type& __last) :
    t(__t),
    lo(__lo),
    hi(__hi),
    first(__first),
    last(__last)
  {}

};

template <typename Type>
inline
thresholds<Type*>
make_thresholds(const Type& t) {
  return thresholds<Type*>(t);
}

template <typename Type>
inline
thresholds<Type*>
make_thresholds(const Type& t, const Type& lo, const Type& hi) {
  return thresholds<Type*>(t, lo, hi);
}

template <typename ForwardIterator>
inline
thresholds<ForwardIterator>
make_thresholds(
    const typename std::iterator_traits<ForwardIterator>::value_type& t,
    const typename std::iterator_traits<ForwardIterator>::value_type& lo,
    const typename std::iterator_traits<ForwardIterator>::value_type& hi,
    const ForwardIterator& first,
    const ForwardIterator& last) {
  return thresholds<ForwardIterator>(t, lo, hi, first, last);
}

template <typename ForwardIterator,
          typename Algorithm,
          typename... Types>
inline
void
project(
    ForwardIterator first,
    ForwardIterator last,
    Algorithm compute,
    Types... params
    ) {
  using Type = typename std::iterator_traits<ForwardIterator>::value_type;
  std::vector<Type> aux(first, last);
  auto t = compute(aux.begin(), aux.end(), params...);
  std::for_each(first, last, [=](Type& x){
    x = std::max(t.lo, std::min(x - t.t, t.hi)); });
}

template <typename ForwardIterator,
          typename Algorithm,
          typename... Types>
inline
void
project(
    ForwardIterator first,
    ForwardIterator last,
    ForwardIterator aux_first,
    ForwardIterator aux_last,
    Algorithm compute,
    Types... params
    ) {
  using Type = typename std::iterator_traits<ForwardIterator>::value_type;
  std::copy(first, last, aux_first);
  auto t = compute(aux_first, aux_last, params...);
  std::for_each(first, last, [=](Type& x){
    x = std::max(t.lo, std::min(x - t.t, t.hi)); });
}

}

#endif

