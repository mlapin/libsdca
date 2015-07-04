#ifndef SDCA_PROX_PROXDEF_H
#define SDCA_PROX_PROXDEF_H

#include <iterator>
#include <limits>

namespace sdca {

enum class projection_case {
  zero = 0,
  constant,
  general
};

template <class ForwardIterator>
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
    hi(+std::numeric_limits<value_type>::infinity()),
    first(nullptr),
    last(nullptr)
  {}

  thresholds(const value_type& __t) :
    t(__t),
    lo(-std::numeric_limits<value_type>::infinity()),
    hi(+std::numeric_limits<value_type>::infinity()),
    first(nullptr),
    last(nullptr)
  {}

  thresholds(const value_type& __t,
             const value_type& __lo,
             const value_type& __hi) :
    t(__t),
    lo(__lo),
    hi(__hi),
    first(nullptr),
    last(nullptr)
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

template <class Type>
inline
thresholds<Type*>
make_thresholds(const Type& t) {
  return thresholds<Type*>(t);
}

template <class Type>
inline
thresholds<Type*>
make_thresholds(const Type& t, const Type& lo, const Type& hi) {
  return thresholds<Type*>(t, lo, hi);
}

template <class ForwardIterator>
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

#endif

