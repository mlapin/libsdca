#ifndef SDCA_PROX_PROXDEF_H
#define SDCA_PROX_PROXDEF_H

#include <algorithm>
#include <iterator>
#include <limits>

#include "util/numeric.h"

namespace sdca {

enum class projection {
  zero = 0,
  constant,
  general
};

template <typename Iterator,
          typename Result>
struct thresholds {
  typedef Iterator iterator_type;
  typedef Result result_type;

  result_type t;
  result_type lo;
  result_type hi;
  iterator_type first;
  iterator_type last;

  thresholds() :
    t(0),
    lo(-std::numeric_limits<result_type>::infinity()),
    hi(+std::numeric_limits<result_type>::infinity())
  {}

  thresholds(const result_type __t,
             const result_type __lo,
             const result_type __hi) :
    t(__t),
    lo(__lo),
    hi(__hi)
  {}

  thresholds(const result_type __t,
             const result_type __lo,
             const result_type __hi,
             const iterator_type __first,
             const iterator_type __last) :
    t(__t),
    lo(__lo),
    hi(__hi),
    first(__first),
    last(__last)
  {}

};

template <typename Result>
inline
thresholds<Result*, Result>
make_thresholds(
    const Result t,
    const Result lo,
    const Result hi
  ) {
  return thresholds<Result*, Result>(t, lo, hi);
}

template <typename Iterator,
          typename Result>
inline
thresholds<Iterator, Result>
make_thresholds(
    const Result t,
    const Result lo,
    const Result hi,
    const Iterator first,
    const Iterator last
  ) {
  return thresholds<Iterator, Result>(t, lo, hi, first, last);
}

template <typename Iterator,
          typename Algorithm,
          typename... Types>
inline
void
project(
    Iterator first,
    Iterator last,
    Algorithm compute,
    Types... params
    ) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  std::vector<Data> aux(first, last);
  auto thresholds = compute(aux.begin(), aux.end(), params...);
  Data t(static_cast<Data>(thresholds.t));
  Data lo(static_cast<Data>(thresholds.lo));
  Data hi(static_cast<Data>(thresholds.hi));
  std::for_each(first, last,
    [=](Data& x){ x = std::max(lo, std::min(x - t, hi)); });
}

template <typename Iterator,
          typename Algorithm,
          typename... Types>
inline
void
project(
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    Algorithm compute,
    Types... params
    ) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  std::copy(first, last, aux_first);
  auto thresholds = compute(aux_first, aux_last, params...);
  Data t(static_cast<Data>(thresholds.t));
  Data lo(static_cast<Data>(thresholds.lo));
  Data hi(static_cast<Data>(thresholds.hi));
  std::for_each(first, last,
    [=](Data& x){ x = std::max(lo, std::min(x - t, hi)); });
}

template <typename Iterator,
          typename Algorithm,
          typename... Types>
inline
void
project(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    Algorithm compute,
    Types... params
    ) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  Iterator vec_last = first + dim;
  for (; first != last; vec_last += dim) {
    std::copy(first, vec_last, aux_first);
    auto thresholds = compute(aux_first, aux_last, params...);
    Data t(static_cast<Data>(thresholds.t));
    Data lo(static_cast<Data>(thresholds.lo));
    Data hi(static_cast<Data>(thresholds.hi));
    std::for_each(first, vec_last,
      [=](Data& x){ x = std::max(lo, std::min(x - t, hi)); });
    first = vec_last;
  }
}

}

#endif
