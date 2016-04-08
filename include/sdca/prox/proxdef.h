#ifndef SDCA_PROX_PROXDEF_H
#define SDCA_PROX_PROXDEF_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <vector>

namespace sdca {

enum class projection {
  zero = 0,
  constant,
  general
};

template <typename Type>
struct numeric_defaults {
  static constexpr std::size_t
  max_num_iter() { return 32; }
};

template <>
struct numeric_defaults<float> {
  static constexpr std::size_t
  max_num_iter() { return 32; }
};

template <typename Result,
          typename Iterator>
struct thresholds {
  typedef Result result_type;
  typedef Iterator iterator_type;

  Result t;
  Result lo;
  Result hi;
  Iterator first;
  Iterator last;

  thresholds() :
    t(0),
    lo(-std::numeric_limits<Result>::infinity()),
    hi(+std::numeric_limits<Result>::infinity())
  {}

  thresholds(const Result __t,
             const Result __lo,
             const Result __hi) :
    t(__t),
    lo(__lo),
    hi(__hi)
  {}

  thresholds(const Result __t,
             const Result __lo,
             const Result __hi,
             const Iterator __first,
             const Iterator __last) :
    t(__t),
    lo(__lo),
    hi(__hi),
    first(__first),
    last(__last)
  {}

};


template <typename Result,
          typename Iterator,
          typename Mapping>
struct generalized_thresholds : thresholds<Result, Iterator> {
  typedef thresholds<Result, Iterator> base;
  typedef Mapping map_type;

  map_type map;

  generalized_thresholds(Mapping __map) :
    base(),
    map(__map)
  {}

  generalized_thresholds(const Result __t,
                         const Result __lo,
                         const Result __hi,
                         Mapping __map) :
    base(__t, __lo, __hi),
    map(__map)
  {}

  generalized_thresholds(const Result __t,
                         const Result __lo,
                         const Result __hi,
                         const Iterator __first,
                         const Iterator __last,
                         Mapping __map) :
    base(__t, __lo, __hi, __first, __last),
    map(__map)
  {}

};


template <typename Result>
inline thresholds<Result, Result*>
make_thresholds(
    const Result t,
    const Result lo,
    const Result hi
  ) {
  assert(std::isfinite(t));
  return thresholds<Result, Result*>(t, lo, hi);
}


template <typename Result,
          typename Iterator>
inline thresholds<Result, Iterator>
make_thresholds(
    const Result t,
    const Result lo,
    const Result hi,
    const Iterator first,
    const Iterator last
  ) {
  assert(std::isfinite(t));
  return thresholds<Result, Iterator>(t, lo, hi, first, last);
}


template <typename Result,
          typename Mapping>
inline generalized_thresholds<Result, Result*, Mapping>
make_thresholds(
    const Result t,
    const Result lo,
    const Result hi,
    Mapping map
  ) {
  assert(std::isfinite(t));
  return generalized_thresholds<Result, Result*, Mapping>(t, lo, hi, map);
}


template <typename Result,
          typename Iterator,
          typename Mapping>
inline generalized_thresholds<Result, Iterator, Mapping>
make_thresholds(
    const Result t,
    const Result lo,
    const Result hi,
    const Iterator first,
    const Iterator last,
    Mapping map
  ) {
  assert(std::isfinite(t));
  return generalized_thresholds<Result, Iterator, Mapping>(
    t, lo, hi, first, last, map);
}


template <typename Result,
          typename Iterator>
inline void
prox(
    const sdca::thresholds<Result, Iterator>& thresholds,
    Iterator first,
    Iterator last
    ) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  Data t(static_cast<Data>(thresholds.t));
  Data lo(static_cast<Data>(thresholds.lo));
  Data hi(static_cast<Data>(thresholds.hi));
  std::for_each(first, last,
    [=](Data& x){ x = std::max(lo, std::min(x - t, hi)); });
}


template <typename Result,
          typename Iterator,
          typename Mapping>
inline void
prox(
    const sdca::generalized_thresholds<Result, Iterator, Mapping>& thresholds,
    Iterator first,
    Iterator last
    ) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  Data t(static_cast<Data>(thresholds.t));
  Data lo(static_cast<Data>(thresholds.lo));
  Data hi(static_cast<Data>(thresholds.hi));
  std::for_each(first, last,
    [=](Data& x){ x = std::max(lo, std::min(thresholds.map(x - t), hi)); });
}


template <typename Iterator,
          typename Algorithm,
          typename... Types>
inline void
prox(
    Iterator first,
    Iterator last,
    Algorithm compute,
    Types... params
    ) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  std::vector<Data> aux(first, last);
  auto thresholds = compute(aux.begin(), aux.end(), params...);
  prox(thresholds, first, last);
}


template <typename Iterator,
          typename Algorithm,
          typename... Types>
inline void
prox(
    Iterator first,
    Iterator last,
    Iterator aux,
    Algorithm compute,
    Types... params
    ) {
  std::copy(first, last, aux);
  auto thresholds = compute(aux, aux + std::distance(first, last), params...);
  prox(thresholds, first, last);
}


template <typename Iterator,
          typename Algorithm,
          typename... Types>
inline void
prox(
    Iterator a_first,
    Iterator a_last,
    Iterator b_first,
    Iterator b_last,
    Algorithm compute,
    Types... params
    ) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  std::vector<Data> a_aux(a_first, a_last);
  std::vector<Data> b_aux(b_first, b_last);
  auto thresholds = compute(
    a_aux.begin(), a_aux.end(), b_aux.begin(), b_aux.end(), params...);
  prox(thresholds.first, a_first, a_last);
  prox(thresholds.second, b_first, b_last);
}


template <typename Iterator,
          typename Algorithm,
          typename... Types>
inline void
prox(
    Iterator a_first,
    Iterator a_last,
    Iterator b_first,
    Iterator b_last,
    Iterator a_aux,
    Iterator b_aux,
    Algorithm compute,
    Types... params
    ) {
  std::copy(a_first, a_last, a_aux);
  std::copy(b_first, b_last, b_aux);
  auto thresholds = compute(
    a_aux, a_aux + std::distance(a_first, a_last),
    b_aux, b_aux + std::distance(b_first, b_last), params...);
  prox(thresholds.first, a_first, a_last);
  prox(thresholds.second, b_first, b_last);
}


template <typename Iterator,
          typename Algorithm,
          typename... Types>
inline void
prox(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux,
    Algorithm compute,
    Types... params
    ) {
  Iterator vec_last(first + dim), aux_last(aux + dim);
  for (; first != last; vec_last += dim) {
    std::copy(first, vec_last, aux);
    auto thresholds = compute(aux, aux_last, params...);
    prox(thresholds, first, vec_last);
    first = vec_last;
  }
}


/**
 * Computes the dot product
 *    <prox(x), prox(x)>
 * without computing prox(x) explicitly.
 **/
template <typename Result,
          typename Iterator>
inline Result
dot_prox_prox(
    const thresholds<Result, Iterator>& t,
    Iterator first,
    Iterator last
    ) {
  Result num_hi = static_cast<Result>(std::distance(first, t.first));
  Result num_mi = static_cast<Result>(std::distance(t.first, t.last));
  Result num_lo = static_cast<Result>(std::distance(t.last, last));
  Result sum_mi = std::accumulate(t.first, t.last, static_cast<Result>(0));
  Result dot_mi(0);
  std::for_each(t.first, t.last, [&](const Result x){ dot_mi += x * x; });
  return t.hi * t.hi * num_hi + t.t * t.t * num_mi + t.lo * t.lo * num_lo
    + dot_mi - static_cast<Result>(2) * t.t * sum_mi ;
}


/**
 * Computes the dot product
 *    <x, prox(x)>
 * without computing prox(x) explicitly.
 **/
template <typename Result,
          typename Iterator>
inline Result
dot_x_prox(
    const thresholds<Result, Iterator>& t,
    Iterator first,
    Iterator last
    ) {
  Result sum_hi = std::accumulate(first, t.first, static_cast<Result>(0));
  Result sum_mi = std::accumulate(t.first, t.last, static_cast<Result>(0));
  Result sum_lo = std::accumulate(t.last, last, static_cast<Result>(0));
  Result dot_mi(0);
  std::for_each(t.first, t.last, [&](const Result x){ dot_mi += x * x; });
  return t.hi * sum_hi - t.t * sum_mi + t.lo * sum_lo + dot_mi;
}


template <typename Iterator,
          typename Functor>
inline void
apply(
    Iterator first,
    Iterator last,
    Functor functor
    ) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  std::for_each(first, last, [=](Data& x){ x = functor(x); });
}


template <typename Iterator,
          typename Functor>
inline void
apply(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Functor functor
    ) {
  Iterator vec_last = first + dim;
  for (; first != last; vec_last += dim) {
    apply(first, vec_last, functor);
    first = vec_last;
  }
}

}

#endif
