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

template <typename Iterator,
          typename Result>
struct exp_thresholds : thresholds<Iterator, Result> {
  typedef thresholds<Iterator, Result> base;

  exp_thresholds() : base() {}

  exp_thresholds(const Result __t, const Result __lo, const Result __hi) :
    base(__t, __lo, __hi) {}

  exp_thresholds(const Result __t, const Result __lo, const Result __hi,
                 const Iterator __first, const Iterator __last) :
    base(__t, __lo, __hi, __first, __last) {}
};

template <typename Iterator,
          typename Result>
struct lambert_thresholds : thresholds<Iterator, Result> {
  typedef thresholds<Iterator, Result> base;

  lambert_thresholds() : base() {}

  lambert_thresholds(const Result __t, const Result __lo, const Result __hi) :
    base(__t, __lo, __hi) {}

  lambert_thresholds(const Result __t, const Result __lo, const Result __hi,
                     const Iterator __first, const Iterator __last) :
    base(__t, __lo, __hi, __first, __last) {}
};

template <typename Iterator,
          typename Result>
struct lambert_a_thresholds : lambert_thresholds<Iterator, Result> {
  typedef lambert_thresholds<Iterator, Result> base;
  Result a;

  lambert_a_thresholds() : base(), a(1) {}

  lambert_a_thresholds(const Result __a,
                       const Result __t, const Result __lo, const Result __hi) :
    base(__t, __lo, __hi), a(__a) {}

  lambert_a_thresholds(const Result __a,
                       const Result __t, const Result __lo, const Result __hi,
                       const Iterator __first, const Iterator __last) :
    base(__t, __lo, __hi, __first, __last), a(__a) {}
};

template <typename Result>
inline thresholds<Result*, Result>
make_thresholds(
    const Result t,
    const Result lo,
    const Result hi
  ) {
  return thresholds<Result*, Result>(t, lo, hi);
}

template <typename Iterator,
          typename Result>
inline thresholds<Iterator, Result>
make_thresholds(
    const Result t,
    const Result lo,
    const Result hi,
    const Iterator first,
    const Iterator last
  ) {
  return thresholds<Iterator, Result>(t, lo, hi, first, last);
}

template <typename Result>
inline exp_thresholds<Result*, Result>
make_exp_thresholds(
    const Result t,
    const Result lo,
    const Result hi
  ) {
  return exp_thresholds<Result*, Result>(t, lo, hi);
}

template <typename Iterator,
          typename Result>
inline exp_thresholds<Iterator, Result>
make_exp_thresholds(
    const Result t,
    const Result lo,
    const Result hi,
    const Iterator first,
    const Iterator last
  ) {
  return exp_thresholds<Iterator, Result>(t, lo, hi, first, last);
}

template <typename Result>
inline lambert_thresholds<Result*, Result>
make_lambert_thresholds(
    const Result t,
    const Result lo,
    const Result hi
  ) {
  return lambert_thresholds<Result*, Result>(t, lo, hi);
}

template <typename Iterator,
          typename Result>
inline lambert_thresholds<Iterator, Result>
make_lambert_thresholds(
    const Result t,
    const Result lo,
    const Result hi,
    const Iterator first,
    const Iterator last
  ) {
  return lambert_thresholds<Iterator, Result>(t, lo, hi, first, last);
}

template <typename Result>
inline lambert_a_thresholds<Result*, Result>
make_lambert_a_thresholds(
    const Result a,
    const Result t,
    const Result lo,
    const Result hi
  ) {
  return lambert_a_thresholds<Result*, Result>(a, t, lo, hi);
}

template <typename Iterator,
          typename Result>
inline lambert_a_thresholds<Iterator, Result>
make_lambert_a_thresholds(
    const Result a,
    const Result t,
    const Result lo,
    const Result hi,
    const Iterator first,
    const Iterator last
  ) {
  return lambert_a_thresholds<Iterator, Result>(a, t, lo, hi, first, last);
}

/**
 * Compute <prox(x),prox(x)> given the thresholds for prox(x) and x
 * without actually computing prox(x).
 **/
template <typename Iterator,
          typename Result,
          typename Summation = std_sum<Iterator, Result>>
inline Result
dot_prox_prox(
    const thresholds<Iterator, Result> t,
    Iterator first,
    Iterator last,
    const Summation sum = Summation()
    ) {
  Result num_hi = static_cast<Result>(std::distance(first, t.first));
  Result num_mi = static_cast<Result>(std::distance(t.first, t.last));
  Result num_lo = static_cast<Result>(std::distance(t.last, last));
  Result sum_mi = sum(t.first, t.last, static_cast<Result>(0));
  Result dot_mi = 0, comp = 0;
  std::for_each(t.first, t.last,
    [&](const Result x){ sum.add(x * x, dot_mi, comp); });
  return t.hi * t.hi * num_hi + t.t * t.t * num_mi + t.lo * t.lo * num_lo
    + dot_mi - static_cast<Result>(2) * t.t * sum_mi ;
}

/**
 * Compute <x,prox(x)> given the thresholds for prox(x) and x
 * without actually computing prox(x).
 **/
template <typename Iterator,
          typename Result,
          typename Summation = std_sum<Iterator, Result>>
inline Result
dot_prox(
    const thresholds<Iterator, Result> t,
    Iterator first,
    Iterator last,
    const Summation sum = Summation()
    ) {
  Result sum_hi = sum(first, t.first, static_cast<Result>(0));
  Result sum_mi = sum(t.first, t.last, static_cast<Result>(0));
  Result sum_lo = sum(t.last, last, static_cast<Result>(0));
  Result dot_mi = 0, comp = 0;
  std::for_each(t.first, t.last,
    [&](const Result x){ sum.add(x * x, dot_mi, comp); });
  return t.hi * sum_hi - t.t * sum_mi + t.lo * sum_lo + dot_mi;
}

template <typename Iterator,
          typename Result>
inline void
prox(
    const sdca::thresholds<Iterator, Result> thresholds,
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

template <typename Iterator,
          typename Result>
inline void
prox(
    const sdca::exp_thresholds<Iterator, Result> thresholds,
    Iterator first,
    Iterator last
    ) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  Data t(static_cast<Data>(thresholds.t));
  Data lo(static_cast<Data>(thresholds.lo));
  Data hi(static_cast<Data>(thresholds.hi));
  std::for_each(first, last,
    [=](Data& x){ x = std::max(lo, std::min(std::exp(x - t), hi)); });
}

template <typename Iterator,
          typename Result>
inline void
prox(
    const sdca::lambert_thresholds<Iterator, Result> thresholds,
    Iterator first,
    Iterator last
    ) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  Data t(static_cast<Data>(thresholds.t));
  Data lo(static_cast<Data>(thresholds.lo));
  Data hi(static_cast<Data>(thresholds.hi));
  std::for_each(first, last,
    [=](Data& x){ x = std::max(lo, std::min(lambert_w_exp(x - t), hi)); });
}

template <typename Iterator,
          typename Result>
inline void
prox(
    const sdca::lambert_a_thresholds<Iterator, Result> thresholds,
    Iterator first,
    Iterator last
    ) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  Data a(static_cast<Data>(thresholds.a));
  Data t(static_cast<Data>(thresholds.t));
  Data lo(static_cast<Data>(thresholds.lo));
  Data hi(static_cast<Data>(thresholds.hi));
  std::for_each(first, last,
    [=](Data& x){ x = std::max(lo, std::min(a * lambert_w_exp(x - t), hi)); });
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
    Iterator aux_first,
    Iterator aux_last,
    Algorithm compute,
    Types... params
    ) {
  std::copy(first, last, aux_first);
  auto thresholds = compute(aux_first, aux_last, params...);
  prox(thresholds, first, last);
}

template <typename Iterator,
          typename Algorithm,
          typename... Types>
inline void
prox(
    const typename std::iterator_traits<Iterator>::difference_type dim,
    Iterator first,
    Iterator last,
    Iterator aux_first,
    Iterator aux_last,
    Algorithm compute,
    Types... params
    ) {
  Iterator vec_last = first + dim;
  for (; first != last; vec_last += dim) {
    std::copy(first, vec_last, aux_first);
    auto thresholds = compute(aux_first, aux_last, params...);
    prox(thresholds, first, vec_last);
    first = vec_last;
  }
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
