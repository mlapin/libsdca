#ifndef SDCA_PROX_PROXDEF_H
#define SDCA_PROX_PROXDEF_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "sdca/math/compensated.h"
#include "sdca/utility/scratch.h"

namespace sdca {

// How the multilabel hinge projection (two-simplex) solves. float64: in double,
// kept only if its acceptance check passes, else the hardened extended-precision
// solver. extended: always the hardened solver.
// The top-k hinge projections use the hardened solver in both modes.
enum class projection_precision { float64, extended };

enum class projection { zero = 0, constant, general };

namespace detail {

// Unsigned keys ordered like the finite floats, with -0 just below +0.
template <typename T>
struct float_key {};
template <>
struct float_key<float> {
  using type = std::uint32_t;
};
template <>
struct float_key<double> {
  using type = std::uint64_t;
};

template <typename T, typename Key = typename float_key<T>::type>
inline Key to_key(const T x) {
  Key bits;
  std::memcpy(&bits, &x, sizeof bits);
  const Key sign = Key(1) << (8 * sizeof(Key) - 1);
  return (bits & sign) ? Key(~bits) : Key(bits | sign);
}

template <typename T, typename Key = typename float_key<T>::type>
inline T from_key(const Key key) {
  const Key sign = Key(1) << (8 * sizeof(Key) - 1);
  const Key bits = (key & sign) ? Key(key & ~sign) : Key(~key);
  T x;
  std::memcpy(&x, &bits, sizeof x);
  return x;
}

// The largest finite x with !test(x), or -inf if there is none, for a test
// that is monotone in x; the search starts at guess (any value).
template <typename T, typename Guess, typename Test>
inline T monotone_cut(const Guess guess, Test test) {
  using Key = typename float_key<T>::type;
  const Key lowest = to_key(std::numeric_limits<T>::lowest());
  const Key highest = to_key(std::numeric_limits<T>::max());
  if (test(std::numeric_limits<T>::lowest())) return -std::numeric_limits<T>::infinity();
  if (!test(std::numeric_limits<T>::max())) return std::numeric_limits<T>::max();
  // Clamp before narrowing: an out-of-range conversion is undefined.
  T start(0);
  if (guess >= Guess(std::numeric_limits<T>::max()))
    start = std::numeric_limits<T>::max();
  else if (guess <= Guess(std::numeric_limits<T>::lowest()))
    start = std::numeric_limits<T>::lowest();
  else if (guess == guess)
    start = T(guess);
  // Gallop from the guess to a bracket [no, yes], then bisect it.
  Key no = to_key(start), yes = no;
  Key step = 1;
  const auto next = [](Key s) { return s > std::numeric_limits<Key>::max() / 2 ? s : 2 * s; };
  if (test(start)) {
    for (;; step = next(step)) {
      no = yes - lowest > step ? yes - step : lowest;
      if (!test(from_key<T>(no))) break;
      yes = no;
    }
  } else {
    for (;; step = next(step)) {
      yes = highest - no > step ? no + step : highest;
      if (test(from_key<T>(yes))) break;
      no = yes;
    }
  }
  while (yes - no > 1) {
    const Key middle = no + (yes - no) / 2;
    if (test(from_key<T>(middle)))
      yes = middle;
    else
      no = middle;
  }
  return from_key<T>(no);
}

// std::partition by a test that is monotone in the (finite) values. Long
// ranges compare against the cut of the test instead of evaluating it:
// the same decisions, so the same order, at the cost of one comparison.
template <typename Iterator, typename Test, typename Guess>
inline Iterator partition_monotone(Iterator first, Iterator last, Test test, const Guess guess,
                                   std::true_type) {
  using Data = typename std::iterator_traits<Iterator>::value_type;
  if (std::distance(first, last) < 32) return std::partition(first, last, test);
  const Data cut = monotone_cut<Data>(guess, test);
  return std::partition(first, last, [cut](const Data x) { return x > cut; });
}

template <typename Iterator, typename Test, typename Guess>
inline Iterator partition_monotone(Iterator first, Iterator last, Test test, const Guess,
                                   std::false_type) {
  return std::partition(first, last, test);
}

template <typename Iterator, typename Test, typename Guess>
inline Iterator partition_monotone(Iterator first, Iterator last, Test test, const Guess guess) {
  using Data = typename std::iterator_traits<Iterator>::value_type;
  using Exact = std::integral_constant<bool, std::is_same<Data, float>::value ||
                                                 std::is_same<Data, double>::value>;
  return partition_monotone(first, last, test, guess, Exact());
}

// lo n_lo + hi n_hi as s + e: opposing bound masses cancel exactly, so a
// small mass beside them survives (fma recovers the products' rounding).
template <typename T>
inline T bound_mass(const T lo, const T n_lo, const T hi, const T n_hi, T& e) {
  const T p_lo = lo * n_lo, p_hi = hi * n_hi;
  if (p_lo == 0 || p_hi == 0) {
    e = 0;
    return p_lo + p_hi;
  }
  const T s = two_sum(p_lo, p_hi, e);
  e += std::fma(lo, n_lo, -p_lo) + std::fma(hi, n_hi, -p_hi);
  return s;
}

// The prefix sums of value(0), ..., value(n - 1) with their rounding errors:
// the sum of [i, j) loses nothing to the magnitude of the values before i.
template <typename T>
struct prefix_sums {
  entropy_buffer<T> hi, lo;

  template <typename Value>
  prefix_sums(const std::size_t n, Value value) : hi(n + 1), lo(n + 1) {
    T *h = hi.data(), *l = lo.data();
    h[0] = l[0] = T(0);
    for (std::size_t i = 0; i < n; ++i) {
      T e;
      h[i + 1] = two_sum(h[i], value(i), e);
      l[i + 1] = l[i] + e;
    }
  }

  T operator()(const std::size_t i, const std::size_t j) const {
    return (hi.data()[j] - hi.data()[i]) + (lo.data()[j] - lo.data()[i]);
  }
};

// A power of two u with u * magnitude * factor well below overflow; one
// when no scaling is needed, so that ordinary inputs are solved unscaled.
template <typename T>
inline T overflow_unit(const T magnitude, T factor) {
  factor = std::min(std::max(factor, T(1)), T(18446744073709551616.0));  // 2^64
  if (!(magnitude * factor > std::numeric_limits<T>::max() / 16)) return T(1);
  const int exponent = std::ilogb(magnitude) + std::ilogb(factor) + 6;
  return std::ldexp(T(1), std::numeric_limits<T>::max_exponent - exponent);
}

// The largest x with !test(x) for a test monotone in x, by bisection of the
// float keys where they exist and of the values otherwise.
template <typename T, typename Test>
inline T monotone_root(const T guess, Test test, std::true_type) {
  return monotone_cut<T>(guess, test);
}

template <typename T, typename Test>
inline T monotone_root(const T, Test test, std::false_type) {
  T no = std::numeric_limits<T>::lowest(), yes = std::numeric_limits<T>::max();
  if (test(no)) return -std::numeric_limits<T>::infinity();
  if (!test(yes)) return yes;
  for (;;) {
    const T middle = no / 2 + yes / 2;
    if (middle == no || middle == yes) return no;
    (test(middle) ? yes : no) = middle;
  }
}

template <typename T, typename Test>
inline T monotone_root(const T guess, Test test) {
  using Keyed =
      std::integral_constant<bool, std::is_same<T, float>::value || std::is_same<T, double>::value>;
  return monotone_root(guess, test, Keyed());
}

}  // namespace detail

template <typename Type>
struct numeric_defaults {
  static constexpr std::size_t max_num_iter() { return 32; }

  static constexpr Type epsilon_relative() { return static_cast<Type>(0.001); }
};

template <typename Result, typename Iterator>
struct thresholds {
  typedef Result result_type;
  typedef Iterator iterator_type;

  Result t;
  // Ordinary projections retain a small threshold correction separately.
  Result origin = 0;
  Result correction = 0;
  Result lo;
  Result hi;
  Iterator first;
  Iterator last;

  thresholds()
      : t(0),
        lo(-std::numeric_limits<Result>::infinity()),
        hi(+std::numeric_limits<Result>::infinity()) {}

  thresholds(const Result t_, const Result lo_, const Result hi_)
      : t(t_), correction(t_), lo(lo_), hi(hi_) {}

  thresholds(const Result t_, const Result lo_, const Result hi_, const Iterator first_,
             const Iterator last_)
      : t(t_), correction(t_), lo(lo_), hi(hi_), first(first_), last(last_) {}
};

template <typename Result, typename Iterator, typename Mapping>
struct generalized_thresholds : thresholds<Result, Iterator> {
  typedef thresholds<Result, Iterator> base;
  typedef Mapping map_type;

  map_type map;

  generalized_thresholds(Mapping map_) : base(), map(map_) {}

  generalized_thresholds(const Result t_, const Result lo_, const Result hi_, Mapping map_)
      : base(t_, lo_, hi_), map(map_) {}

  generalized_thresholds(const Result t_, const Result lo_, const Result hi_, const Iterator first_,
                         const Iterator last_, Mapping map_)
      : base(t_, lo_, hi_, first_, last_), map(map_) {}
};

template <typename Result>
inline thresholds<Result, Result*> make_thresholds(const Result t, const Result lo,
                                                   const Result hi) {
  assert(std::isfinite(t));
  return thresholds<Result, Result*>(t, lo, hi);
}

template <typename Result, typename Iterator>
inline thresholds<Result, Iterator> make_thresholds(const Result t, const Result lo,
                                                    const Result hi, const Iterator first,
                                                    const Iterator last) {
  assert(std::isfinite(t));
  return thresholds<Result, Iterator>(t, lo, hi, first, last);
}

template <typename Result, typename Mapping>
inline generalized_thresholds<Result, Result*, Mapping> make_thresholds(const Result t,
                                                                        const Result lo,
                                                                        const Result hi,
                                                                        Mapping map) {
  assert(std::isfinite(t));
  return generalized_thresholds<Result, Result*, Mapping>(t, lo, hi, map);
}

template <typename Result, typename Iterator, typename Mapping>
inline generalized_thresholds<Result, Iterator, Mapping> make_thresholds(
    const Result t, const Result lo, const Result hi, const Iterator first, const Iterator last,
    Mapping map) {
  assert(std::isfinite(t));
  return generalized_thresholds<Result, Iterator, Mapping>(t, lo, hi, first, last, map);
}

template <typename Result, typename ThresholdIterator, typename Iterator>
inline void prox(const sdca::thresholds<Result, ThresholdIterator>& thresholds, Iterator first,
                 Iterator last) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  using Work = typename std::common_type<Data, Result>::type;
  const Work origin = Work(thresholds.origin), t = Work(thresholds.correction),
             lo = Work(thresholds.lo), hi = Work(thresholds.hi);
  std::for_each(first, last, [=](Data& x) {
    x = static_cast<Data>(std::max(lo, std::min((Work(x) - origin) - t, hi)));
  });
}

template <typename Result, typename ThresholdIterator, typename Iterator, typename Mapping>
inline void prox(const sdca::generalized_thresholds<Result, ThresholdIterator, Mapping>& thresholds,
                 Iterator first, Iterator last) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  using Work = typename std::common_type<Data, Result>::type;
  const Work t = Work(thresholds.t), lo = Work(thresholds.lo), hi = Work(thresholds.hi);
  std::for_each(first, last, [=](Data& x) {
    x = static_cast<Data>(std::max(lo, std::min(Work(thresholds.map(Work(x) - t)), hi)));
  });
}

template <typename Iterator, typename Algorithm, typename... Types>
inline void prox(Iterator first, Iterator last, Algorithm compute, Types... params) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  if (first == last) throw std::invalid_argument("Empty projection input");
  std::vector<Data> aux(first, last);
  auto thresholds = compute(aux.begin(), aux.end(), params...);
  prox(thresholds, first, last);
}

template <typename Iterator, typename Algorithm, typename... Types>
inline void prox(Iterator first, Iterator last, Iterator aux, Algorithm compute, Types... params) {
  if (first == last) throw std::invalid_argument("Empty projection input");
  std::copy(first, last, aux);
  auto thresholds = compute(aux, aux + std::distance(first, last), params...);
  prox(thresholds, first, last);
}

template <typename Iterator, typename Algorithm, typename... Types>
inline void prox(Iterator a_first, Iterator a_last, Iterator b_first, Iterator b_last,
                 Algorithm compute, Types... params) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  if (a_first == a_last || b_first == b_last) throw std::invalid_argument("Empty projection block");
  std::vector<Data> a_aux(a_first, a_last);
  std::vector<Data> b_aux(b_first, b_last);
  auto thresholds = compute(a_aux.begin(), a_aux.end(), b_aux.begin(), b_aux.end(), params...);
  prox(thresholds.first, a_first, a_last);
  prox(thresholds.second, b_first, b_last);
}

template <typename Iterator, typename Algorithm, typename... Types>
inline void prox(Iterator a_first, Iterator a_last, Iterator b_first, Iterator b_last,
                 Iterator a_aux, Iterator b_aux, Algorithm compute, Types... params) {
  if (a_first == a_last || b_first == b_last) throw std::invalid_argument("Empty projection block");
  std::copy(a_first, a_last, a_aux);
  std::copy(b_first, b_last, b_aux);
  auto thresholds = compute(a_aux, a_aux + std::distance(a_first, a_last), b_aux,
                            b_aux + std::distance(b_first, b_last), params...);
  prox(thresholds.first, a_first, a_last);
  prox(thresholds.second, b_first, b_last);
}

template <typename Iterator, typename Algorithm, typename... Types>
inline void prox(const typename std::iterator_traits<Iterator>::difference_type dim, Iterator first,
                 Iterator last, Iterator aux, Algorithm compute, Types... params) {
  if (dim <= 0 || std::distance(first, last) % dim != 0)
    throw std::invalid_argument("Invalid projection batch dimensions");
  if (first == last) return;
  Iterator vec_last(first + dim), aux_last(aux + dim);
  for (; first != last;) {
    std::copy(first, vec_last, aux);
    auto thresholds = compute(aux, aux_last, params...);
    prox(thresholds, first, vec_last);
    first = vec_last;
    if (first != last) vec_last = first + dim;
  }
}

template <typename Iterator, typename Algorithm, typename... Types>
inline void prox(const typename std::iterator_traits<Iterator>::difference_type dim,
                 const typename std::iterator_traits<Iterator>::difference_type p, Iterator first,
                 Iterator last, Iterator aux, Algorithm compute, Types... params) {
  if (dim <= 0 || p <= 0 || p >= dim || std::distance(first, last) % dim != 0)
    throw std::invalid_argument("Invalid two-block projection dimensions");
  if (first == last) return;
  Iterator vec_mid(first + p), vec_last(first + dim);
  Iterator aux_mid(aux + p), aux_last(aux + dim);
  for (; first != last;) {
    std::copy(first, vec_last, aux);
    auto thresholds = compute(aux, aux_mid, aux_mid, aux_last, params...);
    prox(thresholds.first, first, vec_mid);
    prox(thresholds.second, vec_mid, vec_last);
    first = vec_last;
    if (first != last) {
      vec_mid = first + p;
      vec_last = first + dim;
    }
  }
}

/**
 * Computes the dot product
 *    <prox(x), prox(x)>
 * without computing prox(x) explicitly.
 **/
template <typename Result, typename Iterator>
inline Result dot_prox_prox(const thresholds<Result, Iterator>& t, Iterator first, Iterator last) {
  Result total(0);
  for (; first != last; ++first) {
    Result value = std::max(t.lo, std::min((Result(*first) - t.origin) - t.correction, t.hi));
    total += value * value;
  }
  return total;
}

/**
 * Computes the dot product
 *    <x, prox(x)>
 * without computing prox(x) explicitly.
 **/
template <typename Result, typename Iterator>
inline Result dot_x_prox(const thresholds<Result, Iterator>& t, Iterator first, Iterator last) {
  Result total(0);
  for (; first != last; ++first) {
    Result x = static_cast<Result>(*first);
    total += x * std::max(t.lo, std::min((x - t.origin) - t.correction, t.hi));
  }
  return total;
}

/**
 * Computes the smoothed hinge value
 *    (<x, prox(x)> - <prox(x), prox(x)> / 2) / gamma
 * term by term, scaling prox(x) by 1 / gamma before the products so that
 * none of them overflows or underflows.
 **/
template <typename Result, typename Iterator>
inline Result dot_smooth_hinge(const thresholds<Result, Iterator>& t, Iterator first, Iterator last,
                               const Result gamma) {
  Result total(0);
  for (; first != last; ++first) {
    Result x = static_cast<Result>(*first);
    Result p = std::max(t.lo, std::min((x - t.origin) - t.correction, t.hi));
    total += (x - p / 2) * (p / gamma);
  }
  return total;
}

template <typename Iterator, typename Functor>
inline void apply(Iterator first, Iterator last, Functor functor) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  std::for_each(first, last, [=](Data& x) { x = functor(x); });
}

template <typename Iterator, typename Functor>
inline void apply(const typename std::iterator_traits<Iterator>::difference_type dim,
                  Iterator first, Iterator last, Functor functor) {
  if (dim <= 0 || std::distance(first, last) % dim != 0)
    throw std::invalid_argument("Invalid batch dimensions");
  if (first == last) return;
  Iterator vec_last = first + dim;
  for (; first != last;) {
    apply(first, vec_last, functor);
    first = vec_last;
    if (first != last) vec_last = first + dim;
  }
}

}  // namespace sdca

#endif
