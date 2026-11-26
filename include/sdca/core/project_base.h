#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_CORE_PROJECT_BASE_H)
#if SDCA_ISA_ID == 1
#define SDCA_CORE_PROJECT_BASE_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_CORE_PROJECT_BASE_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_CORE_PROJECT_BASE_H3 1
#else
#define SDCA_CORE_PROJECT_BASE_H4 1
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "sdca/core/lambert.h"
#include "sdca/core/limits.h"
#include "sdca/core/project_numeric.h"
#include "sdca/core/lanes.h"

SDCA_ISA_BEGIN
namespace core {

// The solution x = max(lo, min((a - origin) - correction, hi)) of an ordinary projection
// (t = origin + correction), and [first, last) the free entries where the solver reports
// them.
template <typename Result, typename Iterator>
struct thresholds {
  typedef Result result_type;
  typedef Iterator iterator_type;

  Result t;
  Result origin = 0;
  Result correction = 0;
  Result lo;
  Result hi;
  Iterator first;
  Iterator last;

  thresholds()
      : t(0),
        lo(-std::numeric_limits<Result>::infinity()),
        hi(+std::numeric_limits<Result>::infinity()),
        first(),
        last() {}

  thresholds(const Result t_, const Result lo_, const Result hi_)
      : t(t_), correction(t_), lo(lo_), hi(hi_), first(), last() {}

  thresholds(const Result t_, const Result lo_, const Result hi_, const Iterator first_,
             const Iterator last_)
      : t(t_), correction(t_), lo(lo_), hi(hi_), first(first_), last(last_) {}
};

// x = max(lo, min(map(a - t), hi)) of an entropic projection.
template <typename Result, typename Iterator, typename Mapping>
struct generalized_thresholds : thresholds<Result, Iterator> {
  typedef thresholds<Result, Iterator> base;
  typedef Mapping map_type;

  map_type map;

  explicit generalized_thresholds(Mapping map_) : base(), map(map_) {}

  generalized_thresholds(const Result t_, const Result lo_, const Result hi_, const Iterator first_,
                         const Iterator last_, Mapping map_)
      : base(t_, lo_, hi_, first_, last_), map(map_) {}
};

template <typename Result, typename Iterator>
inline thresholds<Result, Iterator> make_thresholds(const Result t, const Result lo,
                                                    const Result hi, const Iterator first,
                                                    const Iterator last) {
  assert(std::isfinite(t));
  return thresholds<Result, Iterator>(t, lo, hi, first, last);
}

template <typename Result, typename Iterator, typename Mapping>
inline generalized_thresholds<Result, Iterator, Mapping> make_thresholds(
    const Result t, const Result lo, const Result hi, const Iterator first, const Iterator last,
    Mapping map) {
  assert(std::isfinite(t));
  return generalized_thresholds<Result, Iterator, Mapping>(t, lo, hi, first, last, map);
}

// thresholds with the threshold kept as origin + correction.
template <typename Result, typename Iterator, typename W>
inline thresholds<Result, Iterator> split_thresholds(const W origin, const W correction,
                                                     const Result lo, const Result hi,
                                                     const Iterator first, const Iterator last) {
  thresholds<Result, Iterator> t =
      make_thresholds(Result(origin + correction), lo, hi, first, last);
  t.origin = Result(origin);
  t.correction = Result(correction);
  return t;
}

// The maps of the entropic projections. W(exp(x + shift)) / alpha without an underflowing W
// or an overflowing 1 / alpha.
template <typename T>
struct scaled_lambert_w_exp_map {
  T alpha, shift, log_alpha;
  explicit scaled_lambert_w_exp_map(T a, T s = 0) : alpha(a), shift(s), log_alpha(std::log(a)) {}
  T operator()(const T x) const {
    const T arg = x + shift;
    return lambert_probability_at(arg, arg - log_alpha, alpha);
  }
};

// The capped entropy probability exp(min(x + shift, bound)), with W(exp(.)) if quadratic.
template <typename T>
struct capped_entropy_map {
  T shift, cap, bound;
  bool quadratic;
  capped_entropy_map(T s, T c, bool q)
      : shift(s), cap(c), bound(c > 0 ? std::log(c) + (q ? c : T(0)) : T(0)), quadratic(q) {}
  T operator()(const T x) const {
    if (cap == 0) return 0;
    const T a = std::min(x + shift, bound);
    return quadratic ? lambert_w_exp(a) : std::exp(a);
  }
};

// Writes the solution of thresholds over the scores [first, last) in place.
template <typename Result, typename ThresholdIterator, typename Iterator>
inline void apply(const thresholds<Result, ThresholdIterator>& t, Iterator first, Iterator last) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  typedef typename std::common_type<Data, Result>::type Work;
  const Work origin = Work(t.origin), c = Work(t.correction), lo = Work(t.lo), hi = Work(t.hi);
  for (; first != last; ++first)
    *first = static_cast<Data>(std::max(lo, std::min((Work(*first) - origin) - c, hi)));
}

template <typename Result, typename ThresholdIterator, typename Iterator, typename Mapping>
inline void apply(const generalized_thresholds<Result, ThresholdIterator, Mapping>& t,
                  Iterator first, Iterator last) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  typedef typename std::common_type<Data, Result>::type Work;
  const Work origin = Work(t.t), lo = Work(t.lo), hi = Work(t.hi);
  for (; first != last; ++first)
    *first = static_cast<Data>(std::max(lo, std::min(Work(t.map(Work(*first) - origin)), hi)));
}

// One entry of an ordinary projection.
template <typename Result, typename Iterator>
inline Result projected(const thresholds<Result, Iterator>& t, const Result x) {
  return std::max(t.lo, std::min((x - t.origin) - t.correction, t.hi));
}

// The smoothed hinge value (<a, prox(a)> - <prox(a), prox(a)> / 2) / gamma, term by term with
// prox(a) scaled by 1 / gamma first so that no product over- or underflows; without writing
// prox(a).
template <typename Result, typename Iterator, typename It>
inline Result dot_smooth_hinge(const thresholds<Result, Iterator>& t, It first, It last,
                               const Result gamma) {
  Result total(0);
  for (; first != last; ++first) {
    const Result x = Result(*first), p = projected(t, x);
    total += (x - p / 2) * (p / gamma);
  }
  return total;
}

// Unsigned keys ordered like the finite floats, with -0 just below +0.
template <typename T>
struct float_key {};
template <>
struct float_key<float> {
  typedef std::uint32_t type;
};
template <>
struct float_key<double> {
  typedef std::uint64_t type;
};

template <typename T>
struct keyed : std::integral_constant<bool, std::is_same<T, float>::value ||
                                                std::is_same<T, double>::value> {};

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

// The largest finite x with !test(x), or -inf if there is none, for a test monotone in x:
// a gallop from the guess (any value) to a bracket, then a bisection of the float keys
// (float and double) or of the values (other types).
template <typename T, typename Guess, typename Test>
inline T monotone_cut(const Guess guess, Test test, std::true_type) {
  typedef typename float_key<T>::type Key;
  const Key lowest = to_key(std::numeric_limits<T>::lowest());
  const Key highest = to_key(std::numeric_limits<T>::max());
  if (test(std::numeric_limits<T>::lowest())) return -std::numeric_limits<T>::infinity();
  if (!test(std::numeric_limits<T>::max())) return std::numeric_limits<T>::max();
  // Clamped before narrowing: an out-of-range conversion is undefined.
  T start(0);
  if (guess >= Guess(std::numeric_limits<T>::max()))
    start = std::numeric_limits<T>::max();
  else if (guess <= Guess(std::numeric_limits<T>::lowest()))
    start = std::numeric_limits<T>::lowest();
  else if (guess == guess)
    start = T(guess);
  Key no = to_key(start), yes = no;
  Key step = 1;
  const Key cap = std::numeric_limits<Key>::max() / 2;
  if (test(start)) {
    for (;; step = step > cap ? step : 2 * step) {
      no = yes - lowest > step ? yes - step : lowest;
      if (!test(from_key<T>(no))) break;
      yes = no;
    }
  } else {
    for (;; step = step > cap ? step : 2 * step) {
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

template <typename T, typename Guess, typename Test>
inline T monotone_cut(const Guess, Test test, std::false_type) {
  T no = std::numeric_limits<T>::lowest(), yes = std::numeric_limits<T>::max();
  if (test(no)) return -std::numeric_limits<T>::infinity();
  if (!test(yes)) return yes;
  for (;;) {
    const T middle = no / 2 + yes / 2;
    if (middle == no || middle == yes) return no;
    (test(middle) ? yes : no) = middle;
  }
}

template <typename T, typename Guess, typename Test>
inline T monotone_cut(const Guess guess, Test test) {
  return monotone_cut<T>(guess, test, keyed<T>());
}

// The midpoint of a bracket lo <= hi of finite values: of the float keys (float and double),
// which closes any bracket to adjacent values in limits::key_bisection_steps<T>() halvings,
// else of the values, in limits::value_bisection_steps<T>() (bisection_steps). lo itself
// where the bracket is one value (+0 and -0 included).
template <typename T>
inline T bisect(const T lo, const T hi, std::true_type) {
  typedef typename float_key<T>::type Key;
  if (!(lo < hi)) return lo;
  const Key a = to_key(lo), b = to_key(hi);
  return from_key<T>(Key(a + (b - a) / 2));
}

template <typename T>
inline T bisect(const T lo, const T hi, std::false_type) {
  return lo / 2 + hi / 2;
}

template <typename T>
inline T bisect(const T lo, const T hi) {
  return bisect(lo, hi, keyed<T>());
}

template <typename T>
constexpr int bisection_steps() {
  return keyed<T>::value ? limits::key_bisection_steps<T>() : limits::value_bisection_steps<T>();
}

// The steps of a safeguarded root search on a bracket of T: free steps (Newton, Halley, or
// bisect where those leave the bracket) as many as a bisection of float keys takes, then only
// bisections, which close the bracket. Each step moves an end of the bracket to its point, so
// after the free steps the bracket is adjacent within bisection_steps<T>() + 1 steps. The
// counts are template arguments: a loop bound that calls the recursive limits functions
// would evaluate them on every iteration.
template <typename T>
struct root_steps {
  static constexpr int free_steps() {
    return std::integral_constant<int, limits::key_bisection_steps<T>()>::value;
  }
  static constexpr int total() {
    return std::integral_constant<int, free_steps() + bisection_steps<T>() + 1>::value;
  }
};

// The count of entries x > cut.
template <typename Iterator, typename C>
inline typename std::iterator_traits<Iterator>::difference_type count_above(Iterator first,
                                                                            Iterator last,
                                                                            const C cut) {
  typename std::iterator_traits<Iterator>::difference_type c = 0;
  for (; first != last; ++first) c += C(*first) > cut ? 1 : 0;
  return c;
}

// The count of entries x >= bound; finite turns false on NaN or infinite entries.
template <typename Iterator, typename C>
inline typename std::iterator_traits<Iterator>::difference_type count_at_least(Iterator first,
                                                                               Iterator last,
                                                                               const C bound,
                                                                               bool& finite) {
  typename std::iterator_traits<Iterator>::difference_type c = 0;
  for (; first != last; ++first) {
    finite &= *first - *first == 0;
    c += C(*first) >= bound ? 1 : 0;
  }
  return c;
}

// The sum of (x - origin) - t in four partial sums: it decides, so its order is free.
template <typename Work, typename Iterator>
inline Work shifted_sum(Iterator first, Iterator last, const Work origin, const Work t) {
  typedef typename std::iterator_traits<Iterator>::difference_type Index;
  const Index n = last - first;
  Work s0(0), s1(0), s2(0), s3(0);
  Index i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += (Work(first[i]) - origin) - t;
    s1 += (Work(first[i + 1]) - origin) - t;
    s2 += (Work(first[i + 2]) - origin) - t;
    s3 += (Work(first[i + 3]) - origin) - t;
  }
  for (; i < n; ++i) s0 += (Work(first[i]) - origin) - t;
  return (s0 + s1) + (s2 + s3);
}

// The sum of x - origin in four partial sums, the largest magnitude (from magnitude) and
// finite, which turns false on NaN or infinite entries.
template <typename Work, typename Iterator>
inline Work scan_scores(Iterator first, Iterator last, const Work origin, Work& magnitude,
                        bool& finite) {
  typedef typename std::iterator_traits<Iterator>::difference_type Index;
  const Index n = last - first;
  Work s0(0), s1(0), s2(0), s3(0), m0 = magnitude, m1 = magnitude, z0(0), z1(0);
  Index i = 0;
  for (; i + 4 <= n; i += 4) {
    const Work a = Work(first[i]), b = Work(first[i + 1]);
    const Work c = Work(first[i + 2]), d = Work(first[i + 3]);
    z0 += (a - a) + (c - c);
    z1 += (b - b) + (d - d);
    m0 = std::max(m0, std::max(std::abs(a), std::abs(c)));
    m1 = std::max(m1, std::max(std::abs(b), std::abs(d)));
    s0 += a - origin;
    s1 += b - origin;
    s2 += c - origin;
    s3 += d - origin;
  }
  for (; i < n; ++i) {
    const Work a = Work(first[i]);
    z0 += a - a;
    m0 = std::max(m0, std::abs(a));
    s0 += a - origin;
  }
  finite &= z0 + z1 == 0;
  magnitude = std::max(m0, m1);
  return (s0 + s1) + (s2 + s3);
}

// std::partition of a random-access range, entry for entry: both standard libraries swap the
// i-th entry failing the test from the left with the i-th passing it from the right while the
// first lies left of the second, which are the failing entries left of the partition point
// (the count c of passing entries) and the passing ones right of it, paired in order. Those
// pairs are collected in blocks without branches and swapped; the sums over the partitioned
// ranges, and so the projections, are unchanged.
template <typename Iterator, typename Test>
__attribute__((noinline)) Iterator
partition_counted(Iterator first, Iterator last, Test test,
                  const typename std::iterator_traits<Iterator>::difference_type c) {
  typedef typename std::iterator_traits<Iterator>::difference_type Index;
  const Index block = 128;
  Index left[block], right[block], i = 0, j = (last - first) - 1;
  while (i < c) {
    Index l = 0, r = 0;
    for (; l < block && i < c; ++i) {
      left[l] = i;
      l += test(first[i]) ? 0 : 1;
    }
    for (; r < l; --j) {
      right[r] = j;
      r += test(first[j]) ? 1 : 0;
    }
    for (Index p = 0; p < l; ++p) std::iter_swap(first + left[p], first + right[p]);
  }
  return first + c;
}

template <typename Iterator, typename Test>
inline Iterator partition(Iterator first, Iterator last, Test test) {
  if (last - first < 16) return std::partition(first, last, test);
  typename std::iterator_traits<Iterator>::difference_type c = 0;
  for (Iterator it = first; it != last; ++it) c += test(*it) ? 1 : 0;
  return partition_counted(first, last, test, c);
}

// core::partition by a test monotone in the (finite) values. Long float and double ranges
// compare against the cut of the test instead of evaluating it: the same decisions, so the
// same order, at the cost of one comparison.
template <typename Iterator, typename Test, typename Guess>
inline Iterator partition_monotone(Iterator first, Iterator last, Test test, const Guess guess) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  if (!keyed<Data>::value || std::distance(first, last) < 32)
    return core::partition(first, last, test);
  const Data cut = monotone_cut<Data>(guess, test);
  return partition_counted(
      first, last, [cut](const Data x) { return x > cut; }, count_above(first, last, cut));
}

// lo n_lo + hi n_hi as s + e: opposing bound masses cancel exactly, so that a small mass
// beside them survives.
template <typename T>
inline T bound_mass(const T lo, const T n_lo, const T hi, const T n_hi, T& e) {
  const T p_lo = lo * n_lo, p_hi = hi * n_hi;
  if (p_lo == 0 || p_hi == 0) {
    e = 0;
    return p_lo + p_hi;
  }
  const T s = sum_error(p_lo, p_hi, e);
  e += std::fma(lo, n_lo, -p_lo) + std::fma(hi, n_hi, -p_hi);
  return s;
}

// The prefix sums of value(0), ..., value(n - 1) with their rounding errors: the sum of
// [i, j) loses nothing to the magnitude of the values before i.
template <typename T>
struct prefix_sums {
  row_buffer<T> hi, lo;

  template <typename Value>
  prefix_sums(const std::size_t n, Value value) : hi(n + 1), lo(n + 1) {
    T *h = hi.data(), *l = lo.data();
    h[0] = l[0] = T(0);
    for (std::size_t i = 0; i < n; ++i) {
      T e;
      h[i + 1] = sum_error(h[i], value(i), e);
      l[i + 1] = l[i] + e;
    }
  }

  T operator()(const std::size_t i, const std::size_t j) const {
    return (hi.data()[j] - hi.data()[i]) + (lo.data()[j] - lo.data()[i]);
  }
};

// A power of two u with u * magnitude * factor well below overflow; one where no scaling is
// needed, so that ordinary inputs are solved unscaled.
template <typename T>
inline T overflow_unit(const T magnitude, T factor) {
  factor = std::min(std::max(factor, T(1)), T(18446744073709551616.0));  // 2^64
  if (!(magnitude * factor > std::numeric_limits<T>::max() / 16)) return T(1);
  const int exponent = std::ilogb(magnitude) + std::ilogb(factor) + 6;
  return std::ldexp(T(1), std::numeric_limits<T>::max_exponent - exponent);
}

template <typename Iterator>
inline void check_scores(Iterator first, Iterator last, const char* message) {
  bool finite = true;
  for (; first != last; ++first) finite &= *first - *first == 0;
  if (!finite) throw std::invalid_argument(message);
}

}  // namespace core
SDCA_ISA_END

#endif  // SDCA_CORE_PROJECT_BASE_H
