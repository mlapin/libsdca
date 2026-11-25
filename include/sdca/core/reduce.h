#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_CORE_REDUCE_H)
#if SDCA_ISA_ID == 1
#define SDCA_CORE_REDUCE_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_CORE_REDUCE_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_CORE_REDUCE_H3 1
#else
#define SDCA_CORE_REDUCE_H4 1
#endif

// Row loops and reductions over n entries of a storage type S read as lanes A (Lanes<W, Isa,
// K>): the extrema with their validity, the largest entry, sums of exponentials, logsumexp,
// dot products and the gradient map of exponential halves. Sums of float work add float terms
// in float lanes (four per lane) into a double sum; sums of double work are compensated
// (TwoSum every 16 terms per lane). Masked classes (-inf) take the same loops.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "lanes.h"
#include "math.h"
#include "storage.h"

SDCA_ISA_BEGIN
namespace core {

// hi + lo += x, exactly (TwoSum).
template <typename T>
SDCA_LANE void two_sum(T& hi, T& lo, const T& x) {
  const T t = hi + x, b = t - hi;
  lo = lo + ((hi - (t - b)) + (x - b));
  hi = t;
}

// s + e == a + b exactly (core::two_sum), unless a + b overflows.
template <typename T>
inline T sum_error(const T a, const T b, T& e) {
  T s = a;
  e = T(0);
  two_sum(s, e, b);
  return s;
}

// p + e == a * b exactly (TwoProduct), unless a * b overflows or underflows.
template <typename T>
inline T two_product(const T a, const T b, T& e) {
  const T p = a * b;
  e = std::fma(a, b, -p);
  return p;
}

template <>
inline float two_product(const float a, const float b, float& e) {
  const double exact = static_cast<double>(a) * static_cast<double>(b);
  const float p = static_cast<float>(exact);
  e = static_cast<float>(exact - static_cast<double>(p));
  return p;
}

// Dekker's product: p + e == a * b exactly without fma, unless the Veltkamp split (|a|, |b|
// near overflow) or the product over- or underflows.
template <typename T>
inline T split_product(const T a, const T b, T& e) {
  const T factor = T((1ull << ((std::numeric_limits<T>::digits + 1) / 2)) + 1);
  const T sa = factor * a, sb = factor * b;
  const T a_hi = sa - (sa - a), a_lo = a - a_hi, b_hi = sb - (sb - b), b_lo = b - b_hi;
  const T p = a * b;
  e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
  return p;
}

// A compensated scalar sum kept as hi + lo: lo collects the exact rounding errors of the
// additions, within an ulp of the exact sum of finite terms. Neumaier's sum (Fast2Sum on the
// larger operand) collects the same errors in the same order, so it is this sum in double.
template <typename T>
struct compensated_sum {
  T hi, lo;
  compensated_sum() : hi(0), lo(0) {}
  SDCA_LANE void add(const T x) { two_sum(hi, lo, x); }
  SDCA_LANE T value() const { return hi + lo; }
};
typedef compensated_sum<double> Neumaier;

// A sum without compensation, with compensated_sum's interface.
template <typename T>
struct plain_sum {
  T hi;
  plain_sum() : hi(0) {}
  void add(const T x) { hi += x; }
  T value() const { return hi; }
};

// A sum in T of Term terms: compensated where they are as wide as T; a wider T keeps the
// narrower terms' sum below their own epsilon.
template <typename Term, typename T>
using row_sum =
    typename std::conditional<std::is_same<Term, T>::value, compensated_sum<T>, plain_sum<T>>::type;

// Lane sums in double of lanes A. Float lanes add up to four terms in float, then widen;
// double lanes add 16 terms plainly, then TwoSum the block into the total.
template <typename A, typename W = typename A::value_type>
struct Accumulator;

template <typename A>
struct Accumulator<A, float> {
  typedef typename A::isa_type Isa;
  typedef Lanes<double, Isa, A::chains * Widening<Isa>::R> D;
  typedef D lanes_type;
  D total;
  A block;
  int count;
  Accumulator() : total(D::zero()), block(A::zero()), count(0) {}
  SDCA_LANE void add(const A& x) {
    block += x;
    if (++count == 4) {
      total += widen(block);
      block = A::zero();
      count = 0;
    }
  }
  // The total of each lane.
  SDCA_LANE D lanes() const { return total + widen(block); }
  SDCA_LANE double value() const { return sum(lanes()); }
};

template <typename A>
struct Accumulator<A, double> {
  typedef A lanes_type;
  A hi, lo, block;
  int count;
  Accumulator() : hi(A::zero()), lo(A::zero()), block(A::zero()), count(0) {}
  SDCA_LANE void add(const A& x) {
    block += x;
    if (++count == 16) {
      two_sum(hi, lo, block);
      block = A::zero();
      count = 0;
    }
  }
  SDCA_LANE A lanes() const {
    A h = hi, l = lo;
    two_sum(h, l, block);
    return h + l;
  }
  SDCA_LANE double value() const {
    A h = hi, l = lo;
    two_sum(h, l, block);
    double s = 0, c = 0;
    for (int i = 0; i < A::L; ++i) two_sum(s, c, h[i]);
    for (int i = 0; i < A::L; ++i) c += l[i];
    return s + c;
  }
};

// body(v, j, count) over [0, n) of x: full lanes, then the last count < L entries padded.
template <typename A, typename S, typename F>
SDCA_LANE void each(const S* x, std::size_t n, typename A::value_type pad, F&& body) {
  std::size_t j = 0;
  for (; j + A::L <= n; j += A::L) body(load<A>(x + j), j, static_cast<int>(A::L));
  if (j < n) body(load<A>(x + j, static_cast<int>(n - j), pad), j, static_cast<int>(n - j));
}

// The mask of lanes A set where the bytes m[0, count) are nonzero (count <= L; unset beyond).
template <typename A>
SDCA_LANE typename A::mask_type mask_lanes(const uint8_t* m, std::size_t count) {
  typedef typename Vector<uint8_t, A::B>::type Bytes;
  typedef typename A::mask_type M;
  uint8_t lane[A::L] = {};
  std::memcpy(lane, m, count < sizeof lane ? count : sizeof lane);
  M r;
  for (int k = 0; k < A::chains; ++k) {
    Bytes b;
    std::memcpy(&b, lane + k * A::B, sizeof b);
    r.v[k] = (typename M::V)(__builtin_convertvector(b, typename M::V) != 0);
  }
  return r;
}

// Lanes of v with lane skip - j (if within the lanes) set to x.
template <typename A>
SDCA_LANE A with_lane(const A& v, std::size_t j, std::size_t skip, typename A::value_type x) {
  typedef typename A::mask_type M;
  typedef typename M::value_type I;
  if (!(skip >= j && skip - j < static_cast<std::size_t>(A::L))) return v;
  return select(
      iota<typename A::value_type, typename A::isa_type, A::chains>() == static_cast<I>(skip - j),
      A::splat(x), v);
}

// Whether x[0, n) holds no NaN and no +inf (-inf, a masked class, is valid): the second pass
// of the loops below for rows whose lane sums are not below +inf.
template <typename A, typename S>
inline bool below_infinity(const S* x, std::size_t n) {
  typedef typename A::value_type T;
  const T inf = std::numeric_limits<T>::infinity();
  typename A::mask_type bad = A::mask_type::zero();
  each<A>(x, n, T(0), [&](const A& v, std::size_t, int) { bad = bad | ~(v < inf); });
  return !any(bad);
}

// The largest of x[j] over j != skip (-inf if none; skip >= n skips none); valid turns false
// on NaN or +inf. A lane sum of the entries is -inf or finite where they hold no NaN or +inf;
// one that is not (also a finite row whose sum overflows) takes a second pass. maxima,
// when requested, receives the maximum of each lane before the horizontal reduction.
template <typename A, typename S>
SDCA_LANE typename A::value_type max_except(const S* x, std::size_t n, std::size_t skip,
                                            bool& valid, A* maxima = nullptr) {
  typedef typename A::value_type T;
  const T inf = std::numeric_limits<T>::infinity();
  A top = A::splat(-inf), total = A::zero();
  each<A>(x, n, -inf, [&](A v, std::size_t j, int) {
    total += v;
    top = max(top, with_lane(v, j, skip, -inf));
  });
  if (!all(total < inf)) valid = valid && below_infinity<A>(x, n);
  if (maxima) *maxima = top;
  return max_lane(top);
}

// The extrema of x[0, n): hi the largest entry, lo the lowest above -inf (+inf if none);
// valid turns false on NaN or +inf, as max_except checks them. lo takes v * 0 + v, NaN where v
// is not finite, which min leaves out. *unbounded (unless null) turns true where a lane sum is
// not finite: an entry of -inf (or finite entries whose sum overflows).
template <typename A, typename S>
SDCA_LANE void extrema(const S* x, std::size_t n, typename A::value_type& lo,
                       typename A::value_type& hi, bool& valid, bool* unbounded = 0) {
  typedef typename A::value_type T;
  const T inf = std::numeric_limits<T>::infinity();
  A top = A::splat(-inf), bottom = A::splat(inf), total = A::zero();
  const T pad = n > 0 ? to_work<T>(x[0]) : T(0);
  each<A>(x, n, pad, [&](const A& v, std::size_t, int) {
    top = max(top, v);
    bottom = min(v * T(0) + v, bottom);
    total += v;
  });
  lo = min_lane(bottom), hi = max_lane(top);
  if (!all(total < inf)) valid = valid && below_infinity<A>(x, n);
  if (unbounded && !all(total > -inf)) *unbounded = true;
}

// The largest entry of x[0, n), valid as extrema.
template <typename A, typename S>
SDCA_LANE typename A::value_type maximum(const S* x, std::size_t n, bool& valid) {
  typedef typename A::value_type T;
  const T inf = std::numeric_limits<T>::infinity();
  A top = A::splat(-inf), total = A::zero();
  const T pad = n > 0 ? to_work<T>(x[0]) : T(0);
  each<A>(x, n, pad, [&](const A& v, std::size_t, int) {
    top = max(top, v);
    total += v;
  });
  if (!all(total < inf)) valid = valid && below_infinity<A>(x, n);
  return max_lane(top);
}

// The first j in [from, n) with x[j] >= floor (n if none).
template <typename A, typename S>
inline std::size_t find_at_least(const S* x, std::size_t from, std::size_t n,
                                 typename A::value_type floor) {
  std::size_t j = from;
  for (; j + A::L <= n; j += A::L)
    if (any(load<A>(x + j) >= floor)) break;
  for (; j < n; ++j)
    if (to_work<typename A::value_type>(x[j]) >= floor) return j;
  return n;
}

const std::size_t kBlock = 2048;

// Sum at most kBlock nonnegative wide float exponentials, including a maximum's 2^64.
// Rounding an entry to a multiple of ulp(bias) leaves at most 2^23 / terms integer
// units. Each lane can therefore sum its high words exactly. The residuals sum in float
// at that smaller scale; the two lane totals are combined in double.
template <typename A>
struct BlockExpSum {
  A high, low;
  BlockExpSum() : high(A::zero()), low(A::zero()) {}
  SDCA_LANE void add(const A& value) {
    const float bias = static_cast<float>(1.5 * (kBlock / A::L) / Wide<float>::unwide());
    const A rounded = (value + bias) - bias;
    high += rounded;
    low += value - rounded;
  }
  SDCA_LANE double value() const { return sum(to_double(high)) + sum(to_double(low)); }
};

// The sum in double of exp(x[j] - shift) over j != skip (skip >= n skips none).
// If saved is present, keep the wide exponentials, including zero at skip, for the gradient.
template <typename A, bool Block = false, typename S>
SDCA_LANE double exp_sum_except(const S* x, std::size_t n, std::size_t skip,
                                typename A::value_type shift, typename A::value_type* saved = 0) {
  typedef typename A::value_type T;
  const T inf = std::numeric_limits<T>::infinity();
  typename std::conditional<Block && std::is_same<T, float>::value, BlockExpSum<A>,
                            Accumulator<A>>::type total;
  each<A>(x, n, -inf, [&](const A& v, std::size_t j, int count) {
    const A e = exp_nonpositive_wide(with_lane(v, j, skip, -inf) - shift);
    if (saved) store(saved + j, e, count);
    total.add(e);
  });
  return total.value() * static_cast<double>(Wide<T>::unwide());
}
template <typename A, typename S>
SDCA_LANE double exp_sum(const S* x, std::size_t n, typename A::value_type shift) {
  return exp_sum_except<A>(x, n, n, shift);
}

// The row's largest entry m and the sum s of exp(x - m) in one sweep over blocks of kBlock
// entries (each read twice from L1): per block its largest entry, then its exponentials from
// the running maximum (the earlier sum rescaled where that rises). skip as max_except.
// A single block can retain its exponentials at the final maximum in saved.
template <typename A, typename S>
SDCA_FLAT inline double max_exp_sum(const S* x, std::size_t n, std::size_t skip,
                                    typename A::value_type& maximum, bool& valid,
                                    typename A::value_type* saved = 0) {
  typedef typename A::value_type T;
  const T inf = std::numeric_limits<T>::infinity();
  if (n <= kBlock) {
    maximum = max_except<A>(x, n, skip, valid);
    return maximum > -inf ? exp_sum_except<A, true>(x, n, skip, maximum, saved) : 0.0;
  }
  maximum = -inf;
  double sum = 0;
  for (std::size_t b = 0; b < n; b += kBlock) {
    const std::size_t m = n - b < kBlock ? n - b : kBlock;
    const std::size_t at = skip >= b ? skip - b : m;
    const T top = max_except<A>(x + b, m, at, valid);
    if (top > maximum) {
      if (sum != 0) sum *= exp(static_cast<double>(maximum) - static_cast<double>(top));
      maximum = top;
    }
    if (maximum > -inf) sum += exp_sum_except<A>(x + b, m, at, maximum);
  }
  return sum;
}

// log sum_j exp(x[j]) (-inf if every entry is -inf); NaN for NaN or +inf entries.
template <typename A, typename S>
inline double logsumexp(const S* x, std::size_t n) {
  typename A::value_type m;
  bool valid = true;
  const double s = max_exp_sum<A>(x, n, n, m, valid);
  if (!valid) return std::numeric_limits<double>::quiet_NaN();
  if (!(m > -std::numeric_limits<typename A::value_type>::infinity())) return m;
  return static_cast<double>(m) + log(s);
}

// sum_j a[j] b[j] in double (compensated for double work).
template <typename A, typename S, typename P>
inline double dot(const S* a, const P* b, std::size_t n) {
  typedef typename A::value_type T;
  Accumulator<A> total;
  std::size_t j = 0;
  for (; j + A::L <= n; j += A::L) total.add(load<A>(a + j) * load<A>(b + j));
  if (j < n) {
    const int count = static_cast<int>(n - j);
    total.add(load<A>(a + j, count, T(0)) * load<A>(b + j, count, T(0)));
  }
  return total.value();
}

// Apply f and power to the terms of a row, rounding into G and making each zero positive.
template <typename A, typename S, typename G, typename F>
SDCA_LANE void scale_terms(const S* x, std::size_t n, typename A::value_type pad,
                           typename A::value_type f, typename A::value_type power, G* out, F term) {
  typedef typename A::value_type T;
  if (power == T(1))
    each<A>(x, n, pad, [&](const A& v, std::size_t j, int count) {
      store(out + j, term(v) * f + T(0), count);
    });
  else
    each<A>(x, n, pad, [&](const A& v, std::size_t j, int count) {
      store(out + j, (term(v) * f) * power + T(0), count);
    });
}

// out[j] = exp(x[j] - center) factor, rounded once into the work type (Wide: subnormal below
// its normal range, infinite beyond it) and into G, zeros +0.0 for either sign of factor; out
// may be x (each entry is read before it is written). False, writing nothing, where factor
// exceeds the range of the work type times 2^Wide::shift. With saved, reuse the wide
// exponentials retained by exp_sum_except.
template <typename A, typename S, typename G>
SDCA_FLAT inline bool exp_scaled(const S* x, std::size_t n, typename A::value_type center,
                                 double factor, G* out, const typename A::value_type* saved = 0) {
  typedef typename A::value_type T;
  T f, power;
  if (!Wide<T>::split(factor, f, power)) return false;
  if (saved)
    scale_terms<A>(saved, n, T(0), f, power, out, [](const A& v) { return v; });
  else
    scale_terms<A>(x, n, center, f, power, out,
                   [&](const A& v) { return exp_nonpositive_wide(v - center); });
  return true;
}

}  // namespace core
SDCA_ISA_END

#endif  // SDCA_CORE_REDUCE_H
