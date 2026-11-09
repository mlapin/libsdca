#ifndef SDCA_PROX_TWO_SIMPLEX_H
#define SDCA_PROX_TWO_SIMPLEX_H

#include "sdca/math/compensated.h"
#include "sdca/prox/knapsack_eq.h"

namespace sdca {

namespace detail {

// Michelot's simplex support of mass rhs over [first, last) in double; the
// support moves to [first, last). False if rounding empties it.
template <typename W, typename Iterator>
inline bool simplex_support(Iterator first, Iterator& last, const W rhs) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  for (;;) {
    W sum(0);
    for (auto it = first; it != last; ++it) sum += W(*it);
    const W t = (sum - rhs) / W(std::distance(first, last));
    const Iterator it = std::partition(first, last, [=](const Data x) { return W(x) > t; });
    if (it == last) return true;
    if (it == first) return false;
    last = it;
  }
}

// A threshold o + c relative to the score origin o: c = (sum (x - o) - sum
// (y + o) - rhs) / n over supports x and y, summed with TwoSum, rounded to
// Result, and a bound on its error (4 eps |c|, the second-order terms and the
// rounding to Result).
template <typename Result, typename W, typename Iterator>
inline W support_correction(Iterator x_first, Iterator x_last, Iterator y_first, Iterator y_last,
                            const W o, const W rhs, W& error) {
  compensated_sum<W> sum;
  W magnitude = std::abs(rhs);
  for (auto it = x_first; it != x_last; ++it) {
    sum.add(W(*it));
    sum.add(-o);
    magnitude += std::abs(W(*it)) + std::abs(o);
  }
  for (auto it = y_first; it != y_last; ++it) {
    sum.add(-W(*it));
    sum.add(-o);
    magnitude += std::abs(W(*it)) + std::abs(o);
  }
  sum.add(-rhs);
  const W n = W(std::distance(x_first, x_last) + std::distance(y_first, y_last));
  const W eps = std::numeric_limits<W>::epsilon(), c = sum.value() / n, r = W(Result(c));
  error = 4 * eps * std::abs(c) + 8 * n * eps * eps * magnitude + std::abs(r - c);
  return r;
}

// Whether every entry x of [first, last) has its output (x - o) - c clearly
// signed (eps: of the type it is formed in), beyond 16 times its error bound: positive on the support [first,
// middle), negative elsewhere. The supports are then exact.
template <typename W, typename Iterator>
inline bool support_separated(Iterator first, Iterator middle, Iterator last, const W o, const W c,
                              const W error, const W eps) {
  // The margin grows with |x - o| no faster than the bound: the entries next
  // to the threshold (the smallest support and the largest other) decide.
  auto separated = [=](const W x, const bool support) {
    const W d = x - o, v = d - c, bound = 16 * (eps * (std::abs(d) + std::abs(v)) + error);
    return support ? v > bound : v < -bound;
  };
  W low = std::numeric_limits<W>::infinity(), high = -low;
  for (auto it = first; it != middle; ++it) low = std::min(low, W(*it));
  for (auto it = middle; it != last; ++it) high = std::max(high, W(*it));
  return (first == middle || separated(low, true)) && (middle == last || separated(high, false));
}

// v0.2.0's search for the supports when <1, x> = <1, y> < rhs (s = -t), in
// double; false if a support empties.
template <typename W, typename Iterator>
inline bool inactive_supports(Iterator a_first, Iterator& x_last, Iterator b_first,
                              Iterator& y_last) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  W sum_x(0), sum_y(0), terms(0);
  for (auto it = a_first; it != x_last; ++it) sum_x += W(*it), terms += std::abs(W(*it));
  for (auto it = b_first; it != y_last; ++it) sum_y += W(*it), terms += std::abs(W(*it));
  const W tol = 8 * std::numeric_limits<W>::epsilon() * terms;
  auto m = std::distance(a_first, x_last) + std::distance(b_first, y_last);
  W u = (sum_x - sum_y) / W(m);
  for (;;) {
    auto x_it = std::partition(a_first, x_last, [=](const Data x) { return W(x) > u; });
    auto y_it = std::partition(b_first, y_last, [=](const Data y) { return W(y) > -u; });
    W part_x(0), part_y(0);
    for (auto it = x_it; it != x_last; ++it) part_x += W(*it);
    for (auto it = y_it; it != y_last; ++it) part_y += W(*it);
    const auto n_x = std::distance(x_it, x_last), n_y = std::distance(y_it, y_last);
    const W infeas = part_x - part_y - W(n_x + n_y) * u;
    W uu;
    if (n_y > 0 && infeas > tol) {
      y_last = y_it;
      uu = W(m) * u + part_y;
      m -= n_y;
    } else if (n_x > 0 && infeas < -tol) {
      x_last = x_it;
      uu = W(m) * u - part_x;
      m -= n_x;
    } else {
      x_last = x_it;
      y_last = y_it;
      return true;
    }
    if (m <= 0) return false;
    u = uu / W(m);
  }
}

// The largest entries of a and b and the largest magnitude (with rhs).
template <typename W, typename Iterator>
inline W two_simplex_scan(Iterator a_first, Iterator a_last, Iterator b_first, Iterator b_last,
                          const W rhs, W& max_a, W& max_b) {
  W scale = rhs;
  max_a = max_b = -std::numeric_limits<W>::infinity();
  for (auto it = a_first; it != a_last; ++it) {
    max_a = std::max(max_a, W(*it));
    scale = std::max(scale, std::abs(W(*it)));
  }
  for (auto it = b_first; it != b_last; ++it) {
    max_b = std::max(max_b, W(*it));
    scale = std::max(scale, std::abs(W(*it)));
  }
  return scale;
}

// A threshold o + c of the fast path, its origin and correction kept apart.
template <typename Result, typename Iterator, typename W>
inline thresholds<Result, Iterator> fast_thresholds(const W o, const W c, const Result rhs,
                                                    Iterator first, Iterator last) {
  auto t = make_thresholds(Result(o + c), Result(0), rhs, first, last);
  t.origin = Result(o);
  t.correction = Result(c);
  return t;
}

// The inactive-sum case of the fast path: supports, then t = o + c about the
// origin o halfway between the largest scores of the two sides (as extended).
template <typename Result, typename W, typename Iterator>
inline bool two_simplex_fast_inactive(Iterator a_first, Iterator a_last, Iterator b_first,
                                      Iterator b_last, Iterator& x_last, Iterator& y_last,
                                      const W max_a, const W max_b, const W rhs, const W eps_out,
                                      W& o, W& c) {
  if (!inactive_supports<W>(a_first, x_last, b_first, y_last)) return false;
  W error;
  o = W(Result(max_a / 2 - max_b / 2));
  c = support_correction<Result>(a_first, x_last, b_first, y_last, o, W(0), error);
  W mass(0);
  for (auto it = a_first; it != x_last; ++it) mass += (W(*it) - o) - c;
  const W slack =
      4 * W(std::distance(a_first, x_last)) * (error + std::numeric_limits<W>::epsilon() * mass);
  return support_separated(a_first, x_last, a_last, o, c, error, eps_out) &&
         support_separated(b_first, y_last, b_last, -o, -c, error, eps_out) && mass < rhs - slack;
}

// projection_precision::float64: v0.2.0's search in double for the supports,
// then each threshold as a score origin o (the largest score, or the midpoint
// of both sides' largest when the sum is inactive) plus a correction c
// recomputed with compensated sums, as the extended solver stores it. The
// acceptance check: every entry 16 error bounds from its threshold (exact
// supports) and a clear active-sum case. Each output (x - o) - c is then within
// a few ulps of |x - o| + |c|. False otherwise (the caller runs the hardened
// solver), with the entries reordered. Needs a nonempty a and b and rhs > 0.
template <typename Result, typename Iterator>
inline bool two_simplex_fast(
    Iterator a_first, Iterator a_last, Iterator b_first, Iterator b_last, const Result rhs,
    std::pair<thresholds<Result, Iterator>, thresholds<Result, Iterator>>& out) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  typedef typename std::common_type<Result, Data, double>::type W;
  const W eps = std::numeric_limits<W>::epsilon();
  // The outputs are formed in common_type<Result, Data> (see prox).
  const W eps_out = std::max(
      eps, W(std::numeric_limits<typename std::common_type<Result, Data>::type>::epsilon()));
  W max_a, max_b;
  const W scale = two_simplex_scan(a_first, a_last, b_first, b_last, W(rhs), max_a, max_b);
  if (!(scale <= std::numeric_limits<W>::max() / 1024 && scale >= W(1e-200))) return false;
  if (W(Result(max_a)) != max_a || W(Result(max_b)) != max_b) return false;
  if (max_a + max_b <= 0) {
    // Every pairwise margin is nonpositive: the zero solution (exact).
    out = std::make_pair(fast_thresholds(max_a, W(0), rhs, a_first, a_first),
                         fast_thresholds(-max_a, W(0), rhs, b_first, b_first));
    return true;
  }
  Iterator x_last = a_last, y_last = b_last;
  if (!simplex_support(a_first, x_last, W(rhs)) || !simplex_support(b_first, y_last, W(rhs)))
    return false;
  // The largest score lies in its support: the origins.
  W o_a = max_a, o_b = max_b, error_a, error_b;
  W c_a = support_correction<Result>(a_first, x_last, b_first, b_first, o_a, W(rhs), error_a);
  W c_b = support_correction<Result>(b_first, y_last, a_first, a_first, o_b, W(rhs), error_b);
  const W sum = (o_a + o_b) + (c_a + c_b),
          sum_error = 2 * eps * (std::abs(o_a) + std::abs(o_b) + std::abs(c_a) + std::abs(c_b)) +
                      error_a + error_b;
  bool ok;
  if (sum > 4 * sum_error) {
    ok = support_separated(a_first, x_last, a_last, o_a, c_a, error_a, eps_out) &&
         support_separated(b_first, y_last, b_last, o_b, c_b, error_b, eps_out);
  } else {
    // v0.2.0's inactive-sum search starts from the active-sum supports.
    ok = sum < -4 * sum_error &&
         two_simplex_fast_inactive<Result>(a_first, a_last, b_first, b_last, x_last, y_last, max_a,
                                           max_b, W(rhs), eps_out, o_a, c_a);
    o_b = -o_a;
    c_b = -c_a;
  }
  if (!(ok && std::isfinite(c_a) && std::isfinite(c_b))) return false;
  out = std::make_pair(fast_thresholds(o_a, c_a, rhs, a_first, x_last),
                       fast_thresholds(o_b, c_b, rhs, b_first, y_last));
  return true;
}

}  // namespace detail

namespace detail {

// The hardened extended-precision solver (projection_precision::extended).
template <typename Result, typename Iterator>
inline std::pair<thresholds<Result, Iterator>, thresholds<Result, Iterator>> two_simplex_extended(
    Iterator a_first, Iterator a_last, Iterator b_first, Iterator b_last, const Result rhs) {
  if (a_first == a_last || b_first == b_last || !(rhs >= 0) || !std::isfinite(rhs))
    throw std::invalid_argument("Invalid two-simplex dimensions or bound");
  Result lo(0), hi(rhs);
  if (rhs == 0) {
    return std::make_pair(make_thresholds(Result(0), lo, hi, a_first, a_last),
                          make_thresholds(Result(0), lo, hi, b_first, b_last));
  }
  // The active-sum case is two ordinary simplex projections.
  // Use a strictly larger per-entry cap so a singleton support stays free.
  const Result cap =
      rhs <= std::numeric_limits<Result>::max() / 2 ? 2 * rhs : std::numeric_limits<Result>::max();
  auto a = thresholds_knapsack_eq(a_first, a_last, lo, cap, rhs);
  auto b = thresholds_knapsack_eq(b_first, b_last, lo, cap, rhs);
  a.hi = rhs;
  b.hi = rhs;
  // The sum is active unless the thresholds add up below zero, to the
  // rounding of their own terms: a fixed floor would swallow a tiny margin.
  const Result terms = std::max(rhs, (std::abs(a.origin) + std::abs(b.origin)) +
                                         (std::abs(a.correction) + std::abs(b.correction)));
  if ((a.origin + b.origin) + (a.correction + b.correction) >=
      -std::numeric_limits<Result>::epsilon() * terms)
    return std::make_pair(a, b);
  Iterator x_last = a.last, y_last = b.last;

  assert(std::distance(a_first, x_last) > 0);
  assert(std::distance(b_first, y_last) > 0);

  using Data = typename std::iterator_traits<Iterator>::value_type;
  using Work = typename std::common_type<Result, Data, double>::type;
  using Accum = typename std::common_type<Work, long double>::type;
  const Result stored_origin = Result(Accum(*std::max_element(a_first, x_last)) / 2 -
                                      Accum(*std::max_element(b_first, y_last)) / 2);
  const Work origin = Work(stored_origin);
  auto sum = [](Iterator begin, Iterator end, Work offset) {
    Accum value(0);
    for (; begin != end; ++begin) value += Accum(*begin) - Accum(offset);
    return value;
  };
  // The fixing test resolves the sums to their own rounding, not to the bound's
  // or to 1 (tiny scores); one pass also forms the first sums.
  Accum scale(0), sum_a(0), sum_b(0);
  for (auto it = a_first; it != x_last; ++it) {
    const Accum d = Accum(*it) - Accum(origin);
    scale += std::abs(d);
    sum_a += d;
  }
  for (auto it = b_first; it != y_last; ++it) {
    const Accum d = Accum(*it) + Accum(origin);
    scale += std::abs(d);
    sum_b += d;
  }
  const Accum tol = 8 * std::numeric_limits<Accum>::epsilon() * scale;
  auto m = std::distance(a_first, x_last) + std::distance(b_first, y_last);
  Accum t = (sum_a - sum_b) / static_cast<Accum>(m);
  for (;;) {
    Work t_w = static_cast<Work>(t);
    auto x_it = detail::partition_monotone(
        a_first, x_last, [=](const Data x) { return Work(x) - origin > t_w; }, t_w + origin);
    auto y_it = detail::partition_monotone(
        b_first, y_last, [=](const Data y) { return Work(y) + origin > -t_w; }, -t_w - origin);
    // Nothing leaves the support: t stands, without re-summing.
    if (x_it == x_last && y_it == y_last) break;
    auto sum_x = sum(x_it, x_last, origin);
    auto n_x = std::distance(x_it, x_last);
    auto sum_y = sum(y_it, y_last, -origin);
    auto n_y = std::distance(y_it, y_last);

    Accum tt, infeas = sum_x - sum_y - static_cast<Accum>(n_x + n_y) * t;
    if (n_y > 0 && infeas > tol) {
      y_last = y_it;
      tt = static_cast<Accum>(m) * t + sum_y;
      m -= n_y;
    } else if (n_x > 0 && infeas < -tol) {
      x_last = x_it;
      tt = static_cast<Accum>(m) * t - sum_x;
      m -= n_x;
    } else {
      x_last = x_it;
      y_last = y_it;
      break;
    }

    if (m > 0) {
      t = tt / static_cast<Accum>(m);
    } else {
      break;
    }
  }

  auto ta = make_thresholds(Result(Accum(origin) + t), lo, hi, a_first, x_last);
  auto tb = make_thresholds(Result(-Accum(origin) - t), lo, hi, b_first, y_last);
  ta.origin = Result(origin);
  ta.correction = Result(t);
  tb.origin = Result(-origin);
  tb.correction = Result(-t);
  return std::make_pair(ta, tb);
}

}  // namespace detail

/**
 * Solve
 *    min_{x,y} ||x - a||^2 + ||y - b||^2
 *              <1, x> = <1, y> <= rhs
 *              0 <= x_i,  0 <= y_j
 *
 * The solution is
 *    x = max(0, a - t)
 *    y = max(0, b - s)
 **/
template <typename Result = double, typename Iterator>
inline std::pair<thresholds<Result, Iterator>, thresholds<Result, Iterator>> thresholds_two_simplex(
    Iterator a_first, Iterator a_last, Iterator b_first, Iterator b_last, const Result rhs = 1,
    const projection_precision mode = projection_precision::float64) {
  std::pair<thresholds<Result, Iterator>, thresholds<Result, Iterator>> fast;
  if (mode == projection_precision::float64 && a_first != a_last && b_first != b_last && rhs > 0 &&
      std::isfinite(rhs) && detail::two_simplex_fast(a_first, a_last, b_first, b_last, rhs, fast))
    return fast;
  return detail::two_simplex_extended(a_first, a_last, b_first, b_last, rhs);
}

// [a_first, a_last) and [b_first, b_last) in place, through temporary copies.
template <typename Result = double, typename Iterator>
inline void prox_two_simplex(Iterator a_first, Iterator a_last, Iterator b_first, Iterator b_last,
                             const Result rhs = 1,
                             const projection_precision mode = projection_precision::float64) {
  prox(a_first, a_last, b_first, b_last,
       thresholds_two_simplex<
           Result,
           typename std::vector<typename std::iterator_traits<Iterator>::value_type>::iterator>,
       rhs, mode);
}

// [a_first, a_last) and [b_first, b_last) in place; a_aux and b_aux are as long.
template <typename Result = double, typename Iterator>
inline void prox_two_simplex(Iterator a_first, Iterator a_last, Iterator b_first, Iterator b_last,
                             Iterator a_aux, Iterator b_aux, const Result rhs = 1,
                             const projection_precision mode = projection_precision::float64) {
  prox(a_first, a_last, b_first, b_last, a_aux, b_aux, thresholds_two_simplex<Result, Iterator>,
       rhs, mode);
}

// [first, middle) and [middle, last) in place, through a temporary copy.
template <typename Result = double, typename Iterator>
inline void prox_two_simplex(Iterator first, Iterator middle, Iterator last, const Result rhs = 1,
                             const projection_precision mode = projection_precision::float64) {
  prox(first, middle, middle, last,
       thresholds_two_simplex<
           Result,
           typename std::vector<typename std::iterator_traits<Iterator>::value_type>::iterator>,
       rhs, mode);
}

// [first, first + p) and [first + p, last) in place; aux is as long.
template <typename Result = double, typename Iterator>
inline void prox_two_simplex(const typename std::iterator_traits<Iterator>::difference_type p,
                             Iterator first, Iterator last, Iterator aux, const Result rhs = 1,
                             const projection_precision mode = projection_precision::float64) {
  prox(first, first + p, first + p, last, aux, aux + p, thresholds_two_simplex<Result, Iterator>,
       rhs, mode);
}

// Each block of dim consecutive entries of [first, last), its first p entries
// the first set; aux holds dim entries.
template <typename Result = double, typename Iterator>
inline void prox_two_simplex(const typename std::iterator_traits<Iterator>::difference_type dim,
                             const typename std::iterator_traits<Iterator>::difference_type p,
                             Iterator first, Iterator last, Iterator aux, const Result rhs = 1,
                             const projection_precision mode = projection_precision::float64) {
  prox(dim, p, first, last, aux, thresholds_two_simplex<Result, Iterator>, rhs, mode);
}

}  // namespace sdca

#endif
