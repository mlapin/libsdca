#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_CORE_PROJECT_TWO_SIMPLEX_H)
#if SDCA_ISA_ID == 1
#define SDCA_CORE_PROJECT_TWO_SIMPLEX_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_CORE_PROJECT_TWO_SIMPLEX_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_CORE_PROJECT_TWO_SIMPLEX_H3 1
#else
#define SDCA_CORE_PROJECT_TWO_SIMPLEX_H4 1
#endif

#include "sdca/core/project_knapsack.h"
#include "sdca/core/lanes.h"

SDCA_ISA_BEGIN
namespace core {
namespace two_simplex {

template <typename Result, typename Iterator>
using pair = std::pair<thresholds<Result, Iterator>, thresholds<Result, Iterator>>;

// Michelot's support of the simplex of mass rhs over [first, last), moved to [first, last);
// false if rounding empties it.
template <typename W, typename Iterator>
inline bool simplex_support(Iterator first, Iterator& last, const W rhs) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  for (;;) {
    W sum(0);
    for (Iterator it = first; it != last; ++it) sum += W(*it);
    const W t = (sum - rhs) / W(std::distance(first, last));
    const Iterator it = std::partition(first, last, [=](const Data x) { return W(x) > t; });
    if (it == last) return true;
    if (it == first) return false;
    last = it;
  }
}

// A threshold o + c relative to the score origin o: c = (sum (x - o) - sum (y + o) - rhs) / n
// over the supports x and y, compensated, rounded to Result, and a bound on its error
// (4 eps |c|, the second-order terms and the rounding to Result).
template <typename Result, typename W, typename Iterator>
inline W support_correction(Iterator x_first, Iterator x_last, Iterator y_first, Iterator y_last,
                            const W o, const W rhs, W& error) {
  compensated_sum<W> sum;
  W magnitude = std::abs(rhs);
  for (Iterator it = x_first; it != x_last; ++it) {
    sum.add(W(*it));
    sum.add(-o);
    magnitude += std::abs(W(*it)) + std::abs(o);
  }
  for (Iterator it = y_first; it != y_last; ++it) {
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

// Whether every entry x of [first, last) has its output (x - o) - c clearly signed (eps of the
// type it is formed in), 16 error bounds from zero: positive on the support [first, middle),
// negative elsewhere. The supports are then exact. The margin grows with |x - o| no faster
// than the bound, so the entries next to the threshold decide.
template <typename W, typename Iterator>
inline bool support_separated(Iterator first, Iterator middle, Iterator last, const W o, const W c,
                              const W error, const W eps) {
  auto separated = [=](const W x, const bool support) {
    const W d = x - o, v = d - c, bound = 16 * (eps * (std::abs(d) + std::abs(v)) + error);
    return support ? v > bound : v < -bound;
  };
  W low = std::numeric_limits<W>::infinity(), high = -low;
  for (Iterator it = first; it != middle; ++it) low = std::min(low, W(*it));
  for (Iterator it = middle; it != last; ++it) high = std::max(high, W(*it));
  return (first == middle || separated(low, true)) && (middle == last || separated(high, false));
}

// The supports where <1, x> = <1, y> < rhs (s = -t), in W; false if a support empties.
// Accum sums the fixing test; the threshold of the partitions is rounded to W. The fixing
// test compares its sum with a running bound of its rounding (u of Accum, to first order):
// the total m t = sum (x - o) - sum (y + o) over the supports errs by E, which starts at
// m_a u |x - o| + m_b u |y + o| + u |total| (the differences and the sums) and grows by those
// of the sums it drops; the test sum_x - sum_y - n t errs by (n_x + 1) u |x - o| and
// (n_y + 1) u |y + o| (its sums), n (E / m + u |t|) and u |n t| (its product), and u of each
// difference.
template <typename W, typename Accum, typename Iterator>
inline bool inactive_supports(Iterator a_first, Iterator& x_last, Iterator b_first,
                              Iterator& y_last, const W origin, Accum& t) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  const Accum u = limits::roundoff<Accum>();
  // The sum of x - offset over [begin, end) and the sum of their magnitudes.
  auto sum = [](Iterator begin, Iterator end, const W offset, Accum& size) {
    Accum value(0);
    size = 0;
    for (; begin != end; ++begin) {
      const Accum d = Accum(*begin) - Accum(offset);
      value += d;
      size += std::abs(d);
    }
    return value;
  };
  Accum size_a, size_b;
  const Accum sum_a = sum(a_first, x_last, origin, size_a),
              sum_b = sum(b_first, y_last, -origin, size_b);
  const auto m_a = std::distance(a_first, x_last), m_b = std::distance(b_first, y_last);
  auto m = m_a + m_b;
  Accum total = sum_a - sum_b;
  Accum error = u * (Accum(m_a) * size_a + Accum(m_b) * size_b + std::abs(total));
  t = total / Accum(m);
  for (;;) {
    const W t_w = W(t);
    Iterator x_it = partition_monotone(
        a_first, x_last, [=](const Data x) { return W(x) - origin > t_w; }, t_w + origin);
    Iterator y_it = partition_monotone(
        b_first, y_last, [=](const Data y) { return W(y) + origin > -t_w; }, -t_w - origin);
    // Nothing leaves the supports: t stands, without re-summing.
    if (x_it == x_last && y_it == y_last) return true;
    Accum size_x, size_y;
    const Accum sum_x = sum(x_it, x_last, origin, size_x),
                sum_y = sum(y_it, y_last, -origin, size_y);
    const auto n_x = std::distance(x_it, x_last), n_y = std::distance(y_it, y_last);
    const Accum n = Accum(n_x + n_y), nt = n * t, difference = sum_x - sum_y;
    const Accum infeas = difference - nt;
    const Accum tol = u * (Accum(n_x + 1) * size_x + Accum(n_y + 1) * size_y + std::abs(nt) +
                           std::abs(difference) + std::abs(infeas)) +
                      n * (error / Accum(m) + u * std::abs(t));
    if (n_y > 0 && infeas > tol) {
      y_last = y_it;
      total += sum_y;
      error += u * (Accum(n_y + 1) * size_y + std::abs(total));
      m -= n_y;
    } else if (n_x > 0 && infeas < -tol) {
      x_last = x_it;
      total -= sum_x;
      error += u * (Accum(n_x + 1) * size_x + std::abs(total));
      m -= n_x;
    } else {
      x_last = x_it;
      y_last = y_it;
      return true;
    }
    if (m <= 0) return false;
    t = total / Accum(m);
  }
}

// The largest entries of a and b and the largest magnitude (with rhs).
template <typename W, typename Iterator>
inline W scan(Iterator a_first, Iterator a_last, Iterator b_first, Iterator b_last, const W rhs,
              W& max_a, W& max_b) {
  W scale = rhs;
  max_a = max_b = -std::numeric_limits<W>::infinity();
  for (Iterator it = a_first; it != a_last; ++it) {
    max_a = std::max(max_a, W(*it));
    scale = std::max(scale, std::abs(W(*it)));
  }
  for (Iterator it = b_first; it != b_last; ++it) {
    max_b = std::max(max_b, W(*it));
    scale = std::max(scale, std::abs(W(*it)));
  }
  return scale;
}

// The supports in double, then each threshold as a score
// origin o (the largest score, or the midpoint of both sides' largest when the sum is
// inactive) plus a correction c recomputed with compensated sums, as the extended solver
// stores it. Accepted if every entry lies 16 error bounds from its threshold (exact supports)
// and the active-sum case is clear: each output (x - o) - c is then within a few ulps of
// |x - o| + |c|. False otherwise, with the entries reordered. Needs a nonempty a and b and
// rhs > 0.
template <typename Result, typename Iterator>
inline bool solve_fast(Iterator a_first, Iterator a_last, Iterator b_first, Iterator b_last,
                       const Result rhs, pair<Result, Iterator>& out) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  typedef typename std::common_type<Result, Data, double>::type W;
  const W eps = std::numeric_limits<W>::epsilon();
  // The outputs are formed in common_type<Result, Data> (see apply).
  const W eps_out = std::max(
      eps, W(std::numeric_limits<typename std::common_type<Result, Data>::type>::epsilon()));
  W max_a, max_b;
  const W scale = scan(a_first, a_last, b_first, b_last, W(rhs), max_a, max_b);
  // The sums of support_correction reach (2 n + 1) scale, and its error bound 8 n eps^2 scale
  // must stay normal.
  const W n = W(std::distance(a_first, a_last) + std::distance(b_first, b_last));
  if (!(scale <= std::numeric_limits<W>::max() / (2 * n + 1) &&
        scale >= limits::min_over_eps<W>() / eps))
    return false;
  if (W(Result(max_a)) != max_a || W(Result(max_b)) != max_b) return false;
  const Result zero(0);
  if (max_a + max_b <= 0) {
    // Every pairwise margin is nonpositive: the zero solution (exact).
    out = std::make_pair(split_thresholds(max_a, W(0), zero, rhs, a_first, a_first),
                         split_thresholds(-max_a, W(0), zero, rhs, b_first, b_first));
    return true;
  }
  Iterator x_last = a_last, y_last = b_last;
  if (!simplex_support(a_first, x_last, W(rhs)) || !simplex_support(b_first, y_last, W(rhs)))
    return false;
  // The largest score lies in its support: the origins.
  W o_a = max_a, o_b = max_b, error_a, error_b;
  W c_a = support_correction<Result>(a_first, x_last, b_first, b_first, o_a, W(rhs), error_a);
  W c_b = support_correction<Result>(b_first, y_last, a_first, a_first, o_b, W(rhs), error_b);
  const W sum = (o_a + o_b) + (c_a + c_b);
  const W sum_error =
      2 * eps * (std::abs(o_a) + std::abs(o_b) + std::abs(c_a) + std::abs(c_b)) + error_a + error_b;
  bool ok;
  if (sum > 4 * sum_error) {
    ok = support_separated(a_first, x_last, a_last, o_a, c_a, error_a, eps_out) &&
         support_separated(b_first, y_last, b_last, o_b, c_b, error_b, eps_out);
  } else {
    // The inactive-sum search starts from the active-sum supports.
    ok = sum < -4 * sum_error;
    if (ok) {
      W t;
      const W center = W(max_a / 2 - max_b / 2);
      ok = inactive_supports<W, W>(a_first, x_last, b_first, y_last, center, t);
      if (ok) {
        W error;
        o_a = W(Result(center));
        c_a = support_correction<Result>(a_first, x_last, b_first, y_last, o_a, W(0), error);
        W mass(0);
        for (Iterator it = a_first; it != x_last; ++it) mass += (W(*it) - o_a) - c_a;
        const W slack = 4 * W(std::distance(a_first, x_last)) * (error + eps * mass);
        ok = support_separated(a_first, x_last, a_last, o_a, c_a, error, eps_out) &&
             support_separated(b_first, y_last, b_last, -o_a, -c_a, error, eps_out) &&
             mass < W(rhs) - slack;
      }
    }
    o_b = -o_a;
    c_b = -c_a;
  }
  if (!(ok && std::isfinite(c_a) && std::isfinite(c_b))) return false;
  out = std::make_pair(split_thresholds(o_a, c_a, zero, rhs, a_first, x_last),
                       split_thresholds(o_b, c_b, zero, rhs, b_first, y_last));
  return true;
}

// The extended-precision solve, where solve_fast declines.
template <typename Result, typename Iterator>
inline pair<Result, Iterator> solve_extended(Iterator a_first, Iterator a_last, Iterator b_first,
                                             Iterator b_last, const Result rhs) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  typedef typename std::common_type<Result, Data, double>::type Work;
  typedef typename std::common_type<Work, long double>::type Accum;
  if (a_first == a_last || b_first == b_last || !(rhs >= 0) || !std::isfinite(rhs))
    throw std::invalid_argument("Invalid two-simplex dimensions or bound");
  const Result lo(0), hi(rhs);
  if (rhs == 0)
    return std::make_pair(make_thresholds(Result(0), lo, hi, a_first, a_last),
                          make_thresholds(Result(0), lo, hi, b_first, b_last));
  // The active-sum case is two simplex projections, with a per-entry cap strictly above rhs
  // so that a singleton support stays free.
  const Result cap =
      rhs <= std::numeric_limits<Result>::max() / 2 ? 2 * rhs : std::numeric_limits<Result>::max();
  thresholds<Result, Iterator> a = thresholds_knapsack_eq(a_first, a_last, lo, cap, rhs);
  thresholds<Result, Iterator> b = thresholds_knapsack_eq(b_first, b_last, lo, cap, rhs);
  a.hi = b.hi = rhs;
  // The sum is active unless the thresholds add up below zero, to the rounding of their own
  // terms: a fixed floor would swallow a tiny margin.
  const Result terms = std::max(rhs, (std::abs(a.origin) + std::abs(b.origin)) +
                                         (std::abs(a.correction) + std::abs(b.correction)));
  if ((a.origin + b.origin) + (a.correction + b.correction) >=
      -std::numeric_limits<Result>::epsilon() * terms)
    return std::make_pair(a, b);
  Iterator x_last = a.last, y_last = b.last;
  assert(std::distance(a_first, x_last) > 0 && std::distance(b_first, y_last) > 0);
  const Result stored = Result(Accum(*std::max_element(a_first, x_last)) / 2 -
                               Accum(*std::max_element(b_first, y_last)) / 2);
  const Work origin = Work(stored);
  Accum t;
  inactive_supports<Work, Accum>(a_first, x_last, b_first, y_last, origin, t);
  thresholds<Result, Iterator> ta =
      make_thresholds(Result(Accum(origin) + t), lo, hi, a_first, x_last);
  thresholds<Result, Iterator> tb =
      make_thresholds(Result(-Accum(origin) - t), lo, hi, b_first, y_last);
  ta.origin = Result(origin);
  ta.correction = Result(t);
  tb.origin = Result(-origin);
  tb.correction = Result(-t);
  return std::make_pair(ta, tb);
}

}  // namespace two_simplex

/**
 * x = max(0, a - t), y = max(0, b - s) minimizing ||x - a||^2 + ||y - b||^2
 * subject to <1, x> = <1, y> <= rhs, x, y >= 0: in double where its acceptance check
 * passes (solve_fast), else in extended precision.
 **/
template <typename Result = double, typename Iterator>
inline two_simplex::pair<Result, Iterator> thresholds_two_simplex(Iterator a_first, Iterator a_last,
                                                                  Iterator b_first, Iterator b_last,
                                                                  const Result rhs = 1) {
  two_simplex::pair<Result, Iterator> fast;
  if (a_first != a_last && b_first != b_last && rhs > 0 && std::isfinite(rhs) &&
      two_simplex::solve_fast(a_first, a_last, b_first, b_last, rhs, fast))
    return fast;
  return two_simplex::solve_extended(a_first, a_last, b_first, b_last, rhs);
}

}  // namespace core
SDCA_ISA_END

#endif  // SDCA_CORE_PROJECT_TWO_SIMPLEX_H
