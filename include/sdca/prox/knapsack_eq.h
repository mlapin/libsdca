#ifndef SDCA_PROX_KNAPSACK_EQ_H
#define SDCA_PROX_KNAPSACK_EQ_H

#include <cmath>

#include "sdca/prox/proxdef.h"
#include "sdca/utility/scratch.h"

/*
 * Based on the Algorithm 3.1 in Kiwiel, K. C.,
 * "Variable fixing algorithms for the continuous quadratic knapsack problem.",
 * Journal of Optimization Theory and Applications 136.3 (2008): 445-458.
 */

namespace sdca {

namespace detail {

// The thresholds of the fixed sets [first, m_first) at hi and [m_last, last)
// at lo: the free mass is recomputed in Accum (wider than the data) relative
// to the smallest free score (Lemma 5.3 of Kiwiel), within the bounds of the
// fixed sets. remaining is the mass left for the free set.
template <typename Work, typename Accum, typename Result, typename Iterator>
inline thresholds<Result, Iterator> knapsack_eq_finish(Iterator first, Iterator last,
                                                       Iterator m_first, Iterator m_last,
                                                       const Work origin, const Result lo,
                                                       const Result hi, const Accum remaining) {
  const Work stored_origin =
      (m_first != m_last) ? Work(*std::min_element(m_first, m_last)) : origin;
  Work t_lo = std::numeric_limits<Work>::lowest();
  Work t_hi = std::numeric_limits<Work>::max();
  if (m_last != last) {
    t_lo = Work((static_cast<Accum>(*std::max_element(m_last, last)) - static_cast<Accum>(lo)) -
                static_cast<Accum>(stored_origin));
  }
  if (first != m_first) {
    t_hi = Work((static_cast<Accum>(*std::min_element(first, m_first)) - static_cast<Accum>(hi)) -
                static_cast<Accum>(stored_origin));
  }
  Work correction;
  if (m_first != m_last) {
    Accum sum(0);
    for (auto it = m_first; it != m_last; ++it) {
      sum += static_cast<Accum>(*it) - static_cast<Accum>(stored_origin);
    }
    correction = Work((sum - remaining) / static_cast<Accum>(std::distance(m_first, m_last)));
    correction = std::max(t_lo, std::min(correction, t_hi));
  } else {
    correction = t_lo / 2 + t_hi / 2;
  }
  auto res = make_thresholds(Result(stored_origin + correction), lo, hi, m_first, m_last);
  res.origin = Result(stored_origin);
  res.correction = Result(correction);
  return res;
}

// first_sum, if given, is the sum of x - *first over all x, as the first
// iteration would compute it.
template <typename Work, typename Accum, typename Result, typename Iterator>
inline thresholds<Result, Iterator> thresholds_knapsack_eq_impl(Iterator first, Iterator last,
                                                                const Result lo, const Result hi,
                                                                const Result rhs,
                                                                const Work* first_sum = nullptr) {
  using Data = typename std::iterator_traits<Iterator>::value_type;
  Iterator m_first = first, m_last = last;
  Work origin(0), t(0);
  // With nothing fixed at hi, lo = +0 and hi >= +0, both products are +0 and
  // the extended sum is exactly rhs (the simplex case).
  const bool simplex = lo == 0 && !std::signbit(lo) && !std::signbit(hi);
  auto remaining = [&]() {
    if (simplex && m_first == first) return static_cast<Accum>(rhs);
    const Accum n_hi = static_cast<Accum>(std::distance(first, m_first));
    const Accum n_lo = static_cast<Accum>(std::distance(m_last, last));
    const Accum p_hi = static_cast<Accum>(hi) * n_hi;
    const Accum p_lo = static_cast<Accum>(lo) * n_lo;
    if (p_hi == 0 || p_lo == 0) return (static_cast<Accum>(rhs) - p_hi) - p_lo;
    // Combine opposite saturated masses exactly before subtracting them from rhs.
    Accum e, f;
    const Accum s =
        detail::bound_mass(static_cast<Accum>(lo), n_lo, static_cast<Accum>(hi), n_hi, e);
    const Accum r = detail::two_sum(static_cast<Accum>(rhs), -s, f);
    return r + (f - e);
  };
  for (;;) {
    // Recenter and recompute after removing a bound set so large saturated
    // scores cannot erase the remaining free mass.
    origin = Work(*m_first);
    const Work rem = Work(remaining());
    Work sum(0);
    if (first_sum != nullptr) {
      sum = *first_sum;
      first_sum = nullptr;
    } else {
      for (auto it = m_first; it != m_last; ++it) sum += Work(*it) - origin;
    }
    t = (sum - rem) / Work(std::distance(m_first, m_last));
    auto it_lo = detail::partition_monotone(
        m_first, m_last, [=](Data x) { return (Work(x) - origin) - t > Work(lo); },
        (Work(lo) + t) + origin);
    auto it_hi = detail::partition_monotone(
        m_first, it_lo, [=](Data x) { return (Work(x) - origin) - t > Work(hi); },
        (Work(hi) + t) + origin);
    // Evaluate feasibility from bounded coordinates rather than cancelling
    // sums of raw scores, and scale eps only by active bound sets in this step.
    Work mass = Work(hi) * Work(std::distance(m_first, it_hi)) +
                Work(lo) * Work(std::distance(it_lo, m_last));
    for (auto it = it_hi; it != it_lo; ++it) mass += (Work(*it) - origin) - t;
    const Work error = mass - rem;
    const Work step_scale = std::abs(rem) +
                            std::abs(Work(hi)) * Work(std::distance(m_first, it_hi)) +
                            std::abs(Work(lo)) * Work(std::distance(it_lo, m_last));
    const Work eps = Work(8) * std::numeric_limits<Work>::epsilon() * step_scale;
    if (it_hi != m_first && (it_lo == m_last || error < -eps)) {
      m_first = it_hi;
    } else if (it_lo != m_last && (it_hi == m_first || error > eps)) {
      m_last = it_lo;
    } else {
      m_first = it_hi;
      m_last = it_lo;
      break;
    }
    if (m_first == m_last) break;
  }

  return knapsack_eq_finish<Work, Accum>(first, last, m_first, m_last, origin, lo, hi, remaining());
}

// One interior entry takes the whole mass: the general search's thresholds.
template <typename Work, typename Result, typename Iterator>
inline thresholds<Result, Iterator> knapsack_eq_single(Iterator first, Iterator last,
                                                       const Result lo, const Result hi,
                                                       const Result rhs) {
  const Work origin = Work(*first), correction = Work(0) - Work(rhs);
  auto res = make_thresholds(Result(origin + correction), lo, hi, first, last);
  res.origin = Result(origin);
  res.correction = Result(correction);
  return res;
}

}  // namespace detail

/**
 * Solve
 *    min_x 0.5 * <x, x> - <a, x>
 *          <1, x> = rhs
 *          lo <= x_i <= hi
 *
 * The solution is
 *    x = max(lo, min(a - t, hi))
 **/
template <typename Result = double, typename Iterator>
inline thresholds<Result, Iterator> thresholds_knapsack_eq(Iterator first, Iterator last,
                                                           const Result lo = 0, const Result hi = 1,
                                                           const Result rhs = 1) {
  using Data = typename std::iterator_traits<Iterator>::value_type;
  using FastWork = typename std::common_type<Result, Data, double>::type;
  using WideWork = typename std::common_type<Result, Data, long double>::type;
  using FastAccum = typename detail::wider<typename std::common_type<Result, Data>::type>::type;
  const auto count = std::distance(first, last);
  if (count <= 0 || !std::isfinite(lo) || !std::isfinite(hi) || !std::isfinite(rhs) || lo > hi) {
    throw std::invalid_argument("Infeasible knapsack projection");
  }
  const WideWork mean = WideWork(rhs) / WideWork(count);
  const WideWork slack_lo = WideWork(4) *
                            static_cast<WideWork>(std::numeric_limits<Result>::epsilon()) *
                            std::abs(WideWork(lo));
  const WideWork slack_hi = WideWork(4) *
                            static_cast<WideWork>(std::numeric_limits<Result>::epsilon()) *
                            std::abs(WideWork(hi));
  if (mean < WideWork(lo) - slack_lo || mean > WideWork(hi) + slack_hi) {
    throw std::invalid_argument("Infeasible knapsack projection");
  }
  // One branch-free pass checks the scores, bounds their magnitude and forms
  // the first iteration's sum (in its order, so the result is unchanged).
  FastWork magnitude = std::max(std::abs(FastWork(lo)), std::abs(FastWork(hi)));
  const FastWork origin = FastWork(*first);
  FastWork sum(0);
  bool finite = true;
  for (auto it = first; it != last; ++it) {
    const FastWork x = FastWork(*it);
    finite &= x - x == 0;
    magnitude = std::max(magnitude, std::abs(x));
    sum += x - origin;
  }
  if (!finite) {
    throw std::invalid_argument("Knapsack scores must be finite");
  }
  // Tolerate outward rounding of a represented bound (e.g. hi = 1/3),
  // but never snap a genuinely interior mean to that endpoint.
  if (mean <= WideWork(lo)) return make_thresholds(Result(0), lo, lo, first, first);
  if (mean >= WideWork(hi)) return make_thresholds(Result(0), hi, hi, last, last);
  const double span = std::abs(static_cast<double>(hi) - static_cast<double>(lo));
  const double abs_rhs = std::abs(static_cast<double>(rhs));
  // Bound the centered sums as well as individual scores before using the
  // faster partitions. Tiny scales retain extended working precision.
  const FastWork safe = std::numeric_limits<FastWork>::max() / (FastWork(4) * FastWork(count));
  if (magnitude <= safe && span >= 1e-290 && (rhs == Result(0) || abs_rhs >= 1e-290)) {
    if (count == 1) return detail::knapsack_eq_single<FastWork>(first, last, lo, hi, rhs);
    return detail::thresholds_knapsack_eq_impl<FastWork, FastAccum, Result>(first, last, lo, hi,
                                                                            rhs, &sum);
  }
  if (count == 1) return detail::knapsack_eq_single<WideWork>(first, last, lo, hi, rhs);
  return detail::thresholds_knapsack_eq_impl<WideWork, WideWork, Result>(first, last, lo, hi, rhs);
}

// [first, last) in place, through a temporary copy.
template <typename Result = double, typename Iterator>
inline void prox_knapsack_eq(Iterator first, Iterator last, const Result lo = 0,
                             const Result hi = 1, const Result rhs = 1) {
  prox(first, last,
       thresholds_knapsack_eq<
           Result,
           typename std::vector<typename std::iterator_traits<Iterator>::value_type>::iterator>,
       lo, hi, rhs);
}

// [first, last) in place; aux (as long) is scratch.
template <typename Result = double, typename Iterator>
inline void prox_knapsack_eq(Iterator first, Iterator last, Iterator aux, const Result lo = 0,
                             const Result hi = 1, const Result rhs = 1) {
  prox(first, last, aux, thresholds_knapsack_eq<Result, Iterator>, lo, hi, rhs);
}

// Each block of dim consecutive entries of [first, last) in place; aux holds dim
// entries.
template <typename Result = double, typename Iterator>
inline void prox_knapsack_eq(const typename std::iterator_traits<Iterator>::difference_type dim,
                             Iterator first, Iterator last, Iterator aux, const Result lo = 0,
                             const Result hi = 1, const Result rhs = 1) {
  prox(dim, first, last, aux, thresholds_knapsack_eq<Result, Iterator>, lo, hi, rhs);
}

}  // namespace sdca

#endif
