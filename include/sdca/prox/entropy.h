#ifndef SDCA_PROX_ENTROPY_H
#define SDCA_PROX_ENTROPY_H

#include "sdca/math/functor.h"
#include "sdca/math/log_exp.h"
#include "sdca/prox/proxdef.h"
#include "sdca/utility/scratch.h"

namespace sdca {

namespace detail {

template <typename Result, typename Iterator>
using capped_entropy_work =
    typename std::common_type<Result, typename std::iterator_traits<Iterator>::value_type,
                              double>::type;

template <typename Result, typename Iterator>
using capped_entropy_thresholds =
    generalized_thresholds<capped_entropy_work<Result, Iterator>, Iterator,
                           capped_entropy_map<capped_entropy_work<Result, Iterator>>>;

// log(sum_i exp(a_i - origin)) over [first, last), the largest term kept
// apart from the log1p of the others.
template <typename Work, typename Iterator>
inline Work shifted_log_sum_exp(Iterator first, Iterator last, Work origin) {
  auto im = std::max_element(first, last);
  Work m = Work(*im) - origin, tail(0);
  for (auto it = first; it != last; ++it) {
    if (it != im) tail += std::exp((Work(*it) - origin) - m);
  }
  return m + std::log1p(tail);
}

// Pure entropy: the exact finite active-set solution, capping the scores
// above the threshold until none is left. The origin stays apart from the
// log1p(tail) correction, without heap allocations.
template <typename Result, typename Iterator>
inline capped_entropy_thresholds<Result, Iterator> capped_entropy_active_set(Iterator first,
                                                                             Iterator last,
                                                                             Result hi,
                                                                             Result rhs) {
  using Data = typename std::iterator_traits<Iterator>::value_type;
  using Work = capped_entropy_work<Result, Iterator>;
  const Work eps = Work(16) * std::numeric_limits<Work>::epsilon();
  const Work log_hi = std::log(Work(hi));
  Iterator m_first = first;
  Work origin = Work(*std::max_element(m_first, last));
  // The free mass is the requested sum less the caps, in extended
  // precision: a positive remainder far below epsilon is valid mass.
  auto remaining = [&]() {
    return static_cast<long double>(rhs) -
           static_cast<long double>(hi) * static_cast<long double>(std::distance(first, m_first));
  };
  Work lse = shifted_log_sum_exp(m_first, last, origin) - std::log(Work(remaining()));
  for (;;) {
    Work tt = lse + log_hi + eps;
    auto it = std::partition(m_first, last, [=](Data a) { return (Work(a) - origin) > tt; });
    if (it == m_first) break;
    m_first = it;
    if (it == last) break;
    origin = Work(*std::max_element(m_first, last));
    const long double r = remaining();
    if (!(r > 0)) {
      lse = -exp_traits<Work>::min_arg() + Work(1);
      break;
    }
    lse = shifted_log_sum_exp(m_first, last, origin) - std::log(Work(r));
  }
  return make_thresholds(origin, Work(0), Work(hi), m_first, last,
                         capped_entropy_map<Work>(-lse, Work(hi), false));
}

// The residual of the quadratic capped entropy equation in the threshold t
// over centered scores, with what a step needs.
template <typename Work>
struct capped_entropy_residual {
  const Work *v, *end;
  Work hi, rhs, bound;
  // Set by operator(): the uncapped sum, its t-derivatives and the free mass.
  Work sum, derivative, curvature, free;

  capped_entropy_residual(const Work* v_, const Work* end_, Work hi_, Work rhs_)
      : v(v_), end(end_), hi(hi_), rhs(rhs_), bound(capped_entropy_map<Work>(0, hi_, true).bound) {}

  Work operator()(Work t) {
    capped_entropy_map<Work> map(-t, hi, true);
    std::size_t capped = 0;
    sum = derivative = curvature = 0;
    for (const Work* it = v; it != end; ++it) {
      // Capped entries need no Lambert W.
      if (*it - t >= map.bound) {
        ++capped;
        continue;
      }
      const Work probability = std::min(hi, map(*it)), denom = Work(1) + probability,
                 slope = probability / denom;
      sum += probability;
      derivative -= slope;
      curvature += slope / (denom * denom);
    }
    // The free mass rhs - capped * hi in extended precision: the residual and
    // its tolerance scale with the uncapped entries, not with rhs.
    free = Work(static_cast<long double>(rhs) -
                static_cast<long double>(capped) * static_cast<long double>(hi));
    return sum - free;
  }

  Work tolerance() const {
    return Work(32) * std::numeric_limits<Work>::epsilon() * std::abs(free);
  }

  // The lowest capped and the highest uncapped score at t.
  Work lowest_capped(Work t) const {
    Work u = std::numeric_limits<Work>::infinity();
    for (const Work* it = v; it != end; ++it)
      if (*it - t >= bound) u = std::min(u, *it);
    return u;
  }
  Work highest_uncapped(Work t) const {
    Work u = -std::numeric_limits<Work>::infinity();
    for (const Work* it = v; it != end; ++it)
      if (*it - t < bound) u = std::max(u, *it);
    return u;
  }

  // The largest t that caps a score u.
  Work cap_edge(Work u) const {
    Work t = u - bound;
    while (u - t < bound) t = std::nextafter(t, -std::numeric_limits<Work>::infinity());
    return t;
  }

  // The next threshold from t with residual f, before safeguarding.
  Work step(Work t, Work f) const {
    if (!(free > 0)) {
      // No root while these stay capped: release the lowest, a Newton step on its branch.
      return (lowest_capped(t) - bound) + f / (hi / (Work(1) + hi) - derivative);
    }
    if (sum > 2 * free || 2 * sum < free) {
      // Far from the root the free mass is near exponential in t: Newton on its log.
      return t + std::log(sum / free) * sum / -derivative;
    }
    // Halley's correction of the Newton step, bounded to keep it a damping.
    const Work newton = -f / derivative,
               correction = (f * curvature) / (Work(2) * derivative * derivative);
    return t + (std::isfinite(correction)
                    ? newton / (Work(1) - std::max(Work(-.5), std::min(Work(.5), correction)))
                    : newton);
  }
};

// Entropy plus a quadratic term: safeguarded Halley on the threshold, the
// scores centered at the score of the expected last capped coordinate.
template <typename Result, typename Iterator>
inline capped_entropy_thresholds<Result, Iterator> capped_entropy_newton(Iterator first,
                                                                         Iterator last, Result hi,
                                                                         Result rhs) {
  using Data = typename std::iterator_traits<Iterator>::value_type;
  using Work = capped_entropy_work<Result, Iterator>;
  const auto n = std::distance(first, last);
  const std::size_t size = static_cast<std::size_t>(n);
  // Stack scratch for small blocks; v keeps the input order of the sums.
  detail::entropy_buffer<Work> scores(size), ordered(size);
  Work *v = scores.data(), *end = v + size;
  std::copy(first, last, v);
  std::copy(v, end, ordered.data());
  const std::size_t pivot = std::min(size - 1, static_cast<std::size_t>(Work(rhs) / Work(hi)));
  std::nth_element(ordered.data(), ordered.data() + pivot, ordered.data() + size,
                   std::greater<Work>());
  const Work origin = ordered.data()[pivot];
  for (Work* it = v; it != end; ++it) *it -= origin;
  const Work mean = Work(rhs) / Work(n);
  Work uniform = -std::log(mean) - mean;
  Work upper = *std::max_element(v, end) + uniform;
  Work lower = *std::min_element(v, end) + uniform;
  // Start at the pure entropy threshold, clamped to the bracket.
  const Work lse = shifted_log_sum_exp(v, end, Work(0)) - std::log(Work(rhs)) + mean;
  Work t = std::max(lower, std::min(upper, lse));
  capped_entropy_residual<Work> residual(v, end, Work(hi), Work(rhs));
  bool converged = false;
  for (int it = 0; it < 200; ++it) {
    Work f = residual(t);
    const Work tolerance = residual.tolerance();
    converged = std::abs(f) <= tolerance;
    // An entry within rounding of its cap (x = hi - d at top - t ~ bound - d (1 + 1 / hi))
    // can hide a root on its capped side; otherwise capping it keeps the smaller residual.
    const Work top = converged ? residual.highest_uncapped(t) : Work(0);
    if (converged && top - t >= residual.bound - 2 * tolerance * (Work(1) + Work(1) / Work(hi))) {
      const Work edge = residual.cap_edge(top), capped = residual(edge);
      if (capped < 0) {
        converged = false;
        f = residual(t = upper = edge);
      } else if (capped < std::abs(f)) {
        t = edge;
      }
    }
    if (converged) break;
    if (f > 0)
      lower = t;
    else
      upper = t;
    Work next = residual.step(t, f);
    if (!(lower < next && next < upper) || !std::isfinite(next)) next = lower + (upper - lower) / 2;
    if (next == t) break;
    t = next;
  }
  if (!converged) {
    // The root is within rounding of a cap edge: try capping the largest uncapped entry.
    const Work current = residual(t), edge = residual.cap_edge(residual.highest_uncapped(t));
    if (std::isfinite(edge) && std::abs(residual(edge)) < std::abs(current)) t = edge;
  }
  capped_entropy_map<Work> map(-t, Work(hi), true);
  auto middle =
      std::partition(first, last, [&](Data x) { return Work(x) - origin - t >= map.bound; });
  return make_thresholds(origin, Work(0), Work(hi), middle, last, map);
}

}  // namespace detail

/**
 * Solve
 *    min_x <x, log(x)> - <a, x> [+ 0.5 <x, x> if quadratic]
 *          <1, x> = rhs
 *          0 <= x_i <= hi
 *
 * The solution is
 *    x = max(0, min(exp(a - t), hi))  (with the Lambert map if quadratic).
 * Throws std::invalid_argument for nonfinite scores or an infeasible
 * (hi, rhs); reorders [first, last).
 **/
template <typename Result, typename Iterator>
inline detail::capped_entropy_thresholds<Result, Iterator> thresholds_capped_entropy(
    Iterator first, Iterator last, Result hi, Result rhs, bool quadratic) {
  using Work = detail::capped_entropy_work<Result, Iterator>;
  const auto n = std::distance(first, last);
  if (n <= 0 || !std::isfinite(hi) || !std::isfinite(rhs) || hi < 0 || rhs < 0 ||
      Work(rhs) / Work(n) > Work(hi)) {
    throw std::invalid_argument("Infeasible capped entropy projection");
  }
  for (auto it = first; it != last; ++it) {
    if (!std::isfinite(*it)) {
      throw std::invalid_argument("Entropy scores must be finite");
    }
  }
  if (rhs == 0 || Work(rhs) / Work(n) == Work(hi)) {
    const Work value = Work(rhs) / Work(n);
    return make_thresholds(Work(0), value, value, first, last,
                           capped_entropy_map<Work>(0, value, quadratic));
  }
  return quadratic ? detail::capped_entropy_newton(first, last, hi, rhs)
                   : detail::capped_entropy_active_set(first, last, hi, rhs);
}

template <typename Result = double, typename Iterator>
inline auto thresholds_entropy(Iterator first, Iterator last, const Result hi = 1,
                               const Result rhs = 1)
    -> decltype(thresholds_capped_entropy(first, last, hi, rhs, false)) {
  return thresholds_capped_entropy(first, last, hi, rhs, false);
}

// [first, last) in place, through a temporary copy.
template <typename Result = double, typename Iterator>
inline void prox_entropy(Iterator first, Iterator last, const Result hi = 1, const Result rhs = 1) {
  prox(first, last,
       thresholds_entropy<
           Result,
           typename std::vector<typename std::iterator_traits<Iterator>::value_type>::iterator>,
       hi, rhs);
}

// [first, last) in place; aux (as long) is scratch.
template <typename Result = double, typename Iterator>
inline void prox_entropy(Iterator first, Iterator last, Iterator aux, const Result hi = 1,
                         const Result rhs = 1) {
  prox(first, last, aux, thresholds_entropy<Result, Iterator>, hi, rhs);
}

// Each block of dim consecutive entries of [first, last) in place; aux holds dim
// entries.
template <typename Result = double, typename Iterator>
inline void prox_entropy(const typename std::iterator_traits<Iterator>::difference_type dim,
                         Iterator first, Iterator last, Iterator aux, const Result hi = 1,
                         const Result rhs = 1) {
  prox(dim, first, last, aux, thresholds_entropy<Result, Iterator>, hi, rhs);
}

}  // namespace sdca

#endif
