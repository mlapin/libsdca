#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_CORE_PROJECT_ENTROPY_H)
#if SDCA_ISA_ID == 1
#define SDCA_CORE_PROJECT_ENTROPY_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_CORE_PROJECT_ENTROPY_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_CORE_PROJECT_ENTROPY_H3 1
#else
#define SDCA_CORE_PROJECT_ENTROPY_H4 1
#endif

#include <functional>

#include "sdca/core/project_softmax.h"
#include "sdca/core/project_topk_entropy.h"
#include "sdca/core/lanes.h"

SDCA_ISA_BEGIN
namespace core {
namespace entropy {

// 1 / e, the largest |d| exp(d) for d <= 0 (at d = -1): it bounds the error a term's rounded
// exponent leaves in the sums below.
const long double kInvE = 0.36787944117144232159552377016146086744581113103176L;

template <typename Result, typename Iterator>
using work = typename std::common_type<Result, typename std::iterator_traits<Iterator>::value_type,
                                       double>::type;

template <typename Result, typename Iterator>
using capped_thresholds = generalized_thresholds<work<Result, Iterator>, Iterator,
                                                 capped_entropy_map<work<Result, Iterator>>>;

template <typename Result, typename Iterator>
using scaled_thresholds = generalized_thresholds<work<Result, Iterator>, Iterator,
                                                 scaled_lambert_w_exp_map<work<Result, Iterator>>>;

// log sum exp(a_i - origin) over [first, last), as m + log1p(tail) with m the largest score
// less origin: its term exp(0) = 1 is exact, so that the sum runs over every term, in blocks of
// kLogSumBlock plain additions with TwoSum between them, and tail = (total - 1) + lo (total - 1
// exact below 2). tail, if given, receives tail.
const int kLogSumBlock = 16;
template <typename Work, typename Iterator>
inline Work shifted_log_sum_exp(Iterator first, Iterator last, const Work origin,
                                Work* tail_out = nullptr) {
  const Work m = Work(*std::max_element(first, last)) - origin;
  Work hi(0), lo(0);
  for (Iterator it = first; it != last;) {
    Work block(0);
    for (int b = 0; b < kLogSumBlock && it != last; ++b, ++it)
      block += std::exp((Work(*it) - origin) - m);
    two_sum(hi, lo, block);
  }
  const Work tail = (hi - 1) + lo;
  if (tail_out) *tail_out = tail;
  return m + std::log1p(tail);
}

// A bound on the rounding of shifted_log_sum_exp of n scores with origin their largest (m = 0)
// and tail tail, value lse (u of Work): a term exp(d), d <= 0 rounded once, errs by u (|d| + 1)
// exp(d) <= u (1 / e + exp(d)) (|d| exp(d) <= 1 / e); a block of b terms by (b - 1) u of its
// sum, the blocks' TwoSum by (2 + n u) u of the total 1 + tail, total - 1 and the addition of
// lo by u of tail each; log1p by that over 1 + tail and u of itself.
template <typename Work>
inline Work log_sum_error(const Work n, const Work tail, const Work lse) {
  const Work u = limits::roundoff<Work>();
  return u * ((n * Work(kInvE) + 1 + tail + (kLogSumBlock + 1 + n * u) * (1 + tail) + 2 * tail) /
                  (1 + tail) +
              std::abs(lse));
}

// Pure entropy: the exact finite active-set solution, capping the scores above the threshold
// until none is left, the origin kept apart from the log1p(tail) correction.
template <typename Result, typename Iterator>
inline capped_thresholds<Result, Iterator> capped_active_set(Iterator first, Iterator last,
                                                             const Result hi, const Result rhs) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  typedef work<Result, Iterator> Work;
  const Work u = limits::roundoff<Work>(), log_hi = std::log(Work(hi));
  Iterator m_first = first;
  Work origin = Work(*std::max_element(m_first, last));
  // The free mass, the requested sum less the caps, in extended precision: a positive
  // remainder far below epsilon is valid mass.
  auto remaining = [&]() {
    return static_cast<long double>(rhs) -
           static_cast<long double>(hi) * static_cast<long double>(std::distance(first, m_first));
  };
  // lse = log-sum-exp - log(r), its parts for its error bound: that of the log-sum-exp,
  // log(r)'s of the rounded r, u (1 + |log(r)|), and the difference's, u |lse|.
  Work lse, tail, log_sum, log_r;
  auto normalize = [&](const long double r) {
    log_r = std::log(Work(r));
    log_sum = shifted_log_sum_exp(m_first, last, origin, &tail);
    lse = log_sum - log_r;
  };
  normalize(remaining());
  for (;;) {
    // A score is capped where (a - origin) - (lse + log(hi)) exceeds the error of both sides
    // (of the threshold, and u of each near it); none can where the threshold is at least the
    // largest, 0.
    const Work threshold = lse + log_hi;
    if (!(threshold < 0)) break;
    const Work lse_error = log_sum_error(Work(std::distance(m_first, last)), tail, log_sum) +
                           u * (1 + std::abs(log_r) + std::abs(lse));
    const Work tt = threshold + (lse_error + u * (std::abs(log_hi) + 2 * std::abs(threshold)));
    Iterator it = std::partition(m_first, last, [=](Data a) { return (Work(a) - origin) > tt; });
    if (it == m_first) break;
    m_first = it;
    if (it == last) break;
    origin = Work(*std::max_element(m_first, last));
    const long double r = remaining();
    if (!(r > 0)) {
      lse = Work(1) - std::log(std::numeric_limits<Work>::min());
      break;
    }
    normalize(r);
  }
  return make_thresholds(origin, Work(0), Work(hi), m_first, last,
                         capped_entropy_map<Work>(-lse, Work(hi), false));
}

// The bound on the Halley correction c of capped_residual::step: 1 / (1 - c) stays within
// [2 / 3, 2], a damping of the Newton step.
const double kHalleyDamping = 0.5;

// The residual of the quadratic capped entropy equation in the threshold t over centered
// scores, with what a step needs. w holds each entry's W(exp(x - t)) of the last evaluation (0
// where capped): an evaluation within lambert_warm_reach of it starts each W there.
template <typename Work>
struct capped_residual {
  const Work *v, *end;
  Work* w;
  Work hi, rhs, bound;
  // Set by operator(): the uncapped sum, its t-derivatives, the free mass, and the sum over
  // the uncapped entries of their probabilities times their roundings (in u).
  Work sum, derivative, curvature, free, rounding, sum_units;
  std::size_t uncapped;
  Work t_last;
  bool warm;

  capped_residual(const Work* v_, const Work* end_, Work* w_, Work hi_, Work rhs_)
      : v(v_),
        end(end_),
        w(w_),
        hi(hi_),
        rhs(rhs_),
        bound(capped_entropy_map<Work>(0, hi_, true).bound),
        sum(0),
        derivative(0),
        curvature(0),
        free(0),
        rounding(0),
        sum_units(0),
        uncapped(0),
        t_last(0),
        warm(false) {}

  Work operator()(const Work t) {
    const bool near = warm && std::abs(t - t_last) <= lambert_warm_reach<Work>();
    std::size_t capped = 0;
    compensated_sum<Work> total;
    derivative = curvature = rounding = 0;
    const std::size_t n = std::size_t(end - v);
    for (std::size_t i = 0; i < n; ++i) {
      const Work a = v[i] - t;
      // Capped entries need no Lambert W.
      if (a >= bound) {
        ++capped;
        w[i] = 0;
        continue;
      }
      const Work wi = near && w[i] > 0 ? lambert_w_exp_from(a, w[i]) : lambert_w_exp(a);
      w[i] = wi;
      const Work probability = std::min(hi, wi), denom = Work(1) + probability,
                 slope = probability / denom;
      // W errs by its own error and by that of its argument, rounded once: u |x - t|.
      if (probability > 0) rounding += probability * std::abs(a);
      total.add(probability);
      derivative -= slope;
      curvature += slope / (denom * denom);
    }
    t_last = t;
    warm = true;
    // The free mass in extended precision: the residual and its tolerance scale with the
    // uncapped entries, not with rhs.
    free = Work(static_cast<long double>(rhs) -
                static_cast<long double>(capped) * static_cast<long double>(hi));
    uncapped = n - capped;
    sum = total.value();
    sum_units = 2 + Work(uncapped) * limits::roundoff<Work>();
    rounding += kLambertWRoundings * sum;
    return sum - free;
  }

  // The residual's rounding: its terms', their compensated sum's ((2 + n u) u of it), the
  // rounded free mass's and the difference's (u of each).
  Work tolerance() const {
    const Work u = limits::roundoff<Work>();
    return u * (rounding + sum_units * sum + std::abs(free) + std::abs(sum - free));
  }

  // A bound on the exact residual at t + delta for the Halley step delta that step(t, f) takes
  // from the evaluation at t (infinite where another kind of step, a step beyond one, or where
  // an entry crosses its cap in between: the residual is smooth only between the caps). The
  // quadratic model f + f' d + f'' d^2 / 2 is f c^2 / (1 - c)^2 at the step (c the Halley
  // correction); f' and f'' err by their terms' rounding and their sums' (at most u (rounding +
  // (n + 8) |f'|), and alike, the terms below the probabilities), over |delta| and delta^2 / 2;
  // the third-order term is at most curvature e^|delta| |delta|^3 / 6: |d^3 W(e^a) / da^3| =
  // W |1 - 2 W| / (1 + W)^5 <= W / (1 + W)^3, a term of f'', which grows by at most e^|delta|
  // over the step (|d log(W / (1 + W)^3) / da| <= 1), and e^d <= 1 + d + d^2 for d <= 1.
  Work predicted(const Work t, const Work f, const Work delta, const Work tolerance) const {
    const Work inf = std::numeric_limits<Work>::infinity(), u = limits::roundoff<Work>();
    if (!(free > 0) || sum > 2 * free || 2 * sum < free) return inf;
    const Work c = (f * curvature) / (Work(2) * derivative * derivative);
    const Work d = std::abs(delta);
    if (!(std::abs(c) < Work(kHalleyDamping)) || !(d <= 1)) return inf;
    const Work n = Work(uncapped), q = c / (1 - c);
    const Work estimate = std::abs(f) * q * q +
                          u * (rounding + (n + 8) * std::abs(derivative)) * d +
                          u * (rounding + (n + 8) * curvature) * d * d / 2 +
                          curvature * (1 + d + d * d) * d * d * d / 6;
    if (!(estimate <= tolerance)) return estimate;
    // No cap crossed, beyond the rounding of the tests x - t >= bound.
    const Work next = t + delta, margin = 2 * u * (std::abs(next) + std::abs(bound));
    const Work top = highest_uncapped(t), low = lowest_capped(t);
    if (!(top - next < bound - (margin + 2 * u * std::abs(top))) ||
        !(low - next >= bound + (margin + 2 * u * std::abs(low))))
      return inf;
    return estimate;
  }

  Work lowest_capped(const Work t) const {
    Work x = std::numeric_limits<Work>::infinity();
    for (const Work* it = v; it != end; ++it)
      if (*it - t >= bound) x = std::min(x, *it);
    return x;
  }
  Work highest_uncapped(const Work t) const {
    Work x = -std::numeric_limits<Work>::infinity();
    for (const Work* it = v; it != end; ++it)
      if (*it - t < bound) x = std::max(x, *it);
    return x;
  }

  // The largest t that caps a score x.
  Work cap_edge(const Work x) const {
    Work t = x - bound;
    while (x - t < bound) t = std::nextafter(t, -std::numeric_limits<Work>::infinity());
    return t;
  }

  // The next threshold from t with residual f, before safeguarding.
  Work step(const Work t, const Work f) const {
    // No root while these stay capped: the lowest is released, a Newton step on its branch.
    if (!(free > 0)) return (lowest_capped(t) - bound) + f / (hi / (Work(1) + hi) - derivative);
    // Far from the root the free mass is near exponential in t: Newton on its log.
    if (sum > 2 * free || 2 * sum < free) return t + std::log(sum / free) * sum / -derivative;
    // Halley's correction of the Newton step, bounded to keep it a damping.
    const Work newton = -f / derivative,
               correction = (f * curvature) / (Work(2) * derivative * derivative),
               damping = Work(kHalleyDamping);
    return t + (std::isfinite(correction)
                    ? newton / (Work(1) - std::max(-damping, std::min(damping, correction)))
                    : newton);
  }
};

// Entropy plus a quadratic term: safeguarded Halley on the threshold, the scores centered at
// the score of the expected last capped coordinate.
template <typename Result, typename Iterator>
inline capped_thresholds<Result, Iterator> capped_newton(Iterator first, Iterator last,
                                                         const Result hi, const Result rhs) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  typedef work<Result, Iterator> Work;
  const auto n = std::distance(first, last);
  const std::size_t size = std::size_t(n);
  // v keeps the input order of the sums.
  row_buffer<Work> scores(size), ordered(size);
  Work *v = scores.data(), *end = v + size;
  std::copy(first, last, v);
  std::copy(v, end, ordered.data());
  const std::size_t pivot = std::min(size - 1, std::size_t(Work(rhs) / Work(hi)));
  std::nth_element(ordered.data(), ordered.data() + pivot, ordered.data() + size,
                   std::greater<Work>());
  const Work origin = ordered.data()[pivot];
  for (Work* it = v; it != end; ++it) *it -= origin;
  const Work mean = Work(rhs) / Work(n), uniform = -std::log(mean) - mean;
  Work upper = *std::max_element(v, end) + uniform, lower = *std::min_element(v, end) + uniform;
  // The start: the pure entropy threshold, clamped to the bracket.
  const Work lse = shifted_log_sum_exp(v, end, Work(0)) - std::log(Work(rhs)) + mean;
  Work t = std::max(lower, std::min(upper, lse));
  row_buffer<Work> lambert(size);
  capped_residual<Work> residual(v, end, lambert.data(), Work(hi), Work(rhs));
  // At a converged t whose residual is at most f_abs: an entry within rounding of its cap
  // (x = hi - d at top - t ~ bound - d (1 + 1 / hi)) can hide a root on its capped side (false,
  // with t and f there); otherwise capping it keeps the smaller residual.
  auto near_cap = [&](Work& t_, Work& f_, const Work f_abs, const Work tolerance) {
    const Work top = residual.highest_uncapped(t_);
    if (top - t_ >= residual.bound - 2 * tolerance * (Work(1) + Work(1) / Work(hi))) {
      const Work edge = residual.cap_edge(top), capped = residual(edge);
      if (capped < 0) {
        f_ = residual(t_ = upper = edge);
        return false;
      }
      if (capped < f_abs) t_ = edge;
    }
    return true;
  };
  bool converged = false;
  for (int it = 0; it < root_steps<Work>::total(); ++it) {
    Work f = residual(t);
    const Work tolerance = residual.tolerance();
    converged = std::abs(f) <= tolerance && near_cap(t, f, std::abs(f), tolerance);
    if (converged) break;
    (f > 0 ? lower : upper) = t;
    const bool free_step = it < root_steps<Work>::free_steps();
    Work next = free_step ? residual.step(t, f) : bisect(lower, upper);
    bool halley = free_step;
    if (!(lower < next && next < upper) || !std::isfinite(next)) {
      next = bisect(lower, upper);
      halley = false;
    }
    if (next == t) break;
    // A Halley step whose residual is bound within the tolerance ends the solve without
    // evaluating it.
    const Work bound = halley ? residual.predicted(t, f, next - t, tolerance)
                              : std::numeric_limits<Work>::infinity();
    t = next;
    if (bound <= tolerance) {
      converged = near_cap(t, f, bound, tolerance);
      if (converged) break;
    }
  }
  if (!converged) {
    // The root is within rounding of a cap edge: capping the largest uncapped entry.
    const Work current = residual(t), edge = residual.cap_edge(residual.highest_uncapped(t));
    if (std::isfinite(edge) && std::abs(residual(edge)) < std::abs(current)) t = edge;
  }
  const capped_entropy_map<Work> map(-t, Work(hi), true);
  Iterator middle =
      std::partition(first, last, [&](Data x) { return Work(x) - origin - t >= map.bound; });
  return make_thresholds(origin, Work(0), Work(hi), middle, last, map);
}

// The capped entropy projection, with a quadratic term if quadratic.
template <typename Result, typename Iterator>
inline capped_thresholds<Result, Iterator> capped(Iterator first, Iterator last, const Result hi,
                                                  const Result rhs, const bool quadratic) {
  typedef work<Result, Iterator> Work;
  const auto n = std::distance(first, last);
  if (n <= 0 || !std::isfinite(hi) || !std::isfinite(rhs) || hi < 0 || rhs < 0 ||
      Work(rhs) / Work(n) > Work(hi))
    throw std::invalid_argument("Infeasible capped entropy projection");
  check_scores(first, last, "Entropy scores must be finite");
  if (rhs == 0 || Work(rhs) / Work(n) == Work(hi)) {
    const Work value = Work(rhs) / Work(n);
    return make_thresholds(Work(0), value, value, first, last,
                           capped_entropy_map<Work>(0, value, quadratic));
  }
  return quadratic ? capped_newton(first, last, hi, rhs) : capped_active_set(first, last, hi, rhs);
}

// The number u of capped scores among the n scores v (partially sorted here), the sum of the
// capped ones relative to wmax, and the log-sum-exp of the others relative to the largest of
// them, v[u]. Only the k largest can be capped: the rest are summed once, relative to
// v[k - 1], and the sums of the scores after each of the k largest relative to it follow from
// one backward recurrence, rest[a] = exp(v[a + 1] - v[a]) (1 + rest[a + 1]), whose factors are
// at most one.
//
// The test of each candidate allows for the rounding of its threshold (u of Work), from a
// running bound of the absolute error of rest: a term exp(d), d <= 0, errs by u (|d| + 1) of
// itself (d rounds once), at most u (1 / e + exp(d)) (|d| exp(d) <= 1 / e), their sum by its
// count times u of itself, and each step of the recurrence by its factor times the previous
// error, the factor's rounding, u (|d| + 1) times the product (at most u (1 + rest) / e + u
// rest), and u twice of the result; log1p(rest) errs by that over 1 + rest and u of itself,
// log(k - a) and the difference by u of each.
template <typename Work>
inline std::size_t topk_active_set(Work* v, const std::size_t n, const std::size_t k,
                                   const Work wmax, Work& sum, Work& lse) {
  const Work K = Work(k), u = limits::roundoff<Work>();
  topk_entropy::sort_largest(v, k, v + n, std::greater<Work>());
  Work tail_k(0);
  const Work vk = v[k - 1], count = Work(n - k), inv_e = Work(kInvE);
  for (std::size_t j = k; j < n; ++j) tail_k += std::exp(v[j] - vk);
  row_buffer<Work> rest_buffer(k), error_buffer(k);
  Work *rest = rest_buffer.data(), *error = error_buffer.data();
  rest[k - 1] = tail_k;
  error[k - 1] = u * (count * inv_e + (count + 1) * tail_k);
  for (std::size_t a = k - 1; a > 1; --a) {
    const Work d = v[a] - v[a - 1], factor = std::exp(d), one = 1 + rest[a];
    rest[a - 1] = factor * one;
    error[a - 1] = factor * error[a] + u * (one * inv_e + 3 * rest[a - 1]);
  }
  sum = v[0] - wmax;
  for (std::size_t active = 1; active < k; ++active) {
    const Work l = std::log1p(rest[active]), log_k = std::log(K - Work(active));
    const Work threshold = l - log_k;
    const Work tol = error[active] / (1 + rest[active]) +
                     u * (std::abs(l) + std::abs(log_k) + std::abs(threshold));
    if (Work(0) <= threshold + tol) {
      // The log-sum-exp of the free scores from a direct sum.
      const Work va = v[active], scale = std::exp(vk - va);
      Work direct(0);
      if (scale > 0 && std::isfinite(tail_k)) {
        for (std::size_t j = active + 1; j < k; ++j) direct += std::exp(v[j] - va);
        direct += scale * tail_k;
      } else {
        for (std::size_t j = active + 1; j < n; ++j) direct += std::exp(v[j] - va);
      }
      lse = std::log1p(direct);
      return active;
    }
    sum += v[active] - wmax;
  }
  throw std::runtime_error("Top-k entropy active set did not converge");
}

}  // namespace entropy

/**
 * x = max(0, min(exp(a - t), hi)) minimizing <x, log(x)> - <a, x>
 * subject to <1, x> = rhs, 0 <= x_i <= hi; reorders [first, last).
 **/
template <typename Result = double, typename Iterator>
inline entropy::capped_thresholds<Result, Iterator> thresholds_entropy(Iterator first,
                                                                       Iterator last,
                                                                       const Result hi = 1,
                                                                       const Result rhs = 1) {
  return entropy::capped(first, last, hi, rhs, false);
}

/**
 * x = max(0, min(lambert_w_exp(a - t), hi)) minimizing 0.5 <x, x> + <x, log(x)> - <a, x>
 * subject to <1, x> = rhs, 0 <= x_i <= hi; reorders [first, last).
 **/
template <typename Result = double, typename Iterator>
inline entropy::capped_thresholds<Result, Iterator> thresholds_entropy_norm(Iterator first,
                                                                            Iterator last,
                                                                            const Result hi = 1,
                                                                            const Result rhs = 1) {
  return entropy::capped(first, last, hi, rhs, true);
}

/**
 * x = max(0, min(exp(a + offset - t), hi)) minimizing
 *    <x, log(x)> + (1 - s) log(1 - s) - <a + offset, x>
 * subject to <1, x> = s <= 1, 0 <= x_i <= s / k, with t the largest uncapped score, so that
 * differences among the uncapped ones survive a far larger capped one.
 **/
template <typename Result = double, typename Iterator>
inline entropy::capped_thresholds<Result, Iterator> thresholds_topk_entropy_offset(
    Iterator first, Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1, const Result offset = 0) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  typedef entropy::work<Result, Iterator> Work;
  const auto n = std::distance(first, last);
  if (k < 1 || k > n || !std::isfinite(offset))
    throw std::invalid_argument("Invalid top-k projection dimensions");
  check_scores(first, last, "Top-k entropy scores must be finite");
  if (k == n) {
    const Work mean = Work(
        topk_entropy::entropy_mean_offset(n, [&](decltype(n) j) { return *(first + j); }, offset));
    const Work cap = std::exp(-softplus(-(mean + std::log(Work(k))))) / Work(k);
    return make_thresholds(Work(0), cap, cap, last, last,
                           capped_entropy_map<Work>(Work(0), cap, false));
  }
  Iterator max_it = std::max_element(first, last);
  const Work wmax = Work(*max_it), origin = wmax + Work(offset), K = Work(k);
  const Work u = limits::roundoff<Work>();
  // Nothing capped (always for k = 1), without sorting or allocating; the test allows for
  // the rounding of thresh0, as in topk_active_set.
  Work tail0(0);
  for (Iterator it = first; it != last; ++it)
    if (it != max_it) tail0 += std::exp(Work(*it) - wmax);
  const Work lse0 = std::log1p(tail0), log_k0 = std::log(K), thresh0 = lse0 - log_k0;
  const Work count0 = Work(n - 1);
  const Work tol0 = u * ((count0 * Work(entropy::kInvE) + (count0 + 1) * tail0) / (1 + tail0) +
                         std::abs(lse0) + std::abs(log_k0) + std::abs(thresh0));
  if (k == 1 || Work(0) <= thresh0 + tol0) {
    const Work odds = origin + thresh0 + std::log(K);
    const Work logs = -softplus(-odds), cap = std::exp(logs) / K;
    return make_thresholds(wmax, Work(0), cap, first, last,
                           capped_entropy_map<Work>(logs - lse0, cap, false));
  }
  const std::size_t un = std::size_t(n), uk = std::size_t(k);
  row_buffer<Work> v_buf(un);
  Work* v = v_buf.data();
  // The scores themselves, compared with each other rather than with wmax: free scores far
  // below a capped one keep their differences.
  std::copy(first, last, v);
  Work sum(0), lse(0);
  const std::size_t ua = entropy::topk_active_set(v, un, uk, wmax, sum, lse);
  const auto active = typename std::iterator_traits<Iterator>::difference_type(ua);
  const Work remaining = K - Work(ua);
  // The first free score is the map's origin, so that apply keeps its differences too.
  const Work origin_free = v[ua];
  const Work odds = origin +
                    (sum + remaining * ((origin_free - wmax) + lse - std::log(remaining))) / K +
                    std::log(K);
  const Work logs = -softplus(-odds), cap = std::exp(logs) / K;
  const Work shift = logs + std::log(remaining / K) - lse;
  Iterator middle = first + active;
  if (active > 0) std::nth_element(first, middle, last, std::greater<Data>());
  return make_thresholds(origin_free, Work(0), cap, middle, last,
                         capped_entropy_map<Work>(shift, cap, false));
}

template <typename Result = double, typename Iterator>
inline entropy::capped_thresholds<Result, Iterator> thresholds_topk_entropy(
    Iterator first, Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1) {
  return thresholds_topk_entropy_offset<Result>(first, last, k, Result(0));
}

/**
 * x = max(0, min(lambert_w_exp(a - t) / alpha, hi)) minimizing
 *    0.5 alpha (<x, x> + s^2) + <x, log(x)> + (1 - s) log(1 - s) - <a, x>
 * subject to <1, x> = s <= 1, 0 <= x_i <= s / k.
 **/
template <typename Result = double, typename Iterator>
inline entropy::scaled_thresholds<Result, Iterator> thresholds_topk_entropy_biased(
    Iterator first, Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1, const Result alpha = 1) {
  typedef entropy::work<Result, Iterator> Work;
  typedef scaled_lambert_w_exp_map<Work> Map;
  if (!(alpha > 0) || !std::isfinite(alpha) || k < 1 || k > std::distance(first, last))
    throw std::invalid_argument("Invalid top-k entropy dimensions or scale");
  const Work a = Work(alpha), lo(0);
  const std::size_t n = std::size_t(std::distance(first, last));
  if (k == 1) {
    // No entry reaches the bound s / k: the softmax step of a target of score 0 and the
    // inputs as the other scores, x_i = W(alpha exp(a_i - t)) / alpha and s = sum_i x_i.
    row_buffer<Work> v(n + 1);
    v.data()[0] = Work(0);
    std::copy(first, last, v.data() + 1);
    softmax_prox<Work> block(n + 1, a);
    block.set_scores(v.data(), [](std::size_t j) { return j == 0 ? Work(1) : Work(0); });
    block.solve();
    block.balance();
    const Work t = block.normalizer() - std::log(a);
    return make_thresholds(Work(0), lo, block.r[0], first, last, Map(a, -t));
  }
  // k > 1: in the logit of the total mass s, with the normalizer and cap of
  // x_i = min(V(log(alpha) + a_i - t) / alpha, s / k).
  row_buffer<Work> v(n), x(n);
  std::copy(first, last, v.data());
  Work t(0), cap(0);
  topk_entropy_free<Work> largest_free;
  topk_entropy_block(n, v.data(), std::size_t(k), a, x.data(), &t, &cap, Work(0), Work(1),
                     &largest_free);
  if (largest_free.index == n)
    return make_thresholds(Work(0), lo, cap, first, last, Map(a, std::log(a) - t));
  // The map takes the scores relative to the largest free one, which keeps their differences
  // beside a far larger capped score.
  return make_thresholds(v.data()[largest_free.index], lo, cap, first, last,
                         Map(a, std::log(a) - largest_free.normalizer));
}

/**
 * The two-block entropic projection of [a_first, a_last) (p entries) and [b_first, b_last):
 *    min_{x,y} 0.5 alpha ||x - a / alpha - 1 / p||^2 + <x, log(x)> +
 *              0.5 alpha ||y - b / alpha||^2 + <y, log(y)>
 *    subject to <1, x> + <1, y> = 1, x, y >= 0,
 * the softmax step of the target uniform on the first block; v holds both blocks' entries.
 **/
template <typename Iterator, typename Work>
inline softmax_prox<Work>& solve_two_entropy(softmax_prox<Work>& block, Iterator a_first,
                                             Iterator a_last, Iterator b_first, Iterator b_last,
                                             Work* v) {
  const std::size_t p = std::size_t(std::distance(a_first, a_last));
  std::copy(a_first, a_last, v);
  std::copy(b_first, b_last, v + p);
  block.set_labels(v, p, [=](std::size_t j) { return j < p; });
  block.solve();
  return block;
}

template <typename Result = double, typename Iterator>
inline std::pair<entropy::scaled_thresholds<Result, Iterator>,
                 entropy::scaled_thresholds<Result, Iterator>>
thresholds_two_entropy(Iterator a_first, Iterator a_last, Iterator b_first, Iterator b_last,
                       const Result alpha = 1) {
  typedef entropy::work<Result, Iterator> Work;
  typedef scaled_lambert_w_exp_map<Work> Map;
  const std::size_t p = std::size_t(std::distance(a_first, a_last));
  const std::size_t n = p + std::size_t(std::distance(b_first, b_last));
  if (p == 0 || p == n) throw std::invalid_argument("Empty entropy block");
  row_buffer<Work> v(n);
  softmax_prox<Work> block(n, Work(alpha));
  solve_two_entropy(block, a_first, a_last, b_first, b_last, v.data());
  const Work q = Work(1) / Work(p), a = Work(alpha);
  // Relative to the largest score, so that differences survive a far-off block.
  const Work shift = std::log(a) - block.relative_normalizer();
  return std::make_pair(
      make_thresholds(block.v_max, Work(0), Work(1), a_first, a_last, Map(a, shift + a * q)),
      make_thresholds(block.v_max, Work(0), Work(1), b_first, b_last, Map(a, shift)));
}

}  // namespace core
SDCA_ISA_END

#endif  // SDCA_CORE_PROJECT_ENTROPY_H
