#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_CORE_PROJECT_TOPK_ENTROPY_H)
#if SDCA_ISA_ID == 1
#define SDCA_CORE_PROJECT_TOPK_ENTROPY_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_CORE_PROJECT_TOPK_ENTROPY_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_CORE_PROJECT_TOPK_ENTROPY_H3 1
#else
#define SDCA_CORE_PROJECT_TOPK_ENTROPY_H4 1
#endif

#include <functional>
#include <vector>

#include "sdca/core/project_base.h"
#include "sdca/core/lanes.h"

SDCA_ISA_BEGIN
namespace core {
namespace topk_entropy {

// The k largest of [first, last) in order under comp at the front: partial_sort for a small k, else
// nth_element and a sort of the k, O(n + k log k) instead of a heap's O(n log k). The rest is
// left in an unspecified order, so sums over it can differ in the last bits from another
// selection's.
template <typename It, typename Compare>
void sort_largest(It first, std::size_t k, It last, Compare comp) {
  if (k <= 16) {
    std::partial_sort(first, first + k, last, comp);
    return;
  }
  std::nth_element(first, first + (k - 1), last, comp);
  std::sort(first, first + k, comp);
}

// Sum of represented values plus an offset, as sum + correction. Accumulate
// the correction separately: cancelling large scores must not erase a small third score.
template <typename Int, typename Value, typename T>
long double entropy_sum_offset(Int n, Value value, T offset, long double& correction) {
  long double sum(0);
  correction = 0;
  for (Int j = 0; j < n; ++j) {
    const long double v = static_cast<long double>(value(j)),
                      off = static_cast<long double>(offset);
    const long double term = v + off, dv = term - v;
    // TwoSum preserves the offset lost before compensated accumulation.
    const long double error = (v - (term - dv)) + (off - dv);
    const long double next = sum + term;
    correction +=
        (std::abs(sum) >= std::abs(term) ? (sum - next) + term : (term - next) + sum) + error;
    sum = next;
  }
  return sum;
}

// Mean of represented values plus an offset (see entropy_sum_offset). Where long
// double is double, the sums of finite scores can overflow: the power-of-two
// scale (exact) keeps every sum below the largest score, so finite means stay finite.
template <typename Int, typename Value, typename T>
long double entropy_mean_offset(Int n, Value value, T offset) {
  long double correction;
  const long double sum = entropy_sum_offset(n, value, offset, correction);
  const long double mean = (sum + correction) / static_cast<long double>(n);
  if (std::isfinite(mean)) return mean;
  const long double scale = std::ldexp(1.0L, -(std::ilogb(static_cast<long double>(n)) + 2));
  const long double scaled = entropy_sum_offset(
      n, [&](Int j) { return static_cast<long double>(value(j)) * scale; },
      static_cast<long double>(offset) * scale, correction);
  return (scaled + correction) / static_cast<long double>(n) / scale;
}

// Start of the log-odds iteration of all_capped_log_mass, relative to center.
template <typename T>
inline T all_capped_start(T mean, T coefficient, T root_logk, T center, bool unshifted, T lo,
                          T hi) {
  // In the small-mass regime, log(s)+coefficient*s=mean+log(k) gives
  // a Lambert-W start. A raw Newton step from s~1/coefficient otherwise
  // overshoots by hundreds of log units when the coefficient is near max().
  // The start drops -log1p(-s) <= s, a 1 / coefficient part of the linear term: past 100 the
  // start is within 1% (a choice of start; the root does not depend on it).
  if (coefficient > 100 && mean <= std::log(coefficient))
    return (mean - center) +
           (root_logk - lambert_w_exp(mean + (root_logk + std::log(coefficient))));
  return unshifted ? std::log(mean) - std::log(coefficient - mean)
                   : std::max(lo, std::min(hi, root_logk - std::log1p(coefficient)));
}

// log(s) of the mass s of an all-capped block, the root of
//    logit(s) + coefficient s = mean + log(k).
template <typename T>
inline T all_capped_log_mass(T mean, T coefficient, T logk) {
  // Near unit mass, solve for its complement. Otherwise subtracting a
  // coefficient-sized shift from the log-odds destroys the small root.
  // logit(1-s)+coefficient*(1-s) = coefficient-mean-log(k).
  const bool complement = mean > coefficient / T(2);
  const T root_logk = complement ? -logk : logk;
  if (complement) mean = coefficient - mean;
  const bool unshifted = mean > 0 && mean < coefficient;
  const T center = unshifted ? T(0) : mean;
  T lo = unshifted ? (mean - coefficient) + root_logk : root_logk - coefficient;
  T hi = unshifted ? mean + root_logk : root_logk;
  T r = all_capped_start(mean, coefficient, root_logk, center, unshifted, lo, hi);
  for (int it = 0; it < root_steps<T>::total(); ++it) {
    const T logit = center + r, logs = -softplus(-logit), s = std::exp(logs);
    const T one_minus_s = std::exp(-softplus(logit)), ds = s * one_minus_s;
    // Cancel the large terms first; adding log(k) to mean loses it.
    const T residual = unshifted ? coefficient * s - mean : coefficient * s;
    const T f = residual + (r - root_logk);
    const T magnitude = std::abs(residual) + std::abs(r - root_logk) + (unshifted ? mean : T(0));
    // f's rounding: s = exp(-softplus(-logit)) errs by u (|logit| (1 - s) + 2 |log(s)| + 2)
    // relative (logit's rounding through d log(s) / d logit = 1 - s, softplus's own and exp's),
    // coefficient s by u more, and each of the three sums by u of itself; coefficient s <=
    // magnitude.
    const T s_error =
        (one_minus_s > 0 ? std::abs(logit) * one_minus_s : T(0)) + 2 * std::abs(logs) + 3;
    if (std::abs(f) <= limits::roundoff<T>() * (s_error * magnitude + std::abs(residual) +
                                                std::abs(r - root_logk) + std::abs(f)))
      break;
    if (f < 0)
      lo = r;
    else
      hi = r;
    T next = it < root_steps<T>::free_steps() ? r - f / (1 + coefficient * ds) : bisect(lo, hi);
    // Test resolution before the strict bracket check. A rounded Newton
    // step at the new endpoint must not jump back across an enormous bracket.
    if (next == r) break;
    if (!(lo < next && next < hi) || !std::isfinite(next)) next = bisect(lo, hi);
    if (next == r) break;
    r = next;
  }
  return complement ? -softplus(center + r) : -softplus(-(center + r));
}

// All competitors are capped, so the block depends only on their mean.
// Shared with the objective's direct path to avoid allocating/sorting scratch
// or scanning synthetic constant scores. Keep the score and log(k) separate.
template <typename T>
inline void topk_entropy_all_capped(std::size_t n, T mean, T alpha, T output_scale, T& cap,
                                    T& scaled_cap, T& logs) {
  if (!std::isfinite(mean) || !(alpha > 0) || !std::isfinite(alpha) || !(output_scale > 0) ||
      !std::isfinite(output_scale))
    throw std::invalid_argument("Invalid top-k entropy block");
  const T K = T(n), logk = std::log(K), factor = 1 + 1 / K;
  if (alpha > std::numeric_limits<T>::max() / factor) {
    // Finite alpha can still overflow alpha*(1+1/k). Retry this rare
    // scalar solve in a wider exponent range instead of saturating on NaN.
    using Wide = typename wider<T>::type;
    if (std::numeric_limits<T>::max_exponent >= std::numeric_limits<Wide>::max_exponent)
      throw std::overflow_error("Top-k entropy coefficient exceeds the working range");
    Wide wide_cap(0), wide_scaled(0), wide_logs(0);
    topk_entropy_all_capped(n, static_cast<Wide>(mean), static_cast<Wide>(alpha),
                            static_cast<Wide>(output_scale), wide_cap, wide_scaled, wide_logs);
    cap = T(wide_cap);
    scaled_cap = T(wide_scaled);
    logs = T(wide_logs);
    return;
  }
  logs = all_capped_log_mass(mean, alpha * factor, logk);
  cap = std::min(T(1), std::exp(logs)) / K;
  scaled_cap = output_scale > 1 && cap < std::numeric_limits<T>::min()
                   ? std::exp(std::log(output_scale) + (logs - logk))
                   : output_scale * cap;
}

// Exact remainder of the rounded sum center + r (TwoSum).
template <typename T>
inline T log_odds_remainder(T center, T r, T sum) {
  const T rr = sum - center, cc = sum - rr;
  return (center - cc) + (r - rr);
}

// The largest free score of a top-k entropy block, v[index] (index n when all are capped),
// and the normalizer relative to it: free x = W(exp(log(alpha) + (v - v[index]) - normalizer))
// / alpha. Differences from it keep the free scores' digits beside a far larger capped one.
template <typename T>
struct topk_entropy_free {
  std::size_t index;
  T normalizer;
  topk_entropy_free() : index(0), normalizer(0) {}
  topk_entropy_free(std::size_t i, T t) : index(i), normalizer(t) {}
};

// The scores of a top-k entropy block, v_i + offset, relative to their
// maximum wmax so that score differences stay exact even when the offset
// is huge, with the k largest in order (the candidates for the cap).
template <typename T>
struct topk_scores {
  std::size_t n, k;
  const T* v;
  std::size_t* order;
  T K, logk, wmax, vmax, offset, offset_low;
  // Optional low parts of v: the scores are v + low.
  const T* low;
  // suffix[j]: the sum of exp(v - wmax) over order[j..n), for j <= k.
  row_buffer<T> suffix;

  topk_scores(std::size_t n_, const T* v_, T offset_, std::size_t k_, std::size_t* order_,
              const T* low_ = nullptr, T offset_low_ = 0)
      : n(n_),
        k(k_),
        v(v_),
        order(order_),
        K(static_cast<T>(k_)),
        logk(std::log(K)),
        wmax(*std::max_element(v_, v_ + n_)),
        vmax(wmax + offset_),
        offset(offset_),
        offset_low(offset_low_),
        low(low_),
        suffix(k_ + 1) {
    for (std::size_t j = 0; j < n; ++j) order[j] = j;
    // With low parts, scores of equal high parts order by them.
    if (low)
      sort_largest(order, k, order + n, [&](std::size_t i, std::size_t j) {
        return v[i] > v[j] || (v[i] == v[j] && low[i] > low[j]);
      });
    else
      sort_largest(order, k, order + n, [&](std::size_t i, std::size_t j) { return v[i] > v[j]; });
    T* tail = suffix.data();
    tail[k] = 0;
    for (std::size_t j = k; j < n; ++j) tail[k] += std::exp(v[order[j]] - wmax);
    for (std::size_t j = k; j > 0; --j) tail[j - 1] = tail[j] + std::exp(v[order[j - 1]] - wmax);
  }

  // The j-th largest score relative to wmax.
  T sorted(std::size_t j) const { return v[order[j]] - wmax; }

  // The j-th largest score itself, and its low part.
  T score(std::size_t j) const { return v[order[j]]; }
  T score_low(std::size_t j) const { return low ? low[order[j]] : T(0); }

  // In the softmax limit (alpha -> 0) the free mass h s is shared in
  // proportion to exp(v_j), so u caps are too few iff the largest free
  // score would get more than s / k. That predicts the active set.
  std::size_t predicted_caps() const {
    std::size_t u = 0;
    while (u + 1 < k && std::exp(sorted(u)) * (1 - T(u) / K) * K > suffix.data()[u]) ++u;
    return u;
  }

  // The limit root r = h log(sum_free exp(v)) - h log(h) + (u/k) log(k) +
  // sum_capped(v) / k (relative to vmax), which starts every candidate.
  T limit_root(std::size_t caps, T sum_capped) const {
    const T hh = 1 - T(caps) / K;
    // The sum underflows only when its largest term does: use that term.
    const T log_free = std::max(std::log(suffix.data()[caps]), sorted(caps));
    return hh * log_free - hh * std::log(hh) + (T(caps) / K) * logk + sum_capped / K;
  }
};

// One candidate active set of a top-k entropy block: the u largest scores
// capped at s / k, the others free, x_j = W(exp(v_j + offset - t)) / alpha.
// Its scalar root is the log-odds of the mass s, relative to a center (the
// limit root at first, the current log-odds once the iteration stalls at
// the resolution of their sum), with the threshold t relative to the
// largest free score t_c. Newton steps on log(sum) - log(mass) are exact
// where the Lambert terms are exponential; free terms below eps mass / n
// are skipped during the iteration.
//
// Each evaluation bounds its rounding (u of T, to first order): logs = -softplus(-logit) errs
// by u (|logit| (1 - s) + 2 |logs| + 1) (logit's rounding through d logs / d logit = 1 - s,
// and softplus's own), s by u more relative; tau by those through A s (A and h round by 3 u
// and 2 u) and u of each sum and quotient. A free score d_j - tau errs by u |score| (d_j
// is formed once, a fixed perturbation of the scores), its probability by that and its own
// error (kLambertProbabilityRoundings u) relative, |score| at most the larger magnitude of
// the counted scores' extremes; the compensated sum of the counted ones by (2 + n u) u of
// it, the skipped ones by eps mass, mass by s's and 3 u, and f by u of itself.
// tau's own error moves every free score alike, so that it moves the root, not the residual
// at it: the threshold's tolerance takes it, and the resolution of the log-odds ends the
// steps that it hides.
template <typename T>
struct topk_candidate {
  const topk_scores<T>& scores;
  std::size_t n, u;
  T alpha, logalpha, machine, unit_roundoff, infinity;
  T* x;
  T* d;  // the sorted scores relative to the first free one (t_c relative to wmax)
  // h t = A s - h log s + (logit s - vmax) + c0; relative to t_c,
  // h (t - t_c) = A s - h log s + r + base, base = (center - anchor), with the anchor
  // vmax - c0 + h t_c (see the constructor).
  T h, A, c0, t_c, center, shift, base, r, anchor;
  // The evaluation at the last log-odds.
  T f = 0, slope = 0, magnitude = 0, threshold = 0, cap = 0, tau = 0, tau_shifted = 0;
  T sum = 0, mass = 0, rate = 0, dt = 0, s = 0, ds = 0, one_minus_s = 0, logs = 0;
  // The bounds of the last evaluation: f's, tau's, and s's (relative).
  T f_error = 0, tau_error = 0, s_error = 0;
  bool skipped = false;
  T lo, hi, previous;

  topk_candidate(const topk_scores<T>& scores_, std::size_t u_, T sum_capped, T alpha_, T* x_,
                 T* d_)
      : scores(scores_),
        n(scores_.n),
        u(u_),
        alpha(alpha_),
        logalpha(std::log(alpha_)),
        machine(std::numeric_limits<T>::epsilon()),
        unit_roundoff(limits::roundoff<T>()),
        infinity(std::numeric_limits<T>::infinity()),
        x(x_),
        d(d_) {
    const T K = scores.K;
    h = 1 - T(u) / K;
    A = alpha * (1 + T(u) / (K * K));
    c0 = -(T(u) / K) * scores.logk - sum_capped / K;
    t_c = scores.sorted(u);
    // The scores relative to the largest free one, formed once from the scores
    // themselves (not from wmax, which a far larger capped score would round
    // them to), and never regrouped with the threshold by a reassociating compiler.
    const T first_free = scores.score(u), first_low = scores.score_low(u);
    if (scores.low)
      for (std::size_t j = 0; j < n; ++j)
        d[j] = (scores.score(j) - first_free) + (scores.score_low(j) - first_low);
    else
      for (std::size_t j = 0; j < n; ++j) d[j] = scores.score(j) - first_free;
    if (u == 0) {
      const T r0 = scores.limit_root(u, sum_capped);
      anchor = scores.vmax;
      center = scores.vmax + r0;
      shift = center - scores.vmax;
      base = ((shift - scores.offset_low) + c0) - h * t_c;
      r = r0 - shift;
    } else {
      // Below capped scores the anchor is formed from differences with the first free
      // score: (first_free + offset) + (u/k) log k + sum_capped(v - first_free) / k. Its
      // parts from wmax cancel exactly, which keeps the threshold's digits beside a far
      // larger capped score; there the log-odds are saturated and their rounding moot.
      T above(0);
      for (std::size_t j = 0; j < u; ++j) above += d[j];
      anchor = ((first_free + scores.offset) + (first_low + scores.offset_low)) +
               ((T(u) / K) * scores.logk + above / K);
      center = anchor;
      shift = center - scores.vmax;
      base = 0;
      // The limit root relative to the anchor: h (log sum_free exp(d) - log h).
      const T free_log = std::max(std::log(scores.suffix.data()[u]) - t_c, T(0));
      r = h * free_log - h * std::log(h);
    }
    lo = -infinity;
    hi = infinity;
    previous = infinity;
  }

  // mass - sum of the free x_j at the given log-odds (relative to center).
  T evaluate(T logodds) {
    const T K = scores.K;
    const T logit = center + logodds;
    logs = -softplus(-logit);
    const T log1s = -softplus(logit);
    s = std::exp(logs);
    one_minus_s = std::exp(log1s);
    ds = s * one_minus_s;
    cap = s / K;
    // Keep the threshold's log-mass correction separate. Subtracting and
    // re-adding logs can erase the entire cap threshold in a tiny tail.
    tau_shifted = (A * s + (logodds + base)) / h;
    tau = tau_shifted - logs;
    const T uu = unit_roundoff, as = A * s, lb = logodds + base;
    // (1 - s) |logit| is zero where s is one, logit infinite included.
    const T sensitivity = one_minus_s > 0 ? std::abs(logit) * one_minus_s : T(0);
    const T logs_error = uu * (sensitivity + 2 * std::abs(logs) + 1);
    s_error = logs_error + uu;
    tau_error = (as * (s_error + 4 * uu) + uu * (std::abs(lb) + std::abs(as + lb))) / h +
                3 * uu * std::abs(tau_shifted) + logs_error + uu * std::abs(tau);
    dt = (A * ds + (T(u) / K) * one_minus_s + s) / h;
    // The free x_j reaches the cap at the score t + log(cap) + alpha cap.
    threshold = tau_shifted - scores.logk + alpha * cap;
    mass = h * s;
    // x_j <= exp(v_j - t): terms below eps * mass / n cannot matter.
    const T cut = mass > 0 ? std::log(machine * mass / T(n - u)) : -infinity;
    // Local accumulators: x could alias the members for the compiler.
    compensated_sum<T> free_sum;
    T free_rate(0), counted(0), low = infinity, high = -infinity;
    T* const xx = x;
    const T* const dd = d;
    for (std::size_t j = 0; j < u; ++j) xx[j] = cap;
    for (std::size_t j = u; j < n; ++j) {
      const T score = dd[j] - tau;
      if (score < cut) {
        xx[j] = 0;
        skipped = true;
        continue;
      }
      const T xj = lambert_probability(score, alpha, logalpha);
      xx[j] = xj;
      free_sum.add(xj);
      free_rate += xj / (1 + alpha * xj);
      low = std::min(low, score);
      high = std::max(high, score);
      counted += 1;
    }
    sum = free_sum.value();
    rate = free_rate;
    slope = rate * dt + h * ds;
    magnitude = sum + mass;
    const T result = mass - sum;
    const T reach = counted > 0 ? std::max(std::abs(low), std::abs(high)) : T(0);
    f_error = uu * (reach + kLambertProbabilityRoundings + 2 + counted * uu) * sum +
              mass * (s_error + 3 * uu) + machine * mass + uu * std::abs(result);
    return result;
  }

  // Continue from a new center: r restarts at the remainder, the bracket
  // moves along.
  // same_logit: center + r is unchanged (only its split), so base moves with r.
  void recenter(T new_center, T rem, bool same_logit) {
    if (lo > -infinity) lo = (center - new_center) + lo;
    if (hi < infinity) hi = (center - new_center) + hi;
    if (u == 0) {
      base = (((new_center - scores.vmax) - scores.offset_low) + c0) - h * t_c;
    } else if (same_logit) {
      // new_center + rem = center + r exactly (TwoSum): base + r stays the same.
      base = base + (r - rem);
    } else {
      base = new_center - anchor;
    }
    center = new_center;
    shift = center - scores.vmax;
    r = rem;
    previous = infinity;
  }

  // Equal-share model: the largest free score carries h s / (n - u) at the
  // threshold t(s), that is D s + logit(s) / h = E. Newton in its log-odds
  // lambda is exact where the alpha terms dominate, where Newton in the
  // block's log-odds overshoots from a saturated mass or crawls. It starts
  // from the mass E / D where that is not small, otherwise from the
  // small-mass root lambda = h E - W(h D exp(h E)): a mass near 1/D (C
  // beyond 1e150) is out of reach of bisections in the mass itself.
  bool share_logit(T& lambda) const {
    const T D = A / h + alpha * h / T(n - u);
    const T E = t_c + (scores.vmax - c0) / h - std::log(h / T(n - u));
    const T hD = h * D, hE = h * E;
    T lo_l = hE - hD, hi_l = hE;
    // The mass E / D drops logit(s) / h, the small-mass root log1p(-s) <= s: past a mass of
    // 1e-3 the first (a choice of start; the root does not depend on it).
    lambda = E > T(1e-3) * D && E < D ? std::log(E) - std::log(D - E)
                                      : hE - lambert_w_exp(hE + std::log(hD));
    for (int it = 0; it < root_steps<T>::total() && std::isfinite(lambda); ++it) {
      // sigmoid(lambda) and 1 - sigmoid(lambda) from one exponential.
      const T e = std::exp(-std::abs(lambda)), small = e / (1 + e), large = 1 / (1 + e);
      const T share = lambda < 0 ? small : large, rest = lambda < 0 ? large : small;
      const T g = (hD * share + lambda) - hE;
      if (g > 0)
        hi_l = lambda;
      else
        lo_l = lambda;
      T next = it < root_steps<T>::free_steps() ? lambda - g / (1 + hD * share * rest)
                                                : bisect(lo_l, hi_l);
      // Converged before the bracket test: a step that rounds onto an
      // endpoint must not restart bisection over the whole bracket. The
      // scalar model is cheap; solving it fully saves block evaluations.
      // A step within its own rounding (that of g, over a slope of at least one).
      if (std::abs(next - lambda) <=
          unit_roundoff * (hD * share + std::abs(lambda) + std::abs(hE) + std::abs(g))) {
        lambda = next;
        break;
      }
      if (!(lo_l < next && next < hi_l)) next = bisect(lo_l, hi_l);
      lambda = next;
    }
    return std::isfinite(lambda);
  }

  // At the resolution of r: done unless the log-odds have spare
  // resolution, then continue from a center there.
  bool stalled() {
    const T logit = center + r;
    if (!(std::abs(r) > std::abs(logit))) return true;
    recenter(logit, log_odds_remainder(center, r, logit), true);
    return false;
  }

  // A step below the resolution of both the log-odds center + r (the
  // mass) and r + base (the threshold) leaves the evaluation unchanged,
  // however many digits r itself has left.
  bool unresolved(T step) const {
    return step == r || (center + step == center + r && step + base == r + base);
  }

  // Where the alpha terms bend t(logit) (alpha s (1 - s) large), or when
  // the step leaves the saturated mass, step in the mass of the equal-share
  // model instead and center the log-odds there. Returns whether it moved.
  bool move_to_share(T next) {
    if (!(std::abs(next - r) > T(0.5) && alpha * s > 1 && (A * ds > 1 || center + next < 30)))
      return false;
    T logit;
    if (!share_logit(logit)) return false;
    const T at = logit - center;
    if (!(lo < at && at < hi && at != r)) return false;
    recenter(logit, 0, false);
    return true;
  }

  // The next log-odds after an evaluation; returns true instead if the
  // center moved to the equal-share root.
  bool propose(T& next, bool& stepped) {
    // d/dr [log(sum) - log(h s)] = -(rate dt / sum + (1 - s)); when the
    // sum underflowed, the largest free score bounds its log.
    T g, gp;
    if (sum > 0) {
      g = std::log(sum / mass);
      gp = rate * dt / sum + one_minus_s;
    } else {
      g = -tau - std::log(mass);
      gp = dt + one_minus_s;
    }
    // A Newton step that did not reduce |g| (the mass and the sum trade
    // dominance between the two sides of the root) is followed by bisection.
    if (!(std::abs(g) < previous) && lo > -infinity && hi < infinity) {
      next = lo + (hi - lo) / 2;
      stepped = true;
    } else if (mass > 0 && std::isfinite(g) && gp > 0) {
      next = r + g / gp;
      stepped = true;
      if (move_to_share(next)) return true;
    } else if (slope > 0) {
      next = r - f / slope;
      stepped = true;
    }
    previous = std::abs(g);
    return false;
  }

  // Bisection, or a doubling step while the bracket is open.
  T fallback() const {
    if (lo > -infinity && hi < infinity) return bisect(lo, hi);
    return f < 0 ? r + std::max(T(1), std::abs(r)) : r - std::max(T(1), std::abs(r));
  }

  // Solves for the log-odds; returns whether the iteration converged. The steps of root_steps,
  // and the doubling steps of an open bracket before its bisection: |r| >= 1 doubles until the
  // bracket closes or r overflows, within max_exponent steps.
  bool solve() {
    const int steps = root_steps<T>::total() + std::numeric_limits<T>::max_exponent;
    for (int it = 0; it < steps; ++it) {
      f = evaluate(r);
      if (std::isnan(f) || std::abs(f) <= f_error) return std::isfinite(f);
      if (f < 0)
        lo = r;
      else
        hi = r;
      T next = r;
      bool stepped = false;
      if (it < root_steps<T>::free_steps()) {
        if (propose(next, stepped)) continue;
        if (stepped && unresolved(next)) {
          if (stalled()) return std::isfinite(f);
          continue;
        }
      }
      if (!stepped || !(lo < next && next < hi) || !std::isfinite(next)) {
        next = fallback();
        if (unresolved(next)) {
          if (stalled()) return std::isfinite(f);
          continue;
        }
      }
      r = next;
    }
    return false;
  }

  // The tolerance of the tests d_j > threshold, which stand for x_j > cap at the evaluation as
  // computed: its rounding is a fixed perturbation of the problem, whose active set they test
  // (backward error), so that tau's own error, common to both sides, is not in it. They differ
  // by the roundings of each path: the threshold's (log(k), alpha cap with s's error and 2 u,
  // the two sums), and the probability's (tau - logs and d_j - tau, u of |tau| and of the score
  // there, at most |threshold| + |tau|, and its own kLambertProbabilityRoundings u); and the root
  // is within |f| / slope of r, times the threshold's rate (A ds + 1) / h + alpha ds / k.
  T threshold_tolerance() const {
    const T uu = unit_roundoff, ac = alpha * cap, ts = tau_shifted - scores.logk;
    const T rate_of_root = (A * ds + 1) / h + alpha * ds / scores.K;
    return uu * (std::abs(scores.logk) + std::abs(ts) + 2 * std::abs(threshold) +
                 2 * std::abs(tau) + kLambertProbabilityRoundings) +
           ac * (s_error + 2 * uu) + (slope > 0 ? rate_of_root * std::abs(f) / slope : infinity);
  }

  // Writes output_scale x in the caller's order, after checking the mass
  // independently of the threshold's conditioning: a huge threshold
  // tolerance must never authorize a huge feasibility error. The total errs
  // by the residual's rounding f_error, by the residual within a step of the
  // log-odds' resolution where the steps stalled (2 u |center + r| of the
  // mass's rate h ds, 2 u |r + base| of the sum's rate dt), and by its own
  // rounding, (n + 1) u of it; a
  // free entry it clips at the cap by its own rounding and the cap's. Beyond,
  // a free entry exceeds the cap (a wrong active set). mass_relative_tolerance
  // is the rounding of the caller's type, for a solve in a wider one.
  void write(T* result, T output_scale, T mass_relative_tolerance) {
    if (skipped)
      for (std::size_t j = u; j < n; ++j)
        if (x[j] == 0) x[j] = lambert_probability(d[j] - tau, alpha, logalpha);
    // Where the steps stalled (|f| beyond its rounding), the root lies below the log-odds'
    // resolution, at delta = -f / slope; at a tiny mass a relative u |log s| of s takes the
    // free sum alpha s times along. One Newton step in the values themselves, which resolve
    // it: x_j moves by -x_j / (1 + alpha x_j) along tau's dt delta, s by ds delta, and the
    // residual is left second order in delta.
    if (std::abs(f) > f_error && slope > 0 && std::isfinite(f)) {
      const T delta = -f / slope, move = dt * delta;
      for (std::size_t j = u; j < n; ++j) x[j] -= x[j] / (1 + alpha * x[j]) * move;
      tau += move;
      s += ds * delta;
      cap = s / scores.K;
      for (std::size_t j = 0; j < u; ++j) x[j] = cap;
    }
    T total = T(u) * cap, clipped(0);
    for (std::size_t j = u; j < n; ++j) {
      if (x[j] > cap) {
        const T score = d[j] - tau;
        clipped += x[j] * unit_roundoff * (std::abs(score) + kLambertProbabilityRoundings) +
                   cap * (s_error + unit_roundoff);
      }
      x[j] = std::min(x[j], cap);
      total += x[j];
    }
    const T mass_rate = h * ds, sum_rate = std::abs(rate * dt);
    const T resolution = 2 * unit_roundoff *
                         ((mass_rate > 0 ? mass_rate * std::abs(center + r) : T(0)) +
                          (sum_rate > 0 ? sum_rate * std::abs(r + base) : T(0)));
    const T tolerance_mass =
        std::max(f_error + resolution + unit_roundoff * T(n + 1) * total + clipped,
                 mass_relative_tolerance * std::max(total, s));
    if (!std::isfinite(total) || !(std::abs(total - s) <= tolerance_mass))
      throw std::runtime_error("Top-k entropy cap boundary lost normalization precision");
    for (std::size_t j = 0; j < n; ++j) {
      T value = output_scale * x[j];
      // A subnormal probability can still have a normal scaled value.
      if (output_scale > 1 && x[j] < std::numeric_limits<T>::min()) {
        T tail_offset = -scores.logk;
        if (j >= u) {
          const T score_shifted = d[j] - tau_shifted;
          const T score = score_shifted + logs;
          tail_offset = std::min(tail_offset, score_shifted - lambert_w_exp(score + logalpha));
        }
        value = std::exp((std::log(output_scale) + logs) + tail_offset);
      }
      result[scores.order[j]] = value;
    }
  }
};

// The search over the number u of capped scores: the feasible u lies in
// (lower, upper) once a lower bound has been found.
struct topk_active_set {
  std::size_t lower = 0, upper;
  bool have_lower = false;
  explicit topk_active_set(std::size_t k) : upper(k) {}

  // The next candidate after one with u caps of which `above` free scores
  // exceed the threshold or `below` capped ones fall short of it.
  std::size_t next(std::size_t u, std::size_t above, std::size_t below) {
    if (above > 0) {
      lower = u;
      have_lower = true;
      return std::min(u + above, upper - 1);
    }
    if (below > 0) {
      upper = u;
      const std::size_t candidate = u > below ? u - below : 0;
      return have_lower && candidate <= lower ? lower + 1 : candidate;
    }
    return u;
  }
};

// An all-capped block (k = n) in the layout of topk_entropy_core.
template <typename T>
inline void topk_entropy_core_all_capped(std::size_t n, const T* v, T offset, T alpha, T* result,
                                         T* normalizer_out, T* upper_out, T output_scale,
                                         topk_entropy_free<T>* free_out) {
  if (free_out) *free_out = topk_entropy_free<T>{n, T(0)};
  const T wmax = *std::max_element(v, v + n), vmax = wmax + offset;
  const T mean = T(entropy_mean_offset(n, [&](std::size_t j) { return v[j]; }, offset));
  T cap(0), scaled_cap(0), logs(0);
  topk_entropy_all_capped(n, mean, alpha, output_scale, cap, scaled_cap, logs);
  for (std::size_t j = 0; j < n; ++j) result[j] = scaled_cap;
  if (upper_out) *upper_out = cap;
  if (normalizer_out) {
    T min_v = *std::min_element(v, v + n);
    *normalizer_out = (cap > 0) ? ((min_v - wmax) + vmax - std::log(cap) - alpha * cap) : T(0);
  }
}

// Top-k entropy block in the total non-target mass s and its active set on
// the non-target scores v_i + offset (see topk_candidate): one scalar root
// per candidate set of capped scores, the next candidate given by the free
// scores above the threshold (or the capped ones below it), not u + 1.
// order and x are scratch of n.
template <typename T>
inline void topk_entropy_core(std::size_t n, const T* v, T offset, std::size_t k, T alpha,
                              T* result, T* normalizer_out, T* upper_out, std::size_t* order, T* x,
                              T output_scale, T mass_relative_tolerance,
                              topk_entropy_free<T>* free_out, const T* low, T offset_low) {
  if (k == n) {
    topk_entropy_core_all_capped(n, v, offset, alpha, result, normalizer_out, upper_out,
                                 output_scale, free_out);
    return;
  }
  const topk_scores<T> scores(n, v, offset, k, order, low, offset_low);
  row_buffer<T> d(n);
  std::size_t u = scores.predicted_caps();
  T sum_capped(0);
  for (std::size_t j = 0; j < u; ++j) sum_capped += scores.sorted(j);
  topk_active_set active(k);
  for (std::size_t attempt = 0; attempt < k + 2; ++attempt) {
    topk_candidate<T> candidate(scores, u, sum_capped, alpha, x, d.data());
    if (!candidate.solve())
      throw std::runtime_error("Top-k entropy normalization did not converge");
    // Free scores above the threshold need the cap; capped ones below it do not.
    const T threshold = candidate.threshold, tolerance = candidate.threshold_tolerance();
    std::size_t above = 0, below = 0;
    for (std::size_t j = u; j < k && d.data()[j] > threshold + tolerance; ++j) ++above;
    for (std::size_t j = u; j > 0 && d.data()[j - 1] < threshold - tolerance; --j) ++below;
    const std::size_t next = active.next(u, above, below);
    // A move to a candidate already ruled out means a score lies on the
    // cap boundary within rounding: this candidate is the solution.
    if (next == u || next >= active.upper) {
      candidate.write(result, output_scale, mass_relative_tolerance);
      // t = (v_free + offset) + tau: the free score itself keeps tau's digits.
      if (normalizer_out)
        *normalizer_out =
            ((scores.score(u) + offset) + (scores.score_low(u) + offset_low)) + candidate.tau;
      if (upper_out) *upper_out = candidate.cap;
      if (free_out) *free_out = topk_entropy_free<T>{order[u], candidate.tau};
      return;
    }
    sum_capped = 0;
    for (std::size_t j = 0; j < next; ++j) sum_capped += scores.sorted(j);
    u = next;
  }
  throw std::runtime_error("Top-k entropy active set did not converge");
}

// The top-k block of topk_entropy_block solved in Wide, and in a wider type
// still if that fails too; the mass tolerance stays that of T.
template <typename Wide, typename T>
void topk_entropy_wide(std::size_t n, const T* v, T offset, std::size_t k, T alpha, T* result,
                       T* normalizer, T* upper, std::size_t* order, T output_scale,
                       T mass_tolerance, topk_entropy_free<T>* free_out, const T* low,
                       T offset_low) {
  row_buffer<Wide> wide_v(n), wide_x(n), wide_result(n), wide_low(low ? n : 1);
  for (std::size_t j = 0; j < n; ++j) wide_v.data()[j] = static_cast<Wide>(v[j]);
  if (low)
    for (std::size_t j = 0; j < n; ++j) wide_low.data()[j] = static_cast<Wide>(low[j]);
  Wide wide_normal = 0, wide_upper = 0;
  topk_entropy_free<Wide> wide_free;
  try {
    topk_entropy_core(n, wide_v.data(), static_cast<Wide>(offset), k, static_cast<Wide>(alpha),
                      wide_result.data(), &wide_normal, &wide_upper, order, wide_x.data(),
                      static_cast<Wide>(output_scale), static_cast<Wide>(mass_tolerance),
                      &wide_free, low ? wide_low.data() : nullptr, static_cast<Wide>(offset_low));
  } catch (const std::runtime_error&) {
    using Wider = typename wider<Wide>::type;
    if (std::numeric_limits<Wide>::digits >= std::numeric_limits<Wider>::digits) throw;
    topk_entropy_wide<Wider>(n, v, offset, k, alpha, result, normalizer, upper, order, output_scale,
                             mass_tolerance, free_out, low, offset_low);
    return;
  }
  for (std::size_t j = 0; j < n; ++j) result[j] = static_cast<T>(wide_result.data()[j]);
  if (normalizer) *normalizer = static_cast<T>(wide_normal);
  if (upper) *upper = static_cast<T>(wide_upper);
  if (free_out) *free_out = topk_entropy_free<T>{wide_free.index, T(wide_free.normalizer)};
}

}  // namespace topk_entropy

using topk_entropy::topk_entropy_free;

// Solve on v + offset. Results are multiplied by output_scale (default 1)
// before a tiny probability can underflow. Optional normalizer and upper stay
// in probability units: x = min(W(exp(log(alpha)+v+offset-t))/alpha, upper);
// free_out gives the largest free score and t relative to it. Optional low parts
// (low, offset_low) extend v and offset to double-double scores.
template <typename T>
inline void topk_entropy_block(std::size_t n, const T* v, std::size_t k, T alpha, T* result,
                               T* normalizer = nullptr, T* upper = nullptr, T offset = 0,
                               T output_scale = 1, topk_entropy_free<T>* free_out = nullptr,
                               const T* low = nullptr, T offset_low = 0) {
  if (n < 1 || k < 1 || k > n || !(alpha > 0) || !std::isfinite(alpha) || !std::isfinite(offset) ||
      !std::isfinite(offset_low) || !(output_scale > 0) || !std::isfinite(output_scale)) {
    throw std::invalid_argument("Invalid top-k entropy block");
  }
  for (std::size_t j = 0; j < n; ++j) {
    if (!std::isfinite(v[j])) {
      throw std::invalid_argument("Top-k entropy scores must be finite");
    }
  }
  row_buffer<std::size_t> order(n);
  row_buffer<T> x(n);
  // The rounding of a sum of n entries and its comparison in T.
  const T mass_tolerance = limits::roundoff<T>() * T(n + 1);
  try {
    topk_entropy::topk_entropy_core(n, v, offset, k, alpha, result, normalizer, upper, order.data(),
                                    x.data(), output_scale, mass_tolerance, free_out, low,
                                    offset_low);
  } catch (const std::runtime_error&) {
    // Cap ties and cancellation can exceed the root's working precision.
    // Retry only this exceptional path in a wider type (float in double,
    // double in long double where it is wider). The same mass check applies
    // to the retry; never return an infeasible block merely because the
    // active-set search revisits a set.
    using Wide = typename wider<T>::type;
    if (std::numeric_limits<T>::digits >= std::numeric_limits<Wide>::digits) throw;
    topk_entropy::topk_entropy_wide<Wide>(n, v, offset, k, alpha, result, normalizer, upper,
                                          order.data(), output_scale, mass_tolerance, free_out, low,
                                          offset_low);
  }
}

}  // namespace core
SDCA_ISA_END

#endif  // SDCA_CORE_PROJECT_TOPK_ENTROPY_H
