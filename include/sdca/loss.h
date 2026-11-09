#ifndef SDCA_LOSS_H
#define SDCA_LOSS_H

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include "sdca/math/compensated.h"
#include "sdca/math/distillation.h"
#include "sdca/math/entropy.h"
#include "sdca/prox/topk_entropy.h"
#include "sdca/prox/topk_simplex.h"
#include "sdca/prox/knapsack_le.h"
#include "sdca/prox/two_simplex.h"
#include "sdca/solver/objective/l2_entropy_topk.h"
#include "sdca/utility/scratch.h"

namespace sdca {

// The ids of the C ABI (sdca_loss); the hinges are smoothed by gamma > 0.
enum class loss_kind {
  cross_entropy = 0,
  topk_cross_entropy = 1,
  multiclass_hinge = 2,
  topk_hinge_alpha = 3,
  topk_hinge_beta = 4,
  topk_hinge = topk_hinge_beta,
  multilabel_hinge = 5,
  multilabel_cross_entropy = 6,
  proximal_cross_entropy = 7
};

namespace detail {

// The proximal entropy loss and its gradient -a at the scores z.
template <typename Data, typename T>
double proximal_entropy_loss(std::size_t n, std::size_t p, const Data* z, Data* gradient,
                             double alpha) {
  if (p == n) throw std::invalid_argument("Proximal entropy needs a non-target class");
  entropy_buffer<T> v(n), a(n);
  const T origin = T(*std::max_element(z, z + n));
  for (std::size_t j = 0; j < n; ++j) v.data()[j] = T(z[j]) - origin;
  const T value = entropy_prox_loss_ptr(n, p, T(alpha), v.data(), a.data());
  for (std::size_t j = 0; j < n; ++j) gradient[j] = Data(-a.data()[j]);
  return double(value);
}

// The multilabel hinge loss (smoothed by gamma if smooth) and its gradient.
template <typename Data, typename T>
double multilabel_hinge_loss(std::size_t n, std::size_t p, const Data* z, Data* gradient,
                             double gamma, bool smooth, const projection_precision mode) {
  if (p == n) throw std::invalid_argument("Multilabel hinge needs a negative label");
  auto pos = std::min_element(z, z + p), neg = std::max_element(z + p, z + n);
  const T low = T(*pos), high = T(*neg), origin = T(z[0]), g = T(gamma);
  if (!smooth) {
    // The compensated margin keeps a tiny 1 + high - low.
    T value = std::max(T(0), margin(high, low));
    if (value > 0) {
      gradient[pos - z] = Data(-1);
      gradient[neg - z] = Data(1);
    }
    return double(value);
  }
  // Margins 1/2 -+ (z_j - z_0), or, when the largest pairwise margin nearly
  // cancels, -(z_j - low) and 1 + z_j - low: that margin sits in one entry.
  const bool recenter =
      std::abs((high - low) + 1) <
      margin_cancellation<T>() * (1 + std::abs(low - origin) + std::abs(high - origin));
  entropy_buffer<T> h_buffer(n), a_buffer(n), aux(n);
  T *h = h_buffer.data(), *a = a_buffer.data();
  for (std::size_t j = 0; j < n; ++j)
    if (recenter)
      a[j] = h[j] = j < p ? low - T(z[j]) : margin(T(z[j]), low);
    else
      a[j] = h[j] = j < p ? T(.5) - (T(z[j]) - origin) : T(.5) + (T(z[j]) - origin);
  prox_two_simplex(a, a + p, a + p, a + n, aux.data(), aux.data() + p, g, mode);
  T value(0);
  for (std::size_t j = 0; j < n; ++j) {
    value += smooth_hinge_term(h[j], a[j], g);
    gradient[j] = Data((j < p ? -a[j] : a[j]) / g);
  }
  return double(value);
}

// The top-k entropy loss from one threshold solve; a receives the gradient of
// the n - 1 non-target scores (their capped probabilities).
template <typename Data, typename T>
double topk_entropy_loss(std::size_t n, const Data* z, std::size_t k, T* a, T* target = nullptr) {
  const std::size_t m = n - 1;
  if (k == m) {
    const long double mean = detail::entropy_mean_offset(
        m, [&](std::size_t j) { return z[j + 1]; }, -static_cast<long double>(z[0]));
    const T odds = T(mean) + std::log(T(k));
    T loss = softplus(odds), mass = std::exp(-softplus(-odds));
    if (odds < -36) {  // as topk_entropy_label
      const T half = std::exp(T(mean) / 2), small = half * (half * T(k));
      loss = std::log1p(small);
      mass = small / (1 + small);
    }
    std::fill(a, a + m, mass / T(k));
    if (target) *target = -mass;
    return double(loss);
  }
  const T maximum = T(*std::max_element(z + 1, z + n));
  entropy_buffer<T> buffer(m);
  T* v = buffer.data();
  // The scores themselves (origin maximum - z[0] below): the thresholds and the
  // map take free scores relative to the largest free one, t, which keeps
  // their differences beside a far larger capped score.
  for (std::size_t j = 0; j < m; ++j) v[j] = a[j] = T(z[j + 1]);
  auto th = thresholds_topk_entropy_offset<T>(v, v + m, static_cast<std::ptrdiff_t>(k), -T(z[0]));
  // As l2_entropy_topk::primal_loss: the u capped scores relative to the
  // maximum and the free ones' log-sum-exp relative to t.
  const T u = static_cast<T>(std::distance(v, th.first)), K = static_cast<T>(k), t = T(th.t);
  compensated_sum<T> head, rest;
  for (T* it = v; it != th.first; ++it) head.add(*it - maximum);
  const T* top = std::max_element(th.first, v + m);
  for (T* it = th.first; it != v + m; ++it)
    if (it != top) rest.add(std::exp(*it - t));
  const T lse = (t - maximum) + std::log1p(rest.value());
  const T gap = maximum - T(z[0]);
  const T mean = (head.value() + (K - u) * (lse - std::log(K - u))) / K;
  const T log_odds = gap + mean + std::log(K);
  prox(th, a, a + m);
  T loss = softplus(log_odds), mass = std::exp(-softplus(-log_odds));
  if (log_odds < -36) {  // as topk_entropy_label
    const T above = std::max(gap, T(0)), half = std::exp((gap - above) / 2);
    const T small = half * (half * (std::exp(mean + above) * K));
    loss = std::log1p(small);
    mass = small / (1 + small);
  }
  if (target) *target = -mass;
  return double(loss);
}

// The nonsmooth top-k hinge losses (alpha averages the k largest margins h,
// beta their positive parts); a receives a subgradient.
template <typename T>
double topk_hinge_loss(bool beta, std::size_t m, const T* h, std::size_t k, T* a) {
  if (k == 1) {
    // Both losses are max(0, max h); the first largest, as the stable sort below.
    std::size_t best = 0;
    for (std::size_t j = 1; j < m; ++j)
      if (h[j] > h[best]) best = j;
    const T value = std::max(T(0), h[best]);
    if (value > 0) a[best] = T(1);
    return double(value);
  }
  // The k largest in stable-sort order: ties by index.
  entropy_buffer<std::size_t> buffer(m);
  std::size_t* order = buffer.data();
  std::iota(order, order + m, std::size_t(0));
  // Only the set matters, the same under this total order either way:
  // partial_sort (m + k log k log(m / k) on random margins) for a small k,
  // else nth_element (linear); the measured crossover is near m / 32.
  auto larger = [&](std::size_t i, std::size_t j) {
    return h[i] > h[j] || (h[i] == h[j] && i < j);
  };
  if (k <= std::max<std::size_t>(16, m / 32))
    std::partial_sort(order, order + k, order + m, larger);
  else
    std::nth_element(order, order + (k - 1), order + m, larger);
  // Summed in double: a float sum of k terms drifts by O(k) float eps.
  using Sum = typename std::common_type<T, double>::type;
  Sum value(0);
  for (std::size_t j = 0; j < k; ++j)
    value += Sum(beta ? std::max(T(0), h[order[j]]) : h[order[j]]) / Sum(k);
  value = std::max(Sum(0), value);
  if (value > 0)
    for (std::size_t j = 0; j < k; ++j)
      if (!beta || h[order[j]] > 0) a[order[j]] = T(1) / T(k);
  return double(value);
}

// The top-k hinge losses smoothed by gamma > 0; a receives the gradient.
template <typename T>
double smooth_topk_hinge_loss(bool beta, std::size_t m, const T* h, std::size_t k, double gamma,
                              T* a) {
  const T g = T(gamma);
  entropy_buffer<T> aux(m);
  std::copy(h, h + m, a);
  if (beta)
    prox_knapsack_le(a, a + m, aux.data(), T(0), g / T(k), g);
  else
    prox_topk_simplex(a, a + m, aux.data(), static_cast<std::ptrdiff_t>(k), g);
  T value(0);
  for (std::size_t j = 0; j < m; ++j) {
    value += smooth_hinge_term(h[j], a[j], g);
    a[j] /= g;
  }
  return double(value);
}

// The nonsmooth top-1 hinge max(0, max 1 + z_j - z_label) over j != label: topk_hinge_loss
// at k = 1 over the others in index order, with the target entry single_target_loss gives.
// A margin is within eps (|z_j| + |z_label| + 1) of 1 + z_j - z_label, so the competitors
// more than 4 eps (|top| + |z_label| + 1) below the largest score top have smaller margins
// than its: the first largest margin is that of the others (all of them where the bound is
// not finite). competitor is max_except(n, z, label); gradient, if non-null, receives the
// subgradient.
template <typename Data, typename T>
double hinge_label(std::size_t n, std::size_t label, const Data* z, Data competitor,
                   Data* gradient) {
  const T target = T(z[label]);
  // Keep a wider score's low bits: rounding its maximum up to double could exclude it.
  using Wide = typename std::common_type<T, double>::type;
  const Wide largest = Wide(competitor);
  const Wide eps = Wide(std::numeric_limits<T>::epsilon());
  const Wide floor = largest - 4 * eps * (std::abs(largest) + std::abs(Wide(target)) + 1);
  // The least Data at or above the floor: z_j >= floor exactly where z_j >= limit.
  Data limit = -std::numeric_limits<Data>::infinity();
  if (std::isfinite(floor)) {
    const bool below = Wide(-floor) > Wide(std::numeric_limits<Data>::max());
    limit = below ? std::numeric_limits<Data>::lowest() : Data(floor);
    if (Wide(limit) < floor) limit = std::nextafter(limit, std::numeric_limits<Data>::infinity());
  }
  std::size_t best = label == 0 ? 1 : 0;
  T top = margin(T(z[best]), target);
  for (std::size_t j = 0; j < n;) {
    j = next_reaching(j, n, z, limit);
    for (const std::size_t end = std::min(n, j + row_lanes); j < end; ++j) {
      if (j == label || !(z[j] >= limit)) continue;
      const T h = margin(T(z[j]), target);
      if (h > top) {
        top = h;
        best = j;
      }
    }
  }
  const T value = std::max(T(0), top);
  if (gradient) {
    std::fill(gradient, gradient + n, Data(0));
    if (value > 0) gradient[best] = Data(1);
    using Sum = typename std::common_type<T, double>::type;
    gradient[label] = Data(-Sum(value > 0 ? 1 : 0));
  }
  return double(value);
}

// A single-target loss: the margins h_j = 1 + z_j - z_0 (entropy: z_j - z_0)
// of the other classes, their gradient a, and the target's from zero sum (top-k
// entropy: minus the competitors' mass).
template <typename Data, typename T>
double single_target_loss(loss_kind kind, std::size_t n, const Data* z, Data* gradient,
                          std::size_t k, double gamma) {
  if (k == 1 && gamma == 0 && kind != loss_kind::topk_cross_entropy) {
    Data lowest;
    return hinge_label<Data, T>(n, 0, z, max_except(n, z, 0, lowest), gradient);
  }
  const std::size_t m = n - 1;
  entropy_buffer<T> h_buffer(m), a_buffer(m);
  T *h = h_buffer.data(), *a = a_buffer.data();
  std::fill(a, a + m, T(0));
  double value(0);
  if (kind == loss_kind::topk_cross_entropy) {
    T target(0);
    value = topk_entropy_loss<Data, T>(n, z, k, a, &target);
    for (std::size_t j = 1; j < n; ++j) gradient[j] = Data(a[j - 1]);
    gradient[0] = Data(target);
    return value;
  }
  // Compensated margins: a tiny 1 + z_j - z_0 survives a large z_j - z_0.
  for (std::size_t j = 1; j < n; ++j) h[j - 1] = margin(T(z[j]), T(z[0]));
  const bool beta = kind == loss_kind::topk_hinge_beta;
  value = gamma > 0 ? smooth_topk_hinge_loss(beta, m, h, k, gamma, a)
                    : topk_hinge_loss(beta, m, h, k, a);
  // The target entry: minus the others' unrounded sum (their float entries can be subnormal
  // or zero where the sum is not), rounded once.
  using Sum = typename std::common_type<T, double>::type;
  compensated_sum<Sum> sum;
  for (std::size_t j = 1; j < n; ++j) {
    gradient[j] = Data(a[j - 1]);
    sum.add(Sum(a[j - 1]));
  }
  gradient[0] = Data(-sum.value());
  return value;
}

// Whether T, if narrower than double, holds gamma (if positive), alpha, their
// reciprocals and the spread of the scores.
template <typename T, typename Data>
bool loss_fits(std::size_t n, const Data* z, double gamma, double alpha) {
  if (std::numeric_limits<T>::digits >= std::numeric_limits<double>::digits) return true;
  const double tiny = double(std::numeric_limits<T>::min());
  // The extrema skip NaN scores; a NaN z[0] does not fit.
  Data lo;
  const Data hi = max_except(n, z, n, lo);
  return (gamma == 0 || (tiny <= gamma && gamma <= 1 / tiny)) && tiny <= alpha &&
         alpha <= 1 / tiny && !std::isnan(z[0]) && std::isfinite(T(hi) - T(lo));
}

// Whether loss_and_gradient accepts these sizes and parameters.
inline bool valid_loss_arguments(std::size_t n, std::size_t p, std::size_t k, double gamma,
                                 double alpha) {
  return n >= 2 && p >= 1 && p <= n && k >= 1 && k < n && gamma >= 0 && std::isfinite(gamma) &&
         alpha > 0 && std::isfinite(alpha);
}
}  // namespace detail

// cross_entropy of the scores z with the single label z[label], as
// loss_and_gradient with that label first. gradient may be null (the loss alone).
template <typename Data, typename Result = double>
double entropy_label_loss(std::size_t n, std::size_t label, const Data* z, Data* gradient) {
  if (n < 2 || label >= n) throw std::invalid_argument("Invalid loss dimensions or parameters");
  using T = typename std::common_type<Data, Result>::type;
  if (!detail::loss_fits<T>(n, z, 0, 1))
    return entropy_label_loss<Data, double>(n, label, z, gradient);
  return double(single_entropy_loss_gradient<T>(n, label, z, gradient));
}

namespace detail {
// Whether a softmax share whose log over 1 / k is threshold exceeds 1 / k (beyond rounding).
template <typename T>
bool topk_entropy_capped(T threshold, T eps) {
  return threshold + eps * std::max(std::abs(threshold), T(1)) < 0;
}

// The k largest competitors of z[label] into top, in descending order: by insertion for small
// k, else selected (n + k log k rather than n k comparisons).
template <typename T, typename Data>
void topk_entropy_top(std::size_t n, std::size_t label, const Data* z, std::size_t k, T* top) {
  if (k > 16) {
    entropy_buffer<T> buffer(n - 1);
    T* v = buffer.data();
    for (std::size_t j = 0, i = 0; j < n; ++j)
      if (j != label) v[i++] = T(z[j]);
    std::nth_element(v, v + (k - 1), v + (n - 1), std::greater<T>());
    std::sort(v, v + k, std::greater<T>());
    std::copy(v, v + k, top);
    return;
  }
  std::fill(top, top + k, -std::numeric_limits<T>::infinity());
  for (std::size_t j = 0; j < n; ++j) {
    const T x = T(z[j]);
    if (j == label || !(x > top[k - 1])) continue;
    std::size_t p = k - 1;
    for (; p > 0 && x > top[p - 1]; --p) top[p] = top[p - 1];
    top[p] = x;
  }
}

// sum exp(z_j - last) over the competitors out of the k largest top (last = top[k - 1]): from
// the terms where they keep every entry that counts (normal down to far below last), else
// from the scores.
template <typename T, typename Data>
T topk_entropy_tail(std::size_t n, std::size_t label, const Data* z, std::size_t k, const T* top,
                    T maximum, const T* terms) {
  const T last = top[k - 1];
  const bool from_terms = maximum - last < -std::log(std::numeric_limits<T>::min()) / 2;
  compensated_sum<T> tail;
  std::size_t equal = 0;  // the competitors at last out of top
  for (std::size_t j = 0; j < n; ++j) {
    const T x = T(z[j]);
    if (j == label || x > last) continue;
    if (x == last)
      ++equal;
    else
      tail.add(from_terms ? terms[j] : std::exp(x - last));
  }
  for (std::size_t i = 0; i < k; ++i) equal -= top[i] == last;
  const T sum = from_terms ? tail.value() * std::exp(maximum - last) : tail.value();
  return sum + T(equal);
}

// The m largest competitors capped at 1 / k while the next one's share of the rest exceeds it:
// active = m, head = sum of their top[i] - maximum, free = top[m], sum the rest's sum
// exp(z_j - free) and lse its log.
template <typename T>
struct topk_entropy_caps {
  std::size_t active = 1;
  T head = 0, free = 0, lse = 0, sum = 1;
};
// The first m in [1, k - 1) whose competitor top[m] is not capped, else k - 1. Being capped is
// monotone in m, so probing m = 1, 2, 4, ... and bisecting the bracket takes log m solves of k
// terms each instead of up to k (one where only the largest is capped, as in most rows).
template <typename T>
topk_entropy_caps<T> topk_entropy_cap(const T* top, std::size_t k, T tail, T maximum, T eps) {
  const T last = top[k - 1];
  auto solve = [&](std::size_t active) {
    topk_entropy_caps<T> caps;
    caps.active = active;
    caps.free = top[active];
    compensated_sum<T> mid;
    for (std::size_t i = active + 1; i < k; ++i) mid.add(std::exp(top[i] - caps.free));
    const T spread = mid.value() + std::exp(last - caps.free) * tail;
    caps.lse = std::log1p(spread);
    caps.sum = 1 + spread;
    return caps;
  };
  // Every m < lo is capped; hi is not (found holds its solve), or hi = k - 1.
  std::size_t lo = 1, hi = k - 1;
  topk_entropy_caps<T> found;
  bool have = false;
  auto capped = [&](std::size_t m) {
    const topk_entropy_caps<T> caps = solve(m);
    if (topk_entropy_capped(caps.lse - std::log(T(k - m)), eps)) return true;
    found = caps;
    have = true;
    return false;
  };
  for (std::size_t probe = 1; probe < hi && !have; probe *= 2) {
    if (capped(probe)) {
      lo = probe + 1;
    } else {
      hi = probe;
    }
  }
  while (lo < hi) {
    const std::size_t m = lo + (hi - lo) / 2;
    if (capped(m)) {
      lo = m + 1;
    } else {
      hi = m;
    }
  }
  topk_entropy_caps<T> caps = have ? found : solve(lo);
  compensated_sum<T> head;
  for (std::size_t i = 1; i < lo; ++i) head.add(top[i] - maximum);
  caps.head = head.value();
  return caps;
}

// The competitors' gradient cap * min(1, exp(z_j - free) / spread), spread the free ones' sum
// of exp(z_j - free) per uncapped slot (a log-domain shift would round at the magnitude of
// log(mass)), the label's -mass from the row statistics; from the terms (times factor =
// exp(maximum - free) / spread) where scaled. One product with cap where that is normal: a
// subnormal one would round at its quantum times a ratio up to spread.
template <typename T, typename Data>
void topk_entropy_gradient(std::size_t n, std::size_t label, const Data* z, T mass, T cap, T free,
                           T spread, T factor, bool scaled, const T* terms, Data* gradient) {
  const T bound = std::log(spread), ratio = scaled ? factor : T(1) / spread;
  const T folded = cap * ratio;
  const bool normal = folded >= std::numeric_limits<T>::min();
  for (std::size_t j = 0; j < n; ++j) {
    if (j == label) continue;
    const T e = scaled ? terms[j] : std::exp(std::min(T(z[j]) - free, bound));
    gradient[j] = Data(normal ? std::min(e * folded, cap) : cap * std::min(e * ratio, T(1)));
  }
  gradient[label] = Data(-mass);
}

// topk_entropy_label_loss in the arithmetic of T; terms holds n values.
template <typename T, typename Data>
double topk_entropy_label(std::size_t n, std::size_t label, const Data* z, std::size_t k,
                          Data* gradient, T* terms) {
  const entropy_row<T> row = single_entropy_row<T>(n, label, z, terms);
  const T eps = 32 * std::numeric_limits<T>::epsilon(), K = T(k);
  // The largest competitor's softmax share 1 / sum is at most 1 / k: entropy.
  if (!topk_entropy_capped(std::log(row.sum) - std::log(K), eps)) {
    single_entropy_gradient(n, label, z, row, T(1), terms, gradient);
    return double(row.loss);
  }
  entropy_buffer<T> top_buffer(k);
  T* top = top_buffer.data();
  topk_entropy_top(n, label, z, k, top);
  const T maximum = row.maximum;
  const T tail = topk_entropy_tail(n, label, z, k, top, maximum, terms);
  const topk_entropy_caps<T> caps = topk_entropy_cap(top, k, tail, maximum, eps);
  const T free = caps.free, lse = caps.lse;
  const T remaining = T(k - caps.active);
  const T rest = remaining * ((free - maximum) + lse - std::log(remaining));
  const T mean = (caps.head + rest) / K, gap = maximum - T(z[label]);
  const T odds = gap + mean + std::log(K);
  // A tail below exp(-36) is its own loss to half an eps, from squared half exponentials
  // rounded once (as single_entropy_row): the log-domain odds round at the magnitude of the gap.
  T loss, mass;
  if (odds < -36) {
    // A positive gap (a capped competitor far above the free ones) joins the mean.
    const T above = std::max(gap, T(0)), half = std::exp((gap - above) / 2);
    const T small = half * (half * (std::exp(mean + above) * K));
    loss = std::log1p(small);
    mass = small / (1 + small);
  } else {
    loss = softplus(odds);
    mass = std::exp(-softplus(-odds));
  }
  const T cap = mass / K, spread = caps.sum / remaining;
  // From the terms where they are all normal and free is near maximum.
  const bool scaled =
      maximum - free <= T(1) && row.lowest - maximum >= std::log(std::numeric_limits<T>::min());
  const T factor = std::exp(maximum - free) / spread;
  topk_entropy_gradient(n, label, z, mass, cap, free, spread, factor, scaled, terms, gradient);
  return double(loss);
}
}  // namespace detail

// topk_cross_entropy (1 < k < n - 1) of the scores z with the single label z[label] and its
// gradient, as loss_and_gradient with that label first, from one pass of exponentials:
// entropy's where no cap binds (the competitors' largest softmax share is at most 1 / k),
// else the closed form of their k largest (the m largest capped at 1 / k, the others sharing
// 1 - m / k by softmax). terms, if given, holds n values of common_type<Data, Result>.
template <typename Data, typename Result = double>
double topk_entropy_label_loss(std::size_t n, std::size_t label, const Data* z, std::size_t k,
                               Data* gradient,
                               typename std::common_type<Data, Result>::type* terms = nullptr) {
  if (n < 2 || label >= n || k < 2 || k + 1 >= n || !gradient)
    throw std::invalid_argument("Invalid loss dimensions or parameters");
  using T = typename std::common_type<Data, Result>::type;
  if (!detail::loss_fits<T>(n, z, 0, 1))
    return topk_entropy_label_loss<Data, double>(n, label, z, k, gradient);
  detail::entropy_buffer<T> buffer(terms ? 0 : n);
  return detail::topk_entropy_label<T>(n, label, z, k, gradient, terms ? terms : buffer.data());
}

// The nonsmooth top-1 hinge (multiclass_hinge; topk_hinge_alpha and topk_hinge_beta
// at k = 1) of the n scores z with the label z[label]: loss_and_gradient with that
// label first, without permuting z. gradient, if non-null, receives the subgradient.
template <typename Data, typename Result = double>
double hinge_label_loss(std::size_t n, std::size_t label, const Data* z, Data* gradient = nullptr) {
  if (n < 2 || label >= n) throw std::invalid_argument("Invalid loss dimensions or parameters");
  using T = typename std::common_type<Data, Result>::type;
  if (!detail::loss_fits<T>(n, z, 0, 1))
    return hinge_label_loss<Data, double>(n, label, z, gradient);
  Data lowest;
  return detail::hinge_label<Data, T>(n, label, z, detail::max_except(n, z, label, lowest),
                                      gradient);
}

// Unregularized loss of the scores z and its gradient (a subgradient for the
// nonsmooth hinge losses) at z, not at an SDCA block's updated scores. The
// first p of the n scores are the labels; k is the top-k parameter (ignored
// by the losses without topk in the name), gamma >= 0 the smoothing of the hinges (0: nonsmooth) and
// alpha the proximal step of proximal_cross_entropy. The arithmetic is common_type<Data, Result>; float
// arithmetic falls back to double where float cannot hold gamma, alpha or the
// score spread. mode selects how the smoothed multilabel hinge projects.
// Throws std::invalid_argument for invalid sizes or parameters.
template <typename Data, typename Result = double>
double loss_and_gradient(loss_kind kind, std::size_t n, std::size_t p, const Data* z,
                         Data* gradient, std::size_t k = 1, double gamma = 0, double alpha = 1,
                         const projection_precision mode = projection_precision::float64) {
  if (!detail::valid_loss_arguments(n, p, k, gamma, alpha))
    throw std::invalid_argument("Invalid loss dimensions or parameters");
  if (kind == loss_kind::multiclass_hinge) k = 1;  // as cross_entropy, it has no top-k
  using T = typename std::common_type<Data, Result>::type;
  if (!detail::loss_fits<T>(n, z, gamma, alpha))
    return loss_and_gradient<Data, double>(kind, n, p, z, gradient, k, gamma, alpha, mode);
  std::fill(gradient, gradient + n, Data(0));
  switch (kind) {
    case loss_kind::cross_entropy:
    case loss_kind::multilabel_cross_entropy:
      return double(detail::entropy_loss_gradient<T>(n, p, z, gradient));
    case loss_kind::proximal_cross_entropy:
      return detail::proximal_entropy_loss<Data, T>(n, p, z, gradient, alpha);
    case loss_kind::multilabel_hinge:
      return detail::multilabel_hinge_loss<Data, T>(n, p, z, gradient, gamma, gamma > 0, mode);
    case loss_kind::topk_cross_entropy:
    case loss_kind::multiclass_hinge:
    case loss_kind::topk_hinge_alpha:
    case loss_kind::topk_hinge_beta:
      if (p != 1) throw std::invalid_argument("This loss needs exactly one target");
      // At k = 1 no cap binds: top-1 entropy is entropy; k < n - 1 has a closed form.
      if (kind == loss_kind::topk_cross_entropy && k == 1)
        return double(detail::entropy_loss_gradient<T>(n, p, z, gradient));
      if (kind == loss_kind::topk_cross_entropy && k + 1 < n) {
        detail::entropy_buffer<T> terms(n);
        return detail::topk_entropy_label<T>(n, 0, z, k, gradient, terms.data());
      }
      return detail::single_target_loss<Data, T>(kind, n, z, gradient, k, gamma);
  }
  throw std::invalid_argument("Unknown loss kind");
}

// distillation_loss H(softmax(teacher / temperature), softmax(student / temperature))
// of the n student and teacher logits, and its gradient (q - p) / temperature
// with respect to student if gradient is non-null.
template <typename Data, typename Result = double>
double distillation_loss(std::size_t n, const Data* student, const Data* teacher,
                         double temperature = 1.0, Data* gradient = nullptr) {
  if (n < 2 || !student || !teacher || !(std::isfinite(temperature) && temperature > 0))
    throw std::invalid_argument("Invalid loss dimensions or parameters");
  detail::entropy_buffer<double> work(gradient ? 7 * n : 6 * n);
  double* w = work.data();
  for (std::size_t j = 0; j < n; ++j) {
    w[j] = double(student[j]);
    w[n + j] = double(teacher[j]);
  }
  const detail::distillation_row row = detail::distillation_double(
      n, temperature, w, w + n, w + 2 * n, w + 3 * n, w + 4 * n, w + 5 * n);
  if (!row.finite) throw std::invalid_argument("Logits must be finite");
  if (gradient) {
    double* g = w + 6 * n;
    detail::distillation_double_gradient(n, row, w + n, w + 5 * n, temperature, 1.0, g);
    for (std::size_t j = 0; j < n; ++j) gradient[j] = Data(g[j]);
  }
  return row.loss;
}

template <typename Data, typename Result = double>
double distillation_loss_and_gradient(std::size_t n, const Data* student, const Data* teacher,
                                      Data* gradient, double temperature = 1.0) {
  if (!gradient) throw std::invalid_argument("Invalid loss dimensions or parameters");
  return distillation_loss<Data, Result>(n, student, teacher, temperature, gradient);
}

// proximal_distillation_loss min_u H(softmax(teacher / temperature), softmax(u))
// + |u - student / temperature|^2 / (2 alpha), and its gradient (pi - p) / temperature
// with respect to student if gradient is non-null.
template <typename Data, typename Result = double>
double proximal_distillation_loss(std::size_t n, const Data* student, const Data* teacher,
                                  double temperature = 1.0, double alpha = 1.0,
                                  Data* gradient = nullptr) {
  if (n < 2 || !student || !teacher || !(std::isfinite(temperature) && temperature > 0) ||
      !(std::isfinite(alpha) && alpha > 0))
    throw std::invalid_argument("Invalid loss dimensions or parameters");
  detail::entropy_buffer<double> work(13 * n + 5 * detail::row_lanes);
  detail::entropy_buffer<std::size_t> index(2 * n);
  double* w = work.data();
  for (std::size_t j = 0; j < n; ++j) {
    w[j] = double(student[j]);
    w[n + j] = double(teacher[j]);
  }
  const detail::distillation_row row = detail::distillation_double(
      n, temperature, w, w + n, w + 2 * n, w + 3 * n, w + 4 * n, w + 5 * n);
  if (!row.finite) throw std::invalid_argument("Logits must be finite");
  double* r = w + 6 * n;
  const double loss =
      detail::proximal_distillation_row(n, alpha, row, w, w + n, w + 2 * n, w + 3 * n, w + 4 * n,
                                        w + 5 * n, r, w + 7 * n, index.data());
  if (gradient) {
    for (std::size_t j = 0; j < n; ++j) gradient[j] = Data(-r[j] / temperature);
  }
  return loss;
}

template <typename Data, typename Result = double>
double proximal_distillation_loss_and_gradient(std::size_t n, const Data* student,
                                               const Data* teacher, Data* gradient,
                                               double temperature = 1.0, double alpha = 1.0) {
  if (!gradient) throw std::invalid_argument("Invalid loss dimensions or parameters");
  return proximal_distillation_loss<Data, Result>(n, student, teacher, temperature, alpha,
                                                  gradient);
}
}  // namespace sdca
#endif
