#ifndef SDCA_SOLVER_OBJECTIVE_ENTROPY_BLOCK_H
#define SDCA_SOLVER_OBJECTIVE_ENTROPY_BLOCK_H

// The SDCA step of an entropy block with a target uniform on p labels (entropy_update): the
// root solve starts at the threshold of the current block, its first evaluation takes the
// current probabilities as guesses of the Lambert W, and evaluations near the last exact one
// expand it to fourth order. On rows of few features these per-coordinate costs dominate an
// epoch.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>

#include "sdca/core/lambert.h"
#include "sdca/core/reduce.h"

SDCA_ISA_BEGIN
namespace solver_detail {
namespace label_block {

// W(exp(x)) by one Householder step from a guess within about 1e-4 (relative); lambert_w_exp
// outside the arguments where that step is its last.
template <typename T>
inline T lambert_w_exp_near(const T x, const T guess) {
  if (!(x > T(-20) && x <= T(4) && guess > 0)) return core::lambert_w_exp(x);
  return core::lambert_w_iter_5(guess, std::exp(x - guess));
}

// Whether exp(h - W(exp(arg))) is exp(h) without W: W(exp(arg)) < exp(arg) is below a quarter
// ulp of |h| >= 1/2, so h - W rounds to h.
template <typename T>
inline bool lambert_negligible(const T arg, const T h) {
  const T cut = -T(std::numeric_limits<T>::digits + 2) * T(0.69314718055994530942L);
  return arg < cut && std::abs(h) >= T(0.5);
}

// The root r in [q - 1, q) of alpha r - log1p(-r / q) = t. Away from r = 0 (|r| >= q / 2) by
// the Lambert inversion of q - r; near it by Halley steps on log1p(-r / q) from a Pade
// estimate, or from a finite guess of r.
template <typename T>
inline T residual(const T t, const T alpha, const T q, const T log_alpha, const T log_q,
                  const T guess = std::numeric_limits<T>::quiet_NaN()) {
  if (t == 0) return 0;
  T lo = q - T(1), hi = q;
  if (t <= alpha * lo + log_q) return lo;
  const T half = alpha * q / T(2);
  const bool warm = std::isfinite(guess);
  if (t > half + T(0.69314718055994530942) || t < -half - T(0.40546510810816438198)) {
    const T h = log_q + (alpha * q - t), arg = log_alpha + h;
    const T w = warm ? lambert_w_exp_near(arg, alpha * (q - guess)) : core::lambert_w_exp(arg);
    const T prob = (arg < T(-18)) ? std::exp(h - w) : w / alpha;
    return std::max(lo, std::min(q, q - prob));
  }
  const T inv_q = T(1) / q;
  T initial = guess;
  if (!warm) {
    const T linear = t / (alpha + inv_q);
    initial = linear / (T(1) + (linear * inv_q) / (T(2) * (T(1) + alpha * q)));
  }
  T r = std::max(lo, std::min(initial, std::nextafter(q, T(0))));
  const T tol = T(8) * std::numeric_limits<T>::epsilon() * std::abs(t);
  for (int it = 0; it < 100; ++it) {
    const T f = alpha * r - std::log1p(-r * inv_q) - t;
    if (std::abs(f) <= tol) break;
    if (f < 0)
      lo = r;
    else
      hi = r;
    const T inv_qr = T(1) / (q - r), d1 = alpha + inv_qr;
    const T denom = d1 - (f * inv_qr * inv_qr) / (T(2) * d1);
    T next = (denom > 0) ? (r - f / denom) : (r - f / d1);
    // A step below the resolution of r ends it: a subnormal r misses tol.
    if (next == r) break;
    if (!(lo < next && next < hi)) next = lo + (hi - lo) / 2;
    if (next == q) return q;
    if (next == r) break;
    r = next;
  }
  return r;
}

// The residuals r0 of an exact evaluation at the threshold base, for the evaluations after
// it. Its probabilities p = W(exp(x)) / alpha expand in a step d of the threshold: with
// u = 1 / (1 + alpha p0), p(base + d) = p0 (1 + d (k1 + d (k2 + d (k3 + d k4)))), the Taylor
// series of W to fourth order. p0 < 0 marks a clipped class. Before the first exact
// evaluation, the residuals of the current block (prior, if any) give guesses instead.
template <typename T>
struct expansion {
  static const std::size_t arrays = 7;
  T local[arrays * 64];
  std::unique_ptr<T[]> heap;
  T *r0, *p0, *k1, *k2, *k3, *k4, *prior = nullptr;
  T base = 0;
  bool accurate = false, expanded = false;
  explicit expansion(const std::size_t n) : heap(n > 64 ? new T[arrays * n] : nullptr) {
    T* data = heap ? heap.get() : local;
    T** parts[] = {&r0, &p0, &k1, &k2, &k3, &k4};
    for (std::size_t i = 0; i < 6; ++i) *parts[i] = data + i * n;
  }
  expansion(const expansion&) = delete;
  expansion& operator=(const expansion&) = delete;
  T* prior_storage(const std::size_t n) { return (heap ? heap.get() : local) + 6 * n; }
};

// The block in residual coordinates r_j = q_j - pi_j for the scores v of p labels (first) and
// n - p other classes, as functions of the threshold relative to the largest score,
// dt = t - v_ref.
template <typename T>
struct solver {
  // f sums in at least double: a float sum of many classes misplaces the root.
  typedef typename std::common_type<T, double>::type Sum;

  std::size_t n, p;
  T alpha, q, log_alpha, log_q, clip_threshold;
  const T* v;
  T* r;
  T v_max_label, v_max_other, v_ref;
  T derivative = 0, curvature = 0, magnitude = 0, other_mass = 0;
  expansion<T>* ex = nullptr;

  solver(const std::size_t n_, const std::size_t p_, const T alpha_, const T* v_, T* r_)
      : n(n_),
        p(p_),
        alpha(alpha_),
        q(T(1) / T(p_)),
        log_alpha(std::log(alpha_)),
        log_q(std::log(q)),
        clip_threshold(alpha_ * (q - T(1)) + log_q),
        v(v_),
        r(r_) {
    v_max_label = *std::max_element(v, v + p);
    v_max_other = *std::max_element(v + p, v + n);
    v_ref = std::max(v_max_label, v_max_other);
  }

  // W(exp(log alpha + h)) / alpha without an underflowing W; a finite guess of W goes to
  // lambert_w_exp_near.
  T probability(const T h, const T guess = std::numeric_limits<T>::quiet_NaN()) const {
    const T arg = h + log_alpha;
    if (lambert_negligible(arg, h)) return std::exp(h);
    const T w = std::isfinite(guess) ? lambert_w_exp_near(arg, guess) : core::lambert_w_exp(arg);
    return (arg < T(-18)) ? std::exp(h - w) : w / alpha;
  }

  void add_slope(const T prob) {
    const T denom = T(1) + alpha * prob, d1 = prob / denom;
    derivative += d1;
    curvature -= d1 / (denom * denom);
  }
  void add_label(const std::size_t j, const T rj, const bool clipped, Sum& f) {
    r[j] = rj;
    f += Sum(rj);
    magnitude += std::abs(rj);
    if (!clipped) add_slope(q - rj);
  }
  void add_other(const std::size_t j, T prob, Sum& f) {
    const bool clipped = prob > T(1);
    if (clipped) prob = T(1);
    r[j] = -prob;
    f -= Sum(prob);
    other_mass += prob;
    magnitude += prob;
    if (!clipped) add_slope(prob);
  }
  void label_exact(const std::size_t j, const T dt, Sum& f,
                   const T guess = std::numeric_limits<T>::quiet_NaN()) {
    const T argument = (v_ref - v[j]) + dt;
    add_label(j, residual(argument, alpha, q, log_alpha, log_q, guess), argument < clip_threshold,
              f);
  }
  void other_exact(const std::size_t j, const T dt, Sum& f,
                   const T guess = std::numeric_limits<T>::quiet_NaN()) {
    add_other(j, probability(((v[j] - v_ref) - dt) + log_q, guess), f);
  }

  // The largest step of the expansion: its truncation, p0 d^5 / 120 times |W^(5) / W| <= 1,
  // stays below eps / 32 of p.
  static T expansion_reach() { return std::ldexp(T(1), -(std::numeric_limits<T>::digits + 4) / 5); }

  // The largest step of a guess of third order (relative error at most d^4 / 24) whose one
  // Householder step (order 5) is accurate. 0 in float, where lambert_w_exp costs little
  // more than that step.
  static T warm_reach() {
    static const T value =
        std::numeric_limits<T>::digits <= std::numeric_limits<float>::digits
            ? T(0)
            : T(std::pow(
                  24 * std::pow(static_cast<long double>(std::numeric_limits<T>::epsilon()) / 64,
                                0.2L),
                  0.25L));
    return value;
  }

  // W(exp(x + s)) / W(exp(x)) - 1 to third order, at u = 1 / (1 + W(exp(x))).
  static T guess_change(const T u, const T s) {
    const T u3 = u * u * u;
    return s * (u + s * (u3 / T(2) + s * (u3 * u * (T(3) * u - T(2)) / T(6))));
  }

  // The base's probability of class j, or -1 where it is clipped.
  T base_probability(const std::size_t j) const {
    const T r0 = ex->r0[j], prob = j < p ? q - r0 : -r0;
    const bool clipped =
        j < p ? (v_ref - v[j]) + ex->base < clip_threshold || r0 == 0 : !(prob < T(1));
    return clipped ? T(-1) : prob;
  }

  void expand_base() {
    for (std::size_t j = 0; j < n; ++j) {
      const T prob = base_probability(j);
      ex->p0[j] = prob;
      if (prob < 0) continue;
      const T u = T(1) / (T(1) + alpha * prob), u2 = u * u, u4 = u2 * u2;
      ex->k1[j] = -u;
      ex->k2[j] = u2 * u / T(2);
      ex->k3[j] = -u4 * (T(3) * u - T(2)) / T(6);
      ex->k4[j] = u4 * u * ((T(15) * u - T(20)) * u + T(6)) / T(24);
    }
    ex->expanded = true;
  }

  // f at base + d from the expansion; a label whose residual would lose its relative accuracy
  // (cancellation, or truncation beside a small residual), and a clipped class, exactly.
  T expand(const T dt) {
    if (!ex->expanded) expand_base();
    const T d = dt - ex->base, d5 = std::abs(d * d * d * d * d),
            eps = std::numeric_limits<T>::epsilon();
    Sum f(0);
    for (std::size_t j = 0; j < n; ++j) {
      const T p0 = ex->p0[j];
      if (!(p0 >= 0)) {
        if (j < p)
          label_exact(j, dt, f);
        else
          other_exact(j, dt, f);
        continue;
      }
      const T s = d * (ex->k1[j] + d * (ex->k2[j] + d * (ex->k3[j] + d * ex->k4[j])));
      const T change = p0 * s;
      if (j >= p) {
        add_other(j, p0 + change, f);
        continue;
      }
      const T r0 = ex->r0[j], rj = r0 - change;
      if (std::abs(change) <= std::abs(r0) / T(2) && p0 * d5 <= eps * std::abs(r0) && rj > q - T(1))
        add_label(j, rj, false, f);
      else
        label_exact(j, dt, f);
    }
    return T(f);
  }

  // The classes at dt from the guesses of the prior residuals: each exact at its own argument
  // (t = alpha r - log1p(-r / q) of a label, x = log(w) + w at w = alpha p of another class).
  void evaluate_from_prior(const T dt, Sum& f) {
    const T* prior = ex->prior;
    for (std::size_t j = 0; j < n; ++j) {
      const T r0 = prior[j], p0 = j < p ? q - r0 : -r0;
      T guess = std::numeric_limits<T>::quiet_NaN();
      if (p0 > 0 && p0 < T(1)) {
        const T u = T(1) / (T(1) + alpha * p0);
        if (j < p) {
          const T shift = ((v_ref - v[j]) + dt) - (alpha * r0 - std::log1p(-r0 / q));
          if (r0 != 0 && std::abs(shift) <= warm_reach()) guess = r0 - p0 * guess_change(u, -shift);
          label_exact(j, dt, f, guess);
        } else {
          const T w0 = alpha * p0, arg = (((v[j] - v_ref) - dt) + log_q) + log_alpha;
          // Outside (-20, 4] lambert_w_exp_near takes no guess.
          const T shift = arg > T(-20) && arg <= T(4) ? arg - (std::log(w0) + w0)
                                                      : std::numeric_limits<T>::infinity();
          if (std::abs(shift) <= warm_reach()) guess = w0 + w0 * guess_change(u, shift);
          other_exact(j, dt, f, guess);
        }
      } else if (j < p) {
        label_exact(j, dt, f);
      } else {
        other_exact(j, dt, f);
      }
    }
  }

  // The residual sum f(dt), concave increasing; fills r and the derivatives. Within
  // expansion_reach() of the last exact evaluation it expands it, within warm_reach() takes
  // its probabilities as guesses.
  T evaluate(const T dt) {
    derivative = curvature = magnitude = other_mass = 0;
    const T distance = ex->accurate ? std::abs(dt - ex->base) : std::numeric_limits<T>::infinity();
    if (distance <= expansion_reach()) return expand(dt);
    Sum f(0);
    if (!ex->accurate && ex->prior) {
      evaluate_from_prior(dt, f);
    } else if (distance <= warm_reach()) {
      const T d = dt - ex->base;
      for (std::size_t j = 0; j < n; ++j) {
        const T p0 = base_probability(j);
        T guess = std::numeric_limits<T>::quiet_NaN();
        if (p0 >= 0) {
          const T s = guess_change(T(1) / (T(1) + alpha * p0), -d);
          guess = j < p ? ex->r0[j] - p0 * s : alpha * p0 + alpha * p0 * s;
        }
        if (j < p)
          label_exact(j, dt, f, guess);
        else
          other_exact(j, dt, f, guess);
      }
    } else {
      for (std::size_t j = 0; j < p; ++j) label_exact(j, dt, f);
      for (std::size_t j = p; j < n; ++j) other_exact(j, dt, f);
    }
    std::copy(r, r + n, ex->r0);
    ex->base = dt;
    ex->accurate = true;
    ex->expanded = false;
    return T(f);
  }

  // Halley step, or a step in the log of the other classes' mass where that mass exceeds one
  // and the Halley step is shorter.
  T step(const T f) const {
    if (!(derivative > 0) || !std::isfinite(derivative)) return T(0);
    const T newton = -f / derivative;
    if (!std::isfinite(newton)) return T(0);
    T correction = (f * curvature) / (T(2) * derivative * derivative);
    T result = newton;
    if (std::isfinite(correction)) {
      correction = std::max(T(-0.5), std::min(T(0.5), correction));
      result = newton / (T(1) - correction);
    }
    if (f < 0 && other_mass > T(1)) {
      const T log_step = std::log(other_mass) + alpha * (other_mass - T(1)) / T(n - p);
      if (std::isfinite(log_step) && log_step > result) result = log_step;
    }
    return result;
  }

  // At lo the largest probability is one, so f(lo) <= 0 and no class is clipped above it; at
  // hi every pi_j <= 1/n, so f(hi) >= 0.
  void bracket(T& lo, T& hi) const {
    lo =
        std::max((v_max_label - v_ref) - alpha * (T(1) - q), (v_max_other - v_ref) - alpha) + log_q;
    hi =
        std::max((v_max_label - v_ref) + alpha * q, (v_max_other - v_ref)) + log_q + std::log(T(n));
    if (!(lo <= hi)) hi = lo;
  }

  // The softmax-limit start in the centered coordinate; confident hard targets skip the exp
  // pass where its bound on the other classes' mass is below sqrt(eps).
  T start() const {
    const bool tiny_tail =
        p == 1 && v_max_label == v_ref &&
        (v_max_other - v_ref) + std::log(T(n)) < std::log(std::numeric_limits<T>::epsilon()) / 2;
    if (tiny_tail) return log_q;
    T tail(0);
    bool maximum_seen = false;
    for (std::size_t j = 0; j < n; ++j) {
      if (!maximum_seen && v[j] == v_ref)
        maximum_seen = true;
      else
        tail += std::exp(v[j] - v_ref);
    }
    return log_q + std::log1p(tail);
  }

  // The threshold dt of the solution; leaves its residuals in r. A finite guess replaces the
  // softmax-limit start (the bracket still applies).
  T solve(const T guess) {
    T lo, hi;
    bracket(lo, hi);
    T dt = std::max(lo, std::min(hi, std::isfinite(guess) ? guess : start()));
    const T tolerance = T(8) * std::numeric_limits<T>::epsilon(),
            tiny = std::numeric_limits<T>::min();
    T f(0);
    for (int it = 0; it < 200; ++it) {
      f = evaluate(dt);
      if (std::abs(f) <= tolerance * std::max(magnitude, tiny)) break;
      if (f < 0)
        lo = dt;
      else
        hi = dt;
      const T delta = step(f);
      T next = dt + delta;
      // A nonzero step below the resolution of dt: dt is the closest root.
      if (delta != 0 && next == dt) {
        lo = hi = dt;
        break;
      }
      if (!(lo < next && next < hi) || !std::isfinite(next)) next = lo + (hi - lo) / 2;
      if (next == dt) break;
      dt = next;
    }
    const T eps = std::numeric_limits<T>::epsilon();
    if (!std::isfinite(f) ||
        (std::abs(f) > T(256) * eps * std::max(magnitude, tiny) &&
         hi - lo > T(64) * eps * std::max({std::abs(lo), std::abs(hi), tiny}))) {
      throw std::runtime_error("Entropy normalization did not converge");
    }
    return dt;
  }

  // The dual block c r, zero-sum up to one rounding in Data: the entry of largest magnitude
  // (the label for p = 1) is formed from the stored others. Where a probability underflows
  // but c pi does not (c > 1), c pi comes from its logarithm.
  template <typename Data>
  void duals(const T c, const T dt, Data* out) const {
    for (std::size_t j = 0; j < n; ++j) {
      out[j] = Data(c * r[j]);
      // A subnormal (or underflowed) probability can still have a normal c p.
      if (j >= p && c > T(1) && -r[j] < std::numeric_limits<T>::min()) {
        const T score = (v[j] - v_ref) + log_q, h = score - dt;
        out[j] = -Data(std::exp(((std::log(c) - dt) + score) - core::lambert_w_exp(h + log_alpha)));
      }
    }
    std::size_t pivot = 0;
    if (p > 1) {
      bool labels_zero = true;
      for (std::size_t j = 0; j < p; ++j) labels_zero = labels_zero && out[j] == 0;
      Sum mass(0);
      if (labels_zero)
        for (std::size_t j = p; j < n; ++j) mass += Sum(out[j]);
      if (mass != 0) {
        // Only recovered mass of the other classes is left: the labels share it.
        for (std::size_t j = 1; j < p; ++j) out[j] = Data(-mass / Sum(p));
      } else {
        for (std::size_t j = 1; j < n; ++j)
          if (std::abs(out[j]) > std::abs(out[pivot])) pivot = j;
      }
    }
    // The others compensated (as softmax_prox::duals): a plain sum of many classes would
    // leave the block off zero by n rounding errors.
    core::row_sum<Data, Sum> others;
    for (std::size_t j = 0; j < n; ++j)
      if (j != pivot) others.add(Sum(out[j]));
    out[pivot] = Data(-others.value());
  }
};

// The threshold dt (relative to the largest of v) at which the probabilities pi = q - a / c of
// the current block solve the block equation of its most probable class:
// dt = (v_j - v_ref) + norm2 a_j - log pi_j. NaN for a zero block or a nonpositive probability.
template <typename T, typename Data>
inline T warm_threshold(const std::size_t n, const std::size_t p, const T c, const T norm2,
                        const Data* a, const T* v) {
  const T q = T(1) / T(p);
  std::size_t best = 0;
  T best_prob = -std::numeric_limits<T>::infinity(), v_ref = v[0];
  bool zero = true;
  for (std::size_t j = 0; j < n; ++j) {
    zero = zero && a[j] == 0;
    v_ref = std::max(v_ref, v[j]);
    const T prob = (j < p ? q : T(0)) - T(a[j]) / c;
    if (prob > best_prob) {
      best_prob = prob;
      best = j;
    }
  }
  if (zero || !(best_prob > 0) || !std::isfinite(best_prob))
    return std::numeric_limits<T>::quiet_NaN();
  return ((v[best] - v_ref) + norm2 * T(a[best])) - std::log(best_prob);
}

// The step of a block of n classes whose target is uniform on the first p (1 <= p < n): a, the
// current duals, receives the new ones; v holds the warm log-odds (finite) and alpha = c norm2.
// Throws std::runtime_error, before writing a, if the normalization does not converge.
template <typename T, typename Data>
inline void entropy_update(const std::size_t n, const std::size_t p, const T c, const T alpha,
                           const Data norm2, Data* a, const T* v) {
  const T guess = warm_threshold(n, p, c, T(norm2), a, v);
  core::row_buffer<T> residuals(n);
  expansion<T> ex(n);
  if (std::isfinite(guess) && solver<T>::warm_reach() > 0) {
    ex.prior = ex.prior_storage(n);
    for (std::size_t j = 0; j < n; ++j) ex.prior[j] = T(a[j]) / c;
  }
  solver<T> s(n, p, alpha, v, residuals.data());
  s.ex = &ex;
  const T dt = s.solve(guess);
  s.duals(c, dt, a);
}

}  // namespace label_block
}  // namespace solver_detail
SDCA_ISA_END

#endif
