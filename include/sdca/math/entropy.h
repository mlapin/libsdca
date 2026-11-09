#ifndef SDCA_MATH_ENTROPY_H
#define SDCA_MATH_ENTROPY_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "sdca/math/compensated.h"
#include "sdca/math/exp_rows.h"
#include "sdca/math/lambert.h"
#include "sdca/math/log_exp.h"
#include "sdca/math/prox_rows.h"
#include "sdca/utility/scratch.h"

namespace sdca {

namespace detail {
// Average gaps to a finite maximum. Divide an overflowing difference in
// parts: its contribution to the mean can still be representable.
template <typename Work, typename Data, typename Int>
Work entropy_average_gap(Int p, const Data* z, Work maximum) {
  Work gap(0);
  for (Int j = 0; j < p; ++j) {
    const Work value = Work(z[j]), difference = maximum - value;
    gap += std::isfinite(difference) ? difference / Work(p) : maximum / Work(p) - value / Work(p);
  }
  return gap;
}

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
}  // namespace detail

// The statistics of a row with one label: the competitors' maximum and
// smallest score, the sum of their exp(z_j - maximum), their probability
// mass and the loss.
template <typename Work>
struct entropy_row {
  Work maximum, lowest, sum, mass, loss;
};

// One label, z[label]. Normalize the competitors on their own scale so their
// total mass survives even when every stored competitor gradient underflows.
// terms, if given, receives exp(z_j - maximum) (0 at the label).
template <typename Work, typename Data, typename Int>
entropy_row<Work> single_entropy_row(Int n, Int label, const Data* z, Work* terms) {
  const std::size_t size = static_cast<std::size_t>(n), skip = static_cast<std::size_t>(label);
  entropy_row<Work> row;
  Data lowest = z[0];
  row.maximum = Work(detail::max_except(size, z, skip, lowest));
  row.lowest = Work(lowest);
  row.sum = detail::exp_sum(size, z, skip, row.maximum, terms, row.lowest - row.maximum >= -708);
  const Work gap = row.maximum - Work(z[label]), relative = gap + std::log(row.sum);
  if (relative < -36) {
    // Multiply the count before the last multiply rounds into subnormals.
    const Work half = std::exp(gap / 2), tail = half * (half * row.sum);
    row.loss = std::log1p(tail);
    row.mass = tail / (1 + tail);
  } else {
    row.loss = sdca::softplus(relative);
    row.mass = std::exp(-sdca::softplus(-relative));
  }
  return row;
}

// The gradient of a row times scale, rounded once into Data, from its
// statistics and terms (recomputed if null).
template <typename Work, typename Data, typename Int>
void single_entropy_gradient(Int n, Int label, const Data* z, const entropy_row<Work>& row,
                             Work scale, const Work* terms, Data* gradient) {
  const std::size_t size = static_cast<std::size_t>(n);
  // One factor where it is normal; else divide first, to round once into subnormals.
  const Work ratio = row.mass / row.sum;
  if (ratio >= std::numeric_limits<Work>::min()) {
    if (terms) {
      for (std::size_t j = 0; j < size; ++j) gradient[j] = Data((terms[j] * ratio) * scale);
    } else {
      detail::exp_scale(size, z, row.maximum, ratio, scale, gradient,
                        row.lowest - row.maximum >= -708);
    }
  } else {
    for (std::size_t j = 0; j < size; ++j) {
      const Work e = terms ? terms[j] : std::exp(Work(z[j]) - row.maximum);
      gradient[j] = Data((row.mass * (e / row.sum)) * scale);
    }
  }
  gradient[label] = Data(-row.mass * scale);
}

// Whether an unscaled derivative of the row is below the smallest normal Work.
template <typename Work>
bool entropy_row_tiny(const entropy_row<Work>& row) {
  const Work tiny = std::numeric_limits<Work>::min();
  return row.mass < tiny || std::exp(row.lowest - row.maximum) * (row.mass / row.sum) < tiny;
}

// The loss of one label, z[label], and its gradient if a buffer is given.
template <typename Result, typename Data, typename Int>
Result single_entropy_loss_gradient(Int n, Int label, const Data* z, Data* gradient) {
  using Work = typename std::common_type<Result, Data, double>::type;
  const std::size_t size = static_cast<std::size_t>(n);
  // The terms go to the gradient when it holds them exactly.
  const bool in_place = std::is_same<Work, Data>::value;
  detail::entropy_buffer<Work> buffer(gradient && !in_place ? size : 0);
  Work* e = !gradient  ? nullptr
            : in_place ? detail::same_type_buffer<Work, Data>::of(gradient)
                       : buffer.data();
  const entropy_row<Work> row = single_entropy_row<Work>(n, label, z, e);
  if (gradient) single_entropy_gradient(n, label, z, row, Work(1), e, gradient);
  return Result(row.loss);
}

// One label, z[0].
template <typename Result, typename Data, typename Int>
Result single_entropy_loss_gradient(Int n, const Data* z, Data* gradient) {
  return single_entropy_loss_gradient<Result>(n, Int(0), z, gradient);
}

// Uniform distribution on the first p labels. Keep the small log correction
// separate from the (possibly large) common offset in the scores.
template <typename Result, typename Data, typename Int>
Result entropy_loss(Int n, Int p, const Data* z) {
  if (p < 1 || p > n) throw std::invalid_argument("Invalid positive label count");
  const Data* im = std::max_element(z, z + n);
  Result m = static_cast<Result>(*im), tail(0);
  for (Int j = 0; j < n; ++j)
    if (z + j != im) tail += std::exp(static_cast<Result>(z[j]) - m);
  return detail::entropy_average_gap(p, z, m) + std::log1p(tail);
}

namespace detail {
// entropy_loss_gradient in the arithmetic of Work.
template <typename Work, typename Data, typename Int>
Work entropy_loss_gradient(Int n, Int p, const Data* z, Data* gradient) {
  if (n < 1 || p < 1 || p > n) throw std::invalid_argument("Invalid positive label count");
  if (p == 1 && n > 1) return single_entropy_loss_gradient<Work>(n, z, gradient);
  const std::size_t size = static_cast<std::size_t>(n);
  const std::size_t top = static_cast<std::size_t>(std::max_element(z, z + n) - z);
  const Work m = static_cast<Work>(z[top]);
  const Work lowest = static_cast<Work>(*std::min_element(z, z + n));
  detail::entropy_buffer<Work> buffer(size);
  Work* e = buffer.data();
  // Compensated sums: C - 1 equal small terms beside a 1 (well-fit rows) keep their digits.
  const Work tail = detail::exp_sum(size, z, top, m, e, lowest - m >= -708);
  e[top] = 1;
  const Work den = tail + 1;
  const Work gap = entropy_average_gap(p, z, m);
  Work neg(0), carry(0), differences(0);
  // Center positive exponentials before summing: equal positive scores retain
  // their equal share of a tiny non-target mass, even when 1/p rounds.
  const Work e0 = e[0];
  for (Int j = 0; j < p; ++j) {
    Work dz = Work(z[j]) - Work(z[0]);
    e[j] = (dz >= 0 ? -e[j] * std::expm1(-dz) : e0 * std::expm1(dz)) / den;
    differences += e[j];
  }
  // The positives balance the unrounded negatives: every entry rounds once into Data.
  for (Int j = p; j < n; ++j) {
    const Work q = e[j] / den, y = q - carry, t = neg + y;
    gradient[j] = Data(q);
    carry = (t - neg) - y;
    neg = t;
  }
  for (Int j = 0; j < p; ++j) gradient[j] = Data(e[j] - (differences + neg) / Work(p));
  return gap + std::log1p(tail);
}
}  // namespace detail

// Loss (as entropy_loss) and gradient at the input logits, not at an SDCA
// block's updated scores; one pass of exponentials serves both.
template <typename Result, typename Data, typename Int>
Result entropy_loss_gradient(Int n, Int p, const Data* z, Data* gradient) {
  using Work = typename std::common_type<Result, Data, double>::type;
  return static_cast<Result>(detail::entropy_loss_gradient<Work>(n, p, z, gradient));
}

// Gradient at the input logits (not at an SDCA block's updated scores).
template <typename Result, typename Data, typename Int>
void entropy_gradient(Int n, Int p, const Data* z, Data* gradient) {
  entropy_loss_gradient<Result>(n, p, z, gradient);
}

// C H(q-a/C), with q uniform on the first p labels. For hard targets the
// complement is obtained from the non-target entries, never from C-a[0].
template <typename Result, typename Data, typename Int>
Result entropy_dual_loss(Int n, Int p, Result c, const Data* a) {
  using Work = typename std::common_type<Result, Data, double>::type;
  const Work C = Work(c), q = Work(1) / Work(p);
  Work balance(0), size(0);
  for (Int j = 0; j < n; ++j) {
    Work aj = Work(a[j]);
    if (!std::isfinite(aj) || (j >= p && aj > 0) ||
        (j < p && aj > C * q * (1 + 32 * Work(std::numeric_limits<Data>::epsilon()))))
      return -std::numeric_limits<Result>::infinity();
    balance += aj;
    size += std::abs(aj);
  }
  if (std::abs(balance) > 32 * Work(std::numeric_limits<Data>::epsilon()) * size)
    return -std::numeric_limits<Result>::infinity();
  Work loss(0), total(0);
  for (Int j = p; j < n; ++j) {
    Work b = -Work(a[j]);
    total += b;
    if (b > 0) loss -= b * (std::log(b) - std::log(C));
  }
  if (p == 1) {
    Work mass = total / C;
    if (mass < std::numeric_limits<Work>::epsilon())
      loss += total;
    else if (mass < 1)
      loss -= C * (1 - mass) * std::log1p(-mass);
  } else {
    for (Int j = 0; j < p; ++j) {
      Work r = Work(a[j]) / C, prob = q - r;
      if (prob > 0) loss -= C * prob * (std::log(q) + std::log1p(-r / q));
    }
  }
  return Result(loss);
}

namespace detail {
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

// W(exp(x)) by the last Householder step of lambert_w_exp, from a guess within
// about 1e-4 (relative) instead of its approximations; lambert_w_exp outside
// the arguments where that step is its last.
template <typename T>
inline T lambert_w_exp_near(T x, T guess) {
  if (!(x > T(-20) && x <= T(4) && guess > 0)) return lambert_w_exp(x);
  return lambert_w_iter_5(guess, std::exp(x - guess));
}

// Invert alpha * r - log1p(-r / q) = t for r in [q - 1, q).
// Away from r = 0 (|r| >= q / 2), q - p is well-conditioned and inverted in
// closed form via Lambert W. Near r = 0 (|r| < q / 2), uses a 2nd-order Pade
// initial guess and Halley refinement on log1p(-r / q). A finite guess of r
// (see lambert_w_exp_near) replaces the initial ones.
template <typename T>
inline T entropy_residual(T t, T alpha, T q, T log_alpha, T log_q,
                          T guess = std::numeric_limits<T>::quiet_NaN()) {
  if (t == 0) return 0;
  T lo = q - T(1), hi = q;
  if (t <= alpha * lo + log_q) return lo;
  const T half = alpha * q / T(2);
  const bool warm = std::isfinite(guess);
  if (t > half + T(0.69314718055994530942) || t < -half - T(0.40546510810816438198)) {
    T h = log_q + (alpha * q - t), arg = log_alpha + h;
    T w = warm ? lambert_w_exp_near(arg, alpha * (q - guess)) : lambert_w_exp(arg);
    T prob = (arg < T(-18)) ? std::exp(h - w) : w / alpha;
    return std::max(lo, std::min(q, q - prob));
  }
  T inv_q = T(1) / q;
  T initial = guess;
  if (!warm) {
    T linear = t / (alpha + inv_q);
    initial = linear / (T(1) + (linear * inv_q) / (T(2) * (T(1) + alpha * q)));
  }
  T r = std::max(lo, std::min(initial, std::nextafter(q, T(0))));
  const T tol = T(8) * std::numeric_limits<T>::epsilon() * std::abs(t);
  for (int it = 0; it < 100; ++it) {
    T f = alpha * r - std::log1p(-r * inv_q) - t;
    if (std::abs(f) <= tol) break;
    if (f < 0)
      lo = r;
    else
      hi = r;
    T inv_qr = T(1) / (q - r);
    T d1 = alpha + inv_qr;
    T denom = d1 - (f * inv_qr * inv_qr) / (T(2) * d1);
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

template <typename T>
inline T entropy_residual(T t, T alpha, T q) {
  return entropy_residual(t, alpha, q, std::log(alpha), std::log(q));
}

}  // namespace detail
namespace detail {

// Whether exp(h - W(exp(arg))) is exp(h) without W: W(exp(arg)) < exp(arg)
// is below a quarter ulp of |h| >= 1/2, so h - W rounds to h.
template <typename T>
inline bool lambert_negligible(const T arg, const T h) {
  const T cut = -T(std::numeric_limits<T>::digits + 2) * T(0.69314718055994530942L);
  return arg < cut && std::abs(h) >= T(0.5);
}

// The residuals r0 of an accurate evaluation of an entropy block at the
// threshold base, for the evaluations after it. Its probabilities
// p = W(exp(x)) / alpha expand in a step d of the threshold (x decreases by d):
// with u = 1 / (1 + alpha p0),
//   p(base + d) = p0 (1 + d (k1 + d (k2 + d (k3 + d k4)))),
// the Taylor series of W to fourth order (W^(k) / W are polynomials in u).
// p0 < 0 marks a class that is clipped at base. Before an accurate evaluation,
// the residuals of a previous solution (prior, if any) give guesses instead.
template <typename T>
struct entropy_expansion {
  static const std::size_t arrays = 7;
  T local[arrays * 64];
  std::unique_ptr<T[]> heap;
  T *r0, *p0, *k1, *k2, *k3, *k4, *prior = nullptr;
  T base = 0;
  bool accurate = false, expanded = false;
  explicit entropy_expansion(std::size_t n) : heap(n > 64 ? new T[arrays * n] : nullptr) {
    T* data = heap ? heap.get() : local;
    T** parts[] = {&r0, &p0, &k1, &k2, &k3, &k4};
    for (std::size_t i = 0; i < 6; ++i) *parts[i] = data + i * n;
  }
  entropy_expansion(const entropy_expansion&) = delete;
  entropy_expansion& operator=(const entropy_expansion&) = delete;
  T* prior_storage(std::size_t n) { return (heap ? heap.get() : local) + 6 * n; }
};

// The model kernels of prox_rows.h on double, and the same sums one at a time on float.
inline void model_moments(std::size_t n, const double* r, const double* u, double* moment) {
  model_moments_kernel(n, r, u, 0.0, moment);
}
template <typename T>
void model_moments(std::size_t n, const T* r, const T* u, double* moment) {
  for (std::size_t j = 0; j < n; ++j) {
    double x = -double(r[j]);
    for (int m = 1; m < model_powers; ++m) moment[m] += (x *= double(u[j]));
  }
}
inline void model_expand(std::size_t n, const double* r, const double* u, const double* b,
                         double* out, double& sum, double& magnitude) {
  model_expand_kernel(n, r, u, 0.0, b, out, &sum, &magnitude);
}
template <typename T, typename Sum>
void model_expand(std::size_t n, const T* r, const T* u, const double* b, T* out, Sum& sum,
                  T& magnitude) {
  for (std::size_t j = 0; j < n; ++j) {
    double s = 0;
    for (int m = model_powers - 1; m > 0; --m) s = (s + b[m]) * double(u[j]);
    out[j] = r[j] + r[j] * T(s);
    sum += Sum(out[j]);
    magnitude += std::abs(out[j]);
  }
}

// The other classes' terms at their probabilities r (others_terms_kernel on double).
inline bool others_terms(std::size_t n, double alpha, double* r, double* weights, double* sums) {
  return others_terms_kernel(n, alpha, r, weights, sums);
}
template <typename T>
bool others_terms(std::size_t n, T alpha, T* r, T* weights, double* sums) {
  bool clipped = false;
  for (std::size_t j = 0; j < n; ++j) {
    const bool over = r[j] > T(1);
    const T prob = over ? T(1) : r[j], u = over ? T(0) : T(1) / (T(1) + alpha * prob);
    weights[j] = u, r[j] = -prob;
    sums[0] += double(prob), sums[1] += double(prob * u), sums[2] -= double((prob * u) * u * u);
    clipped = clipped || over;
  }
  return clipped;
}

// The start's sums (entropy_block_solver::fixed_point_start): of exp(v - ref), its terms in
// e, and of the fixed-point step's exp(y) (fixed_point_kernel), on the row kernels in double.
inline double start_terms(std::size_t n, const double* v, double ref, double* e) {
  return exp_sum_kernel(n, v, n, ref, e, false);
}
template <typename T>
double start_terms(std::size_t n, const T* v, T ref, T* e) {
  compensated_sum<double> total;
  for (std::size_t j = 0; j < n; ++j) {
    const double x = double(v[j]) - double(ref), ej = x >= -708 ? exp_normal(x) : 0.0;
    e[j] = T(ej);
    total.add(ej);
  }
  return total.value();
}
inline double start_step(std::size_t n, std::size_t p, const double* v, double ref, const double* e,
                         double inv_sum, double a, double q) {
  return fixed_point_kernel(n, p, v, ref, e, inv_sum, a, q);
}
template <typename T>
double start_step(std::size_t n, std::size_t p, const T* v, T ref, const T* e, double inv_sum,
                  double a, double q) {
  compensated_sum<double> shifted;
  for (std::size_t j = 0; j < n; ++j) {
    const double y =
        (double(v[j]) - double(ref)) + a * ((j < p ? q : 0.0) - double(e[j]) * inv_sum);
    shifted.add(y >= -708 ? exp_normal(y) : 0.0);
  }
  return shifted.value();
}

// Entropic block solve in residual coordinates r_j = q_j - p_j for the scores
// v of p labels (first) and n - p other classes, as functions of the
// normalization threshold relative to the largest score, dt = t - v_ref:
// centering there avoids cancellation on scores far below the others.
template <typename T, typename Int>
struct entropy_block_solver {
  Int n, p;
  T alpha, q, log_alpha, log_q, clip_threshold;
  const T* v;
  T* r;
  T v_max_label, v_max_other, v_ref;
  // Derivative terms of the residual sum at the last evaluated threshold.
  T derivative = 0, curvature = 0, magnitude = 0, other_mass = 0;
  // The evaluations after an accurate one start from it (see evaluate).
  entropy_expansion<T>* expansion = nullptr;
  // Whether the other classes' probabilities take the row kernels (prox_rows.h) and the solve
  // starts at a fixed-point step (the double loss functions; float rows, which they do not
  // speed up, and SDCA steps keep the scalar functions).
  bool lanes = false;
  // The loss functions' Taylor model (model_root): the weights u_j = 1 / (1 + alpha pi_j) of
  // the last evaluation (0 where clipped) and storage of its expanded residuals; null in SDCA
  // steps. Whether the last evaluation clipped a probability.
  T* weights = nullptr;
  T* expanded = nullptr;
  bool any_clipped = false;

  entropy_block_solver(Int n_, Int p_, T alpha_, const T* v_, T* r_)
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

  // Probability W(exp(log alpha + h)) / alpha without an underflowing W; a
  // finite guess of W goes to lambert_w_exp_near.
  T probability(T h, T guess = std::numeric_limits<T>::quiet_NaN()) const {
    const T arg = h + log_alpha;
    if (lambert_negligible(arg, h)) return std::exp(h);
    const T w = std::isfinite(guess) ? lambert_w_exp_near(arg, guess) : lambert_w_exp(arg);
    return (arg < T(-18)) ? std::exp(h - w) : w / alpha;
  }

  void add_slope(Int j, T prob) {
    const T denom = T(1) + alpha * prob;
    if (weights) {
      const T u = T(1) / denom, d1 = prob * u;
      weights[j] = u;
      derivative += d1;
      curvature -= (d1 * u) * u;
      return;
    }
    const T d1 = prob / denom;
    derivative += d1;
    curvature -= d1 / (denom * denom);
  }
  void add_clipped(Int j) {
    any_clipped = true;
    if (weights) weights[j] = T(0);
  }

  // f sums in at least double: a float sum of many classes misplaces the root.
  using Sum = typename std::common_type<T, double>::type;

  void add_label(Int j, T rj, bool clipped, Sum& f) {
    r[j] = rj;
    f += Sum(rj);
    magnitude += std::abs(rj);
    if (clipped)
      add_clipped(j);
    else
      add_slope(j, q - rj);
  }
  void add_other(Int j, T prob, Sum& f) {
    const bool clipped = (prob > T(1));
    if (clipped) prob = T(1);
    r[j] = -prob;
    f -= Sum(prob);
    other_mass += prob;
    magnitude += prob;
    if (clipped)
      add_clipped(j);
    else
      add_slope(j, prob);
  }
  void label_exact(Int j, T dt, Sum& f, T guess = std::numeric_limits<T>::quiet_NaN()) {
    const T argument = (v_ref - v[j]) + dt;
    add_label(j, entropy_residual(argument, alpha, q, log_alpha, log_q, guess),
              argument < clip_threshold, f);
  }
  void other_exact(Int j, T dt, Sum& f, T guess = std::numeric_limits<T>::quiet_NaN()) {
    add_other(j, probability(((v[j] - v_ref) - dt) + log_q, guess), f);
  }

  // The largest step of the expansion: its truncation, p0 d^5 / 120 times
  // |W^(5) / W| <= 1, stays below eps / 32 of p.
  static T expansion_reach() { return std::ldexp(T(1), -(std::numeric_limits<T>::digits + 4) / 5); }

  // The largest step of a guess of third order (relative error at most d^4 / 24)
  // whose one Householder step (order 5) is accurate. 0 in float, where
  // lambert_w_exp costs little more than that step.
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
  static T guess_change(T u, T s) {
    const T u3 = u * u * u;
    return s * (u + s * (u3 / T(2) + s * (u3 * u * (T(3) * u - T(2)) / T(6))));
  }

  // The base's probability of class j, or -1 where it is clipped.
  T base_probability(Int j) const {
    const entropy_expansion<T>& e = *expansion;
    const T r0 = e.r0[j], prob = j < p ? q - r0 : -r0;
    const bool clipped =
        j < p ? (v_ref - v[j]) + e.base < clip_threshold || r0 == 0 : !(prob < T(1));
    return clipped ? T(-1) : prob;
  }

  // The coefficients of the expansion about its base.
  void expand_base() {
    entropy_expansion<T>& e = *expansion;
    for (Int j = 0; j < n; ++j) {
      const T prob = base_probability(j);
      e.p0[j] = prob;
      if (prob < 0) continue;
      const T u = T(1) / (T(1) + alpha * prob), u2 = u * u, u4 = u2 * u2;
      e.k1[j] = -u;
      e.k2[j] = u2 * u / T(2);
      e.k3[j] = -u4 * (T(3) * u - T(2)) / T(6);
      e.k4[j] = u4 * u * ((T(15) * u - T(20)) * u + T(6)) / T(24);
    }
    e.expanded = true;
  }

  // f at base + d from the expansion; a label whose residual would lose its
  // relative accuracy (cancellation, or truncation beside a small residual),
  // and a clipped class, are evaluated exactly.
  T expand(T dt) {
    entropy_expansion<T>& e = *expansion;
    if (!e.expanded) expand_base();
    const T d = dt - e.base, d5 = std::abs(d * d * d * d * d),
            eps = std::numeric_limits<T>::epsilon();
    Sum f(0);
    for (Int j = 0; j < n; ++j) {
      const T p0 = e.p0[j];
      if (!(p0 >= 0)) {
        if (j < p)
          label_exact(j, dt, f);
        else
          other_exact(j, dt, f);
        continue;
      }
      const T s = d * (e.k1[j] + d * (e.k2[j] + d * (e.k3[j] + d * e.k4[j])));
      const T change = p0 * s;
      if (j >= p) {
        add_other(j, p0 + change, f);
        continue;
      }
      const T r0 = e.r0[j], rj = r0 - change;
      if (std::abs(change) <= std::abs(r0) / T(2) && p0 * d5 <= eps * std::abs(r0) && rj > q - T(1))
        add_label(j, rj, false, f);
      else
        label_exact(j, dt, f);
    }
    return T(f);
  }

  // The classes at dt from the guesses of the prior residuals r of a previous
  // solution: each is exact at its own argument (t = alpha r - log1p(-r / q)
  // of a label, x = log(w) + w at w = alpha p of another class).
  void evaluate_from_prior(T dt, Sum& f) {
    const T* prior = expansion->prior;
    for (Int j = 0; j < n; ++j) {
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

  // The residual sum f(dt), concave increasing; fills r and the derivatives.
  // With an expansion, an evaluation within expansion_reach() of the last
  // accurate one expands it, one within warm_reach() takes its probabilities
  // as guesses.
  T evaluate(T dt) {
    any_clipped = false;
    derivative = 0;
    curvature = 0;
    magnitude = 0;
    other_mass = 0;
    const T distance = expansion && expansion->accurate ? std::abs(dt - expansion->base)
                                                        : std::numeric_limits<T>::infinity();
    if (distance <= expansion_reach()) return expand(dt);
    Sum f(0);
    if (expansion && !expansion->accurate && expansion->prior) {
      evaluate_from_prior(dt, f);
    } else if (distance <= warm_reach()) {
      const T d = dt - expansion->base;
      for (Int j = 0; j < n; ++j) {
        const T p0 = base_probability(j);
        T guess = std::numeric_limits<T>::quiet_NaN();
        if (p0 >= 0) {
          const T s = guess_change(T(1) / (T(1) + alpha * p0), -d);
          guess = j < p ? expansion->r0[j] - p0 * s : alpha * p0 + alpha * p0 * s;
        }
        if (j < p)
          label_exact(j, dt, f, guess);
        else
          other_exact(j, dt, f, guess);
      }
    } else {
      for (Int j = 0; j < p; ++j) label_exact(j, dt, f);
      if (lanes) {
        others_probability(n - p, v + p, v_ref, dt, log_q, r + p);
        if (weights) {
          double sums[3] = {0, 0, 0};
          any_clipped =
              others_terms(std::size_t(n - p), alpha, r + p, weights + p, sums) || any_clipped;
          f -= Sum(sums[0]);
          other_mass += T(sums[0]);
          magnitude += T(sums[0]);
          derivative += T(sums[1]);
          curvature += T(sums[2]);
        } else {
          for (Int j = p; j < n; ++j) add_other(j, r[j], f);
        }
      } else {
        for (Int j = p; j < n; ++j) other_exact(j, dt, f);
      }
    }
    if (expansion) {
      std::copy(r, r + n, expansion->r0);
      expansion->base = dt;
      expansion->accurate = true;
      expansion->expanded = false;
    }
    return T(f);
  }

  // The other classes' probabilities by the row kernel (double), else the scalar function.
  void others_probability(Int count, const double* x, T ref, T dt, T shift, double* out) const {
    probability_kernel(std::size_t(count), x, double(ref), double(dt), double(shift), double(alpha),
                       double(log_alpha), out);
  }
  template <typename X>
  void others_probability(Int count, const X* x, T ref, T dt, T shift, X* out) const {
    for (Int j = 0; j < count; ++j) out[j] = probability(((x[j] - ref) - dt) + shift);
  }

  // Halley step, or a step in the log of the other classes' mass where that
  // mass exceeds one and the Halley step is shorter.
  T step(T f) const {
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

  // The reach of the Taylor model (prox_rows.h): its remainder is below eps / 32 there.
  static T model_reach() {
    return std::numeric_limits<T>::digits <= std::numeric_limits<float>::digits ? T(0.25)
                                                                                : T(0.015625);
  }

  // Near the root, the Taylor model of the probabilities about the last evaluation at dt gives
  // the root and its residuals without another evaluation: true with dt, r, f and magnitude
  // there; else false with the model's step in delta (where finite).
  bool model_root(T& dt, T& delta, T lo, T hi, T& f) {
    double moment[model_powers] = {}, b[model_powers], db[model_powers];
    for (Int j = 0; j < p; ++j) {
      const double u = double(weights[j]);
      double x = double(q - r[j]);
      for (int m = 1; m < model_powers; ++m) moment[m] += (x *= u);
    }
    model_moments(std::size_t(n - p), r + p, weights + p, moment);
    auto slope = [&]() -> double {
      double total = 0;
      for (int m = model_powers - 1; m > 0; --m) total -= db[m] * moment[m];
      return total;
    };
    const T eps = std::numeric_limits<T>::epsilon();
    const double f0 = double(f);
    double d = double(delta);
    for (int it = 0; it < 8; ++it) {
      model_coefficients(d, b, db);
      double change = 0;
      for (int m = model_powers - 1; m > 0; --m) change += b[m] * moment[m];
      const double step = (f0 - change) / slope();
      if (!std::isfinite(step)) return false;
      d -= step;
      if (std::abs(step) <= 1e-3 * double(eps) * std::abs(d)) break;
    }
    const T tiny = std::numeric_limits<T>::min();
    for (int attempt = 0; attempt < 3; ++attempt) {
      const T next = dt + T(d);
      if (!(lo <= next && next <= hi)) return false;
      delta = next - dt;
      if (!(std::abs(delta) <= model_reach())) return false;
      d = double(delta);
      model_coefficients(d, b, db);
      // The remainder relative to eps / 32 of the probability.
      const T bound = T(std::pow(std::abs(d) / double(model_reach()), model_order + 1));
      Sum sum(0);
      T size(0);
      for (Int j = 0; j < p; ++j) {
        const T prob = q - r[j];
        double s = 0;
        for (int m = model_powers - 1; m > 0; --m) s = (s + b[m]) * double(weights[j]);
        const T change = prob * T(s), rj = r[j] - change;
        // A label whose residual would lose its relative accuracy is solved exactly.
        const bool kept = std::abs(change) <= std::abs(r[j]) / 2 &&
                          prob * bound <= std::abs(r[j]) && rj > q - T(1);
        expanded[j] =
            kept ? rj : entropy_residual((v_ref - v[j]) + next, alpha, q, log_alpha, log_q);
        sum += Sum(expanded[j]);
        size += std::abs(expanded[j]);
      }
      model_expand(std::size_t(n - p), r + p, weights + p, b, expanded + p, sum, size);
      if (std::abs(sum) <= Sum(T(8) * eps * std::max(size, tiny))) {
        std::copy(expanded, expanded + n, r);
        dt = next;
        f = T(sum);
        magnitude = size;
        return true;
      }
      d -= double(sum) / slope();
    }
    delta = T(d);
    return false;
  }

  // Analytic bracket: at lo the largest probability is one, so f(lo) <= 0
  // and no class is clipped above it; at hi every p_j <= 1/n, so f(hi) >= 0.
  void bracket(T& lo, T& hi) const {
    lo =
        std::max((v_max_label - v_ref) - alpha * (T(1) - q), (v_max_other - v_ref) - alpha) + log_q;
    hi =
        std::max((v_max_label - v_ref) + alpha * q, (v_max_other - v_ref)) + log_q + std::log(T(n));
    if (!(lo <= hi)) hi = lo;
  }

  // Softmax-limit start in the centered coordinate, so the small log
  // correction is never added to a large score. Confident hard targets
  // converge in two evaluations without it: the exp pass is skipped when its
  // bound on the other classes' mass is below sqrt(machine epsilon).
  T start() const {
    const bool tiny_tail =
        p == 1 && v_max_label == v_ref &&
        (v_max_other - v_ref) + std::log(T(n)) < std::log(std::numeric_limits<T>::epsilon()) / 2;
    if (tiny_tail) return log_q;
    if (lanes) return fixed_point_start();
    T tail(0);
    bool maximum_seen = false;
    for (Int j = 0; j < n; ++j) {
      if (!maximum_seen && v[j] == v_ref)
        maximum_seen = true;
      else
        tail += std::exp(v[j] - v_ref);
    }
    return log_q + std::log1p(tail);
  }

  // The softmax-limit start log q + log sum exp(v - v_ref) and, where alpha (max - min) of
  // q_j - pi_j over its probabilities pi is at most 2, one fixed-point step
  // log q + log sum exp((v - v_ref) + alpha (q_j - pi_j)) (q_j = q on the labels, 0 elsewhere):
  // fewer Halley steps from it. In double, its terms in r.
  T fixed_point_start() const {
    const double sum = start_terms(std::size_t(n), v, v_ref, r);
    const double a = double(alpha), qd = double(q), inv_sum = 1 / sum;
    const double softmax = double(log_q) + std::log1p(sum - 1);
    // The gaps q_j - pi_j: q - pi_j of the labels, -pi_j of the others.
    T smallest;
    const T largest = max_except(std::size_t(n - p), r + p, std::size_t(n - p), smallest);
    double high = 0, low = -double(largest) * inv_sum;
    for (Int j = 0; j < p; ++j) {
      const double gap = qd - double(r[j]) * inv_sum;
      high = std::max(high, gap);
      low = std::min(low, gap);
    }
    if (!(a * (high - low) <= 2)) return T(softmax);
    const double shifted = start_step(std::size_t(n), std::size_t(p), v, v_ref, r, inv_sum, a, qd);
    const double fixed = double(log_q) + std::log(shifted);
    return T(std::isfinite(fixed) ? fixed : softmax);
  }

  // The threshold dt of the solution; leaves its residuals in r. A finite
  // guess replaces the softmax-limit start (the bracket still applies).
  T solve(const T guess = std::numeric_limits<T>::quiet_NaN()) {
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
      T delta = step(f);
      if (weights && !any_clipped && std::abs(delta) <= 2 * model_reach()) {
        T root = delta;
        if (model_root(dt, root, lo, hi, f)) break;
        if (std::isfinite(root) && root != 0) delta = root;
      }
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
    // The resolution of t and of f is eps relative to at least tiny: they can underflow.
    if (!std::isfinite(f) ||
        (std::abs(f) > T(256) * eps * std::max(magnitude, tiny) &&
         hi - lo > T(64) * eps * std::max({std::abs(lo), std::abs(hi), tiny}))) {
      throw std::runtime_error("Entropy normalization did not converge");
    }
    return dt;
  }

  // Normalized probabilities: q - r where r is small, the Lambert inversion
  // where p is (never exp(log q + alpha r - s), whose O(alpha) terms cancel).
  void probabilities(T dt, std::vector<T>& out) const {
    out.resize(static_cast<std::size_t>(n));
    T total(0);
    for (Int j = 0; j < n; ++j) {
      T prob = -r[j];
      if (j < p) {
        prob = q - r[j];
        if (r[j] > q / T(2)) prob = probability(log_q + (alpha * q - ((v_ref - v[j]) + dt)));
      }
      out[static_cast<std::size_t>(j)] = std::min(T(1), prob);
      total += out[static_cast<std::size_t>(j)];
    }
    if (!(total > 0) || !std::isfinite(total)) {
      throw std::runtime_error("Invalid entropy probability normalization");
    }
    for (auto& prob : out) prob /= total;
  }

  // log p_j of a label: log1p where r_j is small, else log(W / alpha) = h - W.
  T label_log_probability(T dt, Int j) const {
    if (r[j] <= q / T(2)) return log_q + std::log1p(-r[j] / q);
    const T h = log_q + (alpha * q - ((v_ref - v[j]) + dt));
    return h - lambert_w_exp(h + log_alpha);
  }

  // The dual block c r, zero-sum up to one rounding in Data: the entry of
  // largest magnitude (the label for p = 1) is formed from the stored others.
  template <typename Data>
  void duals(T c, T dt, Data* out) const {
    for (Int j = 0; j < n; ++j) {
      out[j] = Data(c * r[j]);
      // A subnormal (or underflowed) probability can still have a normal
      // c p. Defer the logarithm of c to this rare path.
      if (j >= p && c > T(1) && -r[j] < std::numeric_limits<T>::min()) {
        const T score = (v[j] - v_ref) + log_q;
        const T h = score - dt;
        out[j] = -Data(std::exp(((std::log(c) - dt) + score) - lambert_w_exp(h + log_alpha)));
      }
    }
    // Sum is at least double: float sums of many classes drift beyond the
    // zero-sum tolerance of the dual objective.
    Int pivot = 0;
    if (p > 1) {
      bool labels_zero = true;
      for (Int j = 0; j < p; ++j) labels_zero = labels_zero && out[j] == 0;
      Sum mass(0);
      if (labels_zero) {
        for (Int j = p; j < n; ++j) mass += Sum(out[j]);
      }
      if (mass != 0) {
        // Only recovered mass of the other classes is left: the labels share it.
        for (Int j = 1; j < p; ++j) out[j] = Data(-mass / Sum(p));
      } else {
        for (Int j = 1; j < n; ++j) {
          if (std::abs(out[j]) > std::abs(out[pivot])) pivot = j;
        }
      }
    }
    Sum sum(0);
    for (Int j = 0; j < n; ++j) {
      if (j != pivot) sum += Sum(out[j]);
    }
    out[pivot] = Data(-sum);
  }
};

}  // namespace detail

namespace detail {

template <typename Result, typename Int>
void check_entropy_block(Int n, Int p, Result alpha, const Result* v) {
  if (p < 1 || p >= n || !(alpha > 0) || !std::isfinite(alpha))
    throw std::invalid_argument("Invalid entropy block dimensions or scale");
  for (Int j = 0; j < n; ++j)
    if (!std::isfinite(v[j])) throw std::invalid_argument("Entropy scores must be finite");
}

}  // namespace detail

// Entropic proximal step of a dual block. The scores v hold p labels (first)
// and n - p other classes; q is uniform on the labels (q_j = 1/p). Solves for
// probabilities pi summing to one with
//   alpha pi_j + log pi_j = v_j + alpha q_j - t   for every class j,
// that is pi = softmax(u) at u = v + alpha (q - pi), and returns the dual
// block out = c (q - pi). Optionally reports pi and the log-normalizer t.
// Throws std::invalid_argument for invalid sizes, scales or nonfinite scores
// and std::runtime_error if the normalization does not converge.
template <typename Data, typename Result, typename Int>
void entropy_prox_dual_ptr(Int n, Int p, Result c, Result alpha, const Result* v, Data* out,
                           std::vector<Result>* probabilities = nullptr,
                           Result* normalizer = nullptr) {
  detail::check_entropy_block(n, p, alpha, v);
  if (!(c > 0) || !std::isfinite(c)) throw std::invalid_argument("Invalid entropy dimensions or C");
  detail::entropy_buffer<Result> residuals(static_cast<std::size_t>(n));
  detail::entropy_block_solver<Result, Int> block(n, p, alpha, v, residuals.data());
  const Result dt = block.solve();
  if (normalizer) *normalizer = block.v_ref + dt;
  if (probabilities) block.probabilities(dt, *probabilities);
  block.duals(c, dt, out);
}

template <typename Data, typename Result, typename Int>
void entropy_prox_dual(Int n, Int p, Result c, Result alpha, const std::vector<Result>& v,
                       Data* out, std::vector<Result>* probabilities = nullptr,
                       Result* normalizer = nullptr) {
  if (v.size() != static_cast<std::size_t>(n))
    throw std::invalid_argument("Invalid entropy dimensions");
  entropy_prox_dual_ptr(n, p, c, alpha, v.data(), out, probabilities, normalizer);
}

// The proximal entropy loss min_u H(q, u) + |u - v|^2 / (2 alpha), and the
// dual out = (u - v) / alpha of its minimizer. H comes from the solved label
// probabilities: u = v + alpha out loses its small part to cancellation.
template <typename Result, typename Int>
Result entropy_prox_loss_ptr(Int n, Int p, Result alpha, const Result* v, Result* out) {
  detail::check_entropy_block(n, p, alpha, v);
  detail::entropy_buffer<Result> residuals(static_cast<std::size_t>(n));
  detail::entropy_block_solver<Result, Int> block(n, p, alpha, v, residuals.data());
  // The error is bounded by 3 max(1, alpha) tail^2. Recover the CE tail
  // where this is below working precision or below a subnormal quantum.
  const Result gap = block.v_max_other - block.v_max_label;
  if (p == 1 && std::isfinite(gap) && gap < std::log(std::numeric_limits<Result>::min())) {
    const Result tail = gap + std::log(Result(n - 1));
    const Result error = std::max(Result(0), block.log_alpha) + tail;
    const Result bound = std::log(std::numeric_limits<Result>::epsilon() / Result(8));
    if (error < bound || error + tail < std::log(std::numeric_limits<Result>::min()) + bound) {
      const Result value = single_entropy_loss_gradient<Result>(n, v, out);
      for (Int j = 0; j < n; ++j) out[j] = -out[j];
      return value;
    }
  }
  block.lanes = std::is_same<Result, double>::value;
  detail::entropy_buffer<Result> weights(static_cast<std::size_t>(n)),
      expanded(static_cast<std::size_t>(n));
  block.weights = weights.data();
  block.expanded = expanded.data();
  const Result dt = block.solve();
  block.duals(Result(1), dt, out);
  Result cross(0), penalty(0);
  for (Int j = 0; j < p; ++j) cross -= block.label_log_probability(dt, j);
  // duals() recovered the one-label mass even if the root residual underflowed.
  if (p == 1 && out[0] <= Result(0.5)) cross = -std::log1p(-out[0]);
  // alpha out_j first: out_j^2 can underflow where the product does not.
  for (Int j = 0; j < n; ++j) penalty += (alpha * out[j]) * out[j];
  if (!std::isfinite(cross)) {
    // A finite mean need not have a representable unnormalized sum. Keep the
    // ordinary accumulation unchanged; divide each term only on overflow.
    cross = Result(0);
    for (Int j = 0; j < p; ++j) cross -= block.label_log_probability(dt, j) / Result(p);
    return cross + penalty / 2;
  }
  return cross / Result(p) + penalty / 2;
}

// Working precision of an SDCA entropy block update, chosen by the caller
// through Result: <float, float> solves in float, <float, double> in double.
template <typename Data, typename Result>
using entropy_block_type = typename std::common_type<Data, Result>::type;

namespace detail {

// Warm log-odds (z - zr) - norm2 (a - ar) in T. Exact differences and product
// in W >= double keep a small residual between huge cancelling score/dual terms.
template <typename T, typename Data>
inline T warm_log_odds(Data z, Data zr, Data a, Data ar, Data norm2) {
  const T dz = T(z) - T(zr), product = T(norm2) * (T(a) - T(ar)), plain = dz - product;
  // Without cancellation the plain rounding error is below 1.5 ulps.
  if (std::abs(dz) + 2 * std::abs(product) <= 2 * std::abs(plain)) return plain;
  using W = typename std::common_type<T, double>::type;
  W dz_lo, da_lo, product_lo, sum_lo;
  const W wdz = two_sum(W(z), -W(zr), dz_lo), da = two_sum(W(a), -W(ar), da_lo);
  const W wproduct = split_product(W(norm2), da, product_lo);
  const W sum = two_sum(wdz, -wproduct, sum_lo);
  const W exact = sum + (sum_lo + (dz_lo - (product_lo + W(norm2) * da_lo)));
  // Keep the plain value within two ulps (and where the split overflows).
  const W tol = 2 * W(std::numeric_limits<T>::epsilon()) * std::abs(W(plain));
  return std::abs(exact - W(plain)) > tol && std::isfinite(exact) ? T(exact) : plain;
}

// warm_log_odds as high + low (both T), exact to the rounding of low.
template <typename T, typename Data>
inline T warm_log_odds_split(Data z, Data zr, Data a, Data ar, Data norm2, T& low) {
  using W = typename std::common_type<T, double>::type;
  W dz_lo, da_lo, product_lo, sum_lo;
  const W wdz = two_sum(W(z), -W(zr), dz_lo), da = two_sum(W(a), -W(ar), da_lo);
  const W wproduct = split_product(W(norm2), da, product_lo);
  const W sum = two_sum(wdz, -wproduct, sum_lo);
  const W rest = sum_lo + (dz_lo - (product_lo + W(norm2) * da_lo));
  const T high = T(sum + rest);
  low = T((sum - W(high)) + rest);
  if (std::isfinite(high) && std::isfinite(low)) return high;
  low = T(0);  // where the split overflows
  return warm_log_odds<T>(z, zr, a, ar, norm2);
}

}  // namespace detail

namespace detail {

// The threshold dt (relative to the largest of v) at which the probabilities
// pi = q - a / c of the current block a solve the block equations, taken at
// its most probable class: alpha (q_j - pi_j) = norm2 a_j, so
//   dt = (v_j - v_ref) + norm2 a_j - log pi_j.
// It is the solution where the block is already optimal (as near the end of
// training), so the root solve starts there. NaN for a zero block (its first
// step, where the softmax-limit start is better) or a nonpositive probability.
template <typename T, typename Data, typename Int>
inline T entropy_warm_threshold(Int n, Int p, T c, T norm2, const Data* a, const T* v) {
  const T q = T(1) / T(p);
  Int best = 0;
  T best_prob = -std::numeric_limits<T>::infinity(), v_ref = v[0];
  bool zero = true;
  for (Int j = 0; j < n; ++j) {
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

}  // namespace detail

// SDCA step of an entropy block: a holds the current block and receives the
// new one, z the scores at the current model. The root solve starts from the
// threshold of the current block (entropy_warm_threshold); its result meets
// the same tolerance as from any start. Evaluations near the last exact one
// expand it (entropy_expansion).
template <typename Data, typename Result, typename Int>
void entropy_update(Int n, Int p, Result c, Data norm2, Data* a, const Data* z) {
  using T = entropy_block_type<Data, Result>;
  if (n < 2 || p < 1 || p >= n) throw std::invalid_argument("Invalid entropy dimensions");
  detail::entropy_buffer<T> storage(static_cast<std::size_t>(n));
  T* v = storage.data();
  // Center before forming the score differences, not only inside the root
  // solve: [-1e16,0,1] otherwise becomes [0,1e16,1e16], losing a real margin.
  const auto reference = std::max_element(z, z + n) - z;
  for (Int j = 0; j < n; ++j)
    v[j] = detail::warm_log_odds<T>(z[j], z[reference], a[j], a[reference], norm2);
  const T cc = T(c), alpha = T(c) * T(norm2);
  detail::check_entropy_block(n, p, alpha, v);
  if (!(cc > 0) || !std::isfinite(cc))
    throw std::invalid_argument("Invalid entropy dimensions or C");
  const T guess = detail::entropy_warm_threshold(n, p, cc, T(norm2), a, v);
  detail::entropy_buffer<T> residuals(static_cast<std::size_t>(n));
  detail::entropy_expansion<T> expansion(static_cast<std::size_t>(n));
  if (std::isfinite(guess) && detail::entropy_block_solver<T, Int>::warm_reach() > 0) {
    expansion.prior = expansion.prior_storage(static_cast<std::size_t>(n));
    for (Int j = 0; j < n; ++j) expansion.prior[j] = T(a[j]) / cc;
  }
  detail::entropy_block_solver<T, Int> block(n, p, alpha, v, residuals.data());
  block.expansion = &expansion;
  const T dt = block.solve(guess);
  block.duals(cc, dt, a);
}

namespace detail {
// Probability inversion shared by the logarithmic top-k mass solve.
template <typename T>
inline T lambert_probability(T score, T alpha, T log_alpha) {
  const T arg = score + log_alpha;
  if (lambert_negligible(arg, score)) return std::exp(score);
  const T w = lambert_w_exp(arg);
  return arg < -18 ? std::exp(score - w) : w / alpha;
}
}  // namespace detail

// Offset-separated top-k iteration with centered log-odds and threshold.
namespace detail {

// Start of the log-odds iteration of all_capped_log_mass, relative to center.
template <typename T>
inline T all_capped_start(T mean, T coefficient, T root_logk, T center, bool unshifted, T lo,
                          T hi) {
  // In the small-mass regime, log(s)+coefficient*s=mean+log(k) gives
  // a Lambert-W start. A raw Newton step from s~1/coefficient otherwise
  // overshoots by hundreds of log units when the coefficient is near max().
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
  const T eps = T(64) * std::numeric_limits<T>::epsilon();
  const bool unshifted = mean > 0 && mean < coefficient;
  const T center = unshifted ? T(0) : mean;
  T lo = unshifted ? (mean - coefficient) + root_logk : root_logk - coefficient;
  T hi = unshifted ? mean + root_logk : root_logk;
  T r = all_capped_start(mean, coefficient, root_logk, center, unshifted, lo, hi);
  for (int it = 0; it < 200; ++it) {
    const T logit = center + r, s = std::exp(-softplus(-logit));
    const T ds = s * std::exp(-softplus(logit));
    // Cancel the large terms first; adding log(k) to mean loses it.
    const T residual = unshifted ? coefficient * s - mean : coefficient * s;
    const T f = residual + (r - root_logk);
    const T magnitude = std::abs(residual) + std::abs(r - root_logk) + (unshifted ? mean : T(0));
    if (std::abs(f) <= eps * magnitude) break;
    if (f < 0)
      lo = r;
    else
      hi = r;
    T next = r - f / (1 + coefficient * ds);
    // Test resolution before the strict bracket check. A rounded Newton
    // step at the new endpoint must not jump back across an enormous bracket.
    if (next == r) break;
    if (!(lo < next && next < hi) || !std::isfinite(next)) next = lo + (hi - lo) / 2;
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

}  // namespace detail

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

namespace detail {

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
  entropy_buffer<T> suffix;

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
      detail::sort_largest(order, k, order + n, [&](std::size_t i, std::size_t j) {
        return v[i] > v[j] || (v[i] == v[j] && low[i] > low[j]);
      });
    else
      detail::sort_largest(order, k, order + n,
                           [&](std::size_t i, std::size_t j) { return v[i] > v[j]; });
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
template <typename T>
struct topk_candidate {
  const topk_scores<T>& scores;
  std::size_t n, u;
  T alpha, logalpha, eps, machine, infinity;
  T* x;
  T* d;  // the sorted scores relative to the first free one (t_c relative to wmax)
  // h t = A s - h log s + (logit s - vmax) + c0; relative to t_c,
  // h (t - t_c) = A s - h log s + r + base, base = (center - anchor), with the anchor
  // vmax - c0 + h t_c (see the constructor).
  T h, A, c0, t_c, center, shift, base, r, anchor;
  // The evaluation at the last log-odds.
  T f = 0, slope = 0, magnitude = 0, threshold = 0, cap = 0, tau = 0, tau_shifted = 0;
  T sum = 0, mass = 0, rate = 0, dt = 0, s = 0, ds = 0, one_minus_s = 0, logs = 0;
  bool skipped = false;
  T lo, hi, previous;

  topk_candidate(const topk_scores<T>& scores_, std::size_t u_, T sum_capped, T alpha_, T* x_,
                 T* d_)
      : scores(scores_),
        n(scores_.n),
        u(u_),
        alpha(alpha_),
        logalpha(std::log(alpha_)),
        eps(T(64) * std::numeric_limits<T>::epsilon()),
        machine(std::numeric_limits<T>::epsilon()),
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
    dt = (A * ds + (T(u) / K) * one_minus_s + s) / h;
    // The free x_j reaches the cap at the score t + log(cap) + alpha cap.
    threshold = tau_shifted - scores.logk + alpha * cap;
    mass = h * s;
    // x_j <= exp(v_j - t): terms below eps * mass / n cannot matter.
    const T cut = mass > 0 ? std::log(machine * mass / T(n - u)) : -infinity;
    // Local accumulators: x could alias the members for the compiler.
    T free_sum(0), free_rate(0);
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
      free_sum += xj;
      free_rate += xj / (1 + alpha * xj);
    }
    sum = free_sum;
    rate = free_rate;
    slope = rate * dt + h * ds;
    magnitude = sum + mass;
    return mass - sum;
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
    lambda = E > T(1e-3) * D && E < D ? std::log(E) - std::log(D - E)
                                      : hE - lambert_w_exp(hE + std::log(hD));
    for (int it = 0; it < 60 && std::isfinite(lambda); ++it) {
      // sigmoid(lambda) and 1 - sigmoid(lambda) from one exponential.
      const T e = std::exp(-std::abs(lambda)), small = e / (1 + e), large = 1 / (1 + e);
      const T share = lambda < 0 ? small : large, rest = lambda < 0 ? large : small;
      const T g = (hD * share + lambda) - hE;
      if (g > 0)
        hi_l = lambda;
      else
        lo_l = lambda;
      T next = lambda - g / (1 + hD * share * rest);
      // Converged before the bracket test: a step that rounds onto an
      // endpoint must not restart bisection over the whole bracket. The
      // scalar model is cheap; solving it fully saves block evaluations.
      if (std::abs(next - lambda) <= 4 * machine * std::max(T(1), std::abs(next))) {
        lambda = next;
        break;
      }
      if (!(lo_l < next && next < hi_l)) next = lo_l + (hi_l - lo_l) / 2;
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
    if (lo > -infinity && hi < infinity) return lo + (hi - lo) / 2;
    return f < 0 ? r + std::max(T(1), std::abs(r)) : r - std::max(T(1), std::abs(r));
  }

  // Solves for the log-odds; returns whether the iteration converged.
  bool solve() {
    for (int it = 0; it < 200; ++it) {
      f = evaluate(r);
      if (!(std::abs(f) > eps * magnitude)) return std::isfinite(f);
      if (f < 0)
        lo = r;
      else
        hi = r;
      T next = r;
      bool stepped = false;
      if (propose(next, stepped)) continue;
      if (stepped && unresolved(next)) {
        if (stalled()) return std::isfinite(f);
        continue;
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

  // The threshold inherits the errors of the root and of alpha s, divided by h.
  T threshold_tolerance() const {
    return eps * std::max(std::max(T(1), std::abs(threshold)), (A * s + std::abs(r)) / h);
  }

  // Writes output_scale x in the caller's order, after checking the mass
  // independently of the threshold's conditioning: a huge threshold
  // tolerance must never authorize a huge feasibility error.
  void write(T* result, T output_scale, T mass_relative_tolerance) {
    if (skipped)
      for (std::size_t j = u; j < n; ++j)
        if (x[j] == 0) x[j] = lambert_probability(d[j] - tau, alpha, logalpha);
    T total = T(u) * cap;
    for (std::size_t j = u; j < n; ++j) {
      x[j] = std::min(x[j], cap);
      total += x[j];
    }
    const T tolerance_mass = mass_relative_tolerance * std::max(total, s);
    if (!std::isfinite(total) || std::abs(total - s) > tolerance_mass)
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
  entropy_buffer<T> d(n);
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
  entropy_buffer<Wide> wide_v(n), wide_x(n), wide_result(n), wide_low(low ? n : 1);
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

}  // namespace detail

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
  detail::entropy_buffer<std::size_t> order(n);
  detail::entropy_buffer<T> x(n);
  const T mass_tolerance = T(64) * std::numeric_limits<T>::epsilon() * T(n);
  try {
    detail::topk_entropy_core(n, v, offset, k, alpha, result, normalizer, upper, order.data(),
                              x.data(), output_scale, mass_tolerance, free_out, low, offset_low);
  } catch (const std::runtime_error&) {
    // Cap ties and cancellation can exceed the root's working precision.
    // Retry only this exceptional path in a wider type (float in double,
    // double in long double where it is wider). The same mass check applies
    // to the retry; never return an infeasible block merely because the
    // active-set search revisits a set.
    using Wide = typename detail::wider<T>::type;
    if (std::numeric_limits<T>::digits >= std::numeric_limits<Wide>::digits) throw;
    detail::topk_entropy_wide<Wide>(n, v, offset, k, alpha, result, normalizer, upper, order.data(),
                                    output_scale, mass_tolerance, free_out, low, offset_low);
  }
}

template <typename T>
inline void topk_entropy_block(const std::vector<T>& v, std::size_t k, T alpha,
                               std::vector<T>& result, T* normalizer = nullptr, T* upper = nullptr,
                               T offset = 0, T output_scale = 1) {
  result.resize(v.size());
  topk_entropy_block(v.size(), v.data(), k, alpha, result.data(), normalizer, upper, offset,
                     output_scale);
}

}  // namespace sdca

#endif
