#ifndef SDCA_MATH_DISTILLATION_H
#define SDCA_MATH_DISTILLATION_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include "sdca/math/compensated.h"
#include "sdca/math/entropy.h"
#include "sdca/math/lambert.h"
#include "sdca/math/prox_rows.h"

namespace sdca {
namespace detail {

// The double work of a row: the loss, the log-norms of q and p, the sum of u
// and the pivot (the first largest p + q); finite: whether every logit is.
struct distillation_row {
  double loss, s_lognorm, t_lognorm, u_sum;
  std::size_t pivot;
  bool finite;
};

// exp(x): the inlined polynomial where the result is normal.
inline double exp_double(double x) {
  return x >= -708 && x <= 709 ? detail::exp_normal(x) : std::exp(x);
}

// (x - y) - (u - v) with the rounding errors of both differences.
inline double offset_difference(double x, double y, double u, double v) {
  double ea, eb;
  const double a = detail::two_sum(x, -y, ea), b = detail::two_sum(u, -v, eb);
  return (a - b) + (ea - eb);
}

// The centered scaled logits of a row, (z - max) / T: the gaps that overflow
// before a large temperature are scaled first. Returns the first maximum's
// index; low receives the smallest scaled logit.
inline std::size_t center_row(std::size_t c, const double* z, double maximum, double temperature,
                              double* out, double& low) {
  std::size_t best = 0;
  low = std::numeric_limits<double>::infinity();
  for (std::size_t j = 0; j < c; ++j) {
    const double d = z[j] - maximum;
    out[j] = temperature > 1 && !std::isfinite(d) ? z[j] / temperature - maximum / temperature
                                                  : d / temperature;
    if (out[j] > out[best]) best = j;
    low = std::min(low, out[j]);
  }
  return best;
}

// The log of the normalizer sum exp(x) of centered logits x whose maximum is at
// best: log1p of the tail, which keeps a small tail; e[j] = exp(x[j]) and sum
// receives the normalizer.
inline double log_normalizer(std::size_t c, const double* x, std::size_t best, bool normal,
                             double* e, double& sum) {
  double top = -std::numeric_limits<double>::infinity();
  for (std::size_t j = 0; j < c; ++j)
    if (j != best && x[j] > top) top = x[j];
  const double origin = std::isfinite(top) ? top : 0;
  detail::compensated_sum<double> all, rest;
  for (std::size_t j = 0; j < c; ++j) {
    e[j] = exp_double(x[j]);
    all.add(e[j]);
    // Tails below the normal range relative to their largest term.
    if (j != best) rest.add(normal ? e[j] : std::exp(x[j] - origin));
  }
  sum = all.value();
  const double half = normal ? 1 : std::exp(top / 2);
  const double tail = half * (half * rest.value());
  return tail < std::numeric_limits<double>::epsilon() / 2 ? tail : std::log1p(tail);
}

// distillation_loss of a row in double from its widened logits cs and ct,
// with s, t, p, q (c each) receiving the centered scaled logits and the
// probabilities, cs the log-ratio log(q / p) and ct u = q - p. Row sums are
// compensated.
inline distillation_row distillation_double(std::size_t c, double temperature, double* cs,
                                            double* ct, double* s, double* t, double* p,
                                            double* q) {
  const double infinity = std::numeric_limits<double>::infinity();
  distillation_row row = {0, 0, 0, 0, 0, true};
  double s_max = -infinity, t_max = -infinity;
  for (std::size_t j = 0; j < c; ++j) {
    row.finite = row.finite && std::isfinite(cs[j]) && std::isfinite(ct[j]);
    s_max = std::max(s_max, cs[j]);
    t_max = std::max(t_max, ct[j]);
  }
  if (!row.finite) return row;
  double s_low, t_low, p_norm, q_norm;
  const std::size_t s_idx = center_row(c, cs, s_max, temperature, s, s_low);
  const std::size_t t_idx = center_row(c, ct, t_max, temperature, t, t_low);
  const bool normal = s_low >= -708 && t_low >= -708;
  row.s_lognorm = log_normalizer(c, s, s_idx, normal, q, q_norm);
  row.t_lognorm = log_normalizer(c, t, t_idx, normal, p, p_norm);
  detail::compensated_sum<double> loss, correction, u_sum;
  // Weighted means: their rounding cancels in the ratio, so they stay plain sums.
  double p_sum = 0, mean_num = 0, relative_num = 0;
  for (std::size_t j = 0; j < c; ++j) {
    p[j] /= p_norm;
    q[j] /= q_norm;
    if (p[j] > 0) {
      loss.add(p[j] * (row.s_lognorm - s[j]));
      p_sum += p[j];
      mean_num += p[j] * offset_difference(cs[j], ct[j], s_max, t_max);
      relative_num += p[j] * (s[j] - t[j]);
    }
  }
  const double mean = mean_num / p_sum;
  // An overflowing raw student/teacher offset (T > 1): centered scaled logits
  // remove that common offset instead.
  const bool over = temperature > 1 && !std::isfinite(mean);
  const double relative = over ? relative_num / p_sum : 0;
  const double shift = over ? -relative - row.t_lognorm : -mean / temperature - row.t_lognorm;
  for (std::size_t j = 0; j < c; ++j) {
    const double dn = over ? (s[j] - t[j]) - relative
                           : (offset_difference(cs[j], ct[j], s_max, t_max) - mean) / temperature;
    cs[j] = dn;
    // exp(log(p) + dn) = exp(s + shift): from the student where dn is not small.
    correction.add(std::fabs(dn) < 1 ? p[j] * std::expm1(dn) : exp_double(s[j] + shift) - p[j]);
  }
  const double log1p_correction = std::log1p(correction.value());
  double top = -1;
  for (std::size_t j = 0; j < c; ++j) {
    const double ratio = cs[j] - log1p_correction;
    cs[j] = ratio;
    ct[j] = std::fabs(ratio) < 1 ? p[j] * std::expm1(ratio) : q[j] - p[j];
    u_sum.add(ct[j]);
    if (p[j] + q[j] > top) top = p[j] + q[j], row.pivot = j;
  }
  row.loss = loss.value();
  row.u_sum = u_sum.value();
  return row;
}

// The double gradient of a row times scale into g: (u - q sum(u)) / T, every
// entry formed on its own; one rounding of the scale over T where finite.
inline void distillation_double_gradient(std::size_t c, const distillation_row& row,
                                         const double* u, const double* q, double temperature,
                                         double scale, double* g) {
  for (std::size_t j = 0; j < c; ++j) g[j] = u[j] - q[j] * row.u_sum;
  const double factor = scale / temperature;
  const bool fits = std::isfinite(factor);
  for (std::size_t j = 0; j < c; ++j) g[j] = fits ? g[j] * factor : g[j] / temperature * scale;
}

// The probability exp(h - W(alpha exp(h))) of a proximal step: exp(h) where W
// is negligible.
inline double prox_probability(double h, double alpha, double log_alpha) {
  const double arg = h + log_alpha;
  if (detail::lambert_negligible(arg, h)) return std::exp(h);
  const double w = lambert_w_exp(arg);
  return arg < -18 ? std::exp(h - w) : w / alpha;
}

// The threshold omega of a proximal row and its residuals r = p - pi: safeguarded Halley
// steps on sum(r(omega)) = 0, each step's residuals solved row_lanes at a time
// (detail::residual_kernel inside the Halley window |r| < p / 8, detail::probability_kernel
// outside it) from the previous step's moved along their slopes pi / (1 + alpha pi). Where
// alpha (max u - min u) <= 1 the steps start at log sum q exp(alpha (p - pi)) for the
// first-order pi = q exp(alpha (p - q) - omega0) (omega0 the same sum at pi = q), with the
// residual guesses of that step. Within the tolerance the residuals move to the Newton root
// omega - f / f'; near it the Taylor model of the probabilities about the last evaluation
// (detail::model_coefficients) gives the root and its residuals without another one.
// lanes: 5 row_lanes doubles; model: 4 c doubles; index: 2 c entries.
struct prox_solve {
  std::size_t c;
  double alpha, log_alpha;
  const double *ratio, *u, *log_p, *p, *q, *score;
  double *r, *slope, *lanes, *model;
  std::size_t* index;

  struct sums {
    double f, magnitude, d1, d2;
    bool clipped;
  };

  // The model's arrays: the weights 1 / (1 + alpha pi) and -pi of the last evaluation, its
  // residuals at the root and the flags of those to solve again.
  double* weights() const { return model; }
  double* neg_pi() const { return model + c; }
  double* next_r() const { return model + 2 * c; }
  unsigned char* flags() const { return reinterpret_cast<unsigned char*>(model + 3 * c); }

  // The residuals at omega (guesses from those at previous where finite) and their sums.
  sums evaluate(double omega, double previous) const {
    const double tiny = std::numeric_limits<double>::min();
    std::size_t* halley = index;
    std::size_t* other = index + c;
    std::size_t nh = 0, no = 0;
    for (std::size_t j = 0; j < c; ++j) {
      // The Halley window where |r| < p / 8 can hold (|arg| < (2 + alpha p) / 8); outside it
      // p - pi keeps r to within 9 rounding errors of pi.
      const double arg = omega - ratio[j];
      const bool window = (p[j] > tiny) & (std::fabs(arg) < (2 + alpha * p[j]) * 0.125);
      halley[nh] = j;
      other[no] = j;
      nh += window;
      no += !window;
    }
    const std::size_t L = detail::row_lanes;
    double *t = lanes, *pl = lanes + L, *lq = lanes + 2 * L, *guess = lanes + 3 * L;
    double* out = lanes + 4 * L;
    const double moved = omega - previous;
    for (std::size_t k = 0; k < nh; k += L) {
      const std::size_t count = nh - k < L ? nh - k : L;
      for (std::size_t l = 0; l < L; ++l) {
        const std::size_t j = halley[k + (l < count ? l : 0)];
        t[l] = omega - ratio[j], pl[l] = p[j], lq[l] = log_p[j];
        guess[l] = r[j] + moved * slope[j];
      }
      detail::residual_kernel(count, t, pl, lq, guess, alpha, out);
      for (std::size_t l = 0; l < count; ++l) {
        const std::size_t j = halley[k + l];
        const double pi = p[j] - out[l];
        r[j] = out[l];
        slope[j] = pi / (1 + alpha * pi);
      }
    }
    bool clipped = false;
    for (std::size_t k = 0; k < no; k += L) {
      const std::size_t count = no - k < L ? no - k : L;
      for (std::size_t l = 0; l < count; ++l) t[l] = score[other[k + l]];
      // h = score - omega.
      detail::probability_kernel(count, t, 0.0, omega, 0.0, alpha, log_alpha, out);
      for (std::size_t l = 0; l < count; ++l) {
        const std::size_t j = other[k + l];
        clipped |= out[l] > 1.0;
        const double pi = std::min(out[l], 1.0);
        // A zero argument has a zero residual, also where p is subnormal.
        r[j] = omega - ratio[j] == 0 ? 0.0 : p[j] - pi;
        slope[j] = pi / (1 + alpha * pi);
      }
    }
    detail::compensated_sum<double> f;
    double magnitude = 0, d1 = 0, d2 = 0;
    double *w = weights(), *minus = neg_pi();
    for (std::size_t j = 0; j < c; ++j) {
      const double pi = p[j] - r[j], uj = 1 / (1 + alpha * pi);
      w[j] = uj, minus[j] = -pi;
      f.add(r[j]);
      magnitude += std::fabs(r[j]);
      d1 += slope[j];
      d2 -= (pi * uj) * (uj * uj);
    }
    const sums result = {f.value(), magnitude, d1, d2, clipped};
    return result;
  }

  // The model's slope in d (at the coefficients db of model_coefficients).
  static double model_slope(const double* moment, const double* db) {
    double total = 0;
    for (int m = detail::model_powers - 1; m > 0; --m) total -= db[m] * moment[m];
    return total;
  }

  // Newton steps on the model's residual sum f - sum_m b_m(d) moment_m from d: false where a
  // step is not finite.
  static bool model_newton(double f, const double* moment, double& d) {
    const double eps = std::numeric_limits<double>::epsilon();
    double b[detail::model_powers], db[detail::model_powers];
    for (int it = 0; it < 8; ++it) {
      detail::model_coefficients(d, b, db);
      double change = 0;
      for (int m = detail::model_powers - 1; m > 0; --m) change += b[m] * moment[m];
      const double step = (f - change) / model_slope(moment, db);
      if (!std::isfinite(step)) return false;
      d -= step;
      if (std::fabs(step) <= 1e-3 * eps * std::fabs(d)) break;
    }
    return true;
  }

  // The flagged residuals of out solved again at next: inside the Halley window by the
  // residual kernel from the model's guess, else from the probability; adds the change of
  // their magnitudes to size and returns the sum of out.
  double solve_flagged(double next, double* out, double& size) const {
    const double tiny = std::numeric_limits<double>::min();
    double *t = lanes, *pl = lanes + detail::row_lanes;
    double *lq = lanes + 2 * detail::row_lanes, *guess = lanes + 3 * detail::row_lanes;
    double* solved = lanes + 4 * detail::row_lanes;
    std::size_t* at = index;
    std::size_t count = 0;
    for (std::size_t j = 0; j < c; ++j) {
      if (!flags()[j]) continue;
      const double arg = next - ratio[j], half = alpha * p[j] / 2, old = out[j];
      if ((p[j] > tiny) & (arg <= half + 0.69314718055994530942) &
          (arg >= -half - 0.40546510810816438198)) {
        at[count++] = j;
      } else {
        const double pi = std::min(prox_probability(score[j] - next, alpha, log_alpha), 1.0);
        out[j] = arg == 0 ? 0.0 : p[j] - pi;
      }
      size += std::fabs(out[j]) - std::fabs(old);
    }
    for (std::size_t k = 0; k < count; k += detail::row_lanes) {
      const std::size_t n = count - k < detail::row_lanes ? count - k : detail::row_lanes;
      for (std::size_t l = 0; l < detail::row_lanes; ++l) {
        const std::size_t j = at[k + (l < n ? l : 0)];
        t[l] = next - ratio[j], pl[l] = p[j], lq[l] = log_p[j], guess[l] = out[j];
      }
      detail::residual_kernel(n, t, pl, lq, guess, alpha, solved);
      for (std::size_t l = 0; l < n; ++l) {
        const std::size_t j = at[k + l];
        size += std::fabs(solved[l]) - std::fabs(out[j]);
        out[j] = solved[l];
      }
    }
    detail::compensated_sum<double> again;
    for (std::size_t j = 0; j < c; ++j) again.add(out[j]);
    return again.value();
  }

  // The root of the Taylor model about the last evaluation at omega (sum f) from its step
  // delta: true with omega and r there where the model's residuals meet the tolerance (those
  // it would not keep to their relative accuracy solved again); else false with its step in
  // delta (where finite).
  bool model_root(double& omega, double& delta, double f, double lo, double hi) const {
    const double eps = std::numeric_limits<double>::epsilon();
    const double tiny = std::numeric_limits<double>::min(), reach = 0.015625;
    double moment[detail::model_powers] = {}, b[detail::model_powers], db[detail::model_powers];
    detail::model_moments_kernel(c, neg_pi(), weights(), 0.0, moment);
    double d = delta;
    if (!model_newton(f, moment, d)) return false;
    double* out = next_r();
    for (int attempt = 0; attempt < 3; ++attempt) {
      const double next = omega + d;
      if (!(lo <= next && next <= hi)) return false;
      delta = next - omega;
      if (!(std::fabs(delta) <= reach)) return false;
      d = delta;
      detail::model_coefficients(d, b, db);
      const double bound = std::pow(std::fabs(d) / reach, detail::model_order + 1);
      double sum_hi = 0, sum_lo = 0, size = 0;
      const std::size_t flagged = detail::model_residuals_kernel(
          c, r, neg_pi(), weights(), b, bound, out, &sum_hi, &sum_lo, &size, flags());
      if (flagged > 0) sum_hi = solve_flagged(next, out, size), sum_lo = 0;
      const double sum = sum_hi + sum_lo;
      if (std::fabs(sum) <= 8 * eps * std::max(size, tiny)) {
        std::copy(out, out + c, r);
        omega = next;
        return true;
      }
      d -= sum / model_slope(moment, db);
    }
    delta = d;
    return false;
  }

  // The start of the steps (0, or the fixed-point step where alpha (max u - min u) <= 1,
  // with its residual guesses in r).
  double start(double lo, double hi) const {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t j = 0; j < c; ++j) r[j] = slope[j] = nan;
    if (!(hi - lo <= 1)) return 0;
    detail::compensated_sum<double> first, second;
    for (std::size_t j = 0; j < c; ++j) first.add(q[j] * exp_double(-alpha * u[j]));
    const double omega0 = std::log(first.value());
    for (std::size_t j = 0; j < c; ++j) {
      const double pi = q[j] * exp_double(alpha * (p[j] - q[j]) - omega0);
      r[j] = q[j] * exp_double(alpha * (p[j] - pi));
      second.add(r[j]);
    }
    const double sum = second.value(), omega = std::log(sum);
    if (!(lo <= omega && omega <= hi)) {
      for (std::size_t j = 0; j < c; ++j) r[j] = nan;
      return 0;
    }
    for (std::size_t j = 0; j < c; ++j) r[j] = p[j] - r[j] / sum, slope[j] = 0;
    return omega;
  }

  // The next threshold of a safeguarded Halley step from omega (the bracket's midpoint where
  // the step leaves it), or omega where the steps stall.
  static double step(double omega, const sums& e, double lo, double hi) {
    const double newton = -e.f / e.d1, halley = (e.f * e.d2) / (2 * e.d1 * e.d1);
    double delta =
        std::isfinite(halley) ? newton / (1 - std::min(std::max(halley, -0.5), 0.5)) : newton;
    if (!(e.d1 > 0 && std::isfinite(e.d1) && std::isfinite(newton))) delta = 0;
    const double next = omega + delta;
    if (delta != 0 && next == omega) return omega;
    if (!(lo <= next && next <= hi && next != omega && std::isfinite(next)))
      return lo + (hi - lo) / 2;
    return next;
  }

  // The bracket of omega: -alpha max(u, 0) and -alpha min(u, 0).
  void bracket(double& lo, double& hi) const {
    double u_max = 0, u_min = 0;
    for (std::size_t j = 0; j < c; ++j) {
      u_max = std::max(u_max, u[j]);
      u_min = std::min(u_min, u[j]);
    }
    lo = -alpha * u_max, hi = -alpha * u_min;
    if (!(lo <= hi)) hi = lo;
  }

  // Within the tolerance: the residuals at the Newton root omega - f / f', to first order.
  void finish(const sums& e) const {
    const double delta = -e.f / e.d1;
    if (delta != 0 && std::isfinite(delta))
      for (std::size_t j = 0; j < c; ++j) r[j] += delta * slope[j];
  }

  // The threshold; leaves its residuals in r.
  double solve() const {
    const double eps = std::numeric_limits<double>::epsilon();
    const double tiny = std::numeric_limits<double>::min();
    double lo, hi;
    bracket(lo, hi);
    double omega = start(lo, hi), previous = omega;
    bool solved = lo == 0 && hi == 0;
    if (solved) evaluate(omega, previous);
    for (int it = 0; it < 100 && !solved; ++it) {
      const sums e = evaluate(omega, previous);
      solved = true;
      if (std::fabs(e.f) <= 8 * eps * std::max(e.magnitude, tiny)) {
        finish(e);
        break;
      }
      (e.f < 0 ? lo : hi) = omega;
      double next = step(omega, e, lo, hi);
      if (!e.clipped && std::fabs(next - omega) <= 2 * 0.015625) {
        double delta = next - omega;
        if (model_root(omega, delta, e.f, lo, hi)) break;
        if (std::isfinite(delta) && delta != 0 && lo <= omega + delta && omega + delta <= hi)
          next = omega + delta;
      }
      if (next == omega) break;
      previous = omega;
      omega = next;
      solved = false;
    }
    // The residuals at omega (the last evaluation's, unless the steps ran out).
    if (!solved) evaluate(omega, previous);
    return omega;
  }
};

// proximal_distillation_loss of a row from its distillation_double work, as
// _tensor_prox.proximal_distillation: the residuals r = p - pi at the
// threshold (into r, the pivot restoring their zero sum); returns the loss.
// work: 6 c + 5 row_lanes doubles, index: 2 c entries.
inline double proximal_distillation_row(std::size_t c, double alpha, const distillation_row& row,
                                        const double* ratio, const double* u, const double* s,
                                        double* t, const double* p, const double* q, double* r,
                                        double* work, std::size_t* index) {
  // t: log p, score: log q + alpha p (the Lambert inversion's h at omega = 0).
  double *slope = work, *score = work + c;
  for (std::size_t j = 0; j < c; ++j) {
    t[j] -= row.t_lognorm;
    score[j] = (s[j] - row.s_lognorm) + alpha * p[j];
  }
  const prox_solve solver = {c,
                             alpha,
                             std::log(alpha),
                             ratio,
                             u,
                             t,
                             p,
                             q,
                             score,
                             r,
                             slope,
                             work + 2 * c,
                             work + 2 * c + 5 * detail::row_lanes,
                             index};
  const double omega = solver.solve();
  detail::compensated_sum<double> others, cross, penalty;
  for (std::size_t j = 0; j < c; ++j)
    if (j != row.pivot) others.add(r[j]);
  r[row.pivot] = -others.value();
  for (std::size_t j = 0; j < c; ++j) {
    if (p[j] > 0) {
      const double h = score[j] - omega;
      // Far above a tiny p, -r / p overflows: log pi = log(p - r) directly.
      const double log_pi = r[j] > p[j] / 2    ? h - lambert_w_exp(h + solver.log_alpha)
                            : r[j] < -p[j] / 2 ? std::log(p[j] - r[j])
                                               : t[j] + std::log1p(-r[j] / p[j]);
      cross.add(-(p[j] * log_pi));
    }
    penalty.add((alpha * r[j]) * r[j]);
  }
  return cross.value() + penalty.value() / 2;
}

}  // namespace detail
}  // namespace sdca

#endif
