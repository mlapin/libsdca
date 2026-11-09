#ifndef SDCA_MATH_PROX_ROWS_H
#define SDCA_MATH_PROX_ROWS_H

// Row kernels of the proximal losses: the probabilities W(alpha exp(h)) / alpha of a
// proximal step and the residuals of its labels, row_lanes entries at a time. As those of
// exp_rows.h they use separate IEEE operations (every ISA clone gives the same bits), with
// selects in place of the scalar functions' branches and the inlined exp_normal and
// log1p_lane in place of the library's exp and log1p; lanes outside their domain (results
// below double's normal range, large Lambert arguments) take the scalar functions.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include "sdca/math/compensated.h"
#include "sdca/math/exp_rows.h"
#include "sdca/math/lambert.h"

namespace sdca {

namespace detail {

// a if c, else b, through a bit mask (GCC does not if-convert double selects).
SDCA_ROW_INLINE double select_double(bool c, double a, double b) {
  const std::uint64_t mask = std::uint64_t(0) - std::uint64_t(c);
  return bits_double((double_bits(a) & mask) | (double_bits(b) & ~mask));
}

// log1p(x) for finite x > -1 within an ulp (musl's): 1 + x = 2^k (1 + f) with 1 + f in
// [sqrt(2) / 2, sqrt(2)), c the rounding error of 1 + x over it, and log(1 + f) by the
// polynomial in s = f / (2 + f). Where 1 + x is already in that range, f = x exactly.
SDCA_ROW_INLINE double log1p_lane(double x) {
  const double u = 1 + x;
  const std::uint64_t bits = double_bits(u);
  const std::uint64_t high = (bits >> 32) + (0x3ff00000u - 0x3fe6a09eu);
  const std::int64_t k = std::int64_t(high >> 20) - 0x3ff;
  const std::uint64_t reduced = (((high & 0x000fffffu) + 0x3fe6a09eu) << 32) | (bits & 0xffffffffu);
  const bool direct = k == 0;
  // double(k) through the bits of 1.5 * 2^52 + k (an int64-to-double conversion would keep
  // the loop scalar before AVX-512).
  const double dk = bits_double(std::uint64_t(k) + 0x4338000000000000u) - 6755399441055744.0;
  const double c0 = select_double(k >= 2, 1 - (u - x), x - (u - 1)) / u;
  const double c = select_double(direct | (k >= 54), 0.0, c0);
  const double f = select_double(direct, x, bits_double(reduced) - 1);
  const double hfsq = 0.5 * f * f, s = f / (2 + f), z = s * s, w = z * z;
  const double t1 = w * (3.999999999940941908e-01 +
                         w * (2.222219843214978396e-01 + w * 1.531383769920937332e-01));
  const double t2 = z * (6.666666666666735130e-01 +
                         w * (2.857142874366239149e-01 +
                              w * (1.818357216161805012e-01 + w * 1.479819860511658591e-01)));
  const double r = t2 + t1;
  return s * (hfsq + r) + (dk * 1.90821492927058770002e-10 + c) - hfsq + f +
         dk * 6.93147180369123816490e-01;
}

// lambert_w_exp(x) for x in [-708, 4], its branches as selects.
SDCA_ROW_INLINE double lambert_lane(double x) {
  const bool pos = x > 0, low = x <= -36, mid = !pos & (x > -20);
  double w = select_double(pos, lambert_w_iter_5(x, 1.0), exp_approx(x));
  w = select_double(mid, lambert_w_iter_5(w, exp_approx(x - w)), w);
  w = lambert_w_iter_5(w, exp_normal(select_double(low, 0.0, x - w)));
  return select_double(low, exp_normal(x), w);
}

// The probability exp(h - W(alpha exp(h))) = W / alpha of a proximal step (as
// tensor::prox_probability and entropy_block_solver::probability) where h + log(alpha) <= 4
// and h >= -700 (the lane domain).
SDCA_ROW_INLINE double probability_lane(double h, double alpha, double log_alpha) {
  const double arg = h + log_alpha;
  const double cut = -double(std::numeric_limits<double>::digits + 2) * 0.69314718055994530942;
  // exp(h) where W is negligible: a zero W takes the exp(h - W) branch.
  const bool negligible = (arg < cut) & (std::fabs(h) >= 0.5);
  const double w =
      select_double(negligible, 0.0, lambert_lane(select_double(negligible, 0.0, arg)));
  return select_double(arg < -18, exp_normal(select_double(arg < -18, h - w, 0.0)), w / alpha);
}

// W(y) / y = exp(-W(y)) for 0 <= y <= exp(-3.5) < 2^-5, by its series
// sum_n (-(n + 1))^n / (n + 1)! y^n, whose terms decay as (e y)^n: 15 are within 2^-56.
SDCA_ROW_INLINE double lambert_ratio_small(double y) {
  double g = 22324.3085127066;
  g = g * y - 9104.500241158019;
  g = g * y + 3741.4497029592385;
  g = g * y - 1551.1605194805195;
  g = g * y + 649.7871723434745;
  g = g * y - 275.5731922398589;
  g = g * y + 118.62522321428571;
  g = g * y - 52.01269841269841;
  g = g * y + 23.343055555555555;
  g = g * y - 10.8;
  g = g * y + 5.208333333333333;
  g = g * y - 2.6666666666666665;
  g = g * y + 1.5;
  g = g * y - 1.0;
  return g * y + 1.0;
}

// The probability of the scalar functions, for lanes outside the domain.
inline double probability_scalar(double h, double alpha, double log_alpha) {
  const double arg = h + log_alpha;
  const double cut = -double(std::numeric_limits<double>::digits + 2) * 0.69314718055994530942;
  if (arg < cut && std::fabs(h) >= 0.5) return std::exp(h);
  const double w = lambert_w_exp(arg);
  return arg < -18 ? std::exp(h - w) : w / alpha;
}

// prob[j] = probability(h[j]) with h[j] = ((v[j] - v_ref) - dt) + shift, j < n; blocks of
// row_lanes in the lane domain on lanes (by the series of W where nearly all are small), the
// others one at a time.
SDCA_ROW_INLINE void probability_lanes(std::size_t n, const double* v, double v_ref, double dt,
                                       double shift, double alpha, double log_alpha, double* prob) {
  for (std::size_t begin = 0; begin < n; begin += row_lanes) {
    const std::size_t count = n - begin < row_lanes ? n - begin : row_lanes;
    double h[row_lanes];
    double low = 0, high = -1000;
    std::size_t large = 0;
    for (std::size_t l = 0; l < count; ++l) {
      h[l] = ((v[begin + l] - v_ref) - dt) + shift;
      low = h[l] < low ? h[l] : low;
      high = h[l] > high ? h[l] : high;
      large += h[l] + log_alpha > -3.5;
    }
    double* out = prob + begin;
    if (low >= -700 && high <= 700 && large <= row_lanes / 8) {
      // W(alpha exp(h)) small but in a few lanes: exp(h) W(y) / y at y = alpha exp(h), the
      // few one at a time.
      for (std::size_t l = 0; l < count; ++l) {
        const double e = exp_normal(h[l]);
        out[l] = e * lambert_ratio_small(alpha * e);
      }
      for (std::size_t l = 0; large && l < count; ++l)
        if (h[l] + log_alpha > -3.5) out[l] = probability_scalar(h[l], alpha, log_alpha);
    } else if (low >= -700 && high + log_alpha <= 4) {
      for (std::size_t l = 0; l < count; ++l) out[l] = probability_lane(h[l], alpha, log_alpha);
    } else {
      for (std::size_t l = 0; l < count; ++l) out[l] = probability_scalar(h[l], alpha, log_alpha);
    }
  }
}

SDCA_ROW_KERNEL void probability_kernel(std::size_t n, const double* v, double v_ref, double dt,
                                        double shift, double alpha, double log_alpha,
                                        double* prob) {
  probability_lanes(n, v, v_ref, dt, shift, alpha, log_alpha, prob);
}

// The residuals r of alpha r - log1p(-r / q) = t of the lanes l < count inside the Halley
// window (|r| < q / 2: entropy_residual's iteration), each by the same safeguarded Halley
// steps from guess where it is finite (else the second-order estimate). Every input and r
// hold row_lanes entries (lanes from count on: any positive normal q, they stay put).
SDCA_ROW_INLINE void residual_lanes(std::size_t count, const double* t, const double* q,
                                    const double* log_q, const double* guess, double alpha,
                                    double* r) {
  double lo[row_lanes], hi[row_lanes], inv_q[row_lanes], tol[row_lanes];
  std::uint64_t run[row_lanes], at_q[row_lanes], floor[row_lanes];
  const double eps = std::numeric_limits<double>::epsilon();
  for (std::size_t l = 0; l < row_lanes; ++l) {
    const double ql = q[l], tl = t[l], gl = guess[l];
    lo[l] = ql - 1, hi[l] = ql, inv_q[l] = 1 / ql, tol[l] = 8 * eps * std::fabs(tl);
    const bool below_floor = tl <= alpha * lo[l] + log_q[l];
    floor[l] = below_floor;
    run[l] = (l < count) & (tl != 0) & !below_floor;
    at_q[l] = 0;
    const double linear = tl / (alpha + inv_q[l]);
    const double estimate = linear / (1 + (linear * inv_q[l]) / (2 * (1 + alpha * ql)));
    const double initial = select_double(gl - gl == 0, gl, estimate);
    // nextafter(q, 0) of a positive normal q.
    const double below_q = bits_double(double_bits(ql) - 1);
    const double clipped = select_double(initial < below_q, initial, below_q);
    r[l] = select_double(clipped > lo[l], clipped, lo[l]);
  }
  for (int it = 0; it < 100; ++it) {
    std::uint64_t any = 0;
    for (std::size_t l = 0; l < row_lanes; ++l) {
      const double ql = q[l], rl = r[l];
      const double f = (alpha * rl - log1p_lane(-rl * inv_q[l])) - t[l];
      bool going = (run[l] != 0) & !(std::fabs(f) <= tol[l]);
      const bool negative = f < 0;
      lo[l] = select_double(going & negative, rl, lo[l]);
      hi[l] = select_double(going & !negative, rl, hi[l]);
      const double inv_qr = 1 / (ql - rl);
      const double d1 = alpha + inv_qr;
      const double denom = d1 - (f * inv_qr * inv_qr) / (2 * d1);
      double next = select_double(denom > 0, rl - f / denom, rl - f / d1);
      going = going & (next != rl);
      next = select_double((lo[l] <= next) & (next <= hi[l]), next, lo[l] + (hi[l] - lo[l]) / 2);
      const bool reached = going & (next == ql);
      at_q[l] |= reached;
      going = going & !reached & (next != rl);
      r[l] = select_double(going, next, rl);
      run[l] = going;
      any |= run[l];
    }
    if (!any) break;
  }
  for (std::size_t l = 0; l < count; ++l) {
    r[l] = select_double(at_q[l] != 0, q[l], r[l]);
    r[l] = select_double(floor[l] != 0, q[l] - 1, r[l]);
    r[l] = select_double(t[l] == 0, 0.0, r[l]);
  }
}

SDCA_ROW_KERNEL void residual_kernel(std::size_t count, const double* t, const double* q,
                                     const double* log_q, const double* guess, double alpha,
                                     double* r) {
  residual_lanes(count, t, q, log_q, guess, alpha, r);
}

// The other classes' terms of the residual sum at their probabilities prob (in place: r =
// -min(prob, 1)): sums[0] += sum min(prob, 1), sums[1] += sum prob u and sums[2] -= sum prob u^3
// over the unclipped, u = 1 / (1 + alpha prob) into weights (0 where clipped). Whether any is.
SDCA_ROW_KERNEL bool others_terms_kernel(std::size_t n, double alpha, double* r, double* weights,
                                         double* sums) {
  double mass[row_lanes], d1[row_lanes], d2[row_lanes];
  std::uint64_t clipped[row_lanes];
  for (std::size_t l = 0; l < row_lanes; ++l) mass[l] = d1[l] = d2[l] = 0, clipped[l] = 0;
  for (std::size_t begin = 0; begin < n; begin += row_lanes) {
    const std::size_t count = n - begin < row_lanes ? n - begin : row_lanes;
    double out[row_lanes], u[row_lanes];
    for (std::size_t l = 0; l < row_lanes; ++l) {
      const double prob = l < count ? r[begin + l] : 0.0;
      const bool over = prob > 1.0;
      const double kept = select_double(over, 1.0, prob), ul = 1.0 / (1.0 + alpha * kept);
      u[l] = select_double(over, 0.0, ul);
      const double slope = kept * u[l];
      out[l] = -kept;
      mass[l] += kept;
      d1[l] += slope;
      d2[l] -= (slope * u[l]) * u[l];
      clipped[l] |= over;
    }
    for (std::size_t l = 0; l < count; ++l) r[begin + l] = out[l], weights[begin + l] = u[l];
  }
  std::uint64_t any = 0;
  for (std::size_t l = 0; l < row_lanes; ++l)
    sums[0] += mass[l], sums[1] += d1[l], sums[2] += d2[l], any |= clipped[l];
  return any != 0;
}

// The sum of exp(y_j) at y_j = (v_j - v_ref) + alpha (target_j - e_j / sum) (target_j = q for
// j < p, else 0): the fixed-point step of the start, compensated as exp_sum_lanes.
SDCA_ROW_KERNEL double fixed_point_kernel(std::size_t n, std::size_t p, const double* v,
                                          double v_ref, const double* e, double inv_sum,
                                          double alpha, double q) {
  double sum[row_lanes] = {}, carry[row_lanes] = {};
  for (std::size_t begin = 0; begin < n; begin += row_lanes) {
    const std::size_t count = n - begin < row_lanes ? n - begin : row_lanes;
    double terms[row_lanes];
    for (std::size_t l = 0; l < row_lanes; ++l) {
      const std::size_t j = begin + (l < count ? l : 0);
      const double y = (v[j] - v_ref) + alpha * ((j < p ? q : 0.0) - e[j] * inv_sum);
      terms[l] = l < count ? exp_term(y, false) : 0.0;
    }
    add_block(terms, sum, carry);
  }
  double total = 0, correction = 0;
  for (std::size_t l = 0; l < 2 * row_lanes; ++l) {
    const double y = (l < row_lanes ? sum[l] : -carry[l - row_lanes]) - correction;
    const double t = total + y;
    correction = (t - total) - y;
    total = t;
  }
  return total;
}

// The Taylor model of the proximal probabilities about an evaluated threshold
// (entropy_block_solver::model_root): at dt + d, pi (1 + s(d, u)) with u = 1 / (1 + alpha pi)
// and s = sum_k P_k(u) (-d)^k / k!, k <= 7 (d log W(e^x) / dx = u and du / dx = u^2 (u - 1):
// P_1 = u, P_{k+1} = u P_k + u^2 (u - 1) P_k'). Its remainder is below eps / 32 of pi for
// |d| <= model_reach (|P_8| <= 1.61 on [0, 1]); s and the residual sum are polynomials in u.
const int model_order = 7, model_powers = 14;
inline const double (&model_table())[model_order][model_powers] {
  // P_k(u), k = 1..7: the coefficients of u^m.
  static const double table[model_order][model_powers] = {
      {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, -2, 3, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 6, -20, 15, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, -24, 130, -210, 105, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 120, -924, 2380, -2520, 945, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, -720, 7308, -26432, 44100, -34650, 10395}};
  return table;
}

// The coefficients b of s(d, u) = sum_m b_m u^m and db of its derivative in d.
inline void model_coefficients(double d, double* b, double* db) {
  const double(&table)[model_order][model_powers] = model_table();
  double t[model_order + 1];
  t[0] = 1;
  for (int k = 1; k <= model_order; ++k) t[k] = t[k - 1] * -d / k;
  for (int m = 0; m < model_powers; ++m) {
    b[m] = db[m] = 0;
    for (int k = 1; k <= model_order; ++k) {
      b[m] += table[k - 1][m] * t[k];
      db[m] -= table[k - 1][m] * t[k - 1];
    }
  }
}

// moment[m] += sum_j pi_j u_j^m (m = 1..13) of pi = target - r over j < n.
SDCA_ROW_KERNEL void model_moments_kernel(std::size_t n, const double* r, const double* u,
                                          double target, double* moment) {
  double acc[model_powers][row_lanes];
  for (int m = 0; m < model_powers; ++m)
    for (std::size_t l = 0; l < row_lanes; ++l) acc[m][l] = 0;
  for (std::size_t begin = 0; begin < n; begin += row_lanes) {
    const std::size_t count = n - begin < row_lanes ? n - begin : row_lanes;
    double x[row_lanes], w[row_lanes];
    for (std::size_t l = 0; l < row_lanes; ++l) {
      x[l] = l < count ? target - r[begin + l] : 0.0;
      w[l] = l < count ? u[begin + l] : 0.0;
    }
    for (int m = 1; m < model_powers; ++m) {
      for (std::size_t l = 0; l < row_lanes; ++l) {
        x[l] *= w[l];
        acc[m][l] += x[l];
      }
    }
  }
  for (int m = 1; m < model_powers; ++m)
    for (std::size_t l = 0; l < row_lanes; ++l) moment[m] += acc[m][l];
}

// out_j = r_j - (target - r_j) s(u_j) over j < n for the coefficients b; adds sum(out) to sum
// and sum |out| to magnitude.
SDCA_ROW_KERNEL void model_expand_kernel(std::size_t n, const double* r, const double* u,
                                         double target, const double* b, double* out, double* sum,
                                         double* magnitude) {
  double total[row_lanes], size[row_lanes];
  for (std::size_t l = 0; l < row_lanes; ++l) total[l] = size[l] = 0;
  for (std::size_t begin = 0; begin < n; begin += row_lanes) {
    const std::size_t count = n - begin < row_lanes ? n - begin : row_lanes;
    double next[row_lanes];
    for (std::size_t l = 0; l < row_lanes; ++l) {
      const double rl = l < count ? r[begin + l] : target, w = l < count ? u[begin + l] : 0.0;
      double s = b[model_powers - 1];
      for (int m = model_powers - 2; m > 0; --m) s = s * w + b[m];
      next[l] = rl - (target - rl) * (s * w);
      total[l] += next[l] - (l < count ? 0.0 : target);
      size[l] += std::fabs(next[l]) - (l < count ? 0.0 : std::fabs(target));
    }
    for (std::size_t l = 0; l < count; ++l) out[begin + l] = next[l];
  }
  for (std::size_t l = 0; l < row_lanes; ++l) *sum += total[l], *magnitude += size[l];
}

// out_j = r_j - pi_j s(u_j) (pi = -neg_pi) over j < n for the coefficients b; the sum of out
// (with TwoSum per lane: hi, lo) and of |out|; flags_j = 1 where r_j would lose its relative
// accuracy (|pi_j s| > |r_j| / 2, or pi_j bound > |r_j| for the remainder pi bound eps / 32).
// The count of those.
SDCA_ROW_KERNEL std::size_t model_residuals_kernel(std::size_t n, const double* r,
                                                   const double* neg_pi, const double* u,
                                                   const double* b, double bound, double* out,
                                                   double* sum_hi, double* sum_lo,
                                                   double* magnitude, unsigned char* flags) {
  double hi[row_lanes], lo[row_lanes], size[row_lanes];
  std::size_t flagged = 0;
  for (std::size_t l = 0; l < row_lanes; ++l) hi[l] = lo[l] = size[l] = 0;
  for (std::size_t begin = 0; begin < n; begin += row_lanes) {
    const std::size_t count = n - begin < row_lanes ? n - begin : row_lanes;
    double next[row_lanes];
    unsigned char bad[row_lanes];
    for (std::size_t l = 0; l < row_lanes; ++l) {
      const bool in = l < count;
      const double rl = in ? r[begin + l] : 0.0, pl = in ? -neg_pi[begin + l] : 0.0;
      const double w = in ? u[begin + l] : 0.0;
      double s = b[model_powers - 1];
      for (int m = model_powers - 2; m > 0; --m) s = s * w + b[m];
      const double change = pl * (s * w);
      next[l] = rl - change;
      const double total = hi[l] + next[l], back = total - hi[l];
      lo[l] += (hi[l] - (total - back)) + (next[l] - back);
      hi[l] = total;
      size[l] += std::fabs(next[l]);
      bad[l] = !(std::fabs(change) <= std::fabs(rl) * 0.5) | !(pl * bound <= std::fabs(rl));
    }
    for (std::size_t l = 0; l < count; ++l) {
      out[begin + l] = next[l];
      flags[begin + l] = bad[l];
      flagged += bad[l];
    }
  }
  for (std::size_t l = 0; l < row_lanes; ++l) {
    double e;
    *sum_hi = two_sum(*sum_hi, hi[l], e);
    *sum_lo += e + lo[l];
    *magnitude += size[l];
  }
  return flagged;
}

}  // namespace detail

}  // namespace sdca

#endif
