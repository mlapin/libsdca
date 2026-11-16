#ifndef SDCA_CPU_PROX_DISTILL_H
#define SDCA_CPU_PROX_DISTILL_H
#include "_cpu_distill.h"
#include "_cpu_prox.h"

namespace {
namespace sdca_cpu {
// Shared SIMD proximal solve. Statistics and exceptional-row dispatch belong
// to the adapter; the numerical iterations and gradient are common.
template <typename G, bool kSum>
double prox_distillation_solve(int64_t c, double alpha, const DistillationStats& st, double* cs,
                               double* ct, double* s, double* t, double* p_arr, double* q_arr,
                               double* work, int32_t* index, G* grad_out) {
  using namespace dlanes;
  double *r = work, *slope = work + c, *score = work + 2 * c, *pi_arr = work + 3 * c;
  // The weights 1 / (1 + alpha pi) of the last evaluation and the model's residuals and
  // probabilities (proxlanes::Model).
  double *weights = work + 4 * c, *next_r = work + 5 * c, *next_pi = work + 6 * c;
  bool clipped = false;
  // The entries the model solves again.
  int32_t* fix = index;
  const double log_alpha = std::log(alpha);
  const double eps = std::numeric_limits<double>::epsilon();
  const double tiny = std::numeric_limits<double>::min();
  double u_max = 0.0, u_min = 0.0;
  for (int64_t j = 0; j < c; ++j) {
    u_max = std::max(u_max, ct[j]);
    u_min = std::min(u_min, ct[j]);
  }
  double lo = -alpha * u_max, hi = -alpha * u_min;
  if (!(lo <= hi)) hi = lo;
  // score: log q + alpha p (the Lambert inversion's h at omega = 0).
  const double shift_s = -st.s_lognorm;
  for (int64_t j = 0; j < c; j += L) {
    const int count = static_cast<int>(std::min<int64_t>(L, c - j));
    put(score + j, (part(s, j, count, 0.0) + shift_s) + alpha * part(p_arr, j, count, 0.0), count);
  }
  struct Sums {
    double f, magnitude, d1, d2;
  };
  using proxlanes::WD;
  constexpr int W = proxlanes::Shape<WD>::L;
  // The residuals r(omega), probabilities and slopes pi / den (den = 1 + alpha pi); the sums
  // of r, |r|, pi / den and -pi / den^3.
  auto evaluate = [&](double omega) -> Sums {
    // Every entry from its probability pi, r = p - pi: within 9 rounding errors of pi where
    // |r| >= p / 8, which |arg| >= (2 + alpha p) / 8 implies. Below (the Halley window), one
    // Newton step on alpha r - log1p(-r / p) = arg gives r to its relative accuracy: from p - pi
    // (error eps p, squared after the step: eps^2 p) where |arg| >= 2^-14, else from the
    // second-order estimate in arg (error |arg|^3 p, then |arg|^6 p).
    using proxlanes::part;
    using proxlanes::put;
    using proxlanes::select;
    using MW = proxlanes::MaskOf<WD>;
    clipped = false;
    for (int64_t j = 0; j < c; j += W) {
      const int count = static_cast<int>(std::min<int64_t>(W, c - j));
      // Padding lanes have a negligible probability.
      const WD h = part<WD>(score, j, count, -1e300), lr = part<WD>(cs, j, count, 0.0);
      const WD p = part<WD>(p_arr, j, count, 0.0), arg = omega - lr;
      const WD prob = proxlanes::prox_probability(h - omega, alpha, log_alpha);
      clipped |= any_lane(prob > 1.0);
      WD pi = proxlanes::min(prob, WD{} + 1.0), rj = p - pi;
      const MW window = (p > tiny) & (proxlanes::abs(arg) < (2.0 + alpha * p) * 0.125);
      if (any_lane(window)) {
        const WD q = select(window, p, WD{} + 1.0), a = select(window, arg, WD{});
        const WD linear = a / (alpha + 1.0 / q);
        const WD estimate = linear / (1.0 + (linear / q) / (2.0 * (1.0 + alpha * q)));
        const WD r0 = select(proxlanes::abs(a) < 0x1p-14, estimate, select(window, rj, WD{}));
        const WD g = (alpha * r0 - proxlanes::log1p(-r0 / q)) - a;
        const WD next = r0 - g / (alpha + 1.0 / (q - r0));
        rj = select(window, next, rj), pi = select(window, p - next, pi);
      }
      // A zero argument has a zero residual, also where p is subnormal.
      rj = select(arg == 0.0, WD{}, rj);
      const WD u = 1.0 / (1.0 + alpha * pi);
      put(r + j, rj, count), put(pi_arr + j, pi, count), put(weights + j, u, count);
      put(slope + j, pi * u, count);
    }
    Accumulator<kSum> sum_f;
    WD magnitude{}, d1{}, d2{};
    for (int64_t j = 0; j < c; j += W) {
      const int count = static_cast<int>(std::min<int64_t>(W, c - j));
      const WD rj = part<WD>(r, j, count, 0.0), slope_j = part<WD>(slope, j, count, 0.0);
      const WD u = part<WD>(weights, j, count, 0.0);
      for (int k = 0; k < proxlanes::kDoubleChains; ++k) sum_f.add(rj.v[k]);
      magnitude += proxlanes::abs(rj);
      d1 += slope_j;
      d2 -= (slope_j * u) * u;
    }
    return {sum_f.value(), proxlanes::total<double>(magnitude), proxlanes::total<double>(d1),
            proxlanes::total<double>(d2)};
  };
  double omega = 0.0;
  // Near the root, the Taylor model about the last evaluation at omega (residual sum f) gives
  // the root and its residuals without another evaluation: true with omega, r and pi_arr
  // there; else false with its step in delta (where finite). Entries whose residual it would
  // not keep to its relative accuracy are solved again (exactly) at the root.
  using Model = proxlanes::Model<double>;
  auto model_root = [&](double f, double& delta) -> bool {
    Model model;
    model.add_probabilities(pi_arr, weights, c);
    double d = delta;
    for (int it = 0; it < 8; ++it) {
      model.at(d);
      const double step = (f - model.change()) / model.slope();
      if (!std::isfinite(step)) return false;
      d -= step;
      if (std::abs(step) <= 1e-3 * eps * std::abs(d)) break;
    }
    for (int attempt = 0; attempt < 3; ++attempt) {
      const double next = omega + d;
      if (!(lo <= next && next <= hi)) return false;
      delta = next - omega;
      if (!(std::abs(delta) <= Model::reach)) return false;
      d = delta;
      model.at(d);
      const double bound = std::pow(std::abs(d) / Model::reach, proxlanes::kModelOrder + 1);
      Accumulator<kSum> total;
      double size = 0.0;
      const int64_t fixes = model.expand_probabilities(r, pi_arr, weights, c, bound, next_r,
                                                       next_pi, total, size, fix);
      double sum = total.value();
      if (fixes > 0) {
        for (int64_t k = 0; k < fixes; ++k) {
          const int64_t j = fix[k];
          const double p = p_arr[j], arg = next - cs[j], half = alpha * p / 2;
          double rj;
          if (p > tiny && arg <= half + 0.69314718055994530942 &&
              arg >= -half - 0.40546510810816438198) {
            rj = prox_residual(arg, alpha, p, log_alpha, t[j], next_r[j]);
            next_pi[j] = p - rj;
          } else {
            next_pi[j] = std::min(prox_probability(score[j] - next, alpha, log_alpha), 1.0);
            rj = arg == 0.0 ? 0.0 : p - next_pi[j];
          }
          size += std::abs(rj) - std::abs(next_r[j]);
          next_r[j] = rj;
        }
        Accumulator<kSum> again;
        for (int64_t j = 0; j < c; j += L)
          again.add(part(next_r, j, static_cast<int>(std::min<int64_t>(L, c - j)), 0.0));
        sum = again.value();
      }
      if (std::abs(sum) <= 8.0 * eps * std::max(size, tiny)) {
        std::copy(next_r, next_r + c, r);
        std::copy(next_pi, next_pi + c, pi_arr);
        omega = next;
        return true;
      }
      d -= sum / model.slope();
    }
    delta = d;
    return false;
  };
  if (hi - lo <= 1.0) {
    Accumulator<false> first, second;
    for (int64_t j = 0; j < c; j += L) {
      const int count = static_cast<int>(std::min<int64_t>(L, c - j));
      const vd q = part(q_arr, j, count, 0.0);
      first.add(q * exp(-alpha * part(ct, j, count, 0.0)));
    }
    const double omega0 = std::log(first.value());
    for (int64_t j = 0; j < c; j += L) {
      const int count = static_cast<int>(std::min<int64_t>(L, c - j));
      const vd q = part(q_arr, j, count, 0.0), p = part(p_arr, j, count, 0.0);
      const vd pi = q * exp(alpha * (p - q) - omega0);
      second.add(q * exp(alpha * (p - pi)));
    }
    const double start = std::log(second.value());
    if (lo <= start && start <= hi) omega = start;
  }
  bool solved = lo == 0.0 && hi == 0.0;
  if (solved) evaluate(omega);
  for (int it = 0; it < 100 && !solved; ++it) {
    const Sums e = evaluate(omega);
    solved = true;
    const double f = e.f;
    if (std::abs(f) <= 8.0 * eps * std::max(e.magnitude, tiny)) {
      // Within the tolerance: the residuals at the Newton root omega - f / d1, to first order.
      const double delta = -f / e.d1;
      if (delta != 0.0 && std::isfinite(delta)) {
        for (int64_t j = 0; j < c; j += L) {
          const int count = static_cast<int>(std::min<int64_t>(L, c - j));
          const vd moved_j = delta * part(slope, j, count, 0.0);
          put(r + j, part(r, j, count, 0.0) + moved_j, count);
          put(pi_arr + j, part(pi_arr, j, count, 0.0) - moved_j, count);
        }
      }
      break;
    }
    (f < 0.0 ? lo : hi) = omega;
    const double newton = -f / e.d1, corr = (f * e.d2) / (2.0 * e.d1 * e.d1);
    double delta =
        std::isfinite(corr) ? newton / (1.0 - std::min(std::max(corr, -0.5), 0.5)) : newton;
    if (!(e.d1 > 0.0 && std::isfinite(e.d1) && std::isfinite(newton))) delta = 0.0;
    if (!clipped && delta != 0.0 && std::abs(delta) <= 2 * Model::reach) {
      double root = delta;
      if (model_root(f, root)) break;
      if (std::isfinite(root) && root != 0.0) delta = root;
    }
    double next = omega + delta;
    if (delta != 0.0 && next == omega) break;
    if (!(lo <= next && next <= hi && next != omega && std::isfinite(next)))
      next = lo + (hi - lo) / 2;
    if (next == omega) break;
    omega = next;
    solved = false;
  }
  // The residuals at omega (the last evaluation's, unless the steps ran out).
  if (!solved) evaluate(omega);
  Accumulator<kSum> other_sum;
  auto add_others = [&](int64_t begin, int64_t end) {
    for (int64_t j = begin; j < end; j += L)
      other_sum.add(part(r, j, static_cast<int>(std::min<int64_t>(L, end - j)), 0.0));
  };
  add_others(0, st.pivot);
  add_others(st.pivot + 1, c);
  r[st.pivot] = -other_sum.value();
  Accumulator<kSum> cross, penalty;
  for (int64_t j = 0; j < c; j += W) {
    const int count = static_cast<int>(std::min<int64_t>(W, c - j));
    using proxlanes::select;
    using MW = proxlanes::MaskOf<WD>;
    const WD p = proxlanes::part<WD>(p_arr, j, count, 0.0);
    const WD rj = proxlanes::part<WD>(r, j, count, 0.0), zero{};
    const MW positive = proxlanes::valid<WD>(count) & (p > 0.0);
    // log pi: log p + log1p(-r / p) where |r| <= p / 2 (pi near p, whose logarithm would lose
    // the small -r / p), else the logarithm of the solved probability (accurate where r is
    // not small: a Lambert inversion's, or p - r), below double's normal range h - W.
    const WD pi = proxlanes::part<WD>(pi_arr, j, count, 1.0);
    const MW near = positive & ~(rj > p / 2.0) & ~(rj < -p / 2.0);
    const MW tiny_pi = positive & ~near & ~(pi >= std::numeric_limits<double>::min());
    const MW logarithm = positive & ~near & ~tiny_pi;
    WD log_pi = zero;
    if (any_lane(logarithm)) log_pi = proxlanes::log(select(logarithm, pi, zero + 1.0));
    if (any_lane(near))
      log_pi = select(
          near,
          proxlanes::part<WD>(t, j, count, 0.0) + proxlanes::log1p(select(near, -rj / p, zero)),
          log_pi);
    if (any_lane(tiny_pi)) {
      const WD h = proxlanes::part<WD>(score, j, count, 0.0) - omega;
      const WD w = proxlanes::lambert_w_exp(select(tiny_pi, h + log_alpha, zero));
      log_pi = select(tiny_pi, h - w, log_pi);
    }
    const WD terms = select(positive, -(p * log_pi), zero), squares = (alpha * rj) * rj;
    for (int k = 0; k < proxlanes::kDoubleChains; ++k)
      cross.add(terms.v[k]), penalty.add(squares.v[k]);
  }
  for (int64_t j = 0; j < c; ++j) grad_out[j] = static_cast<G>(-r[j] / st.temp);
  return cross.value() + penalty.value() / 2;
}

}  // namespace sdca_cpu
}  // namespace
#endif
