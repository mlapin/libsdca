// Smoke test of the top-k and multilabel rows (sdca/loss/topk.h, sdca/loss/multilabel.h):
// sdca::kernel::evaluate of every kind against naive long double references on random, tied,
// masked and invalid rows, in float and double storage and work, with negative upstream
// factors, in place, and with a label set as with a mask. The smoothed losses and top-k
// entropy are referenced by their definitions max_{p in S} <h, p> - gamma / 2 |p|^2 (resp.
// <a, p> + H(1 - sum p, p)): an inner bisection at a fixed mass s, a golden section over s.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include "sdca/kernel.h"

namespace {

// The references' type (x87 long double is slow under emulation: -DSDCA_SMOKE_REFERENCE=double).
#ifndef SDCA_SMOKE_REFERENCE
#define SDCA_SMOKE_REFERENCE long double
#endif
typedef SDCA_SMOKE_REFERENCE R;
const double kInf = std::numeric_limits<double>::infinity();
const double kNaN = std::numeric_limits<double>::quiet_NaN();
int failures = 0, checks = 0;

enum Kind { topk_ce = 1, multiclass = 2, alpha = 3, beta = 4, multilabel = 5 };
const char* const kNames[] = {"",
                              "topk_cross_entropy",
                              "multiclass_hinge",
                              "topk_hinge_alpha",
                              "topk_hinge_beta",
                              "multilabel_hinge",
                              "multilabel_cross_entropy",
                              "proximal_cross_entropy",
                              "distillation",
                              "proximal_distillation"};

struct Case {
  int kind;
  std::size_t k;
  double gamma;
};

void check(bool ok, const Case& c, const char* what, std::size_t n, std::size_t j, double got,
           double want) {
  ++checks;
  if (ok) return;
  if (++failures <= 30)
    std::printf("FAIL %s k=%zu gamma=%g %s: n %zu j %zu got %.17g want %.17g\n", kNames[c.kind],
                c.k, c.gamma, what, n, j, got, want);
}

// argmax of a concave f on [lo, hi] (golden section).
R golden(const std::function<R(R)>& f, R lo, R hi) {
  const R g = (std::sqrt(R(5)) - 1) / 2;
  R a = lo, b = hi, x = b - g * (b - a), y = a + g * (b - a), fx = f(x), fy = f(y);
  for (int it = 0; it < 120; ++it) {
    if (fx < fy) {
      a = x, x = y, fx = fy, y = a + g * (b - a), fy = f(y);
    } else {
      b = y, y = x, fy = fx, x = b - g * (b - a), fx = f(x);
    }
  }
  const R m = (a + b) / 2;
  if (f(lo) >= f(m) && f(lo) >= f(hi)) return lo;
  if (f(hi) >= f(m)) return hi;
  return m;
}

// The t where sum_j p_j(t) = s for p_j decreasing in t (bisection).
R solve_mass(const std::function<R(R)>& sum, R s, R lo, R hi) {
  for (int it = 0; it < 200; ++it) {
    const R t = (lo + hi) / 2;
    (sum(t) > s ? lo : hi) = t;
  }
  return (lo + hi) / 2;
}

// max_{p} <h, p> - gamma / 2 |p|^2 over {0 <= p_j <= cap, sum p = s}: the value, p into p.
R fixed(const std::vector<R>& h, R gamma, R cap, R s, std::vector<R>& p) {
  p.assign(h.size(), 0);
  if (h.empty() || !(s > 0)) return 0;
  R lo = h[0], hi = h[0];
  for (std::size_t j = 0; j < h.size(); ++j) lo = std::min(lo, h[j]), hi = std::max(hi, h[j]);
  auto entry = [&](R x, R t) { return std::min(std::max((x - t) / gamma, R(0)), cap); };
  const R t = solve_mass(
      [&](R tt) {
        R total = 0;
        for (std::size_t j = 0; j < h.size(); ++j) total += entry(h[j], tt);
        return total;
      },
      s, lo - gamma - 1, hi + 1);
  R v = 0;
  for (std::size_t j = 0; j < h.size(); ++j)
    p[j] = entry(h[j], t), v += p[j] * (h[j] - gamma * p[j] / 2);
  return v;
}

// fixed() maximized over s in [0, smax], the cap a function of s.
R smoothed(const std::vector<R>& h, R gamma, const std::function<R(R)>& cap, R smax,
           std::vector<R>& p) {
  p.assign(h.size(), 0);
  if (h.empty() || !(smax > 0)) return 0;
  std::vector<R> q;
  const R s = golden([&](R x) { return fixed(h, gamma, cap(x), x, q); }, 0, smax);
  return fixed(h, gamma, cap(s), s, p);
}

// max_{p} <a, p> + H(1 - s, p) over {0 <= p_j <= s / k, sum p = s}, maximized over s.
R topk_entropy(const std::vector<R>& a, std::size_t k, std::vector<R>& p) {
  p.assign(a.size(), 0);
  if (a.size() < k) return 0;
  R top = a[0];
  for (std::size_t j = 0; j < a.size(); ++j) top = std::max(top, a[j]);
  auto at = [&](R s, std::vector<R>& q) -> R {
    q.assign(a.size(), 0);
    if (!(s > 0)) return 0;
    const R c = s / R(k);
    auto entry = [&](R x, R t) { return std::min(c, std::exp(x - t)); };
    const R t = solve_mass(
        [&](R tt) {
          R total = 0;
          for (std::size_t j = 0; j < a.size(); ++j) total += entry(a[j], tt);
          return total;
        },
        s, top - 800, top + 800);
    R v = s < 1 ? -(1 - s) * std::log(1 - s) : 0;
    for (std::size_t j = 0; j < a.size(); ++j) {
      q[j] = entry(a[j], t);
      if (q[j] > 0) v += q[j] * (a[j] - std::log(q[j]));
    }
    return v;
  };
  std::vector<R> q;
  const R s = golden([&](R x) { return at(x, q); }, 0, 1 - R(1e-30));
  return at(s, p);
}

// k times the gradient of the sum of the k largest of h over k at kinks: 1 above the k-th, its
// remaining places split evenly among the entries equal to it, each times slope(h).
std::vector<R> topk_weights(const std::vector<R>& h, std::size_t k,
                            const std::function<R(R)>& slope) {
  std::vector<R> sorted(h);
  std::sort(sorted.begin(), sorted.end(), std::greater<R>());
  const R last = k <= sorted.size() ? sorted[k - 1] : -R(kInf);
  std::size_t above = 0, tied = 0;
  for (R x : h) above += x > last, tied += x == last;
  std::vector<R> w(h.size(), 0);
  for (std::size_t i = 0; i < h.size(); ++i)
    if (h[i] >= last) w[i] = (h[i] > last ? 1 : R(k - above) / R(tied)) * slope(h[i]);
  return w;
}

// The label balances all competitors, which can carry hundreds of equal small shares.
R sum_weights(const std::vector<R>& weights) {
  R sum = 0, correction = 0;
  for (R weight : weights) {
    const R term = weight - correction, next = sum + term;
    correction = (next - sum) - term;
    sum = next;
  }
  return sum;
}

// The reference value and gradient (scale 1) of a row; NaN for an invalid row. Kinks take the
// mean of their one-sided subgradients.
R reference(const Case& c, const std::vector<double>& z, int64_t y, const std::vector<uint8_t>& m,
            std::vector<R>& g) {
  const std::size_t n = z.size();
  g.assign(n, 0);
  for (std::size_t j = 0; j < n; ++j)
    if (z[j] != z[j] || z[j] == kInf) return kNaN;
  if (c.kind == multilabel) {
    std::vector<std::size_t> pos, neg, masked;
    for (std::size_t j = 0; j < n; ++j) {
      if (m[j]) {
        if (z[j] == -kInf) masked.push_back(j);
        pos.push_back(j);
      } else if (z[j] > -kInf) {
        neg.push_back(j);
      }
    }
    if (pos.empty() || pos.size() == n) return kNaN;
    if (neg.empty()) return masked.size() == pos.size() ? kNaN : 0;
    if (!masked.empty()) {
      // The limit as the masked positives go to -inf together: the whole mass on them.
      if (c.gamma == 0) {
        R hi = -kInf;
        std::size_t ties = 0;
        for (std::size_t j : neg) hi = std::max(hi, R(z[j]));
        for (std::size_t j : neg) ties += R(z[j]) == hi;
        for (std::size_t j : neg)
          if (R(z[j]) == hi) g[j] = 1 / R(ties);
        for (std::size_t i : masked) g[i] = -1 / R(masked.size());
        return kInf;
      }
      std::vector<R> b, w;
      for (std::size_t j : neg) b.push_back(R(z[j]));
      fixed(b, c.gamma, 1, 1, w);
      for (std::size_t j = 0; j < neg.size(); ++j) g[neg[j]] = w[j];
      for (std::size_t i : masked) g[i] = -1 / R(masked.size());
      return kInf;
    }
    if (c.gamma == 0) {
      R lo = kInf, hi = -kInf;
      std::size_t lows = 0, highs = 0;
      for (std::size_t i : pos) lo = std::min(lo, R(z[i]));
      for (std::size_t j : neg) hi = std::max(hi, R(z[j]));
      for (std::size_t i : pos) lows += R(z[i]) == lo;
      for (std::size_t j : neg) highs += R(z[j]) == hi;
      const R h = R(1) + hi - lo, f = h > 0 ? 1 : (h == 0 ? R(0.5) : 0);
      for (std::size_t i : pos)
        if (R(z[i]) == lo) g[i] = -f / R(lows);
      for (std::size_t j : neg)
        if (R(z[j]) == hi) g[j] = f / R(highs);
      return std::max(h, R(0));
    }
    // max over x on the positives and y on the negatives of equal mass s <= 1 of
    // <a, x> + <b, y> - gamma / 2 (|x|^2 + |y|^2), a_i = -z_i, b_j = 1 + z_j.
    std::vector<R> a, b, x, w;
    for (std::size_t i : pos) a.push_back(-R(z[i]));
    for (std::size_t j : neg) b.push_back(1 + R(z[j]));
    const R gamma = c.gamma;
    auto value = [&](R s) { return fixed(a, gamma, s, s, x) + fixed(b, gamma, s, s, w); };
    const R s = golden(value, 0, 1);
    const R v = value(s);
    for (std::size_t i = 0; i < pos.size(); ++i) g[pos[i]] = -x[i];
    for (std::size_t j = 0; j < neg.size(); ++j) g[neg[j]] = w[j];
    return v;
  }
  const std::size_t k = c.k;
  if (y < 0 || y >= int64_t(n) || k < 1 || k >= n) return kNaN;
  const R label = z[y];
  std::vector<std::size_t> idx;
  for (std::size_t j = 0; j < n; ++j)
    if (int64_t(j) != y && z[j] > -kInf) idx.push_back(j);
  if (!(label > -kInf) && !(c.kind == topk_ce && k == 1)) {
    // The limit as z_y goes to -inf over the r competitors left.
    const std::size_t r = idx.size();
    if (r == 0) return kNaN;
    if ((c.kind == alpha || c.kind == topk_ce) && k > 1 && r < k) return 0;
    std::vector<R> a, p;
    for (std::size_t j : idx) a.push_back(R(z[j]));
    const R K = R(k);
    if (c.kind == topk_ce) {
      R top = a[0];
      for (R x : a) top = std::max(top, x);
      auto entry = [&](R x, R t) { return std::min(1 / K, std::exp(x - t)); };
      const R t = solve_mass(
          [&](R tt) {
            R total = 0;
            for (R x : a) total += entry(x, tt);
            return total;
          },
          1, top - 800, top + 800);
      for (R x : a) p.push_back(entry(x, t));
    } else if (c.gamma > 0 && r <= k) {
      p.assign(r, 1 / K);
    } else if (c.gamma > 0) {
      fixed(a, c.gamma, 1 / K, 1, p);
    } else {
      p = topk_weights(a, k, [](R) { return R(1); });
      for (std::size_t i = 0; i < r; ++i) g[idx[i]] = p[i] / K;
      g[y] = -sum_weights(p) / K;
      return kInf;
    }
    R total = 0;
    for (std::size_t i = 0; i < r; ++i) g[idx[i]] = p[i], total += p[i];
    g[y] = -total;
    return kInf;
  }
  if (c.kind == topk_ce && k == 1 && !(label > -kInf)) {
    if (idx.empty()) return kNaN;
    R mx = -kInf, sum = 0;
    for (std::size_t j : idx) mx = std::max(mx, R(z[j]));
    for (std::size_t j : idx) sum += std::exp(R(z[j]) - mx);
    for (std::size_t j : idx) g[j] = std::exp(R(z[j]) - mx) / sum;
    g[y] = -1;
    return kInf;
  }
  if (c.kind == topk_ce) {
    std::vector<R> a, p;
    for (std::size_t j : idx) a.push_back(R(z[j]) - label);
    const R v = topk_entropy(a, k, p);
    R total = 0;
    for (std::size_t i = 0; i < idx.size(); ++i) g[idx[i]] = p[i], total += p[i];
    g[y] = -total;
    return v;
  }
  const bool is_beta = c.kind != alpha || k == 1;
  std::vector<R> h;
  for (std::size_t j : idx) h.push_back(1 + R(z[j]) - label);
  if (c.gamma > 0) {
    std::vector<R> p;
    const R K = R(k);
    const R v =
        is_beta ? smoothed(
                      h, c.gamma, [=](R) { return 1 / K; }, std::min(R(1), R(h.size()) / K), p)
                : smoothed(h, c.gamma, [=](R s) { return s / K; }, h.size() >= k ? R(1) : R(0), p);
    R total = 0;
    for (std::size_t i = 0; i < idx.size(); ++i) g[idx[i]] = p[i], total += p[i];
    g[y] = -total;
    return v;
  }
  std::vector<R> scores;
  for (std::size_t j : idx) scores.push_back(R(z[j]));
  std::vector<R> sorted(scores);
  std::sort(sorted.begin(), sorted.end(), std::greater<R>());
  auto margin = [&](R score) { return (score - label) + 1; };
  R total = 0;
  for (std::size_t i = 0; i < k; ++i) {
    const R hi = i < sorted.size() ? margin(sorted[i]) : -R(kInf);
    total += (is_beta ? std::max(hi, R(0)) : hi) / R(k);
  }
  const R v = is_beta ? total : std::max(total, R(0));
  const R factor = total > 0 ? 1 : (total == 0 ? R(0.5) : 0);
  std::vector<R> w = topk_weights(scores, k, [&](R score) {
    const R x = margin(score);
    return is_beta ? (x > 0 ? 1 : (x == 0 ? R(0.5) : 0)) : factor;
  });
  if (k == 1 && !sorted.empty() && margin(sorted[0]) == 0) {
    // The largest margin ties with the hinge's zero (as optax): 1 / (tied + 1) each.
    const auto tied = std::count(scores.begin(), scores.end(), sorted[0]);
    for (std::size_t i = 0; i < scores.size(); ++i)
      w[i] = scores[i] == sorted[0] ? 1 / R(tied + 1) : 0;
  }
  for (std::size_t i = 0; i < idx.size(); ++i) g[idx[i]] = w[i] / R(k);
  g[y] = -sum_weights(w) / R(k);
  return v;
}

std::vector<uint8_t> mask_of(std::size_t n, std::mt19937& rng, int64_t y) {
  std::vector<uint8_t> m(n, 0);
  std::size_t count = 0;
  for (std::size_t j = 0; j < n; ++j) m[j] = rng() % 4 == 0, count += m[j];
  if (count == 0) m[std::size_t(y)] = 1;
  if (count == n) m[0] = 0;
  return m;
}

// One row of storage S in work precision (1 float, 2 double), upstream factor scale.
template <typename S>
void run_row(const Case& c, const std::vector<double>& z, int64_t y, const std::vector<uint8_t>& m,
             int precision, double scale, bool in_place, bool sets) {
  const std::size_t n = z.size();
  std::vector<S> zs(n);
  std::vector<double> zw(n);
  for (std::size_t j = 0; j < n; ++j) zs[j] = S(z[j]), zw[j] = double(zs[j]);
  std::vector<R> want;
  const R value = reference(c, zw, y, m, want);
  std::vector<S> grad(n, S(7));
  S* out = in_place ? zs.data() : grad.data();
  const sdca::kernel::Loss loss = {c.kind, c.k, c.gamma, 1.0, 0.0};
  const sdca::kernel::Execution e = {precision, 1, false};
  auto evaluate = [&](const S* in, S* to, double& v) {
    if (c.kind == multilabel && sets) {
      std::vector<int64_t> labels, offsets(2, 0);
      for (std::size_t j = 0; j < n; ++j)
        if (m[j]) labels.push_back(int64_t(j));
      offsets[1] = int64_t(labels.size());
      const sdca::kernel::LabelSets t = {labels.data(), offsets.data()};
      return sdca::kernel::evaluate(loss, e, 1, n, in, t, &v, to, 1.0, &scale);
    }
    if (c.kind == multilabel) {
      const sdca::kernel::Mask t = {m.data()};
      return sdca::kernel::evaluate(loss, e, 1, n, in, t, &v, to, 1.0, &scale);
    }
    const sdca::kernel::Labels t = {&y, 0, 0, false};
    return sdca::kernel::evaluate(loss, e, 1, n, in, t, &v, to, 1.0, &scale);
  };
  double got = -1;
  const sdca::kernel::Result r = evaluate(zs.data(), out, got);
  check(r.status == 0, c, "status", n, 0, r.status, 0);
  const bool invalid = value != value;
  check(r.invalid_rows == (invalid ? 1u : 0u), c, "invalid count", n, 0, double(r.invalid_rows),
        invalid);
  if (invalid) {
    check(got != got, c, "invalid value", n, 0, got, kNaN);
    for (std::size_t j = 0; j < n; ++j)
      check(out[j] != out[j], c, "invalid gradient", n, j, double(out[j]), kNaN);
    return;
  }
  const bool fwork = precision == 1;
  const double vtol = (fwork ? 2e-5 : (c.gamma > 0 || c.kind == topk_ce ? 1e-8 : 1e-13)) *
                      std::max(1.0, std::fabs(double(value)));
  if (value == kInf)
    check(got == kInf, c, "value", n, 0, got, double(value));
  else
    check(std::fabs(got - double(value)) <= vtol, c, "value", n, 0, got, double(value));
  const double gtol = std::max(fwork ? 2e-5 : (c.gamma > 0 || c.kind == topk_ce ? 2e-6 : 1e-15),
                               2.0 * double(std::numeric_limits<S>::epsilon()));
  double sum = 0, top = 0;
  for (std::size_t j = 0; j < n; ++j)
    sum += double(out[j]), top = std::max(top, std::fabs(double(out[j])));
  check(std::fabs(sum) <= 1e-6 * std::max(1.0, top) * std::fabs(scale), c, "gradient sum", n, 0,
        sum, 0);
  for (std::size_t j = 0; j < n; ++j) {
    const double gj = double(out[j]);
    if (gj == 0) check(!std::signbit(gj), c, "+0.0", n, j, gj, 0);
    const double w = double(want[j]) * scale;
    check(std::fabs(gj - w) <= gtol * std::fabs(scale) * std::max(1.0, top / std::fabs(scale)), c,
          "gradient", n, j, gj, w);
  }
  if (value == kInf && sizeof(S) == sizeof(double) && !in_place) {
    // A finite stand-in for the masked label (positives) far below the others gives the same
    // gradient.
    double low = kInf;
    for (std::size_t j = 0; j < n; ++j)
      if (zw[j] > -kInf) low = std::min(low, zw[j]);
    std::vector<S> zf(zs);
    for (std::size_t j = 0; j < n; ++j)
      if (zw[j] == -kInf && (c.kind == multilabel ? m[j] != 0 : int64_t(j) == y))
        zf[j] = S(low - 1e4 - 20 * c.gamma);
    std::vector<S> gf(n, S(7));
    double vf = -1;
    evaluate(zf.data(), gf.data(), vf);
    check(vf > 1e3 && vf < kInf, c, "stand-in value", n, 0, vf, 1e4);
    for (std::size_t j = 0; j < n; ++j)
      check(std::fabs(double(gf[j]) - double(out[j])) <= 1e-9 * std::fabs(scale), c,
            "stand-in gradient", n, j, double(gf[j]), double(out[j]));
  }
}

// 16-bit storage H with float gradients equals float storage of the same values.
template <typename H>
void run_row16(const Case& c, const std::vector<double>& z, int64_t y,
               const std::vector<uint8_t>& m) {
  const std::size_t n = z.size();
  std::vector<H> zh(n);
  std::vector<float> zf(n), gh(n, 7), gf(n, 7);
  for (std::size_t j = 0; j < n; ++j)
    zh[j] = sdca::native::core::to_storage<H>(z[j]),
    zf[j] = sdca::native::core::to_work<float>(zh[j]);
  const sdca::kernel::Loss loss = {c.kind, c.k, c.gamma, 1.0, 0.0};
  const sdca::kernel::Execution e = {0, 1, false};
  double vh = 0, vf = 0;
  const double scale = -0.5;
  if (c.kind == multilabel) {
    const sdca::kernel::Mask t = {m.data()};
    sdca::kernel::evaluate(loss, e, 1, n, zh.data(), t, &vh, gh.data(), 1.0, &scale);
    sdca::kernel::evaluate(loss, e, 1, n, zf.data(), t, &vf, gf.data(), 1.0, &scale);
  } else {
    const sdca::kernel::Labels t = {&y, 0, 0, false};
    sdca::kernel::evaluate(loss, e, 1, n, zh.data(), t, &vh, gh.data(), 1.0, &scale);
    sdca::kernel::evaluate(loss, e, 1, n, zf.data(), t, &vf, gf.data(), 1.0, &scale);
  }
  check(vh == vf || (vh != vh && vf != vf), c, "16-bit value", n, 0, vh, vf);
  for (std::size_t j = 0; j < n; ++j)
    check(gh[j] == gf[j] || (gh[j] != gh[j] && gf[j] != gf[j]), c, "16-bit gradient", n, j,
          double(gh[j]), double(gf[j]));
}

// The step's probabilities pi_j(t) = W(alpha exp(v_j - t)) / alpha (alpha pi + log pi = v - t)
// with sum pi = 1, by bisection on t.
std::vector<R> step_probabilities(const std::vector<R>& v, R alpha) {
  auto pi = [&](R x, R t) {
    const R l = std::log(alpha) + x - t;  // w + log w = l, w = alpha pi
    R w = l > 1 ? l - std::log(l) : std::exp(l);
    if (l > -40)
      for (int it = 0; it < 60; ++it) w -= (w + std::log(w) - l) / (1 + 1 / w);
    return w / alpha;
  };
  R top = v[0];
  for (R x : v) top = std::max(top, x);
  const R t = solve_mass(
      [&](R tt) {
        R total = 0;
        for (R x : v) total += pi(x, tt);
        return total;
      },
      1, top - 2 * alpha - 1000, top + 1000);
  std::vector<R> p;
  for (R x : v) p.push_back(pi(x, t));
  return p;
}

// Masked labels, positives and student classes of the cross-entropy losses: multilabel
// cross-entropy (6), proximal cross-entropy (7), distillation (8) and proximal distillation (9),
// against the limits, and a finite stand-in far below for the masked classes.
void masked_entropy_rows(std::mt19937& rng) {
  std::normal_distribution<double> normal(0.0, 1.0);
  for (int kind = 6; kind <= 9; ++kind) {
    for (std::size_t n : {2, 3, 6, 17, 40}) {
      for (int rep = 0; rep < 12; ++rep) {
        const double gamma = rep % 3 == 0 ? 0.5 : (rep % 3 == 1 ? 4.0 : 32.0);
        const double temperature = rep % 2 ? 2.0 : 1.0, spread = rep % 4 < 2 ? 1.0 : 6.0;
        std::vector<double> z(n), t(n);
        std::vector<uint8_t> m(n, 0);
        for (std::size_t j = 0; j < n; ++j) z[j] = spread * normal(rng), t[j] = normal(rng);
        const std::size_t y = rng() % n, other = (y + 1 + rng() % (n - 1)) % n;
        // Masked classes beside the one that makes the loss infinite; one class left.
        for (std::size_t j = 0; j < n; ++j)
          if (j != y && j != other && rng() % 3 == 0) z[j] = -kInf;
        z[y] = -kInf;
        for (std::size_t j = 0; j < n; ++j) m[j] = j == y || rng() % 3 == 0;
        if (kind >= 8)
          for (std::size_t j = 0; j < n; ++j)
            if (j != y && z[j] == -kInf && rng() % 2) t[j] = -kInf;
        // The limit (scale 1).
        std::vector<R> g(n, 0);
        std::vector<std::size_t> kept;
        for (std::size_t j = 0; j < n; ++j)
          if (z[j] > -kInf) kept.push_back(j);
        if (kind == 6 || kind == 8) {
          const R T = kind == 8 ? temperature : 1;
          R top = -kInf, sum = 0;
          for (std::size_t j : kept) top = std::max(top, R(z[j]) / T);
          for (std::size_t j : kept) sum += std::exp(R(z[j]) / T - top);
          std::vector<R> target(n, 0);
          if (kind == 6) {
            std::size_t count = 0;
            for (std::size_t j = 0; j < n; ++j) count += m[j];
            for (std::size_t j = 0; j < n; ++j) target[j] = m[j] ? 1 / R(count) : 0;
          } else {
            R tt = -kInf, ts = 0;
            for (std::size_t j = 0; j < n; ++j) tt = std::max(tt, R(t[j]) / T);
            for (std::size_t j = 0; j < n; ++j) ts += std::exp(R(t[j]) / T - tt);
            for (std::size_t j = 0; j < n; ++j) target[j] = std::exp(R(t[j]) / T - tt) / ts;
          }
          for (std::size_t j = 0; j < n; ++j)
            g[j] = ((z[j] > -kInf ? std::exp(R(z[j]) / T - top) / sum : 0) - target[j]) / T;
        } else {
          const R T = kind == 9 ? temperature : 1;
          std::vector<R> v, p(n, 0);
          if (kind == 9) {
            R tt = -kInf, ts = 0;
            for (std::size_t j = 0; j < n; ++j) tt = std::max(tt, R(t[j]) / T);
            for (std::size_t j = 0; j < n; ++j) ts += std::exp(R(t[j]) / T - tt);
            for (std::size_t j = 0; j < n; ++j) p[j] = std::exp(R(t[j]) / T - tt) / ts;
          } else {
            p[y] = 1;
          }
          for (std::size_t j : kept) v.push_back(R(z[j]) / T + R(gamma) * p[j]);
          const std::vector<R> pi = step_probabilities(v, R(gamma));
          for (std::size_t j = 0; j < n; ++j) g[j] = -p[j] / T;
          for (std::size_t i = 0; i < kept.size(); ++i) g[kept[i]] += pi[i] / T;
        }
        const double scale = rep % 2 ? -0.75 : 1.0;
        const sdca::kernel::Loss loss = {kind, 1, gamma, temperature, 0.0};
        const sdca::kernel::Execution e = {rep % 3 == 2 ? 2 : 0, 1, false};
        // Double storage, and float storage (float work, rerun in double where it asks).
        auto evaluate = [&](const std::vector<double>& in, std::vector<double>& out, double& v,
                            bool single) {
          out.assign(n, 7);
          const int64_t label = int64_t(y);
          std::vector<float> fin(in.begin(), in.end()), ft(t.begin(), t.end()), fout(n, 7);
          sdca::kernel::Result result;
          if (kind == 6) {
            const sdca::kernel::Mask target = {m.data()};
            result = single ? sdca::kernel::evaluate(loss, e, 1, n, fin.data(), target, &v,
                                                     fout.data(), scale)
                            : sdca::kernel::evaluate(loss, e, 1, n, in.data(), target, &v,
                                                     out.data(), scale);
          } else if (kind == 7) {
            const sdca::kernel::Labels target = {&label, 0, 0, false};
            result = single ? sdca::kernel::evaluate(loss, e, 1, n, fin.data(), target, &v,
                                                     fout.data(), scale)
                            : sdca::kernel::evaluate(loss, e, 1, n, in.data(), target, &v,
                                                     out.data(), scale);
          } else if (single) {
            const sdca::kernel::Teacher<float> target = {ft.data()};
            result =
                sdca::kernel::evaluate(loss, e, 1, n, fin.data(), target, &v, fout.data(), scale);
          } else {
            const sdca::kernel::Teacher<double> target = {t.data()};
            result =
                sdca::kernel::evaluate(loss, e, 1, n, in.data(), target, &v, out.data(), scale);
          }
          if (single) out.assign(fout.begin(), fout.end());
          return result;
        };
        std::vector<double> out, finite_out;
        double value = 0, finite_value = 0;
        const Case named = {kind, n, gamma};
        double top = 0;
        for (std::size_t j = 0; j < n; ++j) top = std::max(top, std::fabs(double(g[j])));
        for (int single = 1; single >= 0; --single) {
          const sdca::kernel::Result r = evaluate(z, out, value, single != 0);
          check(r.status == 0 && r.invalid_rows == 0 && value == kInf, named, "value", n, 0, value,
                kInf);
          const double tol = single ? 2e-5 : 1e-9;
          for (std::size_t j = 0; j < n; ++j) {
            const double w = double(g[j]) * scale;
            check(std::fabs(out[j] - w) <= tol * std::max(1.0, top), named, "gradient", n, j,
                  out[j], w);
            if (out[j] == 0) check(!std::signbit(out[j]), named, "+0.0", n, j, out[j], 0);
          }
        }
        // In place (the gradient over the scores): the same gradient.
        {
          std::vector<double> zi(z);
          double vi = 0;
          const int64_t label = int64_t(y);
          if (kind == 6) {
            const sdca::kernel::Mask target = {m.data()};
            sdca::kernel::evaluate(loss, e, 1, n, zi.data(), target, &vi, zi.data(), scale);
          } else if (kind == 7) {
            const sdca::kernel::Labels target = {&label, 0, 0, false};
            sdca::kernel::evaluate(loss, e, 1, n, zi.data(), target, &vi, zi.data(), scale);
          } else {
            const sdca::kernel::Teacher<double> target = {t.data()};
            sdca::kernel::evaluate(loss, e, 1, n, zi.data(), target, &vi, zi.data(), scale);
          }
          check(vi == value, named, "in-place value", n, 0, vi, value);
          for (std::size_t j = 0; j < n; ++j)
            check(zi[j] == out[j], named, "in-place gradient", n, j, zi[j], out[j]);
        }
        // A finite stand-in for the classes that make the loss infinite.
        double low = kInf;
        for (std::size_t j : kept) low = std::min(low, z[j]);
        std::vector<double> zf(z);
        for (std::size_t j = 0; j < n; ++j) {
          const bool infinite = kind == 6 ? m[j] != 0 : (kind == 7 ? j == y : t[j] > -kInf);
          if (z[j] == -kInf && infinite) zf[j] = low - (50 + 3 * gamma) * temperature;
        }
        evaluate(zf, finite_out, finite_value, false);
        check(finite_value > 0 && finite_value < kInf, named, "stand-in value", n, 0, finite_value,
              50);
        for (std::size_t j = 0; j < n; ++j)
          check(std::fabs(finite_out[j] - out[j]) <= 1e-9 * std::max(1.0, top), named,
                "stand-in gradient", n, j, finite_out[j], out[j]);
      }
    }
  }
}

}  // namespace

int main() {
  std::mt19937 rng(20261119);
  std::normal_distribution<double> normal(0.0, 1.0);
  const std::size_t sizes[] = {2, 3, 5, 8, 17, 33, 100, 300};
  std::vector<Case> cases;
  const double gammas[] = {0.0, 0.5, 3.0};
  // Ties in the same SIMD stripe, another stripe, and the partial last vector.
  // A zero-margin tie also shares the subgradient with the constant zero.
  for (double label : {0.0, 1.0}) {
    std::vector<double> z(67, -2.0);
    z[1] = z[17] = z[49] = z[66] = 0.0;
    z[35] = label;
    const std::vector<uint8_t> mask(z.size(), 0);
    for (Kind kind : {multiclass, alpha, beta})
      run_row<float>(Case{kind, 1, 0.0}, z, 35, mask, 1, -0.75, true, false);
  }
  const std::size_t ks[] = {1, 2, 3, 7, 70};
  for (double gamma : gammas) {
    cases.push_back(Case{multiclass, 1, gamma});
    cases.push_back(Case{multilabel, 1, gamma});
    for (std::size_t k : ks) {
      cases.push_back(Case{alpha, k, gamma});
      cases.push_back(Case{beta, k, gamma});
    }
  }
  for (std::size_t k : ks) cases.push_back(Case{topk_ce, k, 0.0});
  for (const Case& c : cases) {
    for (std::size_t n : sizes) {
      if (c.kind != multilabel && c.k >= n) continue;
      const int reps = 3;
      for (int rep = 0; rep < reps; ++rep) {
        for (int shape = 0; shape < 6; ++shape) {
          std::vector<double> z(n);
          const int64_t y = int64_t(rng() % n);
          const double spread = rep == 0 ? 1.0 : (rep == 1 ? 0.2 : 8.0);
          for (std::size_t j = 0; j < n; ++j) z[j] = spread * normal(rng);
          if (shape == 1)  // ties
            for (std::size_t j = 0; j < n; ++j) z[j] = double(int(rng() % 4)) * 0.5;
          if (shape == 2)  // masked classes
            for (std::size_t j = 0; j < n; ++j)
              if (int64_t(j) != y && rng() % 3 == 0) z[j] = -kInf;
          if (shape == 3) {  // few competitors left (fewer than k for larger k)
            for (std::size_t j = 0; j < n; ++j)
              if (int64_t(j) != y && rng() % 8 != 0) z[j] = -kInf;
          }
          if (shape == 4) z[rng() % n] = rng() % 2 ? kNaN : kInf;  // invalid
          if (shape == 5) z[std::size_t(y)] = -kInf;               // masked label / positive
          const std::vector<uint8_t> m = mask_of(n, rng, y);
          const double scale = rep == 1 ? -0.75 : 1.0;
          run_row<double>(c, z, y, m, 2, scale, false, false);
          run_row<float>(c, z, y, m, 1, scale, false, rep == 2);
          run_row<float>(c, z, y, m, 2, scale, rep == 1, false);
          if (rep == 0) run_row<double>(c, z, y, m, 2, scale, true, true);
          if (rep == 0) run_row16<sdca::half>(c, z, y, m), run_row16<sdca::bfloat16>(c, z, y, m);
        }
      }
    }
    // Labels out of range.
    std::vector<double> z(5, 0.25);
    std::vector<uint8_t> m(5, 0);
    m[1] = 1;
    if (c.kind != multilabel) {
      run_row<double>(c, z, -1, m, 2, 1.0, false, false);
      run_row<double>(c, z, 5, m, 2, 1.0, false, false);
    } else {
      std::vector<uint8_t> none(5, 0), all(5, 1);
      run_row<double>(c, z, 0, none, 2, 1.0, false, false);
      run_row<double>(c, z, 0, all, 2, 1.0, false, true);
    }
  }
  masked_entropy_rows(rng);
  std::printf("%s: %d of %d checks failed\n", failures ? "FAILED" : "passed", failures, checks);
  return failures ? 1 : 0;
}
