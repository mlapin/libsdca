// Smoke test of float work on the rows its former retries to double work took: far tails
// (gaps 80 to 110), upstream factors beyond float, targets and parameters beyond float,
// against double work on the same (exact) inputs: |g - ref| <= 64 eps |ref| + eps FLT_MIN (+ 64 eps (p + q) |scale| / T for
// distillation), an infinity of the reference's sign where it rounds beyond float, values
// within 64 eps |ref| (+ 64 eps for rows of a near-zero loss).
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "sdca/kernel.h"

namespace {

int failures = 0;
const double kInf = std::numeric_limits<double>::infinity();
const double kFloatMax = double(FLT_MAX), kFloatMin = double(FLT_MIN);

struct Rows {
  std::size_t n, c;
  std::vector<float> z, t;  // scores; teacher logits or probabilities
  std::vector<int64_t> y;
};

template <typename S, typename T>
sdca::kernel::Result run(const sdca::kernel::Loss& loss, int precision, const Rows& r,
                         const std::vector<S>& z, const std::vector<T>& t, double upstream,
                         std::vector<double>& v, std::vector<S>& g) {
  const sdca::kernel::Execution e = {precision, 1, false};
  v.assign(r.n, 0), g.assign(r.n * r.c, 0);
  if (loss.kind == sdca::kernel::distillation) {
    const sdca::kernel::Teacher<S> teacher = {reinterpret_cast<const S*>(t.data())};
    return sdca::kernel::evaluate(loss, e, r.n, r.c, z.data(), teacher, v.data(), g.data(),
                                  upstream);
  }
  if (!r.t.empty()) {
    const sdca::kernel::Probabilities<T> p = {t.data(), 0};
    return sdca::kernel::evaluate(loss, e, r.n, r.c, z.data(), p, v.data(), g.data(), upstream);
  }
  const sdca::kernel::Labels l = {r.y.data(), 0, 0, false};
  return sdca::kernel::evaluate(loss, e, r.n, r.c, z.data(), l, v.data(), g.data(), upstream);
}

void check(const char* name, const sdca::kernel::Loss& loss, const Rows& r, double upstream,
           bool random = false) {
  std::vector<double> zd(r.z.begin(), r.z.end()), td(r.t.begin(), r.t.end()), vf, vd;
  std::vector<float> gf;
  std::vector<double> gd;
  run(loss, 0, r, r.z, r.t, upstream, vf, gf);
  run(loss, 0, r, zd, td, upstream, vd, gd);
  const double eps = double(FLT_EPSILON), T = loss.temperature;
  double worst = 0, worst_value = 0;
  for (std::size_t i = 0; i < r.n; ++i) {
    const double vb = 64 * eps * (std::fabs(vd[i]) + 1e-30);
    if (!(std::fabs(vf[i] - vd[i]) <= vb))
      worst_value = std::fmax(worst_value, std::fabs(vf[i] - vd[i]) / vb);
    std::vector<double> p(r.c), q(r.c);
    if (loss.kind == sdca::kernel::distillation) {
      double zm = -kInf, tm = -kInf, zs = 0, ts = 0;
      for (std::size_t j = 0; j < r.c; ++j)
        zm = std::fmax(zm, zd[i * r.c + j]), tm = std::fmax(tm, td[i * r.c + j]);
      for (std::size_t j = 0; j < r.c; ++j)
        zs += std::exp((zd[i * r.c + j] - zm) / T), ts += std::exp((td[i * r.c + j] - tm) / T);
      for (std::size_t j = 0; j < r.c; ++j)
        q[j] = std::exp((zd[i * r.c + j] - zm) / T) / zs,
        p[j] = std::exp((td[i * r.c + j] - tm) / T) / ts;
    }
    // Random rows: also 64 eps of the row's largest entry, times sqrt(C / 1000) from 1000
    // classes (the ordinary sweeps of tests/python/conformance/contract.py).
    double row_allowance = 0;
    if (random) {
      for (std::size_t j = 0; j < r.c; ++j)
        row_allowance = std::fmax(row_allowance, std::fabs(gd[i * r.c + j]));
      row_allowance *= 64 * eps * std::fmax(1.0, std::sqrt(double(r.c) / 1000));
    }
    for (std::size_t j = 0; j < r.c; ++j) {
      const double want = gd[i * r.c + j], got = double(gf[i * r.c + j]);
      double ratio;
      if (std::fabs(want) > kFloatMax * (1 + eps / 2)) {
        ratio = std::isinf(got) && (got > 0) == (want > 0) ? 0 : kInf;
      } else {
        const double bound = 64 * eps * std::fabs(want) + eps * kFloatMin + row_allowance +
                             64 * eps * (p[j] + q[j]) / T * std::fabs(upstream);
        ratio = std::fabs(got - want) / bound;
        if (!(ratio == ratio)) ratio = kInf;
      }
      if (ratio > worst) worst = ratio;
    }
  }
  const bool ok = worst <= 1 && worst_value <= 1;
  if (!ok) ++failures;
  if (!ok)
    std::printf("FAIL %s: gradient error/bound %.3g, value error/bound %.3g\n", name, worst,
                worst_value);
}

}  // namespace

int main() {
  std::mt19937 rng(5);
  std::normal_distribution<double> normal(0, 1);
  const sdca::kernel::Loss ce = {sdca::kernel::cross_entropy, 1, 1.0, 1.0, 0.0};
  const int gaps[] = {80, 87, 90, 95, 100, 104, 106, 110};
  const std::size_t widths[] = {10, 33, 1000};
  char name[128];
  // Labels: the label at 0, the competitors at -gap; then random rows at upstreams beyond float.
  for (std::size_t c : widths) {
    Rows r = {8, c, {}, {}, {}};
    for (int gap : gaps) {
      r.z.assign(r.n * c, float(-gap)), r.y.assign(r.n, 0);
      for (std::size_t i = 0; i < r.n; ++i) r.z[i * c + i % c] = 0, r.y[i] = int64_t(i % c);
      for (double up : {1.0, 1e10, 4.0, 1e-10}) {
        std::snprintf(name, sizeof name, "ce labels far tail %d c %zu up %g", gap, c, up);
        if (up == 1.0 || gap == 100) check(name, ce, r, up);
      }
    }
    r.z.resize(r.n * c);
    for (float& x : r.z) x = float(3 * normal(rng));
    for (double up : {1e39, 1e44, 1e60, 1e-25}) {
      std::snprintf(name, sizeof name, "ce labels random c %zu up %g", c, up);
      check(name, ce, r, up, true);
    }
  }
  // Probabilities: the target (one-hot, or halves at 0 and 1) at 0, the others at -gap.
  for (std::size_t c : widths) {
    Rows r = {8, c, {}, {}, {}};
    for (int gap : gaps)
      for (int two = 0; two < 2; ++two) {
        r.z.assign(r.n * c, float(-gap)), r.t.assign(r.n * c, 0.0f);
        for (std::size_t i = 0; i < r.n; ++i) {
          r.z[i * c] = 0;
          if (two)
            r.t[i * c] = 0.5f, r.t[i * c + 1] = 0.5f;
          else
            r.t[i * c] = 1.0f;
        }
        for (double up : {1.0, 1e10, 4.0}) {
          std::snprintf(name, sizeof name, "ce probs far tail %d %s c %zu up %g", gap,
                        two ? "two" : "one", c, up);
          if (up == 1.0 || gap == 100) check(name, ce, r, up);
        }
      }
    r.z.resize(r.n * c), r.t.resize(r.n * c);
    for (std::size_t i = 0; i < r.n; ++i) {
      double total = 0;
      for (std::size_t j = 0; j < c; ++j) {
        r.z[i * c + j] = float(3 * normal(rng)), r.t[i * c + j] = float(std::fabs(normal(rng)));
        total += double(r.t[i * c + j]);
      }
      for (std::size_t j = 0; j < c; ++j) r.t[i * c + j] = float(double(r.t[i * c + j]) / total);
    }
    for (double up : {1e39, 1e44, 1e-25}) {
      std::snprintf(name, sizeof name, "ce probs random c %zu up %g", c, up);
      check(name, ce, r, up, true);
    }
    for (float& x : r.t) x = x > 0.2f / float(c) ? 3e38f : x;
    std::snprintf(name, sizeof name, "ce probs huge targets c %zu", c);
    check(name, ce, r, 1.0);
  }
  // Distillation: student [0, -T gs ...], teacher [0, -T gt ...]; upstreams and temperatures
  // beyond float.
  const int pairs[][2] = {{80, 90},   {90, 80},   {100, 100}, {104, 104},
                          {100, 110}, {110, 100}, {110, 111}};
  for (double T : {0.5, 1.0, 2.0, 4.0})
    for (std::size_t c : widths) {
      sdca::kernel::Loss d = {sdca::kernel::distillation, 1, 1.0, T, 0.0};
      Rows r = {7, c, {}, {}, {}};
      r.z.assign(r.n * c, 0), r.t.assign(r.n * c, 0);
      for (std::size_t i = 0; i < r.n; ++i)
        for (std::size_t j = 1; j < c; ++j)
          r.z[i * c + j] = float(-T * pairs[i][0]), r.t[i * c + j] = float(-T * pairs[i][1]);
      for (double up : {1.0, 1e10, 1e-25}) {
        std::snprintf(name, sizeof name, "distill far tail T %g c %zu up %g", T, c, up);
        check(name, d, r, up);
      }
      for (std::size_t i = 0; i < r.n * c; ++i)
        r.z[i] = float(3 * normal(rng)), r.t[i] = float(3 * normal(rng));
      for (double up : {1e39, 1e45}) {
        std::snprintf(name, sizeof name, "distill random T %g c %zu up %g", T, c, up);
        check(name, d, r, up, true);
      }
    }
  // Proximal cross-entropy: gamma beyond float (per call), upstream beyond float; top-k
  // cross-entropy at an upstream whose root exceeds float (the retry kept).
  for (std::size_t c : widths) {
    Rows r = {8, c, {}, {}, {}};
    r.z.resize(r.n * c), r.y.resize(r.n);
    for (float& x : r.z) x = float(3 * normal(rng));
    for (std::size_t i = 0; i < r.n; ++i) r.y[i] = int64_t(rng() % c);
    for (double gamma : {1e-39, 1e39, 1.0}) {
      const sdca::kernel::Loss px = {sdca::kernel::proximal_cross_entropy, 1, gamma, 1.0, 0.0};
      for (double up : {1.0, 1e40}) {
        if (gamma != 1.0 && up != 1.0) continue;
        std::snprintf(name, sizeof name, "proximal ce gamma %g c %zu up %g", gamma, c, up);
        check(name, px, r, up);
      }
    }
    const sdca::kernel::Loss tk = {sdca::kernel::topk_cross_entropy, 5, 1.0, 1.0, 0.0};
    for (double up : {1.0, 1e80}) {
      std::snprintf(name, sizeof name, "topk ce k 5 c %zu up %g", c, up);
      check(name, tk, r, up);
    }
  }
  std::printf("%s (%d failing cases)\n", failures ? "FAILED" : "ok", failures);
  return failures ? 1 : 0;
}
