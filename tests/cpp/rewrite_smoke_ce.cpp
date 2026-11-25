// Smoke test of the core lanes and the cross-entropy rows: sdca::kernel::evaluate against a
// naive long double reference on random rows, masked rows and invalid rows, in float, double
// and 16-bit storage, float and double work, one and several threads.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "sdca/kernel.h"
#include "sdca/core/select.h"

namespace {

using sdca::bfloat16;
using sdca::half;
using sdca::native::core::to_storage;
using sdca::native::core::to_work;
typedef long double R;
const double kInf = std::numeric_limits<double>::infinity();
int failures = 0;
int test_isa = 0;  // 0: run-time choice; else the namespace of that instruction set

template <typename S, typename G, typename T>
sdca::kernel::Result eval(const sdca::kernel::Loss& loss, const sdca::kernel::Execution& e,
                          std::size_t n, std::size_t c, const S* scores, const T& t, double* values,
                          G* grads, double upstream = 1.0, const double* rows = 0) {
  switch (test_isa) {
#if defined(__x86_64__)
    case 1:
      return sdca::isa_scalar::core::evaluate_kind(loss, e, n, c, scores, t, values, grads,
                                                   upstream, rows);
    case 3:
      return sdca::isa_avx2::core::evaluate_kind(loss, e, n, c, scores, t, values, grads, upstream,
                                                 rows);
    case 4:
      return sdca::isa_avx512::core::evaluate_kind(loss, e, n, c, scores, t, values, grads,
                                                   upstream, rows);
#endif
    default:
      return sdca::kernel::evaluate(loss, e, n, c, scores, t, values, grads, upstream, rows);
  }
}

const double kNaN = std::numeric_limits<double>::quiet_NaN();

void check(bool ok, const char* what, std::size_t c, std::size_t row, double got, double want) {
  if (ok) return;
  if (++failures <= 20)
    std::printf("FAIL %s: classes %zu row %zu: got %.17g want %.17g\n", what, c, row, got, want);
}

// The reference row of target weights u: value and gradient T softmax(z) - u times scale.
void reference(const std::vector<double>& z, const std::vector<R>& u, double scale, R& value,
               std::vector<R>& g) {
  const std::size_t n = z.size();
  R m = -kInf, total = 0, sum = 0;
  for (std::size_t j = 0; j < n; ++j) m = std::max<R>(m, z[j]), total += u[j];
  for (std::size_t j = 0; j < n; ++j) sum += std::exp(R(z[j]) - m);
  const R lse = m + std::log(sum);
  value = 0;
  g.assign(n, 0);
  for (std::size_t j = 0; j < n; ++j) {
    if (u[j] != 0) value += u[j] * (lse - z[j]);
    g[j] = (total * std::exp(R(z[j]) - lse) - u[j]) * scale;
  }
}

template <typename S>
std::vector<S> stored(const std::vector<double>& x) {
  std::vector<S> s(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) s[i] = to_storage<S>(x[i]);
  return s;
}
template <typename S>
std::vector<double> widened(const std::vector<S>& s) {
  std::vector<double> x(s.size());
  for (std::size_t i = 0; i < s.size(); ++i) x[i] = to_work<double>(s[i]);
  return x;
}

enum Kind { kLabels, kSmoothed, kProbabilities, kMultilabel };

template <typename S, typename G>
void run(Kind kind, std::size_t c, int precision, double value_tol, double grad_tol,
         std::mt19937& rng) {
  const std::size_t n = 37;
  std::normal_distribution<double> normal(0.0, 3.0);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::vector<double> z(n * c);
  for (double& x : z) x = normal(rng);
  std::vector<int64_t> labels(n);
  std::vector<uint8_t> mask(n * c, 0);
  std::vector<double> probs(n * c), weights(c);
  for (double& w : weights) w = 0.5 + uniform(rng);
  for (std::size_t i = 0; i < n; ++i) {
    labels[i] = static_cast<int64_t>(rng() % c);
    mask[i * c + static_cast<std::size_t>(labels[i])] = 1;
    mask[i * c + rng() % c] = 1;
    for (std::size_t j = 0; j < c; ++j) probs[i * c + j] = uniform(rng) < 0.3 ? 0.0 : uniform(rng);
  }
  // Masked classes in rows 1 to 4 away from the targets; row 5 masks every competitor,
  // row 6 masks the label (+inf loss); rows 7 and 8 hold NaN and +inf (invalid); row 9 is
  // ignored (labels).
  for (std::size_t i = 1; i <= 4; ++i)
    for (std::size_t j = 0; j < c; ++j)
      if (uniform(rng) < 0.4 && j != static_cast<std::size_t>(labels[i]) && !mask[i * c + j]) {
        z[i * c + j] = -kInf;
        probs[i * c + j] = 0;
      }
  const bool label_kind = kind == kLabels || kind == kSmoothed;
  if (label_kind) {
    for (std::size_t j = 0; j < c; ++j)
      if (j != static_cast<std::size_t>(labels[5])) z[5 * c + j] = -kInf;
    if (kind == kLabels) z[6 * c + static_cast<std::size_t>(labels[6])] = -kInf;
  }
  z[7 * c + 1] = std::numeric_limits<double>::quiet_NaN();
  z[8 * c] = kInf;
  const int64_t ignore = -100;
  if (kind == kLabels) labels[9] = ignore;

  const std::vector<S> scores = stored<S>(z);
  const std::vector<double> zs = widened(scores);
  std::vector<double> values(n);
  std::vector<G> grads(n * c);
  std::vector<double> up(n);
  for (double& u : up) u = uniform(rng) < 0.5 ? -1.5 : 0.75;
  const double eps = kind == kSmoothed ? 0.1 : 0.0;
  sdca::kernel::Loss ce = {sdca::kernel::cross_entropy, 1, 0, 1, eps};
  const sdca::kernel::Loss ml = {sdca::kernel::multilabel_cross_entropy, 1, 0, 1, 0};
  sdca::kernel::Execution e = {precision, 1, false};
  sdca::kernel::Result r;
  const sdca::kernel::Labels labelled = {labels.data(), weights.data(), ignore, kind == kLabels};
  if (kind == kMultilabel) {
    const sdca::kernel::Mask t = {mask.data()};
    r = eval(ml, e, n, c, scores.data(), t, values.data(), grads.data(), 1.0, up.data());
  } else if (kind == kProbabilities) {
    const sdca::kernel::Probabilities<double> t = {probs.data(), weights.data()};
    r = eval(ce, e, n, c, scores.data(), t, values.data(), grads.data(), 1.0, up.data());
  } else {
    r = eval(ce, e, n, c, scores.data(), labelled, values.data(), grads.data(), 1.0, up.data());
  }
  check(r.status == 0 && r.invalid_rows == 2 && r.first_invalid == 7, "invalid count", c, 0,
        double(r.invalid_rows), 2);

  // The same call on four threads, in place where G = S: the same bits.
  if (kind == kLabels) {
    std::vector<G> inplace(n * c);
    for (std::size_t i = 0; i < n * c; ++i) inplace[i] = to_storage<G>(to_work<double>(scores[i]));
    std::vector<double> v4(n);
    const sdca::kernel::Execution e4 = {precision, 4, false};
    eval(ce, e4, n, c, inplace.data(), labelled, v4.data(), inplace.data(), 1.0, up.data());
    bool same = true;
    for (std::size_t i = 0; i < n; ++i)
      same = same && (std::memcmp(&v4[i], &values[i], sizeof(double)) == 0 || i == 7 || i == 8);
    if (sizeof(G) == sizeof(S))
      same = same && std::memcmp(inplace.data(), grads.data(), grads.size() * sizeof(G)) == 0;
    check(same, "threads and in place", c, 0, 0, 0);
    // check_rows: nothing written.
    std::vector<double> untouched(n, 7.0);
    const sdca::kernel::Execution ec = {precision, 2, true};
    const sdca::kernel::Result rc =
        eval(ce, ec, n, c, scores.data(), labelled, untouched.data(), static_cast<G*>(0));
    check(rc.status == -2 && rc.invalid_rows == 2 && untouched[0] == 7.0, "check_rows", c, 0,
          rc.status, -2);
  }

  for (std::size_t i = 0; i < n; ++i) {
    std::vector<double> row(zs.begin() + i * c, zs.begin() + (i + 1) * c);
    const bool invalid = i == 7 || i == 8;
    if (invalid) {
      check(std::isnan(values[i]), "invalid value", c, i, values[i], kNaN);
      check(std::isnan(to_work<double>(grads[i * c])), "invalid gradient", c, i, 0, kNaN);
      continue;
    }
    std::vector<R> u(c, 0);
    double scale = up[i];
    if (kind == kLabels) {
      if (labels[i] == ignore) {
        check(values[i] == 0, "ignored value", c, i, values[i], 0);
        for (std::size_t j = 0; j < c; ++j)
          check(to_work<double>(grads[i * c + j]) == 0, "ignored gradient", c, i, 0, 0);
        continue;
      }
      u[labels[i]] = weights[labels[i]];
    } else if (kind == kSmoothed) {
      for (std::size_t j = 0; j < c; ++j) u[j] = weights[j] * eps / c;
      u[labels[i]] += (1 - eps) * weights[labels[i]];
    } else if (kind == kProbabilities) {
      for (std::size_t j = 0; j < c; ++j) u[j] = weights[j] * probs[i * c + j];
    } else {
      R count = 0;
      for (std::size_t j = 0; j < c; ++j) count += mask[i * c + j];
      for (std::size_t j = 0; j < c; ++j) u[j] = mask[i * c + j] / count;
    }
    R want;
    std::vector<R> g;
    if (label_kind && i == 5) {
      want = 0;
      g.assign(c, 0);
      if (kind == kSmoothed) {
        // Smoothing puts weight on every masked class: +inf.
        check(values[i] == kInf, "masked smoothing", c, i, values[i], kInf);
        continue;
      }
    } else if (kind == kLabels && i == 6) {
      check(values[i] == kInf, "masked label value", c, i, values[i], kInf);
      std::vector<double> rest = row;
      std::vector<R> none(c, 0);
      R ignored;
      reference(rest, none, 0, ignored, g);
      R total = 0, m = -kInf;
      for (double x : rest) m = std::max<R>(m, x);
      for (double x : rest) total += std::exp(R(x) - m);
      const R scale_w = scale * weights[labels[i]];
      for (std::size_t j = 0; j < c; ++j) g[j] = std::exp(R(rest[j]) - m) / total * scale_w;
      g[labels[i]] = -scale_w;
      want = kInf;
    } else {
      reference(row, u, scale, want, g);
    }
    if (want != kInf)
      check(std::fabs(values[i] - double(want)) <=
                value_tol * std::max<double>(1, std::fabs(double(want))),
            "value", c, i, values[i], double(want));
    double largest = 0;
    for (std::size_t j = 0; j < c; ++j) {
      largest = std::max<double>(largest, std::fabs(double(g[j])));
      // T softmax(z) - u cancels: its error is relative to u.
      largest = std::max<double>(largest, std::fabs(double(u[j] * scale)));
    }
    for (std::size_t j = 0; j < c; ++j) {
      const double got = to_work<double>(grads[i * c + j]);
      check(std::fabs(got - double(g[j])) <= grad_tol * std::max(largest, 1e-300), "gradient", c, i,
            got, double(g[j]));
      if (g[j] == 0) check(got == 0 && !std::signbit(got), "+0.0 gradient", c, i, got, 0);
    }
  }
}

// An exponential within rel of want (exact to a unit of the least subnormal below the normal
// range, and max where want overflows T).
template <typename T>
bool near_exp(T e, double want, double rel) {
  const T most = std::numeric_limits<T>::max();
  if (want > static_cast<double>(most)) return e == most;
  return std::fabs(static_cast<double>(e) - want) <=
         rel * want + static_cast<double>(std::numeric_limits<T>::denorm_min());
}

void lane_math() {
  typedef sdca::native::core::Native<double>::type D;
  typedef sdca::native::core::Native<float>::type F;
  // Through the subnormal range to +0.0, and to overflow (the reduction's fast range is
  // [-708.05, 709.44) in double, [-86.99, 88.38) in float).
  const double xs[] = {-800, -745.5, -745, -720, -708.4, -708.05, -708,   -100,  -1,    -1e-9, 0,
                       1e-9, 0.5,    1,    20,   700,    709.43,  709.44, 709.7, 709.8, 710,   800};
  for (double x : xs) {
    const double e = sdca::native::core::exp(D::splat(x))[0], want = std::exp(x);
    check(near_exp(e, want, 4e-16), "exp", 0, 0, e, want);
    const double nonpositive = sdca::native::core::exp_nonpositive(D::splat(x))[0];
    check(x > 0 || nonpositive == e, "exp_nonpositive", 0, 0, nonpositive, e);
    const double m = sdca::native::core::expm1(D::splat(x))[0], wm = std::expm1(x);
    check(m == wm || std::fabs(m - wm) <= 4e-16 * std::fabs(wm), "expm1", 0, 0, m, wm);
    if (x > -1) {
      const double l = sdca::native::core::log1p(D::splat(x))[0], wl = std::log1p(x);
      check(std::fabs(l - wl) <= 4e-16 * std::fabs(wl), "log1p", 0, 0, l, wl);
    }
    if (x > 0) {
      const double l = sdca::native::core::log(D::splat(x))[0], wl = std::log(x);
      check(std::fabs(l - wl) <= 4e-16 * std::fabs(wl), "log", 0, 0, l, wl);
    }
  }
  const float fs[] = {-200,    -104.5f, -104,  -103.5f, -100,   -90, -87.3365479f, -87,
                      -86.99f, -86.6f,  -20,   -1,      0,      1,   20,           88,
                      88.37f,  88.38f,  88.5f, 88.72f,  88.73f, 100};
  for (float f : fs) {
    const float ef = sdca::native::core::exp(F::splat(f))[0];
    const double wf = std::exp(double(f));
    check(near_exp(ef, wf, 3e-7), "float exp", 0, 0, double(ef), wf);
    const float mf = sdca::native::core::exp_or_expm1(F::splat(f), F::splat(f) != F::splat(f))[0];
    check(near_exp(mf, wf, 3e-7), "float exp_or_expm1", 0, 0, double(mf), wf);
  }
  const float negative[] = {-INFINITY, -105, -104, -103.5f, -100, -87.4f, -80, -1, 0};
  for (float x : negative) {
    const float e = sdca::native::core::exp(F::splat(x))[0];
    const float nonpositive = sdca::native::core::exp_nonpositive_full(F::splat(x))[0];
    check(nonpositive == e, "full float exp_nonpositive", 0, 0, double(nonpositive), double(e));
  }
  check(sdca::native::core::log(D::splat(5e-320))[0] == std::log(5e-320) ||
            std::fabs(sdca::native::core::log(D::splat(5e-320))[0] - std::log(5e-320)) < 1e-13,
        "subnormal log", 0, 0, sdca::native::core::log(D::splat(5e-320))[0], std::log(5e-320));
  check(sdca::native::core::exp(D::splat(-kInf))[0] == 0, "exp(-inf)", 0, 0, 0, 0);
  check(sdca::native::core::exp_nonpositive(F::splat(-INFINITY))[0] == 0, "float exp(-inf)", 0, 0,
        0, 0);
}

void selection(std::mt19937& rng) {
  std::normal_distribution<double> normal(0.0, 1.0);
  const std::size_t sizes[] = {3, 17, 100, 1000};
  const std::size_t ks[] = {1, 2, 5, 40};
  for (std::size_t n : sizes)
    for (std::size_t k : ks) {
      if (k >= n) continue;
      std::vector<float> x(n);
      for (float& v : x) v = static_cast<float>(std::round(normal(rng) * 4));  // ties
      const std::size_t skip = n / 2;
      std::vector<double> sorted;
      std::vector<float> topf(k);
      sdca::native::core::top_k<sdca::native::core::Native<float>::type>(x.data(), n, skip, k,
                                                                         topf.data());
      for (std::size_t j = 0; j < n; ++j)
        if (j != skip) sorted.push_back(double(x[j]));
      sdca::native::core::sort_largest(sorted.data(), k, sorted.data() + sorted.size());
      for (std::size_t i = 0; i < k; ++i)
        check(double(topf[i]) == sorted[i], "top_k", n, i, double(topf[i]), sorted[i]);
      const float last = topf[k - 1];
      const std::size_t nth = sdca::native::core::count_of(topf.data(), k, last);
      std::size_t want = n, seen = 0;
      for (std::size_t j = 0; j < n && want == n; ++j)
        if (j != skip && x[j] == last && ++seen == nth) want = j;
      const std::size_t got = sdca::native::core::index_of<sdca::native::core::Native<float>::type>(
          x.data(), n, skip, last, nth);
      check(got == want, "index_of", n, k, double(got), double(want));
    }
}

// multilabel_cross_entropy where the positives' softmax nears 1 / p: tied positives at z0
// beside negatives of total mass N = sum exp(z - z0) have the gradient -(N / p) / (p + N),
// which the plain softmax(z) - 1 / p cancels to a few correct digits.
void cancellation() {
  for (int precision = 0; precision <= 2; precision += 2) {
    const std::size_t c = 12, p = 3;
    std::vector<double> z(c);
    std::vector<uint8_t> mask(c, 0);
    double mass = 0;
    for (std::size_t j = 0; j < c; ++j) {
      if (j % 4 == 1) {
        z[j] = 7.0;
        mask[j] = 1;
      } else {
        z[j] = -9.0 - 0.5 * static_cast<double>(j);
        mass += std::exp(z[j] - 7.0);
      }
    }
    const double want = -(mass / p) / (p + mass);
    const sdca::kernel::Loss ml = {sdca::kernel::multilabel_cross_entropy, 1, 0, 1, 0};
    const sdca::kernel::Execution e = {precision, 1, false};
    const sdca::kernel::Mask t = {mask.data()};
    std::vector<double> g(c);
    std::vector<float> zf(z.begin(), z.end()), gf(c);
    double v;
    eval(ml, e, 1, c, z.data(), t, &v, g.data());
    eval(ml, e, 1, c, zf.data(), t, &v, gf.data());
    for (std::size_t j = 1; j < c; j += 4) {
      check(std::fabs(g[j] - want) <= 1e-14 * std::fabs(want), "anchored double", c, j, g[j], want);
      if (precision == 0)
        check(std::fabs(double(gf[j]) - want) <= 1e-6 * std::fabs(want), "anchored float", c, j,
              double(gf[j]), want);
    }
  }
}

// A call large enough to take threads: the same bits on one and on eight.
void threads(std::mt19937& rng) {
  const std::size_t n = 20000, c = 300;
  std::normal_distribution<float> normal(0.0f, 3.0f);
  std::vector<float> z(n * c), g1(n * c), g8(n * c);
  for (float& x : z) x = normal(rng);
  z[123 * c + 7] = NAN;
  std::vector<int64_t> labels(n);
  for (int64_t& y : labels) y = static_cast<int64_t>(rng() % c);
  std::vector<double> v1(n), v8(n);
  const sdca::kernel::Loss ce = {sdca::kernel::cross_entropy, 1, 0, 1, 0};
  const sdca::kernel::Labels t = {labels.data(), 0, 0, false};
  const sdca::kernel::Execution one = {0, 1, false}, eight = {0, 8, false};
  const sdca::kernel::Result r1 = eval(ce, one, n, c, z.data(), t, v1.data(), g1.data());
  const sdca::kernel::Result r8 = eval(ce, eight, n, c, z.data(), t, v8.data(), g8.data());
  check(r1.invalid_rows == 1 && r8.invalid_rows == 1 && r8.first_invalid == 123, "threads invalid",
        c, 0, double(r8.invalid_rows), 1);
  check(std::memcmp(g1.data(), g8.data(), g1.size() * sizeof(float)) == 0 &&
            std::memcmp(v1.data(), v8.data(), v1.size() * sizeof(double)) == 0,
        "threads bits", c, 0, 0, 0);
}

}  // namespace

int main() {
  std::mt19937 rng(7);
  lane_math();
  selection(rng);
  int isas[] = {0, 0, 0};
  int count = 0;
#if defined(__x86_64__)
  isas[count++] = 1;
  isas[count++] = 3;  // Rosetta runs AVX2 but its CPUID does not say so
  if (std::getenv("SDCA_TEST_AVX512")) isas[count++] = 4;
#else
  isas[count++] = 0;
#endif
  const std::size_t classes[] = {2, 3, 7, 16, 33, 100, 1000, 5000};
  const Kind kinds[] = {kLabels, kSmoothed, kProbabilities, kMultilabel};
  for (int k = 0; k < count; ++k) {
    test_isa = isas[k];
    threads(rng);
    cancellation();
    for (Kind kind : kinds)
      for (std::size_t c : classes) {
        run<double, double>(kind, c, 0, 1e-13, 1e-13, rng);
        run<float, float>(kind, c, 0, 2e-6, 2e-6, rng);
        run<float, float>(kind, c, 2, 1e-13, 2e-7, rng);
        run<half, float>(kind, c, 0, 2e-6, 2e-6, rng);
        run<bfloat16, float>(kind, c, 0, 2e-6, 2e-6, rng);
        run<half, half>(kind, c, 0, 2e-6, 5e-3, rng);
      }
    std::printf("isa %d: %d failures so far\n", test_isa, failures);
  }
  std::printf("%s (%d failures), run-time isa %s\n", failures ? "FAILED" : "ok", failures,
              sdca::kernel::isa_name());
  return failures ? 1 : 0;
}
