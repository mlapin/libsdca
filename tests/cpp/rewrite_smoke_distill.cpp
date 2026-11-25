// Smoke test of the distillation and proximal rows: sdca::kernel::evaluate against a naive long
// double reference on random rows, rows of equal student and teacher, masked and teacher-masked
// rows and invalid rows, in float and double storage, float and double work, one and several
// threads.
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "sdca/kernel.h"

namespace {

// Long double where it is wider (not on arm64 macOS: references there are double).
typedef long double R;
const double kInf = std::numeric_limits<double>::infinity();
const double kNaN = std::numeric_limits<double>::quiet_NaN();
int failures = 0;

void check(bool ok, const char* what, int kind, std::size_t c, std::size_t row, double got,
           double want) {
  if (ok) return;
  if (++failures <= 30)
    std::printf("FAIL %s: kind %d classes %zu row %zu: got %.17g want %.17g\n", what, kind, c, row,
                got, want);
}

// log softmax of x over its finite entries (-inf elsewhere).
std::vector<R> log_softmax(const std::vector<R>& x) {
  R m = -kInf, sum = 0;
  for (R v : x) m = std::max(m, v);
  for (R v : x) sum += std::exp(v - m);
  std::vector<R> l(x.size());
  for (std::size_t j = 0; j < x.size(); ++j) l[j] = x[j] - m - std::log(sum);
  return l;
}

// The entropic proximal step on the kept classes: alpha pi + log pi = v + alpha p - t with
// sum pi = 1; returns -<p, log pi> + alpha / 2 |p - pi|^2 and pi.
R prox_reference(const std::vector<R>& v, const std::vector<R>& p, const std::vector<bool>& kept,
                 R alpha, std::vector<R>& pi) {
  const std::size_t n = v.size();
  std::vector<R> log_pi(n, -kInf);
  pi.assign(n, 0);
  // log x of alpha x + log x = c: Newton from the right on the convex increasing function.
  auto solve = [&](R c) {
    R y = c;
    for (int it = 0; it < 2000; ++it) {
      const R e = alpha * std::exp(y), step = (e + y - c) / (e + 1);
      y -= step;
      if (std::fabs(step) <= 1e-19L * (1 + std::fabs(y))) break;
    }
    return y;
  };
  auto mass = [&](R t) {
    R total = 0;
    for (std::size_t j = 0; j < n; ++j)
      if (kept[j]) total += std::exp(log_pi[j] = solve(v[j] + alpha * p[j] - t));
    return total;
  };
  R lo = -1, hi = 1;
  while (mass(lo) < 1) lo = 2 * lo - 1;
  while (mass(hi) > 1) hi = 2 * hi + 1;
  for (int it = 0; it < 200 && lo < hi; ++it) {
    const R mid = (lo + hi) / 2;
    (mass(mid) > 1 ? lo : hi) = mid;
  }
  mass(hi);
  R total = 0, value = 0;
  for (std::size_t j = 0; j < n; ++j)
    if (kept[j]) total += std::exp(log_pi[j]);
  for (std::size_t j = 0; j < n; ++j) {
    if (!kept[j]) continue;
    log_pi[j] -= std::log(total);
    pi[j] = std::exp(log_pi[j]);
    if (p[j] > 0) value -= p[j] * log_pi[j];
    value += alpha / 2 * (p[j] - pi[j]) * (p[j] - pi[j]);
  }
  return value;
}

template <typename S>
std::vector<S> stored(const std::vector<double>& x) {
  return std::vector<S>(x.begin(), x.end());
}

enum Kind { kProximalCE = 7, kDistillation = 8, kProximalDistillation = 9 };

template <typename S>
void run(Kind kind, std::size_t c, int precision, double value_tol, double grad_tol,
         std::mt19937& rng) {
  const std::size_t n = 37;
  const double temperature = 2.5, gamma = 0.7;
  std::normal_distribution<double> normal(0.0, 3.0);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::vector<double> z(n * c), t(n * c);
  std::vector<int64_t> labels(n);
  for (std::size_t i = 0; i < n * c; ++i) {
    z[i] = normal(rng);
    t[i] = i % 3 == 0 ? normal(rng) : z[i] + 0.3 * normal(rng);
  }
  for (std::size_t i = 0; i < n; ++i) labels[i] = static_cast<int64_t>(rng() % c);
  const bool teacher = kind != kProximalCE;
  // Row 1: equal student and teacher; rows 2 and 3 mask classes in both (and, row 3, classes
  // only the teacher masks); row 4 masks a class only the student masks (proximal CE: the
  // label), loss +inf with the limit of the gradient; rows 5 and 6 hold NaN and +inf, invalid;
  // row 7 masks every teacher class (proximal CE: every class but the label, loss 0).
  for (std::size_t j = 0; j < c; ++j) t[c + j] = z[c + j];
  for (std::size_t i = 2; i <= 3; ++i)
    for (std::size_t j = 0; j < c; ++j) {
      if (j == static_cast<std::size_t>(labels[i])) continue;
      const double u = uniform(rng);
      if (u < 0.3) z[i * c + j] = t[i * c + j] = -kInf;
      if (i == 3 && u > 0.7) t[i * c + j] = -kInf;
    }
  z[4 * c + static_cast<std::size_t>(labels[4])] = -kInf;
  z[5 * c + 1] = std::numeric_limits<double>::quiet_NaN();
  z[6 * c] = kInf;
  for (std::size_t j = 0; j < c; ++j) {
    if (teacher)
      t[7 * c + j] = -kInf;
    else if (j != static_cast<std::size_t>(labels[7]))
      z[7 * c + j] = -kInf;
  }

  const std::vector<S> scores = stored<S>(z), teachers = stored<S>(t);
  std::vector<double> values(n), up(n);
  std::vector<S> grads(n * c);
  for (double& u : up) u = uniform(rng) < 0.5 ? -1.5 : 0.75;
  const sdca::kernel::Loss loss = {kind, 0, gamma, temperature, 0.0};
  const sdca::kernel::Execution e = {precision, 1, false};
  const sdca::kernel::Teacher<S> tt = {teachers.data()};
  const sdca::kernel::Labels lt = {labels.data(), 0, 0, false};
  sdca::kernel::Result r =
      teacher ? sdca::kernel::evaluate(loss, e, n, c, scores.data(), tt, values.data(),
                                       grads.data(), 1.0, up.data())
              : sdca::kernel::evaluate(loss, e, n, c, scores.data(), lt, values.data(),
                                       grads.data(), 1.0, up.data());
  const std::size_t invalid = teacher ? 3 : 2;
  check(r.status == 0 && r.invalid_rows == invalid && r.first_invalid == 5, "invalid count", kind,
        c, 0, double(r.invalid_rows), double(invalid));

  // Four threads, in place, and the values alone: the same bits.
  {
    std::vector<S> inplace = scores;
    std::vector<double> v4(n), alone(n);
    const sdca::kernel::Execution e4 = {precision, 4, false};
    if (teacher) {
      sdca::kernel::evaluate(loss, e4, n, c, inplace.data(), tt, v4.data(), inplace.data(), 1.0,
                             up.data());
      sdca::kernel::evaluate(loss, e, n, c, scores.data(), tt, alone.data(), static_cast<S*>(0));
    } else {
      sdca::kernel::evaluate(loss, e4, n, c, inplace.data(), lt, v4.data(), inplace.data(), 1.0,
                             up.data());
      sdca::kernel::evaluate(loss, e, n, c, scores.data(), lt, alone.data(), static_cast<S*>(0));
    }
    bool same = std::memcmp(inplace.data(), grads.data(), grads.size() * sizeof(S)) == 0;
    for (std::size_t i = 0; i < n; ++i) {
      same = same && std::memcmp(&v4[i], &values[i], sizeof(double)) == 0;
      check(
          alone[i] == values[i] ||
              std::fabs(alone[i] - values[i]) <= value_tol * std::max(1.0, std::fabs(values[i])) ||
              (alone[i] != alone[i] && values[i] != values[i]),
          "value alone", kind, c, i, alone[i], values[i]);
    }
    check(same, "threads and in place", kind, c, 0, 0, 0);
  }

  for (std::size_t i = 0; i < n; ++i) {
    const bool bad = i == 5 || i == 6 || (teacher && i == 7);
    if (bad) {
      check(values[i] != values[i], "invalid value", kind, c, i, values[i], kNaN);
      check(double(grads[i * c]) != double(grads[i * c]), "invalid gradient", kind, c, i, 0, kNaN);
      continue;
    }
    std::vector<R> zs(c), ts(c);
    std::vector<bool> kept(c);
    for (std::size_t j = 0; j < c; ++j) {
      zs[j] = R(double(scores[i * c + j])), ts[j] = R(double(teachers[i * c + j]));
      kept[j] = zs[j] > -kInf;
    }
    R want = 0;
    std::vector<R> g(c, 0), allowed(c, 0);
    const R scale = up[i];
    if (kind == kProximalCE) {
      std::vector<R> p(c, 0), pi;
      p[labels[i]] = 1;
      std::size_t count = 0;
      for (std::size_t j = 0; j < c; ++j) count += kept[j];
      if (count > 1 || !kept[labels[i]]) {
        // A masked label: +inf, the limit step on the classes left and -1 at the label.
        want = prox_reference(zs, p, kept, gamma, pi);
        if (!kept[labels[i]]) want = kInf;
        for (std::size_t j = 0; j < c; ++j) g[j] = ((kept[j] ? pi[j] : 0) - p[j]) * scale;
      }
    } else {
      std::vector<R> zt(c), tt2(c);
      for (std::size_t j = 0; j < c; ++j) zt[j] = zs[j] / temperature, tt2[j] = ts[j] / temperature;
      const std::vector<R> log_q = log_softmax(zt), log_p = log_softmax(tt2);
      std::vector<R> p(c), q(c);
      for (std::size_t j = 0; j < c; ++j) p[j] = std::exp(log_p[j]), q[j] = std::exp(log_q[j]);
      if (kind == kDistillation) {
        for (std::size_t j = 0; j < c; ++j)
          if (p[j] > 0) want -= p[j] * log_q[j];
        for (std::size_t j = 0; j < c; ++j) g[j] = (q[j] - p[j]) / temperature * scale;
        // Float work: 64 float eps of (p + q) / T (docs/API.md, Known limits).
        if (sizeof(S) < sizeof(double) && precision != 2)
          for (std::size_t j = 0; j < c; ++j)
            allowed[j] = 64 * R(FLT_EPSILON) * (p[j] + q[j]) / temperature * std::fabs(scale);
      } else {
        std::vector<R> pi;
        want = prox_reference(zt, p, kept, gamma, pi);
        for (std::size_t j = 0; j < c; ++j)
          if (!kept[j] && p[j] > 0) want = kInf;
        for (std::size_t j = 0; j < c; ++j)
          g[j] = ((kept[j] ? pi[j] : 0) - p[j]) / temperature * scale;
      }
    }
    // Equal student and teacher: the gradient is zero (the reference rounds).
    if (teacher && i == 1) g.assign(c, 0);
    check(values[i] == double(want) || std::fabs(values[i] - double(want)) <=
                                           value_tol * std::max<double>(1, std::fabs(double(want))),
          "value", kind, c, i, values[i], double(want));
    double largest = 0;
    for (std::size_t j = 0; j < c; ++j)
      largest = std::max<double>(largest, std::fabs(double(g[j])));
    for (std::size_t j = 0; j < c; ++j) {
      const double got = double(grads[i * c + j]);
      check(std::fabs(got - double(g[j])) <=
                grad_tol * std::max(largest, 1e-300) + double(allowed[j]),
            "gradient", kind, c, i, got, double(g[j]));
      if ((!kept[j] && g[j] == 0) || (i == 1 && teacher))
        check(got == 0 && !std::signbit(got), "+0.0 gradient", kind, c, i, got, 0);
    }
  }
}

// Exact limiting softmaxes, including a representable weighted loss when the
// unweighted student log-probability overflows.
void extreme_distillation() {
  const double m = DBL_MAX;
  const double tiny = std::ldexp(1.0, -880);
  const double p = 1 / (1 + std::exp(-1.0));
  const double tail = std::exp(-400.0);
  struct Case {
    std::vector<double> z, t, grad;
    double temperature, value;
  };
  const Case cases[] = {
      {{m, -m, -m}, {-m, m, m}, {2, -1, -1}, 0.5, kInf},
      {{m, -m, m, m}, {0, 0, 0, 0}, {1.0 / 12, -0.25, 1.0 / 12, 1.0 / 12}, 1, m / 2},
      {{m, -m}, {0, -800}, {0, 0}, 1, 2 * (m * tail) * tail},
      {{0, -tiny, -m / 4},
       {-kInf, 0, -kInf},
       {p / tiny, -p / tiny, 0},
       tiny,
       1 + std::log1p(std::exp(-1.0))},
  };
  const sdca::kernel::Execution execution = {2, 1, false};
  for (const Case& test : cases) {
    const sdca::kernel::Loss loss = {kDistillation, 0, 0.0, test.temperature, 0.0};
    const sdca::kernel::Teacher<double> target = {test.t.data()};
    std::vector<double> grad(test.z.size());
    for (bool backward : {false, true}) {
      double value;
      sdca::kernel::evaluate(loss, execution, 1, grad.size(), test.z.data(), target, &value,
                             backward ? grad.data() : static_cast<double*>(0));
      check(value == test.value || (std::isfinite(test.value) &&
                                    std::fabs(value - test.value) <= 8 * DBL_EPSILON * test.value),
            "extreme value", kDistillation, grad.size(), 0, value, test.value);
      if (backward)
        for (std::size_t j = 0; j < grad.size(); ++j)
          check(std::fabs(grad[j] - test.grad[j]) <= 8 * DBL_EPSILON * std::fabs(test.grad[j]),
                "extreme gradient", kDistillation, grad.size(), j, grad[j], test.grad[j]);
    }
  }
}

// At a gap of 754 each teacher probability underflows before it is weighted
// by the student gap. The cross term summed over 4096 competitors is still
// representable. A 430-digit evaluation rounds to 218 or 80 subnormal units.
void grouped_subnormal_value() {
  const std::size_t classes = 4097;
  const double quantum = std::numeric_limits<double>::denorm_min();
  const sdca::kernel::Loss loss = {kDistillation, 0, 0.0, 1.0, 0.0};
  const sdca::kernel::Execution execution = {2, 1, false};
  std::vector<double> student(classes, -754), gradient(classes);
  student[0] = 0;
  for (int offset = 0; offset <= 1; ++offset) {
    std::vector<double> teacher(classes, -754 - offset);
    teacher[0] = 0;
    const sdca::kernel::Teacher<double> target = {teacher.data()};
    const double expected = (offset == 0 ? 218 : 80) * quantum;
    for (bool backward : {false, true}) {
      double value = -1;
      sdca::kernel::evaluate(loss, execution, 1, classes, student.data(), target, &value,
                             backward ? gradient.data() : static_cast<double*>(0));
      check(value == expected, "summed subnormal distillation", kDistillation, classes, offset,
            value, expected);
    }
  }
}

}  // namespace

int main() {
  extreme_distillation();
  grouped_subnormal_value();
  std::mt19937 rng(11);
  const std::size_t classes[] = {2, 3, 7, 16, 33, 100, 1000};
  const Kind kinds[] = {kDistillation, kProximalDistillation, kProximalCE};
  for (Kind kind : kinds)
    for (std::size_t c : classes) {
      run<double>(kind, c, 0, 1e-12, 1e-12, rng);
      run<float>(kind, c, 0, kind == kProximalDistillation ? 1e-12 : 3e-6, 3e-6, rng);
      run<float>(kind, c, 2, 1e-12, 2e-7, rng);
    }
  std::printf("%s (%d failures), isa %s\n", failures ? "FAILED" : "ok", failures,
              sdca::kernel::isa_name());
  return failures ? 1 : 0;
}
